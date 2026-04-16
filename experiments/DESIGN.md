# GEPAFewShot Design Notes

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*

This document describes how `GEPAFewShot` extends vanilla `GEPA` with joint
instruction + demonstration optimization.  Every claim is tied to a specific
code location in `dspy/teleprompt/gepa/gepa_fewshot.py` (abbreviated
`gepa_fewshot.py` below) and `dspy/teleprompt/gepa/gepa.py` (`gepa.py`).

---

## 1. Integration boundary: what the upstream `gepa` package owns

The upstream [`gepa`](https://github.com/gepa-ai/gepa) PyPI package provides the
entire optimization engine.  DSPy's `GEPA.compile()` delegates to it via a
single call ([`gepa.py:560`](../dspy/teleprompt/gepa/gepa.py#L560)):

```python
gepa_result: GEPAResult = optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=valset,
    adapter=adapter,           # ← sole integration point
    ...
)
```

Everything inside `gepa.optimize()` — the Pareto frontier, candidate selection,
budget tracking, LM reflection orchestration, checkpointing, W&B/MLflow logging
— is owned by the upstream package.  DSPy never touches it.

The upstream package interacts with DSPy through **exactly two adapter methods**:

| Method | Called when | Returns |
|---|---|---|
| `adapter.build_program(candidate)` | A candidate must be evaluated | Runnable `dspy.Module` |
| `adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)` | GEPA's LM reflection proposes a mutation | `dict[str, str]` — new instructions |

`DspyAdapter` (in `gepa_utils.py`) provides the baseline implementations of
both.  `GEPAFewShotAdapter` subclasses `DspyAdapter` and overrides only these
two methods, always calling `super()` first to preserve the upstream chain.

---

## 2. The demo pool: built once, outside the loop

Before `gepa.optimize()` is called, `GEPAFewShot.compile()` bootstraps a pool
of candidate demonstrations ([`gepa_fewshot.py:476`](../dspy/teleprompt/gepa/gepa_fewshot.py#L476)):

```python
demo_pool = self._bootstrap_demo_pool(student, trainset)
```

### 2.1 Pool construction (`_bootstrap_demo_pool`, line 371)

`_bootstrap_demo_pool` runs `BootstrapFewShot` once on a deep-copy of the
student program ([line 395](../dspy/teleprompt/gepa/gepa_fewshot.py#L395)):

```python
bfs = BootstrapFewShot(
    metric=bfs_metric,
    max_bootstrapped_demos=self.max_bootstrapped_demos,
    max_labeled_demos=0,
)
bfs.compile(student.deepcopy(), trainset=trainset)
```

The pool type is ([line 61](../dspy/teleprompt/gepa/gepa_fewshot.py#L61)):

```python
DemoPool = dict[str, list[tuple[Example, float]]]
#                          ↑ Example   ↑ quality score
```

Two demo tiers are added per predictor ([lines 399–408](../dspy/teleprompt/gepa/gepa_fewshot.py#L399-L408)):

| Tier | Source | Quality score |
|---|---|---|
| Bootstrapped | `bfs.name2traces[name]` — traces that passed the metric | **1.0** |
| Labeled | `bfs.validation[:max_labeled_demos]` — raw trainset examples | **0.5** |

The score difference drives the `metric_based` mutation strategy (§4.2).

### 2.2 Metric bridging (`_wrap_metric_for_bootstrap`, line 356)

GEPA's metric protocol is 5-arg: `(gold, pred, trace, pred_name, pred_trace)`.
`BootstrapFewShot` expects a 3-arg metric: `(gold, pred, trace)`.
The wrapper translates at the boundary ([lines 363–367](../dspy/teleprompt/gepa/gepa_fewshot.py#L363-L367)):

```python
def wrapped(gold, pred, trace=None) -> float:
    result = five_arg(gold, pred, trace, None, None)
    if hasattr(result, "score"):
        return float(result["score"])
    return float(result)
```

---

## 3. The demo registry: shadowing each instruction candidate

`GEPAFewShotAdapter` maintains a side-table ([line 96](../dspy/teleprompt/gepa/gepa_fewshot.py#L96)):

```python
self._demo_registry: dict[tuple, dict[str, list[Example]]] = {}
#                          ↑ candidate key  ↑ predictor → demos
```

The **candidate key** is a sorted, frozen tuple of `(predictor_name, instruction)`
pairs ([line 104](../dspy/teleprompt/gepa/gepa_fewshot.py#L104)):

```python
def _candidate_key(self, candidate: dict[str, str]) -> tuple:
    return tuple(sorted(candidate.items()))
```

This makes the key order-independent and hashable.

Every instruction candidate that GEPA ever proposes gets a companion demo set
registered under this key.  The seed candidate's demo set is initialised lazily
on first access via `_get_demos()` ([lines 106–124](../dspy/teleprompt/gepa/gepa_fewshot.py#L106-L124)):
random draw of `k_demos` examples from the pool (ignoring scores at init time).

---

## 4. The incremental mutation loop

The upstream package drives the loop; `GEPAFewShotAdapter` intercepts at two
points each iteration.

### 4.1 Evaluation: `build_program` (line 135)

```python
def build_program(self, candidate: dict[str, str]) -> Module:
    new_prog = super().build_program(candidate)   # 1. set instructions (upstream)
    demos = self._get_demos(candidate)            # 2. look up companion demos
    for name, pred in new_prog.named_predictors():
        if name in demos:
            pred.demos = list(demos[name])        # 3. inject demos into predictors
    return new_prog
```

Every program the upstream package evaluates carries both its instruction string
*and* its companion demo set.  The Pareto scores GEPA tracks therefore reflect
the joint (instruction, demos) pair, not instruction alone.

### 4.2 Mutation: `propose_new_texts` (line 144)

```python
def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
    # Step 1 — instruction mutation (upstream LM reflection, fully preserved)
    new_instructions = super().propose_new_texts(
        candidate, reflective_dataset, components_to_update
    )
    # Step 2 — demo mutation (our extension)
    current_demos = self._get_demos(candidate)
    new_demos = {}
    for name, demo_list in current_demos.items():
        if name in components_to_update:
            pool = self.demo_pool.get(name, [])
            new_demos[name] = self._mutate_demos(
                current=list(demo_list),
                pool=pool,
                reflective_examples=reflective_dataset.get(name, []),
            )
        else:
            new_demos[name] = list(demo_list)   # carry over unchanged

    # Step 3 — register the new demo set under the new candidate's key
    self._register_demos(new_instructions, new_demos)
    return new_instructions
```

Key invariants:
- `super().propose_new_texts()` is called first and its return value is
  returned unchanged.  The upstream package sees no difference.
- Demo mutation tracks `components_to_update` (the predictor names GEPA chose
  to mutate this iteration).  Predictors not selected by GEPA carry their demos
  forward unchanged.
- The new demo set is registered before `build_program()` is called with the
  new candidate, so the lookup in §4.1 always finds it.

### 4.3 Mutation strategies (`_mutate_demos`, line 182)

`_mutate_demos` dispatches to one of two strategies:

**`random` ([line 199](../dspy/teleprompt/gepa/gepa_fewshot.py#L199)):**
Uniformly picks one of three primitive operations — add, remove, swap — subject
to availability and the `k_demos` cap.  Pool scores are ignored.

**`metric_based` ([line 232](../dspy/teleprompt/gepa/gepa_fewshot.py#L232)):**
Same three operations, but pool-side selection is score-weighted
(`rng.choices(..., weights=pool_scores)`).  Bootstrapped demos (score 1.0) are
2× more likely to be selected than labeled examples (score 0.5).  Removal is
kept uniform: we have no per-demo score for the currently active set.

Both strategies keep the demo count at or below `k_demos`, ensuring the token
overhead per candidate is bounded and predictable.

---

## 5. Data flow summary

```
GEPAFewShot.compile()
│
├─ _bootstrap_demo_pool()          ← runs once, outside the loop
│   ├─ BootstrapFewShot.compile()  ← single-pass trace collection
│   └─ returns DemoPool            ← {predictor: [(Example, score), ...]}
│
└─ gepa.optimize(adapter=GEPAFewShotAdapter, ...)   ← upstream loop
    │
    ├─ [each iteration]
    │   ├─ adapter.propose_new_texts()
    │   │   ├─ DspyAdapter.propose_new_texts()  ← LM reflection (upstream)
    │   │   └─ _mutate_demos()                  ← demo mutation (ours)
    │   │       registers new (instruction, demos) pair in _demo_registry
    │   │
    │   └─ adapter.build_program()
    │       ├─ DspyAdapter.build_program()  ← set instructions (upstream)
    │       └─ inject pred.demos from _demo_registry
    │
    └─ returns GEPAResult (best_candidate = best instruction dict)

adapter.build_program(gepa_result.best_candidate)
    ← final call: applies best instructions + best companion demos
```

---

## 6. Known limitation: duplicated `compile()` body

`GEPAFewShot.compile()` ([line 420](../dspy/teleprompt/gepa/gepa_fewshot.py#L420)) is
structurally identical to `GEPA.compile()` ([`gepa.py:465`](../dspy/teleprompt/gepa/gepa.py#L465))
except for:

1. The `_bootstrap_demo_pool()` call before `gepa.optimize()`.
2. Instantiating `GEPAFewShotAdapter` instead of `DspyAdapter`.

This duplication was chosen deliberately to avoid modifying the parent class
(which is provided by the upstream DSPy/GEPA codebase and should remain
untouched).

**Maintenance consequence:** if `GEPA.compile()` adds a new argument to its
`gepa.optimize()` call — for example, a new upstream kwarg or a new budget
parameter — `GEPAFewShot.compile()` will silently fall behind.  The divergence
would not be caught at import time.

**Mitigation strategy:** before every merge to `main`, run a diff between
the two `compile()` methods and verify that any new `gepa.optimize()` kwargs
in `GEPA.compile()` are also present in `GEPAFewShot.compile()`.  A one-line
comment at the top of `GEPAFewShot.compile()` already documents this:

```
# compile() — mirrors GEPA.compile() with GEPAFewShotAdapter injected
```

The correct long-term fix — should `GEPA.compile()` become more complex — is to
refactor `GEPA.compile()` to accept a factory callable for the adapter object
(`adapter_cls` or `_make_adapter()`), which would allow `GEPAFewShot` to simply
override that factory rather than duplicating the full method body.  This is not
done now to keep the parent class unmodified.

---

## 7. Experiment matrix and results infrastructure

### 7.1 Supported optimizers

`run_experiment.py` supports five optimizer IDs:

| ID | Class | Notes |
|---|---|---|
| `baseline` | — | Zero-shot eval; no optimization |
| `gepa` | `GEPA(use_merge=False)` | Vanilla GEPA, merge disabled |
| `gepa_merge` | `GEPA(use_merge=True)` | GEPA with merge enabled (upstream default) |
| `gepa_fewshot` | `GEPAFewShot(use_merge=False)` | Our extension; merge forced off |
| `miprov2` | `MIPROv2` | DSPy's Bayesian instruction+demo optimizer |

`gepa` and `gepa_merge` are distinguished to support an ablation on the merge
mechanism independent of the few-shot extension.

### 7.2 Reproducibility: `--seed`

`run_experiment.py` accepts `--seed INT`.  When provided:
- `RANDOM_SEED` (used by all GEPA / MIPROv2 internals) is set to the given value.
- Python's `random` module and NumPy are also seeded.
- The seed is included in the `run_tag` directory name and written to `config.json`
  and `results.json`.

When `--seed` is omitted, `RANDOM_SEED` falls back to `int(time.time())` (prior
behaviour), giving a unique but non-reproducible seed.

### 7.3 Per-dataset split configuration

Split sizes differ per dataset and are encoded in `run_matrix.py:DATASET_CONFIGS`:

| Dataset | train | val | test |
|---|---|---|---|
| GSM8K | 100 | 250 | 1500 (≥ full test set) |
| Iris | 15 | 35 | 50 |

`run_experiment.py` CLI flags (`--train-size`, `--val-size`, `--test-size`) override
these for ad-hoc single runs.

### 7.4 Matrix driver (`run_matrix.py`)

`run_matrix.py` enumerates all `(dataset, model, optimizer, seed)` combinations and
calls `run_experiment.py` as a subprocess for process isolation.  Runs are grouped by
model (outer loop) to minimise SGLang server swaps.

Key flags:
- `--dry-run` — print commands without executing
- `--resume` — skip combinations whose `results.json` already exists
- `--keep-going` — continue even if a run fails
- `--summary-only` — recompute `matrix_summary.json` without running new experiments
- `--models / --datasets / --optimizers / --seeds` — subset the matrix

After all runs complete, `write_summary()` scans `results_v1/` and writes:
- `matrix_summary.json` — per-group stats (mean, std, CI, runtime, per-seed breakdown)

### 7.5 `results.json` schema

Each completed run writes `results.json` with:

| Field | Type | Description |
|---|---|---|
| `run_tag` | str | Unique run identifier (includes seed) |
| `seed` | int | Random seed used |
| `dataset` / `optimizer` / `model` / `auto` | str | Run identity |
| `test_score` | float | Accuracy on test set |
| `runtime_opt_s` | float | Optimization wall-clock time (seconds) |
| `runtime_eval_s` | float | Test-set evaluation time (seconds) |
| `total_metric_calls` | int or null | Total GEPA metric calls (GEPA runs only) |
| `optimized_instructions` | dict | Final instruction per predictor |
| `demos_per_predictor` | dict | Demo count per predictor |
| `n_demos_total` | int | Sum of demos across all predictors |
| `train_size` / `val_size` / `test_size` | int | Actual split sizes used |
| `k_demos` | int | Demos per candidate (GEPAFewShot only; 0 otherwise) |

### 7.6 Statistical analysis (`analyze_results.py`)

`--aggregate` mode groups runs by `(dataset, model, optimizer)` and computes:
- Cross-seed accuracy: mean, std, 95% CI (t-distribution for n ≤ 5, z=1.96 for n > 5)
- Optimization time: mean, std, median, min, max (reported in minutes)
- Sample efficiency: mean and std of `total_metric_calls` (GEPA runs only)

Output: printed table + `aggregate_stats.json` written alongside `matrix_summary.json`.
