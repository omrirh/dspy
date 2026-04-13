# ReAct Agent Experiment — Task Bank

## Status snapshot (2026-04-02)

> **results_v2 matrix COMPLETE.** All 20 runs finished under corrected conditions:
> 100/250/1500 train/dev/test splits, `max_bootstrapped_demos=4` equalized across all
> optimizers, `AUTO_CONFIRM=true` for MIPROv2 non-interactive runs, `search_queries`
> trajectory field added. Results archived in `results_v2/`.

**results_v2 headline numbers (mean ± std across seeds 100/200/300):**

| Model | Baseline | ClusterFS | MIPROv2 | BFRS |
|---|---|---|---|---|
| 7B  | 26.3% | 43.34 ± 0.77% | 42.56 ± 1.77% | 45.28 ± 0.95% |
| 14B | 47.9% | **54.65 ± 0.26%** | 51.49 ± 0.69% | 53.76 ± 0.82% |
| Llama-8B | 38.1% | 45.42 ± 2.48% | 45.58 ± 1.02% | 47.53%† |

† BFRS Llama-8B: seed 100 only (s200/s300 pending). MIPROv2 now 3 seeds; variance hides bimodal termination (95 vs 1300 exhausted across seeds).

**Three-regime finding:** 7B optimization is *format teaching*. Llama-8B optimization is
*termination teaching* (finish-demo count in bootstrap pool is the dominant factor; extreme
seed variance for both methods). 14B optimization is *reasoning quality* (ClusterFS
diversity-first selection leads; MIPROv2 instruction tuning does not help). Regime identity
depends on architecture, not scale alone.

**Old pre-fix numbers (struck-through, for reference only):**

| Model | ClusterFS | MIPROv2 | BFRS |
|---|---|---|---|
| 7B | ~~45.8%~~ | ~~39.2%~~ | ~~49.8%~~ |
| 14B | ~~55.4%~~ | ~~48.6%~~ | ~~54.8%~~ |

---

## TODO 1 — ClusterFS 7B re-run  _(SUPERSEDED by results_v2)_

**Resolved:** results_v2 provides 3-seed ClusterFS 7B data (seeds 100/200/300) under the
corrected 100-train split. Mean 43.34% ± 0.77% with 0 type-A failures per seed — format
compliance is stable under the fixed codebase. The Mar 20 anomaly was a pre-fix artifact.
See `optimizer_insights.md` for full results_v2 summary.

---

## TODO 2 — Update `optimizer_insights.md`: overall takeaways  _(DONE)_

**Done:** `optimizer_insights.md` fully rewritten with results_v2 data (2026-03-31).
Covers two-regime finding, per-optimizer trajectory quality, compile-time efficiency, and
cross-optimizer statistical comparison. No further action needed.

---

## TODO 3 — Correlation analysis: format compliance failures vs. performance gain  _(DEFERRED)_

**Status:** Data now available in `results_v2/` JSONs. Analysis is valid but lower priority
given the two-regime finding already characterizes the 7B format compliance effect clearly.
Revisit when building the aggregation pipeline (TODO 8).

---

## TODO 4 — Debug: ReAct format compliance under prompt variants  _(DEFERRED)_

**Status:** Format compliance is no longer a blocking issue — results_v2 7B runs show ≤5
type-A failures per seed across all optimizers. Intervention ablations remain interesting
for paper framing but are not on the critical path. Defer until after TODO 9 and TODO 10.

---

## TODO 5 — Results Infrastructure  _(DONE)_

**Implemented in:** `enh(experiment)` commit — structured JSON output, `--seed`/`--results-dir` CLI args

**Goal**: every run emits a single structured JSON with config, scores, timing, optimizer
metadata, and per-example results. This is the foundation for all downstream analysis.

**Files**: `react_agent_experiment.py`, `react_agent_experiment_driver.sh`

### Steps

| Step | What | Key Detail |
|---|---|---|
| 5.1 | Add `--seed` CLI arg | Default `int(time.time())`, call `random.seed(seed)` at start of `main()`, forward through driver.sh |
| 5.2 | Create `results/` directory structure | `results/<model_basename>/<optimizer>/<seed>.json` — use `os.makedirs(..., exist_ok=True)` |
| 5.3 | Switch `Evaluate` to `return_all_scores=True, return_outputs=True` | Return type becomes `(score, results, all_scores)` where `results` is `[(example, prediction, score), ...]` — see `evaluate.py:224-225` |
| 5.4 | Build per-example results array | From `results` list — extract `question`, `gold` answer, `predicted` answer, `score`, trajectory step count, `finished_via_tool` flag |
| 5.5 | Capture optimizer metadata post-compile | ClusterFS: `selected_encoder`, `N`, bootstrap yield. BFRS: candidate scores/seeds. MIPROv2: `demos_per_predictor` |
| 5.6 | Record timing | Already have `baseline_runtime`, `compile_runtime`, `optimized_runtime`. Add `total_runtime = time.time() - script_start` |
| 5.7 | Save program state | `optimized_program.save(program_path)` using `Module.save()` — save alongside JSON |
| 5.8 | Write JSON file | Assemble all above + `git_sha` (from `subprocess`), `dspy.__version__`, `schema_version` |
| 5.9 | Update driver.sh | Forward `--seed`, co-locate log file in `results/` dir alongside JSON |

### JSON schema (implementation reference)

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "2026-03-26T14:30:00Z",
  "git_sha": "eab133ba...",
  "dspy_version": "2.6.x",
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "optimizer": "clusterfs",
    "seed": 100,
    "train_size": 500,
    "dev_size": 200,
    "test_size": 500,
    "max_iters": 20,
    "colbert_url": "http://localhost:8894/api/search",
    "encoder_device": "cpu"
  },
  "scores": {
    "baseline": 26.2,
    "optimized": 45.8,
    "delta_pp": 19.6,
    "compliant_accuracy": 52.1,
    "compliant_n": 440
  },
  "timing_seconds": {
    "baseline_eval": 312.5,
    "compile_total": 1845.2,
    "optimized_eval": 298.7,
    "total": 2456.4
  },
  "optimizer_meta": {
    "clusterfs": {
      "selected_encoder": "multi-qa-mpnet-base-dot-v1",
      "n_clusters": 7,
      "bootstrap_yield": 101,
      "bootstrap_attempts": 500,
      "winning_strategy": "best_in_cluster"
    }
  },
  "parse_failures": {
    "baseline_eval": {
      "total": 164,
      "rate": 0.328,
      "types": {"A": 80, "B": 50, "C": 20, "D": 14},
      "exhausted_max_iters": 12
    },
    "optimized_eval": {
      "total": 3,
      "rate": 0.006,
      "types": {"A": 1, "B": 2, "C": 0, "D": 0},
      "exhausted_max_iters": 45
    }
  },
  "per_example_results": [
    {
      "idx": 0,
      "question": "Were Scott Derrickson and Ed Wood...",
      "gold": "yes",
      "predicted": "yes",
      "score": 1.0,
      "parse_failures": 0,
      "parse_failure_type": null,
      "trajectory_steps": 3,
      "finished_via_tool": true
    }
  ],
  "program_state_path": "results/Qwen2.5-7B-Instruct/clusterfs/100_program.json"
}
```

---

## TODO 6 — ReAct Parse Failure Instrumentation  _(DONE)_

**Implemented in:** same commit — `count_parse_failures()`, `aggregate_parse_failures()`, `compute_compliant_accuracy()` in `react_agent_experiment.py`

**Goal**: make format compliance measurable per-example without modifying DSPy core.
All logic lives in `react_agent_experiment.py` only — no changes to `dspy/predict/react.py`.

**Key insight**: parse failures in `react.py:83-85` cause `break`, leaving the trajectory
missing `thought_{idx}` onward. A "clean finish" has `tool_name_{N} == "finish"`. Any
trajectory that exits without `"finish"` and before `max_iters` exhaustion hit a parse failure.

### Steps

| Step | What |
|---|---|
| 6.1 | Add `count_parse_failures(prediction, max_iters)` function to experiment script |
| 6.2 | Classify failures: Type A (no fields), B (thought only), C (thought+tool, no args), D (other) |
| 6.3 | Integrate into per-example collection from TODO 5.4 |
| 6.4 | Aggregate: `total_failures`, `failure_rate`, `failure_types` counter, `exhausted_max_iters` |
| 6.5 | "Compliant accuracy" metric: accuracy only on examples with 0 parse failures |

### Detection logic (trajectory-based)

```python
def count_parse_failures(prediction, max_iters=20):
    """Detect parse failures from a ReAct prediction's trajectory.

    Returns:
        (n_failures, failure_type, n_steps, finished_via_tool)

    Failure types:
        A — no trajectory fields at all (immediate failure)
        B — thought present but no tool_name (partial parse)
        C — thought + tool_name present but no tool_args
        D — all fields present but still broke early (other parse issue)
        None — no failure (clean finish or max_iters exhausted)
    """
    trajectory = getattr(prediction, "trajectory", {})
    if not trajectory:
        return 1, "A", 0, False  # immediate failure

    n_steps = sum(1 for k in trajectory if k.startswith("thought_"))
    finished = any(
        trajectory.get(f"tool_name_{j}") == "finish" for j in range(n_steps)
    )

    if finished:
        return 0, None, n_steps, True

    if n_steps == max_iters:
        return 0, None, n_steps, False  # exhausted, not a parse failure

    # Broke early without finishing → parse failure
    # Check what fields exist for the step that would have been next
    has_thought = f"thought_{n_steps}" in trajectory
    has_tool = f"tool_name_{n_steps}" in trajectory
    has_args = f"tool_args_{n_steps}" in trajectory

    if not has_thought:
        ftype = "A"
    elif not has_tool:
        ftype = "B"
    elif not has_args:
        ftype = "C"
    else:
        ftype = "D"

    return 1, ftype, n_steps, False
```

### Aggregation helper

```python
def aggregate_parse_failures(per_example_results):
    """Summarize parse failures across an evaluation run."""
    from collections import Counter
    types = Counter()
    total = 0
    exhausted = 0
    for ex in per_example_results:
        total += ex["parse_failures"]
        if ex["parse_failure_type"]:
            types[ex["parse_failure_type"]] += 1
        if not ex["finished_via_tool"] and ex["parse_failures"] == 0:
            exhausted += 1
    return {
        "total": total,
        "rate": round(total / len(per_example_results), 4) if per_example_results else 0,
        "types": dict(types),
        "exhausted_max_iters": exhausted,
    }
```

---

## TODO 7 — Full Re-run Matrix  _(DONE)_

**Completed 2026-03-31.** All 20 runs finished. Results in `results_v2/`. Matrix runner
`run_react_matrix.sh` used with `--resume` for the 14B block after a WiFi disconnect.

**Goal**: re-run complete experiment matrix under fixed conditions with structured output
from TODOs 5+6. All results go into `results/` as JSON.

### Matrix

| Dimension | Values |
|---|---|
| Models | `Qwen2.5-7B-Instruct`, `Qwen2.5-14B-Instruct` |
| Optimizers | `clusterfs`, `miprov2`, `bfrs` |
| Seeds | `100`, `200`, `300` |
| Baselines | 1 per model (seed-independent) |

**Total**: 3 optimizers × 2 models × 3 seeds + 2 baselines = **20 runs**

### Execution order

Group by model to minimize SGLang server swaps:

```
# --- 7B block (10 runs) ---
1.  baseline     seed=100  (baseline is deterministic, one run suffices)
2.  clusterfs    seed=100
3.  clusterfs    seed=200
4.  clusterfs    seed=300
5.  miprov2      seed=100
6.  miprov2      seed=200
7.  miprov2      seed=300
8.  bfrs         seed=100
9.  bfrs         seed=200
10. bfrs         seed=300

# --- 14B block (10 runs) ---
11. baseline     seed=100
12. clusterfs    seed=100
13. clusterfs    seed=200
14. clusterfs    seed=300
15. miprov2      seed=100
16. miprov2      seed=200
17. miprov2      seed=300
18. bfrs         seed=100
19. bfrs         seed=200
20. bfrs         seed=300
```

### Per-run validation checklist

Before moving to the next run, verify:
- [ ] JSON file written to `results/<model>/<optimizer>/<seed>.json`
- [ ] `schema_version` == `"1.0"`
- [ ] `config.seed` matches CLI arg
- [ ] `config.max_iters` == 20
- [ ] `len(per_example_results)` == `config.test_size` (500)
- [ ] `scores.baseline` is present and plausible (7B: ~20-30%, 14B: ~40-55%)
- [ ] `program_state_path` file exists and is loadable

### Estimated GPU time

- Per run: ~45-65 min on A100-80GB (bootstrap + compile + 2 evals × 500 examples)
- 7B block (10 runs): ~8-11 hours
- 14B block (10 runs): ~9-11 hours
- **Total: ~17-22 A100-hours wall clock**

---

## TODO 8 — Aggregation & Statistical Analysis Pipeline  _(blocked)_

**Blocked by:** TODO 9 (Llama-3.1-8B results). Aggregation is more meaningful once the
cross-architecture comparison is in — avoids writing the pipeline twice with different
model sets. Input will be all JSONs across `results_v2/` (Qwen runs) + Llama runs.

**Goal**: new `aggregate_react_results.py` script that reads all JSON results from `results/`
and produces publication-ready tables, statistical tests, and diagnostic plots.

**Files**: new `aggregate_react_results.py`

### Steps

| Step | What | Detail |
|---|---|---|
| 8.1 | `load_all_results(results_dir)` | Glob `results/**/*.json`, parse, group by `(model, optimizer)` |
| 8.2 | `compute_group_stats()` | Mean/std/CI per group across seeds. Also median for robustness |
| 8.3 | Markdown table generator | For `optimizer_insights.md` — model × optimizer grid with mean±std |
| 8.4 | LaTeX table generator | For paper — same grid, formatted for `\begin{tabular}` |
| 8.5 | CSV flat export | One row per run for Google Sheets / external tools |
| 8.6 | Format compliance R² decomposition | Binary compliant/non-compliant as predictor of accuracy. Uses `parse_failures.*.total` from JSON |
| 8.7 | Paired bootstrap significance test | Paired on `(question, seed)` across optimizer pairs. 10k bootstrap resamples, report p-value + CI |
| 8.8 | Scatter plot: parse failures vs accuracy | Per-candidate (from BFRS multi-seed) and per-example |

### CLI interface

```bash
python aggregate_react_results.py \
    --results-dir results/ \
    --output-dir analysis/
```

### Outputs

```
analysis/
├── summary_table.md              # Markdown: model × optimizer grid
├── summary_table.tex             # LaTeX: same grid for paper
├── full_results.csv              # Flat CSV: one row per run
├── significance_tests.md         # Pairwise p-values + CIs
├── compliance_decomposition.png  # R² bar chart + scatter
└── parse_failures_vs_accuracy.png
```

---

## TODO 9 — Cross-architecture 7B-class model validation  _(PARTIAL — BFRS seeds 200/300 missing)_

**Completed**: baseline (1 seed), ClusterFS (seeds 100/200/300), MIPROv2 (seeds 100/200/300),
BFRS (seed 100 only).
**Missing**: BFRS seeds 200 and 300.

**Key findings** (full analysis in `optimizer_insights.md`):

Llama-3.1-8B defines a **third optimization regime: termination teaching**. The model produces
structurally valid trajectories zero-shot (0 type-A failures) but has a 65.3% exhaustion rate
at baseline — it loops without calling `Finish[]`. This is distinct from both Qwen-7B (format
failures) and Qwen-14B (reasoning quality).

ClusterFS's failure to separate from other methods at 7B scale is **not Qwen-specific**, but
for a different reason: at Qwen-7B all methods converge because any demos fix format compliance;
at Llama-3.1-8B all methods have extreme seed variance because **finish-demo presence/absence in
the bootstrap pool** dominates, making optimizer type secondary.

**Finish-Demo Law**: the number of bootstrapped demos containing a `Finish[]` call is the
single strongest predictor of post-optimization termination. Runs with 0/4 finish demos produce
termination regression; BFRS s100 (0/2 demos, 491 exhausted) is a partial exception — fewer
demos attenuate but don't reverse the law. See `optimizer_insights.md` for full analysis.

**Remaining runs needed** — BFRS on Llama-3.1-8B (seeds 200 and 300):
```bash
./react_agent_experiment_driver.sh --model meta-llama/Llama-3.1-8B-Instruct \
    --optimizer bfrs --seed 200 --sglang-port 7501 \
    --train-size 100 --dev-size 250 --test-size 1500 --no-visuals
```
Repeat for seed 300. Results land in `results_v2/Llama-3.1-8B-Instruct/bfrs/`.

**Key open question**: does BFRS's metric-based selection consistently bootstrap only 2 demos
(as in s100), and does that partial-pool behavior reliably avoid termination regression? If BFRS
s200/s300 also show 0 finish demos with <baseline exhaustion, the law needs a demo-count term.

---

## TODO 10 — Extended 14B seed matrix: ClusterFS + BFRS  _(DEFERRED — v3 scope)_

**Goal**: add seeds 400 and 500 for ClusterFS and BFRS on Qwen2.5-14B to tighten
confidence intervals and determine whether ClusterFS's 0.89pp lead over BFRS is real.

**Motivation**: With only 3 seeds, ClusterFS std=0.26% and BFRS std=0.82% give a
gap that is suggestive but not conclusive. 5-seed std will narrow CIs by ~37% and give
a cleaner picture.

**Matrix**: 2 optimizers × 2 additional seeds = 4 runs (no new baseline needed).

**Expected output** (seeds 100–500, n=5):
- ClusterFS 14B: tighter std, likely still leads BFRS
- BFRS 14B: wider natural variance may persist due to random search sensitivity

**Run with** (add to matrix runner or run individually via driver):
```bash
./react_agent_experiment_driver.sh --model Qwen/Qwen2.5-14B-Instruct \
    --optimizer clusterfs --seed 400 --sglang-port 7501 \
    --train-size 100 --dev-size 250 --test-size 1500 --no-visuals
```
Repeat for seeds 400/500 × optimizers clusterfs/bfrs.

**Update `optimizer_insights.md`** after completing with 5-seed mean/std table.

---

## TODO 11 — Bootstrap metric: penalize non-termination  _(v3 — composite metric experiment)_

**Goal**: Modify the bootstrap metric to require `finished_via_tool=True` so that the demo
candidate pool is guaranteed to contain termination-teaching examples.

**Motivation**: `answer_exact_match` is blind to non-termination. For Llama-3.1-8B, ~31% of
looping (exhausted) trajectories get the correct answer, so the bootstrap candidate pool is
dominated by non-terminating demos. When ClusterFS or MIPROv2 select 3–4 demos, they can
select zero finish-containing examples by chance — causing termination regression vs baseline
(ClusterFS s300: 1247 exhausted vs 980 at baseline; MIPROv2 s100: 1300 exhausted). This is
a systematic bug, not a random failure.

**Proposed fix** in `react_agent_experiment.py`:

```python
def answer_exact_match_with_finish(example, pred, trace=None):
    """Bootstrap metric: requires both correct answer AND clean Finish[] termination."""
    answer_ok = answer_exact_match(example, pred, trace)
    finished = any(
        getattr(pred, 'trajectory', {}).get(f'tool_name_{j}') == 'finish'
        for j in range(20)
    ) if hasattr(pred, 'trajectory') and isinstance(pred.trajectory, dict) else False
    return answer_ok and finished
```

Pass to optimizers as `metric=answer_exact_match_with_finish`.

**Experiment**: re-run Llama-3.1-8B ClusterFS (3 seeds) with composite metric.

**Prediction**:
- All demo pools will contain ≥1 finish-teaching example → no termination regressions
- Exhausted counts across seeds should collapse to a narrow range (vs current 351–1247)
- Accuracy effect: neutral to slightly positive for Llama; no change expected for Qwen-7B
  (demos already contained `Finish[]` traces) or Qwen-14B (low baseline exhaustion)

**Update `optimizer_insights.md`** with composite metric results once available.

---

## Dependency graph

```
TODO 5 (Infrastructure) ──┬──> TODO 7 (Re-runs) ──> TODO 9 (Llama) ──────────────────> TODO 8 (Aggregation)
TODO 6 (Parse Failures) ──┘    DONE ✓               ClusterFS/MIPROv2 DONE              blocked by BFRS s200/300
          DONE ✓                                     BFRS s100 DONE
                                                     BFRS s200/300 ← only gap

                                                    [results_v2 scope ends here]

                                                    TODO 10 (14B seeds)         TODO 11 (composite metric)
                                                    DEFERRED → results_v3       DEFERRED → results_v3

TODO 1 — SUPERSEDED   TODO 2 — DONE   TODO 3 — DEFERRED   TODO 4 — DEFERRED
```

**Critical path**: ~~TODO 5 + TODO 6 → TODO 7~~ DONE → TODO 9 (BFRS s200/300) → TODO 8 → **results_v2 complete**

---

## Open questions (inform paper framing)

- Does ClusterFS's diversity-first mechanism advantage require a minimum reasoning capability
  threshold? results_v2 suggests yes (14B benefits, 7B does not). Llama-3.1-8B confirms
  diversity-first selection cannot overcome the termination-demo deficit in the bootstrap pool.
- Is the format compliance threshold model-specific to Qwen2.5-7B, or a general ≤7B property?
  **Answered (partially)**: Llama-3.1-8B has zero format failures — the format-teaching regime
  is Qwen-7B-specific. Llama is in a termination-teaching regime instead.
- Can a composite bootstrap metric (`exact_match AND finished_via_tool`) guarantee
  termination-teaching demos without requiring cluster-based selection? **TODO 11 tests this.**
- Does the Finish-Demo Law hold for BFRS? BFRS s100 selected only 2 demos (0 finish), yet
  avoided regression (491 exhausted vs 980 baseline). Law appears attenuated at lower demo
  count, not reversed. **Requires BFRS s200/300 to determine if this is consistent.**
- Is the high acc@exhaust for Llama-3.1-8B (0.310 baseline) a general property of 8B-class
  instruction-tuned models, or specific to Llama's training distribution?
