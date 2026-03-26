# ReAct Agent Experiment — Task Bank

## Status snapshot (2026-03-26)

> **All pre-fix results are INVALIDATED.** Five critical bugs were fixed in
> `react_agent_experiment.py` and `programs.py` between Mar 20-25. Any number
> collected before these fixes is unreliable and must be re-run under the fixed
> codebase with structured output (TODO 5).

**Fixes applied:**
1. `max_iters` changed from 5 → 20 (was silently capping agent reasoning depth)
2. `ReactAgentMH.forward()` now passes `max_iters` through to `dspy.ReAct` (was ignored)
3. Content-safety filter applied to all splits (was missing from dev/test)
4. Bootstrap metric switched from default to `answer_exact_match` (was using wrong metric)
5. `dspy.configure(rm=retriever)` moved inside `create_colbert_search_tool` (was missing for bootstrap)

**Old numbers (struck-through, for reference only):**

| Model | ClusterFS | MIPROv2 | BFRS |
|---|---|---|---|
| 7B | ~~45.8%~~ | ~~39.2%~~ | ~~49.8%~~ |
| 14B | ~~55.4%~~ | ~~48.6%~~ | ~~54.8%~~ |

~~Zero-shot 7B baseline: ~26% (true), ~29% (reported, inflated by extract fallback ~7pp).~~

---

## TODO 1 — ClusterFS 7B re-run  _(blocked)_

**Blocked by:** TODO 5 (needs structured JSON output), TODO 7 (part of full re-run matrix)

**Goal**: verify that ClusterFS reproducibly finds format-compliant demos and delivers ~45%+,
and confirm whether the Mar 20 run's anomalous 0 zero-shot failures was a server artifact.

**What we know**: a fresh `--baseline` run (Mar 24) produced 164 parse failures and 26.2% —
consistent with MIPROv2/BFRS baselines. The Mar 20 ClusterFS run's 0 failures is an outlier
that can only be explained by a different SGLang server state at that time, not seed variation
(P(0 failures | ~32% population failure rate) ≈ 0).

**When it completes, record:**
- Baseline score + parse failure count (expect ~26%, ~150-165 failures)
- Optimized score + parse failure count on final test eval (expect ~0 failures if demos are compliant)
- Bootstrap yield (expect ~20%, ~100 traces from 500 attempts)
- Whether `best_in_cluster` vs `top_n` tie holds again
- Update `optimizer_insights.md` §ClusterFewshot 7B with corrected baseline note

---

## TODO 2 — Update `optimizer_insights.md`: overall takeaways  _(blocked)_

**Blocked by:** TODO 7 (needs post-fix data to write credible insights)

Add a **§Format Compliance as Optimizer Confound** section with:

1. **The confound**: on 7B, task accuracy is confounded by whether the selected demos happen
   to be format-compliant. The 23.4pp BFRS candidate spread (24.8%–48.2%) is almost entirely
   explained by this, not by reasoning quality differences.

2. **The mechanism**: 7B operates at the format compliance threshold for 3-field structured
   output. Bootstrapped reasoning traces (complete thought→search→observe→finish chains) shift
   the model above threshold; labeled-only pairs and low-quality single traces do not.
   Evidence: seeds -3/-2 (labeled-only) have 157 failures = identical to zero-shot (156).

3. **Proposed disentangled metric**: report accuracy on format-compliant inferences only
   (exclude all extract-fallback outputs) alongside raw accuracy. Isolates optimizer reasoning
   quality from format scaffolding effect.

4. **Cross-size comparison**: 14B is format-stable (≤3 failures across all optimizers/seeds).
   The format compliance problem is specific to small models (≤7B). Optimizer rankings differ
   between model sizes partly for this reason.

---

## TODO 3 — Correlation analysis: format compliance failures vs. performance gain  _(blocked)_

**Blocked by:** TODO 5 (needs per-example scores in JSON), TODO 6 (needs parse failure counts)

**Goal**: formally quantify the relationship between per-candidate parse failure rate and
accuracy, using the BFRS 9-candidate sweep as a controlled natural experiment. This is
scientifically meaningful because all 9 candidates share the same model, task, base prompt,
and evaluation set — the only variable is the bootstrapped demo set.

**Why it's interesting**:
The correlation measures how much of the apparent performance gap between optimizer candidates
is driven by format compliance (model stays in ReAct mode) vs. reasoning quality (model reasons
well once in ReAct mode). If the correlation is strong (expected R² > 0.9 from the three-tier
structure), it directly supports the claim that prompt optimizers on small models are largely
performing *format compliance selection*, not *reasoning quality selection* — a strong and
publishable reframing of what few-shot optimization achieves on ≤7B models.

**Analysis to run** (all data already in logs, no new experiments needed):

1. **Scatter plot**: x = parse failures per candidate (500-example eval), y = candidate accuracy.
   Plot the 9 BFRS candidates + zero-shot baseline as 10 data points. Fit a regression line.
   Expected result: strong negative correlation with a natural cluster gap around ~10 failures
   separating the two regimes.

2. **Two-component decomposition**:
   - *Between-group variance*: compliant (≤3 failures) vs. non-compliant (~157 failures).
     Quantify how much accuracy variance this binary split explains (expected: ~80-90%).
   - *Within-group variance*: among compliant seeds only (0, -1, 2, 4, 5), plot failures vs.
     accuracy. Residual variance here reflects trace reasoning quality, not format compliance.
     This decomposition makes the "necessary but not sufficient" claim rigorous.

3. **Parse failure rate as early-rejection proxy**: simulate a sequential eval where candidates
   are abandoned after 25 examples if failure rate > 50%. Show which non-compliant candidates
   would be correctly rejected early, and estimate compute savings.

4. **Extend to MIPROv2 minibatch trials** (secondary): 25 trials × 25 examples each, most
   have 0 failures. Check whether the 3 outlier trials (trials 9, 11, 21) with elevated
   failures also have suppressed minibatch scores. Smaller effect expected since 14B was used
   and the variance in MIPROv2 trial scores is driven by minibatch noise more than format.

**Data sources** (no new runs needed):
- BFRS 7B: `react_agent_experiment_Qwen2.5-7B-Instruct_bfrs_2026-03-24.log` — per-seed
  failure counts and scores already extracted (see `optimizer_insights.md` §BFRS 7B table)
- MIPROv2 7B: `react_agent_experiment_Qwen2.5-7B-Instruct_miprov2_2026-03-21.log` — per-trial
  scores and failure counts already mapped

---

## TODO 4 — Debug: ReAct format compliance under prompt variants  _(blocked)_

**Blocked by:** TODO 6 (parse failure Type A/B/C/D classification is implemented there)

**Goal**: understand *when* and *why* 7B fails structured output, and identify minimal
interventions that guarantee compliance. Feeds directly into paper framing.

### Step 1 — Instrument failure types (code change, ~30 min)
In [dspy/predict/react.py:83](dspy/predict/react.py#L83), log raw pred fields at parse failure:
```python
except AttributeError:
    logger.warning(
        "ReAct parse failure. Present fields: %s",
        list(vars(pred).keys()) if hasattr(pred, '__dict__') else repr(pred)
    )
    break
```
Re-run `--baseline` on 7B and classify failures into types:
- **Type A**: no fields at all (model produced prose, not structured output)
- **Type B**: partial fields (`next_thought` present, tool fields missing)
- **Type C**: wrong field names (format drift — model invented key names)
- **Type D**: `next_tool_args` present but not dict-parseable

### Step 2 — Identify failure-prone examples (~1h, needs Step 1)
Run `--baseline` 3× with different seeds. Track which HotPotQA questions fail consistently
across runs. Measure average input token length for failing vs non-failing examples.
Hypothesis: long-passage retrievals consume budget, truncating the structured output.

### Step 3 — Prompt intervention ablations (~3h, needs Step 2)
On the ~50 most consistently-failing examples from Step 2, run each intervention:

| Intervention | What it tests |
|---|---|
| Explicit format reminder in system prompt | Reinforcing field contract at inference time |
| 1 manually-crafted perfect-format demo | Minimal few-shot format anchor |
| Reduce `max_iters` 20→3 | Fewer steps = less chance of format drift |
| Temperature 0.0 | Eliminates stochasticity in format-critical output |
| Grammar-constrained decoding (Outlines) | Hard ceiling — what's achievable with constraint |

### Step 4 — Trace analysis: format-compliant vs non-compliant BFRS seeds (~1h)
Compare bootstrap traces from BFRS seeds 0/-1/4 (compliant) vs seeds 1/3 (non-compliant).
Look for structural differences: avg trajectory length, whether `finish` tool appears,
tool_args JSON validity, trace token length. This directly explains the bimodal effect.

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

## TODO 7 — Full Re-run Matrix  _(READY — run_react_matrix.sh created)_

**Infrastructure done:** `run_react_matrix.sh` created. Run with `--dry-run` to preview, `--resume` to skip completed runs.

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

**Blocked by:** TODO 7 (needs completed results from full matrix)

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

## Dependency graph

```
TODO 5 (Infrastructure) ──┬──> TODO 7 (Re-runs) ──> TODO 8 (Aggregation)
TODO 6 (Parse Failures) ──┘        │                       │
          DONE ✓                    ├──> TODO 1 (ClusterFS re-run)
                              READY ├──> TODO 2 (update insights)
                                    ├──> TODO 3 (correlation analysis)
                                    └──> TODO 4 (debug compliance)
```

**Critical path**: ~~TODO 5 + TODO 6 (parallel) →~~ TODO 7 (execute matrix) → TODO 8

TODOs 1-4 are unblocked once TODO 7 completes (post-fix data available).

---

## Open questions (inform paper framing and TODO 3)

- Does ClusterFS's high bootstrap yield (101 traces) *cause* format compliance, or does it
  merely *correlate* with it? Test: subsample ClusterFS to 10 traces and re-run selection.
- Is the format compliance threshold model-specific to Qwen2.5-7B, or does it generalize to
  other 7B-class models (Llama-3.1-8B, Gemma-3-4B)?
- Can a simple bootstrap filter (discard traces with > N parse failures during collection)
  guarantee format-compliant demos without requiring cluster-based selection?
