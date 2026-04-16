# Project TODOs

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*

Sections: [Experiments to run](#experiments-to-run) · [Implementation](#implementation)
· [Compute / infrastructure](#compute--infrastructure) · [Known bugs / issues](#known-bugs--issues)

---

## Experiments to run

### results_v1 — primary hypothesis matrix (current priority)

- [x] Run full matrix: 2 datasets × 2 models × 5 optimizers × 5 seeds = 100 runs
  - Driver: `python experiments/run_matrix.py`
  - Models: `Llama-3.2-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
  - Optimizers: `baseline`, `miprov2`, `gepa`, `gepa_merge`, `gepa_fewshot`
  - Seeds: 42, 123, 456, 789, 1337
- [x] Run `python experiments/analyze_results.py --log-dir experiments/results_v1 --aggregate`
      to produce `aggregate_stats.json` and the comparison table
- [x] Generate plots:
      `python experiments/analyze_results.py --log-dir experiments/results_v1 --aggregate --plot-dir experiments/plots/results_v1`
- [x] Update `insights.md` with statistically grounded findings once results are in

### results_v2 — re-runs of gepa and gepa_merge (complete)

40 re-runs (gepa + gepa_merge × 2 datasets × 2 models × 5 seeds) completed 2026-04-15.
Results merged into v1 via `analyze_results.py --log-dir results_v1 results_v2`.

- [x] Run gepa / gepa_merge re-runs (40 runs) — completed 2026-04-15
- [x] Merge into v1 and regenerate plots (`--log-dir results_v1 results_v2 --aggregate --plot-dir plots/results_v2`)
- [x] Update `insights.md` with v2-corrected findings (2026-04-16)

Key finding from v2: gepa on GSM8K/3B improved +3.37pp, narrowing gepa_fewshot gap from
4.81pp to 1.44pp (now non-significant).  Iris story unchanged.  gepa_merge weakened on Iris.

### results_v3 — next experiments (scope set by v2 analysis)

- [ ] **Formal statistical tests** — Wilcoxon signed-rank or Mann-Whitney U for key pairwise
      comparisons (gepa_fewshot vs gepa per cell); CI analysis is directional, paper needs p-values
- [ ] **gepa_fewshot with merge enabled** — blocked by `_demo_registry` bug; fix first (see
      Implementation below), then add as ablation point
- [ ] **Larger model (13B+)** — test whether the Iris inversion strengthens further with scale;
      Qwen2.5-14B or Llama-3.1-8B are candidates
- [ ] **GSM8K scaling inversion** — gap now 1.44pp at 3B and 1.26pp at 7B; a 13B+ run would
      test whether it inverts on reasoning tasks at all

---

## Validity fixes before results_v3

These bugs were found during post-v2 audit (2026-04-16).  Fix all three before re-running
gepa/gepa_merge for results_v3 — they affect the fairness of the gepa vs gepa_fewshot comparison.

- [ ] **[BUG] `reflection_minibatch_size` silently dropped for gepa/gepa_merge** ← *fix first*
  `run_experiment.py` `build_optimizer` forwards `reflection_minibatch_size` to `GEPAFewShot`
  but not to `GEPA`.  gepa/gepa_merge ran with GEPA's default of **3**; gepa_fewshot ran with
  **10**.  This gives gepa_fewshot a richer reflection signal per step — a confound.
  Fix: add `reflection_minibatch_size=args.reflection_minibatch_size` to the `GEPA(...)` call
  in `build_optimizer` (unstaged change already present, just needs commit + clean re-run).

- [ ] **[BUG] Iris data split shuffled by experiment seed**
  `IrisDataset(seed=RANDOM_SEED)` is called with the optimizer seed, so each of the 5 seeds
  sees a different train/val/test partition.  Cross-seed variance on Iris mixes optimizer
  randomness with data-split randomness.  GSM8K is unaffected (fixed slices, no shuffle).
  Fix: change to `IrisDataset(seed=0)` always; optimizer seed only controls GEPA internals.

- [ ] **[BUG] Iris val set silently 25 examples, not 35**
  `IrisDataset.dev` has 25 examples; `valset[:35]` returns all 25 without error.  GEPA's
  "medium" budget is calibrated to 25, not 35.  Past results.json (v1/v2) log `val_size: 35`
  (requested size) — the unstaged fix to log `len(valset)` corrects this going forward.
  Fix: commit the unstaged `run_experiment.py` change (already done); verify val_size=25 in
  results_v3 output.

---

## Implementation

### High priority

- [ ] **Fix `gepa_fewshot` merge path** — `GEPAFewShot` currently forces `use_merge=False`
  because merged candidates receive new instruction dicts that are not registered in
  `_demo_registry`, so `build_program` cannot find their companion demo set.
  The fix requires either:
  (a) hooking the merge callback to register a new demo set for the merged candidate, or
  (b) falling back to the seed demo set when a key is missing in `_demo_registry`.
  Once fixed, `gepa_fewshot` with merge enabled would be an additional ablation point.

### Medium priority

- [ ] **Semantic-coverage demo selection** — the current pool sampling is score-weighted
  random.  A diversity/coverage criterion (e.g., cluster centroids or MMR) may reduce
  demo variance across seeds and strengthen the demonstrative signal.  Evaluate after
  `results_v1` to determine if variance is high enough to warrant this.

- [ ] **Per-iteration demo tracking** — capture the active demo set at each optimization
  step to enable post-hoc analysis of how the demo pool evolves.  Requires hooking
  `propose_new_texts` to log outgoing/incoming demo sets.

---

## Compute / infrastructure

> **Note**: Slurm and small-compute support are deferred until `results_v1` is complete.
> Current runs assume an A100 GPU with SGLang serving the model locally.

- [ ] **Slurm cluster support** — adapt `run_matrix.py` to submit individual runs as
  Slurm array jobs on the TAU cluster.
  - Each matrix cell → one `sbatch` job (dataset/model/optimizer/seed passed as env vars)
  - Add `--slurm` flag to `run_matrix.py` that generates and submits batch scripts
    instead of calling `subprocess.run` directly
  - Reference `NLP_project_guidelines.pdf` for queue/resource/time-limit policies
  - Document the Slurm path in `README.md`

- [ ] **Ollama backend** — add an Ollama-compatible serving path for colleagues without
  GPU access.
  - Adapt or replace `ensure_sglang_server` with an `ensure_ollama_server` helper
  - Verify metric parity between Ollama and SGLang serving for the same model
  - Document the Ollama path in `README.md`

---

## Known bugs / issues

- [x] **`experiments/metrics.py` was missing** — `run_experiment.py` imported
  `BETTER_REFLECTION_PROMPT`, `gsm8k_gepa_metric`, and `iris_gepa_metric` from this
  module but the file was never committed.  *(Fixed)*

- [x] **`RANDOM_SEED` non-reproducible** — previously set as `int(time.time())` at module
  level.  Fixed by adding `--seed` CLI arg and moving seed initialization into `main()`.
  *(Fixed)*

- [x] **`run_experiment_driver.sh` optimizer validation gap** — `VALID_OPTIMIZERS` did not
  include `baseline` or `gepa_merge`, causing those to fail validation.  *(Fixed)*

- [x] **`total_metric_calls` not captured in results.json** — GEPA exposes
  `optimized.detailed_results.total_metric_calls` when `track_stats=True` (already set
  for all runs).  Added to `results.json` and surfaced in `analyze_results.py`'s
  aggregate table.  *(Fixed)*

- [ ] **`GEPAFewShot` merge assertion error** — `use_merge=True` triggers an
  `AssertionError` because merged candidates are not registered in `_demo_registry`.
  Workaround in place (`__init__` forces `use_merge=False`).
  Long-term fix tracked above under Implementation.
