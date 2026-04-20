# Project TODOs

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*

Sections: [Experiments to run](#experiments-to-run) · [Implementation](#implementation)
· [Compute / infrastructure](#compute--infrastructure) · [Known bugs / issues](#known-bugs--issues)

---

## Experiments to run

### results_v1 — primary hypothesis matrix ✓ complete

- [x] Run full matrix: 2 datasets × 2 models × 5 optimizers × 5 seeds = 100 runs
- [x] Aggregate and plot results

### results_v2 — gepa / gepa_merge re-runs ✓ complete (2026-04-15)

- [x] 40 re-runs (gepa + gepa_merge × 2 datasets × 2 models × 5 seeds)
- [x] Merge into v1 and regenerate plots

### results_v3 — validity-fixed re-runs ✓ complete (2026-04-20)

All 70 runs complete. Merged with v1+v2 via:
```
python experiments/analyze_results.py \
    --log-dir experiments/results_v1 experiments/results_v2 experiments/results_v3 \
    --aggregate --plot-dir experiments/plots/results_v3 --without miprov2
```

- [x] Iris re-runs (50 runs, all validity fixes: reflection_minibatch_size=10, IrisDataset(seed=0), torch.manual_seed)
- [x] GSM8K gepa/gepa_merge re-runs (20 runs, reflection_minibatch_size=10)
- [x] Clean up stale pre-fix duplicate directories
- [x] Compile final aggregate stats and regenerate plots (score comparison, accuracy vs runtime, accuracy gain vs runtime)
- [x] Update insights.md and todos.md with final findings

Key v3 findings: GSM8K gap shrinks to 0.9pp (3B) and 1.0pp (7B) — confirmed ties. Iris/3B
+14.8pp directional unchanged. GEPAFewShot faster than vanilla GEPA on both Iris cells.

### Pending experiments (post-report, optional)

- [ ] **Formal statistical tests** — Wilcoxon signed-rank or Mann-Whitney U for
      gepa_fewshot vs. gepa per cell. Priority: Iris/3B (+14.8pp directional).
- [ ] **1B model on Iris** — Llama-3.2-1B-Instruct, 3 optimizers × 5 seeds = 15 runs.
      Complementary evidence for the scaling hypothesis: if instruction-following degrades
      further at 1B, the demonstration advantage should be larger and potentially statistically
      significant, anchoring the 1B→3B→7B saturation trend with a third data point.
      Low compute cost (~2–3h on Iris).
- [ ] **gepa_fewshot with merge enabled** — blocked by `_demo_registry` bug (see
      Implementation below). Once fixed, add as ablation point.
- [ ] **Larger model (13B+)** — Qwen2.5-14B or Llama-3.1-8B on Iris. Tests whether
      saturation continues or eventually reverses. Low priority given May 3 deadline.

---

## Validity fixes before results_v3 — all applied ✓

- [x] **[BUG] `reflection_minibatch_size` silently dropped for gepa/gepa_merge** ← *fixed*
- [x] **[BUG] Iris data split shuffled by experiment seed** ← *fixed* (`IrisDataset(seed=0)`)
- [x] **[BUG] Iris val set silently 25 examples, not 35** ← *fixed (logging only)*
- [x] **[BUG] torch not seeded** ← *fixed* (`torch.manual_seed` in `main()`)

---

## Implementation

### High priority

- [ ] **Fix `gepa_fewshot` merge path** — `GEPAFewShot` forces `use_merge=False` because
  merged candidates receive new instruction dicts not registered in `_demo_registry`, so
  `build_program` cannot find their companion demo set.
  Fix options:
  (a) hook the merge callback to register a new demo set for the merged candidate, or
  (b) fall back to the seed demo set when a key is missing in `_demo_registry`.

### Medium priority (post-report)

- [ ] **Instruction-conditioned demo selection** — current mutation uses token-Jaccard to
  score pool demos against the reflective minibatch. A stronger signal: score each pool demo
  against the *current instruction* semantically (e.g., run instruction as prompt on demo
  input, check metric alignment). Addresses the core architectural gap where demo mutation
  is instruction-agnostic. Key challenge: demo staleness as instruction evolves.

- [ ] **Semantic-coverage demo selection** — cluster-centroid or MMR-based selection from
  the pool to reduce demo variance across seeds. Evaluate after results_v3.

- [ ] **Per-iteration demo tracking** — capture the active demo set at each optimization
  step for post-hoc analysis of how the demo pool evolves.

---

## Compute / infrastructure

- [ ] **Slurm cluster support** — adapt `run_matrix.py` to submit individual runs as Slurm
  array jobs. Reference `NLP_project_guidelines.pdf` for queue/resource policies.
- [ ] **Ollama backend** — serving path for colleagues without GPU access.

---

## Known bugs / issues

- [x] `experiments/metrics.py` was missing — fixed
- [x] `RANDOM_SEED` non-reproducible — fixed (`--seed` CLI arg)
- [x] `run_experiment_driver.sh` optimizer validation gap — fixed
- [x] `total_metric_calls` not captured in results.json — fixed
- [ ] **`GEPAFewShot` merge assertion error** — `use_merge=True` triggers `AssertionError`
  because merged candidates are not registered in `_demo_registry`. Workaround in place
  (`__init__` forces `use_merge=False`). Long-term fix tracked under Implementation above.
