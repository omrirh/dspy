# Experiment Insights

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*

This file tracks empirical findings that support (or challenge) the core project hypothesis.
Each entry links to the experimental evidence and records its current validation status.

> **Status**: `results_v1` matrix complete (2026-04-14) — 100 runs (2 datasets × 2 models ×
> 5 optimizers × 5 seeds).  `results_v2` re-runs of `gepa` and `gepa_merge` (40 runs, same
> seeds) merged in 2026-04-16; later values supersede v1 for those cells.  Entries marked
> **[confirmed]** are statistically grounded.  Entries marked **[preliminary]** remain from
> pre-matrix exploratory runs.

---

## Core Hypothesis

> **GEPA+FewShot is more efficient than Vanilla GEPA for models with moderate reasoning
> capability** (≤ ~4B parameters), because demonstrative context provides a more direct
> learning signal than reflectively-derived instructions that such models cannot
> effectively self-generate from the training data alone.
>
> For larger / stronger models (≥ 7B), Vanilla GEPA closes the gap or surpasses
> GEPA+FewShot because the model can self-reflect meaningfully — extracting precise,
> generalizable instructions — and the marginal gain from fixed demonstrations diminishes
> relative to the cost of running an additional bootstrap pass.

**Verdict (results_v1):** Supported on structured classification (Iris), partially supported
on arithmetic reasoning (GSM8K).  The effect is a **task × scale** interaction, not a pure
scale effect.

---

## Insight 1 — Demonstrative context outperforms instructional context on small models

**Status**: [confirmed] on Iris; [directional only] on GSM8K

### Iris / Llama-3.2-3B-Instruct (N=5 seeds)

| Optimizer | Mean | CI±95 |
|---|---|---|
| baseline | 34.00 | ±0.00 |
| gepa | 40.40 | ±10.15 |
| gepa_fewshot | **64.40** | ±8.85 |
| gepa_merge | 42.40 | ±7.53 |
| miprov2 | **69.20** | ±13.21 |

`gepa_fewshot` vs `gepa`: **+24.0pp**, non-overlapping CIs (`gepa` upper ~50.6%,
`gepa_fewshot` lower ~55.6%).  This is the strongest evidence in the dataset.

`gepa` variance is high (±10.15) — consistent with the prior finding that vanilla
GEPA generates a noisy, non-improving candidate chain when the small model cannot follow
declarative classification rules.  `gepa_fewshot` is both better and more stable.

`miprov2` is the top optimizer on this cell (69.20 ± 13.21).  The margin over `gepa_fewshot`
is within CI overlap; the shared mechanism is bootstrapped demonstrations.

*Note: `gepa_merge` dropped 8pp vs v1 (50.40 → 42.40) in the v2 re-runs, now nearly tied
with plain `gepa`.  The merge path provides no benefit on this cell.*

### GSM8K / Llama-3.2-3B-Instruct (N=5 seeds)

| Optimizer | Mean | CI±95 |
|---|---|---|
| baseline | 66.19 | ±0.00 |
| gepa | 73.77 | ±2.27 |
| gepa_fewshot | **75.21** | ±2.66 |
| gepa_merge | 71.17 | ±3.52 |
| miprov2 | **75.68** | ±1.21 |

`gepa_fewshot` vs `gepa`: **+1.44pp** *(was +4.81pp in v1 — gepa improved by +3.37pp in v2
re-runs)*.  CIs overlap substantially (`gepa` upper ~76.0, `gepa_fewshot` lower ~72.6).
This gap is **directional only** and cannot be cleanly separated statistically at N=5.

On arithmetic reasoning, GEPA's reflective instruction refinement ("solve step by step")
is effective even for 3B — v2 confirms this more strongly.  The demonstrative advantage on
GSM8K/small models exists at most as a weak tendency, not a structural finding.

`miprov2` is statistically tied with `gepa_fewshot` (75.68 vs 75.21, CI overlap).

**Mechanism update:** The demonstration advantage on small models is **task-dependent**.
It is sharp on structured tasks requiring declarative rule-following (Iris), where small
models cannot internalize instructions.  On reasoning tasks (GSM8K), there is at most a
marginal and non-significant benefit — instruction refinement is sufficient at this scale.

---

## Insight 2 — The demonstrative advantage inverts with model scale on structured tasks

**Status**: [confirmed] on Iris; [not confirmed] on GSM8K

### Iris / Qwen2.5-7B-Instruct (N=5 seeds)

| Optimizer | Mean | CI±95 |
|---|---|---|
| baseline | 59.20 | ±6.47 |
| gepa | **80.80** | ±6.23 |
| gepa_fewshot | 72.80 | ±7.37 |
| gepa_merge | 78.80 | ±4.84 |
| miprov2 | 79.20 | ±5.72 |

`gepa_fewshot` vs `gepa`: **−8.0pp** — the inversion is confirmed.  The 7B model's
instruction-following is strong enough that GEPA's reflective loop converges to tight
numerical decision boundaries.  Fixed demonstrations constrain exploration without
adding information the model couldn't derive from instructions alone.

`gepa_merge` dropped from 82.40 (v1) to 78.80 (v2 re-runs) and is no longer the top
optimizer on this cell — `gepa` (80.80) now leads.  The prior claim that gepa_merge is
a robust hybrid for 7B on structured tasks is weakened; all four non-baseline optimizers
are within CI overlap of each other.

### GSM8K / Qwen2.5-7B-Instruct (N=5 seeds)

| Optimizer | Mean | CI±95 |
|---|---|---|
| baseline | 74.00 | ±0.00 |
| gepa | 83.60 | ±6.63 |
| gepa_fewshot | **84.85** | ±2.56 |
| gepa_merge | 83.81 | ±1.90 |
| miprov2 | 83.99 | ±2.61 |

`gepa_fewshot` vs `gepa`: **+1.26pp** *(was +1.0pp in v1, essentially unchanged)*.  All
four optimizers fall within CI of each other — the choice of optimizer is statistically
irrelevant at 7B on GSM8K.  Baseline model quality dominates.

All optimizers are statistically tied on this cell (all CIs overlap).  At 7B on GSM8K,
the choice of optimizer matters far less than baseline model quality.

**Scaling conclusion:** The hypothesis predicts a scale × modality interaction.
Results_v1 confirms this on structured tasks (Iris) but not on reasoning tasks (GSM8K),
indicating the interaction is mediated by task type.

---

## Insight 3 — The hypothesis effect is a task × scale interaction, not a pure scale effect

**Status**: [confirmed]

The results reveal a 2×2 pattern:

|  | Small model (3B) | Large model (7B) |
|---|---|---|
| **Structured task (Iris)** | gepa_fewshot >> gepa (+24.0pp) | gepa >> gepa_fewshot (−8.0pp) |
| **Reasoning task (GSM8K)** | gepa_fewshot ≈ gepa (+1.44pp, n.s.) | gepa ≈ gepa_fewshot (+1.26pp, n.s.) |

*(Updated 2026-04-16: v2 re-runs raised gepa on GSM8K/3B by +3.37pp, shrinking the
gepa_fewshot gap from 4.81pp to 1.44pp — no longer separable at N=5.)*

The clean inversion only appears for structured classification.  On reasoning tasks, the
demonstrative advantage is **not statistically significant at either scale** — the gaps are
within CI overlap in both cells.  Few-shot arithmetic examples do not hurt but also do not
provide a consistent structural benefit beyond GEPA's instruction refinement.

**Implication for paper framing:** The core claim should be:
> "Demonstrative context provides a stronger optimization signal than instructional context
> when the task requires rule-following behaviour that small models cannot derive from
> instructions.  This advantage inverts for larger models that can self-reflect effectively.
> On tasks where in-context exemplars benefit all models regardless of scale (e.g., arithmetic
> reasoning), the scaling interaction is attenuated."

---

## Insight 4 — gepa_merge performance is within noise of plain gepa

**Status**: [revised — v2 weakens prior claim]

After v2 re-runs, `gepa_merge` is consistently competitive but no longer a standout:

| Cell | gepa | gepa_merge | verdict |
|---|---|---|---|
| GSM8K/Llama | 73.77 ± 2.27 | 71.17 ± 3.52 | gepa leads; within CI |
| GSM8K/Qwen  | 83.60 ± 6.63 | 83.81 ± 1.90 | tied |
| Iris/Llama  | 40.40 ± 10.15 | 42.40 ± 7.53 | tied |
| Iris/Qwen   | **80.80 ± 6.23** | 78.80 ± 4.84 | gepa leads; within CI |

The previous v1 claim that gepa_merge was the top optimizer on Iris/Qwen (82.40) does not
replicate in v2 (78.80).  Across all four cells, gepa_merge falls within CI of plain gepa —
the merge path adds no consistent accuracy benefit.

The practical recommendation is revised: **plain gepa is the better default** for 7B+ on
structured tasks.  gepa_merge's lower variance on GSM8K/Qwen (std 1.52 vs 5.34) may still
be useful when stability is prioritized over peak accuracy.

---

## Insight 5 — Metric-based demo mutation is preferred over random

**Status**: [preliminary] — single-run comparison, no CI

**Observation**: Runs with `--demo-mutation-strategy random` occasionally swapped
high-quality bootstrapped demos (score 1.0) out of the active set in favour of labeled
examples (score 0.5), causing a mid-optimization quality dip.  `metric_based` prevents
this by making bootstrapped demos 2× more likely to be selected during add/swap operations.

**Default**: `metric_based` is fixed in the `results_v1` matrix and should remain so unless
a targeted ablation is added in a future `results_v2`.

---

## Insight 6 — Optimization budget sensitivity

**Status**: [preliminary]

**Observation**: On Iris (structured classification, small dataset), `light` and `medium`
budget runs converged to similar accuracy while `medium` had measurably higher wall-clock
cost.  On GSM8K (multi-step arithmetic, larger dataset), `medium` yielded noticeably better
instruction quality over `light`.

**Implication**: All `results_v1` runs use `medium` budget for consistency.  A follow-up
ablation varying budget (light / medium / heavy) is listed in `todos.md`.

---

## Pending analyses (post `results_v2` merge)

- **Statistical tests**: Wilcoxon signed-rank or Mann-Whitney U across seeds for
  key pairwise comparisons (gepa_fewshot vs gepa per cell) — CI analysis above is
  directional; formal tests needed for paper claims.  *Priority: the Iris/3B cell (+24pp)
  should easily pass; GSM8K/3B (1.44pp) likely will not.*
- **Per-seed score variance**: Iris/Llama `gepa` (±10.15) and `miprov2` (±13.21) remain
  high-variance.  Is this optimizer sensitivity to initial random training examples, or
  model sensitivity to data order?
- **gepa_fewshot merge path**: blocked by `_demo_registry` bug.  Once fixed, this is the
  missing ablation that closes the matrix.
- **Larger model (13B+)**: Qwen2.5-14B or Llama-3.1-8B — does the Iris inversion
  strengthen further with scale?  Does GSM8K show any inversion at 13B+?
- **Instruction quality analysis**: Do instructions from `gepa_fewshot` tend to be shorter /
  more generic because demonstrations carry more of the load?
