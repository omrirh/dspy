# Experiment Insights

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*

This file tracks empirical findings that support (or challenge) the core project hypothesis.

> **Status**: Results finalised — `results_v1` (100 runs) + `results_v2` (40 gepa/gepa_merge
> re-runs) + `results_v3` (70 runs: all 50 Iris + 20 GSM8K gepa/gepa_merge, validity-fixed).
> All three merged via `analyze_results.py --log-dir results_v1 results_v2 results_v3`.
> Entries marked **[confirmed v3]** reflect the fully corrected, final numbers.
> Entries marked **[directional]** are consistent but not statistically separable at N=5.

---

## Core Hypothesis

> **Bootstrapped few-shot demonstrations provide a stronger optimization signal than
> reflectively-derived instructions for small LMs on structured tasks**, because declarative
> rules are unreliable behavioural constraints for models that cannot follow them faithfully,
> while demonstrations provide in-context templates the model can imitate directly.

**Final framing (results_v3 complete):** The hypothesis holds **conditionally** — on
structured rule-following tasks (Iris) at small model scale (3B), directionally. It does
not hold on reasoning tasks (GSM8K) at any scale. The scaling story is **saturation, not
inversion**: the demonstration advantage narrows to zero as scale increases, but never
flips negative.

A second finding — unanticipated but clean — is that GEPAFewShot achieves this advantage
at **equal or lower optimization runtime** than vanilla GEPA, making it Pareto-dominant
on Iris/3B (better accuracy, same compute).

---

## Final results table (v1+v2+v3 merged, miprov2 excluded from main comparison)

### Iris

| Optimizer | Llama-3.2-3B mean ±CI95 | Qwen2.5-7B mean ±CI95 |
|---|---|---|
| baseline | 36.00 ±0.00 | 48.00 ±0.00 |
| gepa | 42.40 ±4.78 | 84.00 ±9.62 |
| **gepa_fewshot** | **57.20 ±18.30** | **85.20 ±7.77** |
| gepa_merge | 46.40 ±3.68 | 82.40 ±6.18 |

### GSM8K (v3-corrected; reflection_minibatch_size=10 for all)

| Optimizer | Llama-3.2-3B mean ±CI95 | Qwen2.5-7B mean ±CI95 |
|---|---|---|
| baseline | 66.19 ±0.00 | 74.00 ±0.00 |
| gepa | 74.31 ±4.43 | 83.87 ±3.66 |
| **gepa_fewshot** | **75.21 ±2.66** | **84.85 ±2.56** |
| gepa_merge | 72.69 ±2.43 | 84.60 ±4.86 |

### gepa_fewshot vs. gepa summary (2×2)

|  | Llama-3.2-3B | Qwen2.5-7B |
|---|---|---|
| **Iris** | **+14.8pp** (directional, CI overlap) | +1.2pp (tie) ✓ v3 |
| **GSM8K** | +0.9pp (tie) ✓ v3 | +1.0pp (tie) ✓ v3 |

---

## Insight 1 — Demonstrative advantage is task-type conditional

**Status**: [directional] on Iris/3B; [confirmed v3] as tie elsewhere

- On Iris/3B: gepa_fewshot leads by +14.8pp but CI (±18.3pp) overlaps gepa's CI (±4.8pp).
  The gap is real in expectation but unstable across seeds (N=5).
- On Iris/7B: +1.2pp — a clean statistical tie. All three optimizers cluster within noise.
- On GSM8K at both scales: gaps of 0.9pp and 1.0pp — non-significant at any reasonable threshold.
  With equal reflection budgets (v3), vanilla GEPA's instruction refinement is as effective as
  joint instruction+demo optimization for arithmetic reasoning.

**Mechanism:** Small models cannot reliably follow declarative instructions for structured
classification (e.g., numerical decision boundaries). Demonstrations provide imitable
input→label mappings that bypass this failure mode. For arithmetic reasoning, chain-of-thought
instruction refinement ("solve step by step") is effective even at 3B — demonstrations add no
incremental signal.

---

## Insight 2 — Scaling story: saturation, not inversion

**Status**: [confirmed v3] on Iris; [confirmed v3] on GSM8K

The v2 finding of a −8pp inversion at 7B (gepa outperforming gepa_fewshot) does not survive
equal reflection budgets. The corrected pattern:

- Iris/3B: +14.8pp (directional demo advantage)
- Iris/7B: +1.2pp (tie — advantage fully saturated)
- GSM8K/3B: +0.9pp (tie from the start — reasoning task, no demo advantage)
- GSM8K/7B: +1.0pp (tie)

At 7B, instruction refinement alone reaches parity with joint optimization on both tasks.
The saturation hypothesis is consistent with the mechanism: as instruction-following improves
with scale, demonstrations lose their compensatory role and the two approaches converge.

The inversion seen in v2 was an artefact of the reflection budget confound (gepa ran with
`reflection_minibatch_size=3` vs gepa_fewshot's 10), which disadvantaged gepa unfairly.

---

## Insight 3 — GEPAFewShot adds no runtime overhead; is faster on Iris

**Status**: [confirmed v3] — unanticipated finding

Median optimization runtime comparison (gepa_fewshot vs. gepa):

| Cell | gepa_fewshot | gepa | Δ |
|---|---|---|---|
| Iris / Llama-3.2-3B | **2.93 min** | 3.16 min | −0.23 min (faster) |
| Iris / Qwen2.5-7B | **7.62 min** | 8.88 min | −1.26 min (faster) |
| GSM8K / Llama-3.2-3B | 12.68 min | 10.05 min | +2.63 min (slower) |
| GSM8K / Qwen2.5-7B | 22.68 min | 23.37 min | −0.69 min (≈tied) |

On Iris, gepa_fewshot is consistently faster than gepa despite doing more work (bootstrap +
mutation). The fixed `max_metric_calls` budget dominates wall-clock time; bootstrapping is a
one-time upfront cost and mutation is cheap. On Iris, programs with demonstrations may converge
to high scores earlier, allowing the budget to be consumed more efficiently.

**Practical implication:** On Iris/3B, gepa_fewshot is **Pareto-dominant** over gepa:
+14.8pp accuracy gain at lower optimization cost. This is the strongest single argument for
GEPAFewShot as a practical optimizer.

---

## Insight 4 — gepa_merge adds no consistent benefit over plain gepa

**Status**: [confirmed v3]

| Cell | gepa | gepa_merge | Verdict |
|---|---|---|---|
| Iris / Llama-3.2-3B | 42.40 ±4.78 | 46.40 ±3.68 | gepa_merge +4pp, within CI |
| Iris / Qwen2.5-7B | 84.00 ±9.62 | 82.40 ±6.18 | gepa leads, within CI |
| GSM8K / Llama-3.2-3B | 74.31 ±4.43 | 72.69 ±2.43 | gepa leads, within CI |
| GSM8K / Qwen2.5-7B | 83.87 ±3.66 | 84.60 ±4.86 | tied |

Across all cells, gepa_merge falls within CI of plain gepa. The merge path adds no reliable
accuracy benefit and is not recommended as a default.

---

## Insight 5 — Reflection budget is a critical confound in optimizer comparison

**Status**: [confirmed] — methodological contribution

Unequal `reflection_minibatch_size` between gepa/gepa_merge (3) and gepa_fewshot (10) in
v1/v2 inflated the apparent demonstration advantage:

- Iris/3B: apparent gap +24pp (v2) → corrected +14.8pp (v3), Δ = −9.2pp
- Iris/7B: apparent gap −8pp inversion (v2) → corrected +1.2pp tie (v3)
- GSM8K/3B: apparent gap +1.44pp (v2) → corrected +0.90pp (v3)
- GSM8K/7B: apparent gap +1.26pp (v2) → corrected +0.98pp (v3)

Fair comparison of DSPy optimizers requires equalising all shared hyperparameters, including
reflection budget. This finding has methodological value beyond this project.

---

## Pending analyses

- **Formal statistical tests** — Wilcoxon signed-rank or Mann-Whitney U for gepa_fewshot
  vs. gepa per cell. Priority: Iris/3B (+14.8pp) — may achieve significance at N=5 given
  the gap; all other cells unlikely to.
- **1B model experiment** — Llama-3.2-1B-Instruct on Iris only (baseline + gepa +
  gepa_fewshot, 5 seeds = 15 runs). Motivated by scaling trend: if 3B shows directional
  advantage, 1B may show a larger, statistically significant gap that anchors the trend.
- **Instruction quality analysis** — do gepa_fewshot instructions differ systematically
  (shorter, more generic) compared to gepa, consistent with demonstrations carrying the load?
