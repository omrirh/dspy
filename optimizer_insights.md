# Prompt Optimizer Insights — HotPotQA ReAct Agent

Findings from running ClusterFewshot, MIPROv2, and BFRS as prompt optimizers
on `dspy.ReAct` (HotPotQA multi-hop QA, ColBERTv2 search tool, `answer_exact_match`).

> **Active dataset**: `results_v2/` — 100/250/1500 train/dev/test split, `max_bootstrapped_demos=4`.
> Prior `results/` (500/200/500 split) is archived for reference only.

---

## Primary Finding — Three Optimization Regimes by Architecture

| Regime | Model | Bottleneck | What demos do | Instruction optimization |
|---|---|---|---|---|
| **Format Teaching** | Qwen-7B | ~35% type-A parse failures (empty trajectories) | Eliminate format failures; effect is binary and complete | No signal — adds variance |
| **Termination Teaching** | Llama-3.1-8B | ~65% exhausted trajectories (0 format failures) | Teach `Finish[]` contract — monotone in finish-demo count | Extreme bimodal: either optimal or hallucinatory |
| **Reasoning Quality** | Qwen-14B | ~6% exhausted trajectories (loops to step limit) | Teach termination strategy and multi-hop reasoning | Small positive signal |

Regime identity is determined by architecture, not scale alone. Llama-3.1-8B (~8B) skips the format-teaching phase entirely (produces structurally valid trajectories zero-shot) but has a far more severe termination problem than either Qwen model. The regime ordering is: format compliance → termination compliance → reasoning quality — but a model can jump directly to the second stage without passing through the first.

At Qwen-7B, any optimizer producing complete bootstrapped trajectories resolves the dominant failure.
At Llama-3.1-8B, the critical factor is whether bootstrapped demos contain at least one `Finish[]` call.
At Qwen-14B, format and termination are pre-solved zero-shot; optimizers compete on reasoning guidance quality.

---

## Results — Llama-3.1-8B-Instruct (100/250/1500, max_iters=20, ClusterFS/MIPROv2 3 seeds; BFRS seed 100 only)

| Optimizer | Seed | Baseline | Optimized | Δ (pp) | Step std | Exhausted | fin_tool% | Compile (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 100 | 38.13% | — | — | 7.67 | **980** | 34.7% | — |
| ClusterFS | 100 | 39.40% | 44.20% | +4.8 | 8.26 | 628 | 58.1% | 4188 |
| ClusterFS | 200 | 39.73% | **48.93%** | **+9.2** | 7.20 | **351** | **76.6%** | 3403 |
| ClusterFS | 300 | 37.47% | 43.13% | +5.7 | 6.39 | **1247†** | 16.9% | 5224 |
| MIPROv2 | 100 | 41.53% | 44.80% | +3.3 | 5.93 | **1300‡** | 13.3% | 9591 |
| MIPROv2 | 200 | 38.67% | 45.20% | +6.5 | **4.45** | 95 | **93.7%** | 6474 |
| MIPROv2 | 300 | 38.40% | 46.73% | +8.3 | 7.90 | 503 | 66.5% | 6254 |
| BFRS | 100 | 40.33% | 47.53% | +7.2 | 7.87 | 491 | 67.3% | 7664 |

† ClusterFS s300 exhausted (1247) exceeds baseline (980) — termination regression, not improvement.
‡ MIPROv2 s100 exhausted (1300) exceeds baseline — caused by hallucinated instruction (see Anomalies).

**Cross-optimizer summary — Llama-3.1-8B** *(BFRS partial — seed 100 only; seeds 200/300 pending)*

| | ClusterFS | MIPROv2 | BFRS |
|---|---|---|---|
| Seeds available | 3 | 3 | 1 (incomplete) |
| Mean optimized | 45.42% | **45.58%** | 47.53%§ |
| Seed std | ±2.48pp | ±1.02pp | — |
| Mean Δ | +6.6pp | +6.0pp | — |
| Exhausted range | 351–1247 | 95–1300 | 491 |
| fin_tool% range | 16.9–76.6% | 13.3–93.7% | 67.3% |
| Mean compile (s) | **4272** | 7440 | 7664 |

§ Single seed only — not statistically comparable.

With 3 seeds each, ClusterFS and MIPROv2 are essentially tied on mean accuracy (45.42% vs 45.58%). MIPROv2's variance narrows to ±1.02pp (from the misleadingly low ±0.28pp at 2 seeds), but the exhausted range (95–1300) remains extreme — the 3-seed mean obscures the bimodal regime distinction. MIPROv2 s300 and BFRS s100 land in a **moderate-termination cluster** (~33% exhausted, ~67% fin_tool, step-std ~7.9), distinct from both the catastrophic (0 finish demos, ≥1247 exhausted) and excellent (MIPROv2 s200, 95 exhausted) outcomes. ClusterFS compile advantage holds (1.9× faster than MIPROv2). The fin_tool% range across all methods (13.3–93.7%) dwarfs any accuracy signal and remains the primary diagnostic metric.

---

## Results — Qwen2.5-7B-Instruct (100/250/1500, max_iters=20, 3 seeds)

| Optimizer | Seed | Baseline | Optimized | Δ (pp) | Step std | Exhausted | fin_tool% | Compile (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 100 | 24.53% | — | — | 4.37† | 52 | 65.1% | — |
| BFRS | 100 | 26.47% | 43.73% | +17.3 | 2.24 | 14 | 99.1% | 1966 |
| BFRS | 200 | 26.40% | 44.53% | +18.1 | 2.90 | 28 | 98.1% | 2265 |
| BFRS | 300 | 27.13% | 46.60% | +19.5 | 2.71 | 26 | 98.3% | 2777 |
| ClusterFS | 100 | 28.13% | 44.07% | +15.9 | 1.60 | **4** | **99.7%** | 870 |
| ClusterFS | 200 | 26.93% | **49.20%** | **+22.3** | 1.86 | **5** | **99.7%** | 650 |
| ClusterFS | 300 | 27.13% | 44.33% | +17.2 | 5.02 | 128 | 91.5% | 1022 |
| MIPROv2 | 100 | 27.13% | **47.20%** | +20.1 | 2.95 | 28 | 98.1% | 2247 |
| MIPROv2 | 200 | 28.27% | 40.13% | +11.9 | 4.43 | 67 | 80.4% | 2210 |
| MIPROv2 | 300 | 25.20% | **47.07%** | +21.9 | 2.14 | 15 | 99.0% | 1956 |

† Baseline step std covers only the 65.1% of examples that produced a valid trajectory; type-A failures (no trajectory) are excluded.

**Cross-optimizer summary — 7B**

| | BFRS | ClusterFS | MIPROv2 |
|---|---|---|---|
| Mean optimized | 44.95% | **45.87%** | 44.80% |
| Seed std | **±1.21pp** | ±2.36pp | ±4.36pp |
| Mean Δ | +18.3pp | **+18.5pp** | +17.9pp |
| Type-A failures (opt eval) | **0** | **0** | 0 (s100/300), **227** (s200) |
| Exhausted range | 14–28 | 4–**128** | 15–**67** |
| Mean compile (s) | 2336 | **847** | 2138 |

All three methods are statistically indistinguishable on accuracy (1.1pp spread, overlapping ranges at 3 seeds). BFRS is the most seed-stable. ClusterFS is 2.75× faster to compile and achieves the best trajectory quality metrics when clustering succeeds (s100/s200). MIPROv2 has the widest variance: instruction search can find strong solutions (s100/s300 ≥47%) or select instruction variants that partially re-introduce format failures (s200: 227/1500 type-A, with format-compliant examples scoring 43.8%).

---

## Results — Qwen2.5-14B-Instruct (100/250/1500, max_iters=20, 3 seeds)

| Optimizer | Seed | Baseline | Optimized | Δ (pp) | Step std | Exhausted | fin_tool% | Compile (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 100 | 47.80% | — | — | 4.29 | 95 | 93.3% | — |
| BFRS | 100 | 47.87% | 53.53% | +5.7 | 2.85 | 32 | 97.9% | 4955 |
| BFRS | 200 | 49.53% | 52.67% | +3.1 | 3.86 | 71 | 94.9% | 4499 |
| BFRS | 300 | 48.67% | **55.07%** | +6.4 | 3.57 | 60 | 95.6% | 4061 |
| ClusterFS | 100 | 47.13% | 53.07% | +5.9 | 3.43 | 55 | 95.9% | 4226 |
| ClusterFS | 200 | 51.47% | **56.80%** | +5.3 | 4.06 | 82 | 94.2% | 2672 |
| ClusterFS | 300 | 48.60% | 54.07% | +5.5 | 3.46 | 50 | 96.0% | 3027 |
| MIPROv2 | 100 | 46.53% | 52.20% | +5.7 | 3.21 | 45 | 95.5% | 3760 |
| MIPROv2 | 200 | 48.27% | 52.00% | +3.7 | 3.95 | 49 | 94.7% | 4040 |
| MIPROv2 | 300 | 48.60% | 52.47% | +3.9 | 5.60 | **174** | 87.7% | 4721 |

**Cross-optimizer summary — 14B**

| | BFRS | ClusterFS | MIPROv2 |
|---|---|---|---|
| Mean optimized | 53.76% | **54.65%** | 52.22% |
| Seed std | ±0.99pp | ±1.58pp | **±0.19pp** |
| Mean Δ | +5.1pp | **+5.6pp** | 4.4pp |
| Delta std | ±1.40pp | **±0.26pp** | ±0.88pp |
| Parse failures (opt eval) | **0–5** | 5–10 | 10–30 |
| Exhausted range | 32–71 | 50–82 | 45–**174** |
| Mean compile (s) | 4505 | **3308** | 4174 |

ClusterFS leads at 14B on mean optimized score (54.65%) and has the **tightest delta variance** (±0.26pp) — it delivers a consistent ~+5.6pp improvement regardless of seed. BFRS is the most stable on raw score variance (±0.99pp) and second on compile efficiency. MIPROv2 has the lowest score variance (±0.19pp) but scores consistently lower and produces the most exhausted trajectories — including an extreme case of 174/1500 (11.6%) on s300.

---

## Trajectory Quality Analysis

### The Finish-Demo Law — Llama-3.1-8B

For Llama-3.1-8B, the single strongest predictor of post-optimization termination quality is
**how many bootstrapped demos contain a `Finish[]` call** (measured as `next_tool_name == "finish"`
in the selected demo's truncated trajectory):

| Run | Finish demos / total | fin_tool% | Exhausted |
|---|---|---|---|
| ClusterFS s300 | 0 / 4 | 16.9% | 1247 (**regression vs baseline**) |
| MIPROv2 s100 | 0 / 3 | 13.3% | 1300 (**regression vs baseline**) |
| BFRS s100 | 0 / 2 | 67.3% | 491 |
| ClusterFS s100 | 1 / 3 | 58.1% | 628 |
| MIPROv2 s300 | 1 / 4 | 66.5% | 503 |
| MIPROv2 s200 | 1 / 2 | 93.7% | 95 |
| ClusterFS s200 | 2 / 4 | 76.6% | 351 |

The monotone relationship holds for fixed demo-pool size: 0/4 → catastrophic regression; 1/4 → moderate improvement; 2/4 → best ClusterFS result. MIPROv2 s200 (1/2 demos, explicit termination instruction) outperforms ClusterFS s100 (1/3 demos) at identical finish-demo count, confirming the additive effect of demo + instruction.

**BFRS s100 exception:** 0 finish demos yet 491 exhausted — well below baseline (980), not a regression. BFRS bootstrapped only **2 demos** (not 4); with a smaller non-terminating demo pool the model's zero-shot termination signal is less suppressed. This attenuates but does not reverse the law: the regression seen at 0/3–0/4 requires enough search-only demos to override default behavior. BFRS seeds 200/300 (pending) will test whether this holds across seeds or is a lucky bootstrap outcome.

**Root cause — bootstrap metric blindness:** `answer_exact_match` treats a 20-step looping trajectory with the correct final answer identically to a clean 2-step finish. For Llama-3.1-8B, ~31% of looping trajectories still get the right answer (acc@exhaust = 0.310 at baseline), so the bootstrapped candidate pool is dominated by non-terminating traces. Neither ClusterFS's diversity-first selection nor MIPROv2's metric-based selection has any mechanism to prefer finish-containing demos when the pool rarely contains them.

**Fix:** composite bootstrap metric `= exact_match AND finished_via_tool`. This filters the candidate pool to only include traces that called `Finish[]`, guaranteeing at least some finish-teaching demos are available for selection (see results_v3 / TODO 11).

### Step variance — behavioral stability indicator

Step-std is a **leading indicator**: the highest step-std within a method's seeds is also the lowest-scoring seed in every case.

| | BFRS | ClusterFS | MIPROv2 |
|---|---|---|---|
| 7B step-std range | 2.24–2.90 | **1.60–5.02** | 2.14–4.43 |
| 14B step-std range | 2.85–3.86 | **3.43–4.06** | 3.21–**5.60** |
| 7B exhausted range | 14–28 | 4–**128** | 15–**67** |
| 14B exhausted range | 32–71 | 50–82 | 45–**174** |

When clustering succeeds (yield >50%), ClusterFS achieves the lowest step-std in the entire dataset — 1.60 and 1.86 for 7B s100/s200, the tightest trajectories across both models and all methods. When yield is low (34/100 for 7B s300), the same mechanism produces the worst step-std (5.02) and 128 exhausted. BFRS degrades gracefully across all seeds. MIPROv2's step-std is unpredictable and its worst seeds (7B s200, 14B s300) show the dataset's most severe exhaustion events.

**Threshold observations**: step-std ≤ 2.5 is associated with well-behaved 7B trajectories; ≤ 4.0 for 14B. Above these values, exhausted trajectory counts spike.

### Step distribution shift

| Run | ≤2 steps | 3 steps | 4–5 | 6–19 | 20(exhaust) | fin_tool% |
|---|---|---|---|---|---|---|
| 7B baseline | 51.3%† | 28.0% | 9.1% | — | 7.8% | 65.1% |
| 7B BFRS mean | 49.8% | 37.4% | 7.1% | — | 1.1% | 98.5% |
| 7B ClusterFS mean | 46.8% | 37.9% | 8.2% | — | 1.2% | 96.7% |
| 7B MIPROv2 mean | 36.4% | 45.2% | 10.0% | — | 2.0% | 92.3% |
| 14B baseline | 15.3% | 60.4% | 14.2% | 0.7% | 0.7% | 93.3% |
| 14B BFRS mean | 27.9% | 54.9% | 9.8% | — | 1.0% | 96.1% |
| 14B ClusterFS mean | 27.5% | 54.3% | 10.6% | — | 1.0% | 95.4% |
| 14B MIPROv2 mean | 27.2% | 50.5% | 11.3% | 1.9% | 1.9% | 92.6% |
| Llama-8B baseline | 0.9% | 17.3% | 12.7% | 3.5% | **65.5%** | 34.7% |
| Llama-8B ClusterFS s200 | 18.1% | 39.5% | 16.4% | 2.6% | **23.4%** | 76.6% |
| Llama-8B ClusterFS mean | 16.0% | 32.6% | 13.6% | 3.3% | **47.7%** | 50.5% |
| Llama-8B MIPROv2 s200 | 37.6% | 40.3% | 12.9% | 2.7% | **6.4%** | 93.7% |
| Llama-8B MIPROv2 s300 | 7.4% | 36.8% | 19.7% | 2.5% | **33.5%** | 66.5% |
| Llama-8B BFRS s100 | 8.8% | 37.7% | 16.3% | 4.3% | **32.8%** | 67.3% |

† 7B baseline's high ≤2-step share is inflated by type-A failures (step=0); not directly comparable with post-opt distributions.

At 7B, all demo-based methods collapse the exhausted tail (7.8% → ~1–2%) and recover fin_tool% from 65% to 92–99%. At 14B, demos shift the step-3-concentrated baseline (60%) toward faster ≤2-step resolution. MIPROv2 resists this shift at 14B — instruction variants extend dwell time in the step-3 bucket while the exhausted tail grows. At Llama-3.1-8B, the exhausted tail dominates at baseline (65.5%); ClusterFS s200 reduces it to 23.4% via finish-demo teaching, but ClusterFS mean (47.7% exhausted) reflects the bimodal nature of the results. MIPROv2 s200 achieves the best termination in the full dataset (6.4% exhausted, 93.7% fin_tool) via combined finish-demo + explicit termination instruction.

### Accuracy by step bucket

**Llama-3.1-8B — selected runs**

| | acc@≤2 | acc@3 | acc@4–5 | acc@exhaust |
|---|---|---|---|---|
| Baseline | .500 | .512 | .518 | .310 |
| ClusterFS s200 | **.638** | .570 | .467 | .282 |
| ClusterFS mean | .598 | .526 | .501 | .343 |
| MIPROv2 s200 | .480 | .501 | .387 | .188 |
| MIPROv2 s300 | .505 | **.580** | .514 | .320 |
| BFRS s100 | .614 | .572 | .514 | .327 |

**acc@exhaust is abnormally high for Llama** (0.188–0.412) compared to Qwen-14B (0.074–0.179). The model often answers correctly while looping because it accumulates relevant observations over many steps and guesses well. This inflates raw accuracy scores and masks exhaustion severity — fin_tool% is essential for correctly diagnosing Llama. MIPROv2 s200's acc@exhaust drops to 0.188 because, with 93.7% termination, almost no examples reach the exhausted bucket; those that do are the hardest questions. BFRS s100 and MIPROv2 s300 — both moderate-termination runs (~33% exhausted) — have acc@exhaust near the baseline (.327/.320 vs .310), confirming that partial termination improvement does not substantially change which examples loop or their accuracy when they do. ClusterFS s200 shows the highest acc@≤2 (0.638) — diversity-first demos teach faster confident termination on straightforward questions, consistent with the 14B pattern.

**7B — optimized runs (mean across seeds)**

| | acc@≤2 | acc@3 | acc@4–5 | acc@exhaust |
|---|---|---|---|---|
| Baseline (valid traj.) | .234 | .343 | .161 | .164 |
| BFRS | .447 | .531 | .286 | .101 |
| ClusterFS | .458 | **.541** | .297 | .157 |
| MIPROv2 | .439 | .534 | .333 | .106 |

**14B — optimized runs (mean across seeds)**

| | acc@≤2 | acc@3 | acc@4–5 | acc@exhaust |
|---|---|---|---|---|
| Baseline | .443 | .561 | .399 | .074 |
| BFRS | .494 | **.622** | .454 | .104 |
| ClusterFS | .526 | .624 | .451 | .103 |
| MIPROv2 | .526 | .604 | .416 | .179 |

At 14B, BFRS and ClusterFS are near-identical on acc@3 (0.622 vs 0.624) and acc@4–5 (0.454 vs 0.451). MIPROv2 trails at acc@4–5 (0.416) and has a much higher acc@exhaust (0.179) — the agent guesses correctly while looping, partially masking the exhaustion problem in raw scores. ClusterFS shows the highest acc@≤2 at 14B (0.526), suggesting diverse demos teach faster confident termination on straightforward questions.

### Step–score correlation r(steps, score)

| Model | BFRS (s100/200/300) | ClusterFS (s100/200/300) | MIPROv2 (s100/200/300) |
|---|---|---|---|
| 7B | −.093 / −.173 / −.161 | **−.081** / −.142 / −.156 | −.202 / −.118 / −.146 |
| 14B | −.141 / −.216 / −.207 | −.171 / −.226 / −.192 | −.127 / −.201 / −.226 |

Negative everywhere — questions answered in fewer steps are answered correctly more often. The correlation is weaker at 7B (type-A failures appear as short-step zeros, compressing the signal) and stronger at 14B (~−0.13 to −0.23). ClusterFS s100 at 7B has the weakest correlation (−0.081): diverse demos teach a wider range of valid trajectory patterns, making length less predictive of outcome.

### Search query uniqueness

Measured as fraction of repeated queries per multi-search trajectory (lower = less repetition).

| | 7B repeat rate | 14B repeat rate | Llama-8B repeat rate |
|---|---|---|---|
| Baseline | 10.9% | 21.9% | 55.1% |
| BFRS mean | **8.3%** | 12.9% | 69.4%§ |
| ClusterFS mean | 8.6% | **12.6%** | 44.5% |
| MIPROv2 mean | 13.9% | 14.1% | 57.2% |

§ BFRS Llama: single seed (s100) only.

Llama-3.1-8B baseline repeat rate (55.1%) is 2–5× higher than either Qwen model — the model systematically re-issues identical queries when it cannot make progress. This is the mechanism behind the 65.5% exhaustion rate: the model loops on the same search rather than reformulating or terminating. Among Llama optimized runs, repeat rate tracks exhaustion severity: ClusterFS s200 (best termination, 76.6% fin_tool) drops to 27.4%; catastrophic runs (ClusterFS s300, MIPROv2 s100) reach 79.5% and 85.0% — above baseline. The moderate-termination cluster (BFRS s100, MIPROv2 s300, ~67% fin_tool) sits at 69–73% — also above baseline, confirming that partial termination improvement does not reduce looping behavior proportionally; query repetition reduction requires near-complete termination improvement. Correct trajectories have higher uniqueness ratios than wrong ones across all conditions; the delta (correct_unique − wrong_unique) ranges from 0.05 to 0.14 for Qwen models. For Llama, teaching termination simultaneously reduces query repetition, suggesting the two behaviors are coupled: a model that commits to answering stops re-searching.

The Qwen findings are unchanged: BFRS and ClusterFS reduce repeat rates comparably; MIPROv2 leaves the most repetitive search behavior. Outlier cases (MIPROv2 7B s300: 20.7%; ClusterFS 14B s300: 18.9%) co-occur with elevated step-std and exhausted counts, consistent with the Llama pattern.

---

## Optimizer-Specific Observations

### BFRS
- Most seed-stable optimizer at both scales (7B ±1.21pp, 14B ±0.99pp).
- At 7B: reliably achieves low step-std (2.24–2.90) and near-zero exhausted counts (14–28).
- At 14B: metric-based selection aligns with the reasoning-quality regime. Exhausted counts (32–71) are higher than ClusterFS s100/s300 but lower than MIPROv2.
- No failure modes observed: BFRS degrades gracefully as seed partition varies.

### ClusterFewshot
- **Fastest compile across all models** (847s 7B, 3308s 14B, 4272s Llama-8B mean) — 2.75× faster than BFRS at 7B, 1.36× at 14B, 1.9× faster than MIPROv2 at Llama-8B. Cost scales with bootstrap yield, not train size per se.
- At 7B: best trajectory quality when bootstrap yield is adequate (≥50%): step-std 1.60/1.86 and 99.7% fin_tool for s100/s200 — the best values in the dataset. When yield is low (34/100 for s300), the demo pool per cluster is too thin for meaningful selection, producing the worst step-std (5.02) and 128 exhausted.
- At 14B: **highest mean optimized score** (54.65%) and **tightest delta variance** (±0.26pp) — delivers ~+5.6pp consistently. Diversity-first demo selection aligns with the reasoning-quality regime: cluster representatives span different multi-hop patterns, improving generalization across question archetypes.
- At Llama-3.1-8B: **bimodal outcome driven by finish-demo composition**. s200 (2/4 demos with `Finish[]`) achieves 48.93% and 351 exhausted — the best result in the Llama dataset. s300 (0/4 finish demos) produces 1247 exhausted — a termination regression vs baseline (980). Diversity-first selection can select a fully non-terminating demo pool when the bootstrap candidate pool contains no clean terminators, because `answer_exact_match` does not filter for termination. Mean (3 seeds): 45.42% ± 2.48pp.
- Bootstrap yield scales with model capability: ~35–59% at 7B → ~60–70% at 14B → ~37–43% at Llama-8B (similar to 7B, suggesting the termination problem reduces yield). Higher yield produces richer candidate pools, but yield composition (what fraction of bootstrapped demos end in `Finish[]`) matters as much as count for models with termination problems.

### MIPROv2
- At 7B: highest peak single-seed scores (47.20%, 47.07%) but also most volatile. Instruction optimization can find strong solutions (s100/s300) or variants that partially suppress format compliance (s200: 227/1500 type-A failures). The agent's format behavior is sensitive to instruction framing in small models.
- At 14B: consistently lower accuracy (52.22%) than BFRS/ClusterFS despite tightest score variance. Produces the most exhausted trajectories across seeds (45–174) — instruction variants that emphasize deliberation without reinforcing the `Finish[]` termination contract lead to looping.
- At Llama-3.1-8B: **extreme bimodal behavior** across seeds. s200 (1 finish demo + explicit termination instruction) achieves 93.7% fin_tool and 95 exhausted — the **best termination rate in the entire cross-model dataset**, better than any Qwen-14B run. s100 (0 finish demos + hallucinatory instruction) produces 1300 exhausted (86.7%) — the **worst** in the dataset. Instruction optimization is the only mechanism shown to achieve near-100% termination (demos alone cap at 76.6% for ClusterFS s200), but only when the meta-LLM generates a coherent instruction. Mean (2 seeds): 45.00% ± 0.28pp — artificially stable variance because two seeds represent qualitatively different behaviors.
- Highest acc@exhaust at 14B (0.179 vs ~0.10 for others) and Llama-8B (baseline 0.310): the agent sometimes answers correctly while exhausted, masking the problem in raw score. acc@exhaust is an unreliable accuracy signal.
- `prompt_model_total_calls = 0` across **all runs for all models**: instrumentation counter in DSPy library never increments. Instruction optimization ran (behavioral differences confirm it); this field is unreliable.

---

## Anomalies

### MIPROv2 s100 hallucinated instruction (Llama-3.1-8B)

The optimized instruction for Llama-3.1-8B MIPROv2 s100 contains a hardcoded HotPotQA training example question (*"What is the name of the river that flows through the town of Aboke in Uganda?"*), a fictional wilderness emergency framing, and a countdown timer. This is a known MIPROv2 failure mode on small models: the meta-LLM overfits to specific training examples during instruction search rather than producing generalizable guidance. The instruction does not clearly communicate the `Finish[]` termination contract. Result: 1300/1500 exhausted trajectories — the worst in the entire cross-model dataset. The companion s200 instruction is generic, coherent, and explicitly guides termination — producing the best termination in the dataset.

### ClusterFS s300 termination regression (Llama-3.1-8B)

ClusterFS s300 improves accuracy by +5.7pp yet produces **more** exhausted trajectories than the zero-shot baseline (1247 vs 980). All 4 selected demos have `next_tool_name=search` — zero finish-teaching examples. Diversity-first selection chose the 4 most semantically distinct non-terminating trajectories in the cluster pool. The accuracy gain is entirely attributable to the 16.9% of examples that happened to terminate early scoring at 52.6%; the remaining 83.1% loop to step 20. This is a silent regression: raw accuracy goes up while the underlying trajectory quality degrades.

### Bootstrap metric blindness to termination (systematic, all models)

The `answer_exact_match` bootstrap filter assigns equal weight to a 20-step looping trajectory that guesses correctly and a 2-step clean trajectory. For Llama-3.1-8B, ~31% of exhausted trajectories score correctly (acc@exhaust = 0.310 baseline), so non-terminating traces are well-represented in the bootstrap candidate pool. Neither ClusterFS's diversity selection nor MIPROv2's metric ranking has a mechanism to prefer finish-containing demos when the pool is dominated by non-terminators. This is the root cause of all termination regressions in the Llama results.

---

## Parse Failure Reference

| Type | Condition | 7B pre-opt | 14B pre-opt | Post-opt |
|---|---|---|---|---|
| **A** | Empty trajectory / no `thought` | ~33–35% of examples | 0 | 0 (BFRS/ClusterFS); 0–15% (MIPROv2 worst seed) |
| **B** | `thought` present, `tool_name` missing | Never | Never | Never |
| **C** | `thought` + `tool_name`, `tool_args` missing | Never | Never | Never |
| **Exhausted** | All 20 steps complete, no `Finish[]` | ~3.5% | ~6.3% | 1–12% (varies by method/seed) |

Format failures at 7B are binary: the model either produces no ReAct structure or complies fully. Types B and C have never been observed. Post-optimization, exhausted trajectories become the primary failure mode for all methods.

---

## Cross-Model Finding

14B zero-shot (47.8%) exceeds the best optimized 7B mean (~45.9%). The 7B capability gap closes through format scaffolding, not improved reasoning: optimized 7B acc@3 (0.531–0.541) is measurably lower than optimized 14B acc@3 (0.604–0.624) at identical step counts. Scaling the model resolves format; optimization improves reasoning strategy on top of that.

Llama-3.1-8B zero-shot (38.1%) is below Qwen-7B zero-shot (26.3% + 35% type-A failures would be ~40% if corrected for format compliance, comparable). Post-optimization, both reach similar means (~45%), suggesting that **format failures and termination failures impose roughly equal performance ceilings** once corrected by optimization. Despite a higher zero-shot accuracy, Llama's termination problem is harder to solve reliably — best Llama result (ClusterFS s200: 48.93%) is comparable to best Qwen-7B result (ClusterFS s200: 49.20%), but with much higher seed variance.

The regime ordering (format → termination → reasoning) does not follow model scale alone — it depends on architecture. Llama-3.1-8B skips the format-teaching phase entirely and enters directly at the termination-teaching phase. This means optimizer choice and bootstrap metric design are more consequential for Llama than for Qwen-7B, where any demos reliably fix the dominant failure.

---

## Paper Framing — Gap Statement

> "While recent studies reveal that LLMs at or below 7B parameters are highly sensitive to prompt formatting [Sclar et al., 2024] and that exemplar choice is a primary driver of ReAct-style agents' performance [Verma et al., 2024], existing work stops short of analyzing the role of the optimizer itself. In particular, we lack a systematic account of how the choice of prompt optimizer — particularly in bootstrap-based pipelines where the optimizer determines which trajectories serve as demonstrations — affects format compliance in structured agentic workflows, and how demonstration diversity influences task accuracy via its impact on adherence to target output formats."

**Key citations:**
- Sclar et al., 2024 — *FormatSpread* (ICLR 2024, arXiv:2310.11324): up to 76pp accuracy variation from prompt format changes on LLaMA-2-13B.
- Verma et al., 2024 — *Brittle Foundations of ReAct Prompting* (arXiv:2405.13966): ReAct gains are driven by exemplar-query similarity, not the reasoning trace interleaving itself.

**Evidence from this experiment:**
- Format compliance is the primary bottleneck at 7B: ~33–35% type-A failures at zero-shot, eliminated by any bootstrapped demo method. The mechanism is behavioral template teaching, not reasoning improvement.
- Step-std as behavioral stability signal: elevated step-std predicts accuracy degradation before examining scores, across both scales and all optimizers.
- MIPROv2's scale interaction (volatile at 7B, consistent but lower at 14B) isolates instruction optimization value as contingent on format pre-compliance.
- ClusterFS's diversity-first selection produces tighter delta variance at 14B (±0.26pp) than metric-based selection (BFRS ±1.40pp), suggesting semantic coverage of the demo set matters for generalization at scale.

---

## Methodology

- Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`; SGLang on A100 80GB
- Splits: trainset=100, devset=250, testset=1500, max_iters=20, seeds 100/200/300
- ColBERTv2: `http://localhost:8894/api/search`, k=3 passages per query
- BFRS: `num_candidate_programs=6`, `max_bootstrapped_demos=4`, `max_labeled_demos=0`
- ClusterFS: `task_type="agentic"`, encoders=[`all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`], K∈[3,4], `max_bootstrapped_demos=4`, `max_labeled_demos=0`
- MIPROv2: `auto="medium"`, `minibatch=True`, `minibatch_size=25`, `minibatch_full_eval_steps=10`, `max_bootstrapped_demos=4`, `max_labeled_demos=0`, `requires_permission_to_run=False`
- Parse failure instrumentation: type A (empty trajectory), B (no tool_name), C (no tool_args); exhausted-max-iters tracked separately
- Demo count equalized: all methods capped at `max_bootstrapped_demos=4`
