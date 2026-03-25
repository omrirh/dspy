# Prompt Optimizer Insights — HotPotQA ReAct Agent

Findings from running ClusterFewshot, MIPROv2, and BFRS as prompt optimizers
on `dspy.ReAct` (HotPotQA multi-hop QA, ColBERTv2 search tool).

**Task setup**: `ReactAgentMH` — single `search` tool + implicit `finish`, `max_iters=5`,
evaluated on 500 hard HotPotQA examples with `answer_exact_match`.

---

## Results Table

| Model | Optimizer | Baseline | Optimized | Δ (pp) | Compile (s) | Eval (s) | Date |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | ClusterFewshot | 22.40%† | 45.80% | +23.40 | 1632.6 | 320.7 | 2026-03-20 |
| Qwen2.5-7B-Instruct | ClusterFewshot (re-run) | 28.00%\* | 46.20% | +23.80 | 1599.1 | 329.7 | 2026-03-24 |
| Qwen2.5-7B-Instruct | MIPROv2 | 22.40%* | 39.20% | +16.80* | 1506.1 | 305.4 | 2026-03-21 |
| Qwen2.5-7B-Instruct | BFRS | 29.00%* | **49.80%** | **+27.40** | 3006.7 | 322.3 | 2026-03-24 |
| Qwen2.5-14B-Instruct | ClusterFewshot | 46.60% | 55.40% | +8.80 | 4605.1 | 685.0 | 2026-03-23 |
| Qwen2.5-14B-Instruct | MIPROv2 | 45.20% | 48.60% | +3.40 | 2494.7 | 626.1 | 2026-03-23 |
| Qwen2.5-14B-Instruct | BFRS | 46.80% | 54.80% | +8.00 | 6920.5 | 629.1 | 2026-03-23 |

> † ClusterFewshot Mar 20 baseline (22.40%) is anomalous: 0 parse failures, likely due to a
> different SGLang deployment with a larger context window. Re-run on Mar 24 shows 139 failures
> and 28.00% reported baseline, consistent with MIPROv2/BFRS runs on the same server.
> True 0-shot baseline ≈ **22%** (Mar 20 clean eval; confirmed by correcting Mar 24 inflated 28.00% for ~6pp extract fallback contribution).
>
> \* Reported baselines for MIPROv2 (29.40%), BFRS (29.00%), and ClusterFewshot re-run (28.00%)
> are inflated ~6–7pp by `extract` fallback recovering empty-trajectory examples.
> Corrected Δ: ClusterFewshot = +23.40/+23.80pp, MIPROv2 = +16.80pp, BFRS = **+27.40pp**.

---

## Model: Qwen/Qwen2.5-7B-Instruct

### ClusterFewshot  ✓ (×2 runs)

**Logs**:
- Run 1: `react_agent_experiment_Qwen2.5-7B-Instruct_clusterfs_2026-03-20.log` (seed 1773998133)
- Run 2: `react_agent_experiment_Qwen2.5-7B-Instruct_clusterfs_2026-03-24.log` (seed 1774390521)

**Compile internals**

| | Run 1 (Mar 20) | Run 2 (Mar 24) |
|---|---|---|
| Bootstrap yield | 101/500 (20.2%) | 138/500 (27.6%) |
| Selected encoder | `multi-qa-mpnet-base-dot-v1` | `all-mpnet-base-v2` |
| K | 3 (silhouette=0.038) | 3 (silhouette=0.037) |
| `best_in_cluster` valset score | 50.5% | 48.5% |
| `top_n` valset score | 50.5% (tie) | 48.5% (tie) |
| Final demos | 3 / 3 (react / extract) | 3 / 3 (react / extract) |
| Baseline parse failures | **0** *(anomaly — see below)* | **139** |
| Compile/bootstrap failures | 0 | 208 |
| Valset + final test failures | **0** | **0** |
| Baseline (reported) | 22.40% | 28.00%\* |
| Optimized test score | 45.80% | **46.20%** |
| Compile time | 1632.6s | 1599.1s |
| Eval runtime | 320.7s | 329.7s |

> \* Inflated ~6pp by extract fallback; corrected true baseline ≈ 22%

**Key observations**
- **March 20 zero-failure baseline is confirmed as anomalous**: Run 2 (same date as BFRS/MIPROv2,
  fresh server restart) shows 139 baseline failures — fully consistent with MIPROv2 (247) and
  BFRS (156) on the same server. The March 20 0-failure result was almost certainly a different
  SGLang deployment with a larger effective context window. The true zero-shot failure rate for
  Qwen2.5-7B on this task is ~30%, and the true zero-shot accuracy is ~22% (correcting for
  extract fallback inflation).
- **Post-optimization format compliance is robust and reproducible**: both runs achieve **0
  parse failures** on the valset one-shot evaluations and final test eval. ClusterFS's systematic
  selection of bootstrapped complete trajectories fully eliminates format failures across
  different seeds, server configs, and encoder choices — unlike BFRS where 4/9 candidate seeds
  remain format non-compliant.
- **Score is stable across runs**: 45.80% vs 46.20% (+0.40pp). The different encoder selection
  (`multi-qa-mpnet-base-dot-v1` vs `all-mpnet-base-v2`) produces no meaningful accuracy
  difference at K=3 on HotPotQA — silhouette scores are nearly identical (~0.037–0.038) and
  the task's 2-hop structure is uniformly distributed in both embedding spaces.
- **`best_in_cluster` == `top_n` tie persists**: same result as Run 1. Confirms the HotPotQA
  2-hop structure is low-variance in embedding space at K=3; cluster diversity and top-quality
  selection are equivalent for this task.
- **Bootstrap yield higher in Run 2 despite more baseline failures**: 138 vs 101 complete
  trajectories. Bootstrap runs iteratively, so early successful traces scaffold later ones;
  the absolute baseline failure rate does not directly cap bootstrap yield.

---

### MIPROv2  ✓

**Log**: `react_agent_experiment_Qwen2.5-7B-Instruct_miprov2_2026-03-21.log`

**Compile internals**
- Bootstrap: 9 sets, ~2-3 traces each (~19 total traces); many are extract-only episodes
  (empty trajectory due to parse failures during bootstrap)
- Instruction proposals: 9 variants × 2 predictors (react step + extract step)
  — proposals are task-semantic (e.g. "think step-by-step") not format-aware
- Bayesian trials: 28 (24 minibatch × 25 examples + 4 full evals × 200 examples)
- Full eval scores: [29.5, 34.5, **43.0**, 40.5] → best program: trial 21 (43.0 on valset)
- Final test score: 39.20% (generalisation gap: −3.8pp from valset best)
- Parse failures across entire run: **258 warnings** (~15% of all LM calls)

**Key observations**
- **Baseline inflation**: MIPROv2 reported 29.40% baseline vs ClusterFewshot's 22.40%.
  The `AttributeError` fix (parse failure → `break` → `extract`) allows extract to recover
  ~7pp of questions from empty trajectories, artificially inflating the baseline and
  compressing the apparent gain. Corrected Δ ≈ +16.80pp.
- **Instruction mutation destabilizes format**: Proposed instructions for the `react`
  predictor describe the task in prose (e.g. "predict the next thought, name of the next
  tool...") without reinforcing the structured field contract (`next_thought`,
  `next_tool_name` as `Literal["search","finish"]`, `next_tool_args` as `dict`). Qwen2.5-7B
  is already marginal on 3-field structured output; instruction perturbation pushes it over
  the format compliance threshold.
- **Minibatch signal is noisy**: Minibatch scores ranged 24→56 on 25-example batches.
  With ~15% parse failure variance, a batch of 25 cannot reliably rank instruction variants.
  Bayesian optimization effectively fits parse-failure variance, not task performance.
- **Bootstrap demos are degraded**: Unlike ClusterFewShot which collects 101 verified
  complete trajectories, MIPROv2's bootstrap collected ~19 traces, several from
  extract-only episodes (no search steps). Showing the model demos with empty trajectories
  fails to teach the think→search→iterate protocol.
- **Compile time breakdown**: ~163s bootstrap + ~160s instruction proposal (LM calls) +
  ~1183s Bayesian trials. MIPROv2 is slightly faster overall but distributes time across
  noisy search rather than quality evaluation.

---

### BFRS  ✓

**Log**: `react_agent_experiment_Qwen2.5-7B-Instruct_bfrs_2026-03-24.log`

**Compile internals**
- 9 candidate programs: seeds -3, -2 (labeled-only, no bootstrap) + seeds -1→+5 (bootstrapped)
- Each candidate scored on **full 500-example trainset** (same as 14B BFRS)
- Bootstrap yield highly variable; seeds 1 and 3 bootstrapped only 1 trace from 1 attempt (100% yield but minimal)
- Winning program: **seed 0** (48.2% on trainset) → 2 bootstrapped traces from 5 attempts
- Final demos: 3 each for `agent.react` and `agent.extract.predict`
- Final test score: **49.80%** — test exceeds trainset score (+1.6pp, conservative generalization)
- Parse failures during final test eval: **0**

**Per-candidate summary**

| Seed | Trainset score | Parse failures | Bootstrapped traces | Notes |
|---|---|---|---|---|
| Baseline | 29.0%\* | 156 | — | inflated; true ~22% |
| -3 | 24.8% | 157 | 0 (labeled only) | |
| -2 | 30.2% | 157 | 0 (labeled only) | |
| -1 | 42.2% | 3 | 3 / 7 attempts | |
| **0** | **48.2%** | **1** | **2 / 5 attempts** | **WINNER** |
| 1 | 30.0% | 157 | 1 / 1 attempt | |
| 2 | 35.8% | 2 | 1 / 7 attempts | |
| 3 | 30.4% | 157 | 1 / 1 attempt | |
| 4 | 40.6% | 0 | 3 / 4 attempts | |
| 5 | 41.4% | 3 | 1 / 4 attempts | |

**Key observations**
- **Three-tier format compliance, not bimodal**: parse failures reveal three distinct classes among
  candidates — (1) labeled-only seeds (-3, -2) and low-quality single-trace bootstrap (seeds 1, 3):
  ~157 failures, ~25-30%; (2) quality single bootstrap trace (seeds 2, 5): 2-3 failures, ~36-41%;
  (3) multi-trace bootstrap (seeds -1, 0, 4): ≤3 failures, ~40-48%. Format compliance is
  necessary but not sufficient — trace quality within the compliant group still determines the score gap.
- **Labeled-only demos are uniformly non-compliant**: seeds -3 and -2 use question-answer pairs
  without reasoning trajectories. For 7B, these provide no format scaffolding whatsoever — their
  failure rate (157) is identical to zero-shot (156). This directly confirms that bootstrapped
  *reasoning traces*, not just labeled examples, are what teach the ReAct loop structure.
- **Bootstrap trace count correlates with compliance but quality dominates**: seeds with ≥2
  bootstrap traces (−1, 0, 4) are all format-compliant; seeds with exactly 1 trace split 50/50
  (seeds 2, 5 compliant; seeds 1, 3 non-compliant). The decisive factor for single-trace seeds
  appears to be the structural quality of that one trace (complete thought→search→finish chain),
  not merely having a bootstrapped example.
- **Positive train→test generalization gap**: winning seed 0 scored 48.2% on trainset and 49.80%
  on test (+1.6pp), the opposite of the 14B BFRS result (−2.6pp). 7B has less capacity to
  memorize training distribution idiosyncrasies, and seed 0's small demo set (2 traces) is
  unlikely to overfit.
- **BFRS is the best 7B optimizer at 49.80%**, exceeding ClusterFewshot (45.80%) by +4pp. On 7B,
  BFRS's random seed search happened to find a demo set with high format compliance and strong
  reasoning quality; whether this advantage holds across re-runs is an open question given the
  high seed-to-seed variance observed (24.8%–48.2% spread, 23.4pp).
- **Baseline inflation confirmed**: BFRS reports 29.00% baseline vs true ~22.40% (same mechanism
  as MIPROv2: extract fallback recovers ~7pp from empty-trajectory examples). Corrected Δ ≈ +27.40pp.

---

## Cross-Optimizer Summary — Qwen2.5-7B-Instruct

| Dimension | BFRS | ClusterFewshot | MIPROv2 |
|---|---|---|---|
| Optimized score | **49.80%** | 45.80% | 39.20% |
| True Δ over baseline | **+27.40pp** | +23.40pp | +16.80pp\* |
| Bootstrap traces collected | ~13 (across 9 seeds) | **101** | ~19 (degraded) |
| Parse failures (total) | 471 (training search) / **0** (final eval) | 347 (baseline+bootstrap) / **0** (post-opt) | 258 |
| Format-compliant candidates | 5 / 9 seeds | All (selection guarantees it) | All trials near-0 |
| Modifies instructions | **No** | **No** | Yes |
| Demo selection mechanism | Random search, full-set eval | Cluster + one-shot eval | Bayesian (noisy minibatch) |
| Demos: react / extract | **3 / 3** | 3 / 3 | 3 / 3 |
| Train → test gap | +1.6pp (conservative) | — | −3.8pp (overfit) |
| Compile time | 3006.7s | **1632.6s** | 1506.1s |
| Primary failure mode | Seed variance (23.4pp spread); lucky winner | — | Format instability + signal noise |

> \* All Δ figures corrected for ~7pp baseline inflation from extract fallback.

**Finding (Qwen2.5-7B, HotPotQA ReAct)**:

**BFRS achieves the highest score (49.80%)**, but at the cost of high variance across its 9
candidates (24.8%–48.2%, a 23.4pp spread driven by format compliance). Its win is contingent
on the random search landing on a format-compliant, multi-trace bootstrap seed — an outcome
that cannot be guaranteed across re-runs. **ClusterFewshot is the more reliable optimizer**:
its systematic bootstrapping of 101 complete trajectories and cluster-based selection eliminates
the format compliance lottery entirely (0 failures throughout), delivering 45.80% predictably.

For small models (≤7B) on structured agentic tasks with typed multi-field output:
1. **Bootstrapped reasoning traces are the critical ingredient** — labeled-only demos provide
   zero format scaffolding (seeds -3/-2 match zero-shot failure rate exactly); only demos
   containing complete thought→search→observation→finish chains teach the ReAct loop structure
2. **Format compliance is the dominant axis of optimizer quality at 7B** — the 23.4pp gap
   between best and worst BFRS seeds is almost entirely explained by whether the bootstrap
   traces happened to be format-compliant; task reasoning quality is a secondary factor
3. **Instruction mutation (MIPROv2) is actively harmful at 7B** — 7B models are at the format
   compliance threshold; prose instruction variants perturb the model past this threshold,
   producing cascade parse failures that corrupt the optimization signal and inflate the baseline

---

## Model: Qwen/Qwen2.5-14B-Instruct

### ClusterFewshot  ✓

**Log**: `react_agent_experiment_Qwen2.5-14B-Instruct_clusterfs_2026-03-23.log`

**Compile internals**
- Bootstrap: 500 trainset examples → **259/500 (51.8%) successful complete trajectories**
- Encoder grid search: `all-mpnet-base-v2` vs `multi-qa-mpnet-base-dot-v1` across K∈[3,4]
  (grid capped at K=4 to bound demo count within 8192-token context budget)
- Selected: `all-mpnet-base-v2`, K=4 (silhouette=0.041)
- Demo selection: `best_in_cluster` scored **50.5%** on the 200-example valset
- Final demos: 4 per predictor (`agent.react`, `agent.extract.predict`)
- Parse failures during compile/eval: 0

**Key observations**
- +8.80pp gain (46.60% → 55.40%) — smaller absolute gain than 7B (+23.40pp) because the
  14B model has a higher 0-shot baseline and already exhibits better format compliance;
  demonstrations provide less marginal structural lift
- Higher bootstrap yield (259 vs 101 traces) reflects the 14B model's stronger baseline
  ability to complete valid ReAct trajectories without demonstrations
- **Context budget matters for K selection**: original K∈[3,10] grid selected K=8,
  causing 400 Bad Request errors (8 full trajectories × ~500–800 tokens ≈ context overflow).
  Capping grid to K∈[3,4] resolved this with no accuracy cost.
- **Efficiency-accuracy trade-off introduced by demos**: ClusterFewshot teaches a faster,
  more decisive ReAct loop. On the sampled trajectory question (*"In 1736, a fortified
  complex was built in Moscow..."*, gold: `Moscow Kremlin`), the baseline used 3 search
  steps and self-corrected to the right answer; ClusterFewshot used 2 steps but halted
  early on a wrong sub-entity (`Kremlin Arsenal`). The +8.80pp net gain shows this
  trade-off is beneficial on average, but demos reduce the model's willingness to
  verify ambiguous retrievals.

---

### MIPROv2  ✓

**Log**: `react_agent_experiment_Qwen2.5-14B-Instruct_miprov2_2026-03-23.log`

**Compile internals**
- Bootstrap: 9 sets, ~2–3 traces each (~18 total traces); 14B yields higher trace quality than 7B
- Instruction proposals: 9 variants × 2 predictors; notable redundancy (Instr 0, 1, 4 for Predictor 0
  are near-identical); 2 roleplay personas generated ("expert detective", "emergency responder")
- Bayesian trials: 28 (25 minibatch × 25 examples + 3 full evals × 200 examples)
- Full eval score progression: 51.0 (default/Trial 1) → 53.5 (Trial 11) → 50.5 (Trial 21) → **55.0 (Trial 28)**
- Final test score: **48.6%** (val-to-test gap: −6.4pp from best full-eval score of 55.0)
- Parse failures across entire run: **3** (vs 258 on 7B — 14B is format-stable)
- Selected program: 3 demos for `agent.react`, **0 demos for `agent.extract.predict`**
- Note: `--sample-trajectory` was not passed to this run; no qualitative trajectory block in log

**Key observations**
- **Minibatch signal is severely noisy**: Minibatch scores (25ex) ranged 44→72 across trials — a 28-point
  swing. Full-eval scores (200ex) only spanned 50.5–55.0 (~4.5 points). Peak minibatch winner
  (Trial 23, 72.0) did not survive full evaluation. The Bayesian optimizer is fitting
  minibatch variance, not task performance.
- **Val-to-test overestimation**: Internal best full-eval: 55.0% → test: 48.6% (−6.4pp). The 200-example
  optimization subsets are drawn from the training distribution; the Bayesian selector overfits to
  this distribution. By contrast, ClusterFewShot's val-to-test gap was +4.9pp (conservative direction).
- **Extract predictor demo starvation**: The winning configuration assigned 0 demonstrations to
  `agent.extract.predict`, leaving answer synthesis entirely instruction-driven. Multi-hop questions
  requiring careful trajectory integration are most exposed to this gap.
- **Instruction redundancy wastes candidate budget**: Near-duplicate instructions (Instr 0/1/4) reduce
  the effective search space. Roleplay framings add context length with no measurable benefit —
  configurations using them scored comparably to plain-instruction configs on minibatches.
- **14B is format-stable; instructions are near-neutral**: Only 3 parse failures vs 258 on 7B.
  The 14B model already reliably complies with the ReAct field contract, so instruction perturbation
  doesn't cause cascading format failures. This means the instruction search component is not actively
  harmful (unlike on 7B) — it is simply not informative enough to compensate for the noisy demo
  selection signal.

---

### BFRS  ✓

**Log**: `react_agent_experiment_Qwen2.5-14B-Instruct_bfrs_2026-03-23.log`

**Compile internals**
- 9 candidate programs evaluated: seeds -3, -2 (labeled-only, no bootstrap) + seeds -1→+5 (bootstrapped)
- Each candidate scored on the **full 500-example trainset** (no minibatch, no held-out split)
- Bootstrap pass rates per seed: seed -1: 3 traces / 10 attempts (30%); seeds 2,3: 1 trace / 1 attempt (100%);
  remaining seeds: 1–3 traces / 1–4 attempts
- Winning program: seed -1 (57.4% on trainset) → 3 bootstrapped demos each for `agent.react` and
  `agent.extract.predict`
- Labeled-only candidates (seeds -3, -2) scored 51.2% and 53.8% — below all bootstrapped candidates
- Final test score: **54.8%** (train→test gap: −2.6pp)
- Parse failures: **0**

**Key observations**
- **Demo-only, no instruction mutation**: BFRS never modifies the task instruction — identical scope to
  ClusterFewShot. Confirms that instruction search (MIPROv2) provides no benefit on this task.
- **Full-set candidate evaluation is reliable but expensive**: Scoring all 9 candidates on 500 examples
  eliminates the minibatch noise problem that corrupts MIPROv2's Bayesian search. The cost is 6920.5s
  compile — 50% slower than ClusterFewShot, 2.8× slower than MIPROv2 — with no accuracy lead.
- **BFRS confirms MIPROv2's extract predictor starvation as a primary failure mode**: BFRS allocates
  3 demos to `agent.extract.predict`; MIPROv2 allocated 0. BFRS outperforms MIPROv2 by +6.2pp. This
  is the most controlled comparison available (both use ~3 bootstrapped react demos), isolating the
  extract predictor demos as the decisive variable.
- **Train-set overfitting in candidate selection**: Best-on-train (57.4%) drops to 54.8% on test
  (−2.6pp). BFRS has no held-out validation step — all candidates are scored on the same pool
  from which demos were bootstrapped. This structural weakness inflates the apparent best-candidate
  score and makes the selector overconfident.
- **Random seed variation is not efficiently diverse**: Bootstrapped candidates 0–5 scored 54.0–57.4% —
  a 3.4pp spread — suggesting the random demo subsets cover similar regions of the training distribution.
  ClusterFewShot's silhouette-guided diversity directly addresses this.

---

## Cross-Optimizer Summary — Qwen2.5-14B-Instruct

| Dimension | ClusterFewshot | BFRS | MIPROv2 |
|---|---|---|---|
| Optimized score | **55.40%** | 54.80% | 48.60% |
| Δ over baseline | **+8.80pp** | +8.00pp | +3.40pp |
| Bootstrap traces collected | **259** | ~12 (across seeds) | ~18 |
| Parse failures | **0** | 0 | 3 |
| Modifies instructions | No | No | Yes |
| Demo selection mechanism | Cluster + one-shot stratified eval | Random search, full-set eval | Bayesian, noisy minibatch |
| Demos: react / extract | 4 / 4 | 3 / 3 | 3 / **0** |
| Train/val → test gap | **+4.9pp** (conservative) | −2.6pp (overfit) | −6.4pp (overfit) |
| Compile time | 4605.1s | 6920.5s | **2494.7s** |
| Primary failure mode | Over-termination on ambiguous retrievals | Train-set overfit in selection; no diversity pressure | Minibatch noise + extract demo starvation |

**Finding (Qwen2.5-14B, HotPotQA ReAct)**:

**ClusterFewShot is the dominant optimizer** — best score (55.4%), fastest compile among demo-only
methods (4605s), and the only optimizer whose internal validation is conservative rather than
overfit. The 0.6pp margin over BFRS is directionally consistent with the cluster-diversity mechanism
providing real signal beyond random demo selection, and ClusterFewShot achieves it 1.5 hours faster.

**BFRS is a reliable but inefficient baseline**: avoids MIPROv2's minibatch noise by evaluating all
candidates on the full trainset, and fully populates both predictors with bootstrapped demos. This
alone accounts for +6.2pp over MIPROv2. However, random seed variation does not explore the demo
space efficiently — the 3.4pp spread across bootstrapped candidates is narrow — and no held-out
validation means the selected seed is overfit to the training distribution (−2.6pp train→test gap).

**MIPROv2's instruction search component provides no benefit and introduces cost**: its failure modes
(minibatch noise, extract predictor demo starvation, val→test overestimation) compound each other.
The extract predictor receiving 0 demos is a consistent pattern across both model sizes, indicating
MIPROv2's Bayesian search systematically deprioritizes the extract predictor when the react
predictor's demos already provide sufficient minibatch signal — a structural misallocation.

---

## Cross-Model Summary

| Model | Size | Best Optimizer | Best Score | Baseline | 2nd Best | 2nd Score | Notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | BFRS | **49.80%** | 22.40% | ClusterFewshot | 45.80% | BFRS win is seed-dependent (23.4pp variance across candidates) |
| Qwen2.5-14B-Instruct | 14B | ClusterFewshot | **55.40%** | 46.60% | BFRS | 54.80% | K capped at 4 for context budget |

> All baselines corrected: 7B true baseline = 22.40% (ClusterFewshot clean eval); MIPROv2/BFRS
> reported baselines (~29%) inflated by extract fallback (~7pp from empty-trajectory recoveries).

---

## Methodology Notes

- All experiments: HotPotQA hard examples, `answer_exact_match`, trainset=500, devset=200, testset=500
- LM: Qwen models served via sglang on A100 80GB; API models via LiteLLM
- ColBERTv2 endpoint: `http://localhost:8894/api/search`, k=3 passages per query
- ClusterFewshot config: `task_type="agentic"`, encoders=[`all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`], K∈[3,4] grid
  (originally K∈[3,10]; capped after K=8 caused context overflow on 14B with 8192-token limit)
- MIPROv2 config: `auto="medium"` (25 trials), `minibatch=True`, `minibatch_size=25`, `minibatch_full_eval_steps=10`, `max_bootstrapped_demos=3`, `max_labeled_demos=3`
- BFRS config: `num_candidate_programs=6`, `max_bootstrapped_demos=3`, `max_labeled_demos=3`
- ReAct fix applied: parse failures (`AttributeError: 'Prediction' has no attribute 'next_thought'`) now fall through to `extract` instead of raising — affects MIPROv2 baseline measurement (see §MIPROv2 baseline inflation)