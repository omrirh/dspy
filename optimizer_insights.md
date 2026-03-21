# Prompt Optimizer Insights — HotPotQA ReAct Agent

Findings from running ClusterFewshot, MIPROv2, and BFRS as prompt optimizers
on `dspy.ReAct` (HotPotQA multi-hop QA, ColBERTv2 search tool).

**Task setup**: `ReactAgentMH` — single `search` tool + implicit `finish`, `max_iters=5`,
evaluated on 500 hard HotPotQA examples with `answer_exact_match`.

---

## Results Table

| Model | Optimizer | Baseline | Optimized | Δ (pp) | Compile (s) | Eval (s) | Date |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | ClusterFewshot | 22.40% | 45.80% | +23.40 | 1632.6 | 320.7 | 2026-03-20 |
| Qwen2.5-7B-Instruct | MIPROv2 | 22.40%* | 39.20% | +16.80* | 1506.1 | 305.4 | 2026-03-21 |
| Qwen2.5-7B-Instruct | BFRS | — | — | — | — | — | — |
| _(next model)_ | ClusterFewshot | — | — | — | — | — | — |
| _(next model)_ | MIPROv2 | — | — | — | — | — | — |
| _(next model)_ | BFRS | — | — | — | — | — | — |

> \* MIPROv2 reported baseline of 29.40%, but this is inflated by ~7pp due to the
> `extract` fallback recovering empty-trajectory examples (see §Qwen2.5-7B notes).
> True 0-shot baseline is 22.40% (ClusterFewshot clean eval). Corrected Δ = +16.80pp.

---

## Model: Qwen/Qwen2.5-7B-Instruct

### ClusterFewshot  ✓

**Log**: `react_agent_experiment_Qwen2.5-7B-Instruct_clusterfs_2026-03-20.log`

**Compile internals**
- Bootstrap: 500 trainset examples → **101/500 (20.2%) successful complete trajectories**
- Encoder grid search: `all-mpnet-base-v2` vs `multi-qa-mpnet-base-dot-v1` across K∈[3,10]
- Selected: `multi-qa-mpnet-base-dot-v1`, K=3 (silhouette=0.038); QA-tuned encoder edges out general encoder
- Demo selection: `best_in_cluster` and `top_n` both scored **50.5%** on the 200-example valset
  (tie → `best_in_cluster` used as tiebreaker per `task_type="agentic"` policy)
- Final demos: 3 per predictor (`agent.react`, `agent.extract.predict`)
- Parse failures during compile/eval: **0**

**Key observations**
- +23.40pp gain from 3 demonstrations alone, with zero instruction mutation — confirms the
  7B model is highly format-sensitive; demos teach the think→search→observe loop structurally
- `best_in_cluster` == `top_n` score suggests HotPotQA 2-hop structure is relatively uniform
  in embedding space (low silhouette ~0.038 across all K); diversity and quality are equivalent
  at K=3 for this task
- Compile time dominated by one-shot demo selection (~1370s for 101 candidate evals × 18
  questions), not by bootstrapping (~265s)

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

### BFRS  _(pending)_

---

## Cross-Optimizer Summary — Qwen2.5-7B-Instruct

| Dimension | ClusterFewshot | MIPROv2 | BFRS |
|---|---|---|---|
| Optimized score | **45.80%** | 39.20% | — |
| True Δ over baseline | **+23.40pp** | +16.80pp* | — |
| Bootstrap traces collected | **101** | ~19 (degraded) | — |
| Parse failures | **0** | 258 | — |
| Modifies instructions | **No** | Yes | No |
| Demo selection mechanism | Cluster + one-shot eval | Bayesian (noisy minibatch) | Random search |
| Primary failure mode | — | Format instability + signal noise | — |

> \* Corrected for baseline inflation artifact.

**Finding (Qwen2.5-7B, HotPotQA ReAct)**: For small models (≤7B) on structured agentic
tasks with typed multi-field output, demo-only optimizers (ClusterFewshot, BFRS) are
expected to outperform instruction-mutating optimizers (MIPROv2) because:
1. Instruction proposals do not reinforce the format contract; examples do
2. 7B models are at the format compliance boundary — small instruction perturbations
   produce parse failure cascades that corrupt the optimization signal
3. Complete verified trajectories as demonstrations are more information-dense than
   instruction rewrites for teaching the ReAct loop

---

## Cross-Model Summary  _(to be populated)_

| Model | Size | Best Optimizer | Best Score | Baseline | Notes |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | ClusterFewshot | 45.80% | 22.40% | |
| _(next model)_ | — | — | — | — | |

---

## Methodology Notes

- All experiments: HotPotQA hard examples, `answer_exact_match`, trainset=500, devset=200, testset=500
- LM: Qwen models served via sglang on A100 80GB; API models via LiteLLM
- ColBERTv2 endpoint: `http://localhost:8894/api/search`, k=3 passages per query
- ClusterFewshot config: `task_type="agentic"`, encoders=[`all-mpnet-base-v2`, `multi-qa-mpnet-base-dot-v1`], K∈[3,10] grid
- MIPROv2 config: `auto="medium"` (25 trials), `minibatch=True`, `minibatch_size=25`, `minibatch_full_eval_steps=10`, `max_bootstrapped_demos=3`, `max_labeled_demos=3`
- BFRS config: `num_candidate_programs=6`, `max_bootstrapped_demos=3`, `max_labeled_demos=3`
- ReAct fix applied: parse failures (`AttributeError: 'Prediction' has no attribute 'next_thought'`) now fall through to `extract` instead of raising — affects MIPROv2 baseline measurement (see §MIPROv2 baseline inflation)