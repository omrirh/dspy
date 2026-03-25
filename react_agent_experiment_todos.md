# ReAct Agent Experiment — Next Steps

## Status snapshot (2026-03-24)

| Model | ClusterFS | MIPROv2 | BFRS |
|---|---|---|---|
| 7B | ✓ (45.8%) | ✓ (39.2%) | ✓ (49.8%) |
| 14B | ✓ (55.4%) | ✓ (48.6%) | ✓ (54.8%) |

Zero-shot 7B baseline: ~26% (true), ~29% (reported, inflated by extract fallback ~7pp).

---

## TODO 1 — ClusterFS 7B re-run  _(in progress)_

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

## TODO 2 — Update `optimizer_insights.md`: overall takeaways

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

## TODO 3 — Correlation analysis: format compliance failures vs. performance gain

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

## TODO 4 — Debug: ReAct format compliance under prompt variants

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
| Reduce `max_iters` 5→3 | Fewer steps = less chance of format drift |
| Temperature 0.0 | Eliminates stochasticity in format-critical output |
| Grammar-constrained decoding (Outlines) | Hard ceiling — what's achievable with constraint |

### Step 4 — Trace analysis: format-compliant vs non-compliant BFRS seeds (~1h)
Compare bootstrap traces from BFRS seeds 0/-1/4 (compliant) vs seeds 1/3 (non-compliant).
Look for structural differences: avg trajectory length, whether `finish` tool appears,
tool_args JSON validity, trace token length. This directly explains the bimodal effect.

---

## Open questions (inform paper framing and TODO 3)

- Does ClusterFS's high bootstrap yield (101 traces) *cause* format compliance, or does it
  merely *correlate* with it? Test: subsample ClusterFS to 10 traces and re-run selection.
- Is the format compliance threshold model-specific to Qwen2.5-7B, or does it generalize to
  other 7B-class models (Llama-3.1-8B, Gemma-3-4B)?
- Can a simple bootstrap filter (discard traces with > N parse failures during collection)
  guarantee format-compliant demos without requiring cluster-based selection?
