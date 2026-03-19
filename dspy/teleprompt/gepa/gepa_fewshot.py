"""
GEPAFewShot: Extends GEPA with joint instruction + few-shot demonstration optimization.

Design overview
---------------
GEPA evolves instructions via Pareto-based reflective search.  GEPAFewShot adds a
*parallel* demonstration evolution track:

  - A demo pool is bootstrapped once before the main search loop, using
    BootstrapFewShot (metric-filtered) on the training set.
  - Each GEPA candidate (an instruction string per predictor) is *shadowed* by a
    companion demo set stored in GEPAFewShotAdapter._demo_registry.
  - When GEPA proposes a new instruction variant, GEPAFewShotAdapter simultaneously
    mutates the companion demo set using metric-based heuristics (add / remove / swap).
  - build_program() applies both instruction updates AND demo updates, so the Pareto
    search implicitly evaluates (instruction, demos) joint candidates.

Demo mutation strategies
------------------------
  "random"        — uniform random add / remove / swap from pool.
  "metric_based"  — feedback-driven: pool quality priors (1.0 bootstrapped /
                    0.5 labeled) are progressively replaced by empirical per-demo
                    scores accumulated from evaluate() calls.  Operation type
                    (add / remove / swap) is biased by _select_operation() toward
                    the direction of highest improvement potential; selection within
                    add/swap is weighted by empirical quality × token-Jaccard
                    relevance to the current reflective minibatch.

Pareto awareness
----------------
We do not modify GEPA's Pareto frontier logic.  Demo cost (token overhead) is
naturally factored in via the k_demos cap: each candidate carries exactly k demos,
keeping the marginal token cost predictable.

Authors: NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Literal, Optional

from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.gepa.gepa import (
    AUTO_RUN_SETTINGS,
    GEPA,
    DspyGEPAResult,
    GEPAFeedbackMetric,
)
from dspy.teleprompt.gepa.gepa_utils import (
    DSPyTrace,
    DspyAdapter,
    LoggerAdapter,
    PredictorFeedbackFn,
    ScoreWithFeedback,
)
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias: per-predictor demo pool — list of (example, quality_score) pairs
# ---------------------------------------------------------------------------
DemoPool = dict[str, list[tuple[Example, float]]]


# ---------------------------------------------------------------------------
# GEPAFewShotAdapter
# ---------------------------------------------------------------------------

class GEPAFewShotAdapter(DspyAdapter):
    """
    Extends DspyAdapter with per-candidate few-shot demonstration management.

    The adapter maintains a *demo registry*: a mapping from candidate keys
    (frozen instruction tuples) to demo sets.  On every call to
    propose_new_texts() it applies a demo mutation in tandem with the
    instruction mutation produced by the parent class, and registers the
    resulting demo set under the new candidate's key.

    build_program() populates predictor.demos from the registry before
    returning, so every evaluated program carries its companion demo set.
    """

    def __init__(
        self,
        *args,
        demo_pool: DemoPool,
        k_demos: int = 3,
        mutation_strategy: Literal["random", "metric_based"] = "metric_based",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.demo_pool = demo_pool          # predictor_name → [(Example, score), ...]
        self.k_demos = k_demos
        self.mutation_strategy = mutation_strategy

        # candidate_key → {predictor_name: [Example, ...]}
        self._demo_registry: dict[tuple, dict[str, list[Example]]] = {}
        # candidate_key → mean score observed during evaluate() (capture_traces=False)
        self._score_registry: dict[tuple, float] = {}
        # id(demo) → (score_sum, count) accumulated across all non-trace evaluations
        self._demo_score_tracker: dict[int, tuple[float, int]] = {}

        # Rollout counters — incremented by evaluate() and propose_new_texts()
        self.n_trace_evals: int = 0    # capture_traces=True  (reflection rollouts)
        self.n_score_evals: int = 0    # capture_traces=False (Pareto scoring)
        self.n_mutations:   int = 0    # propose_new_texts() calls (candidate proposals)

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _candidate_key(self, candidate: dict[str, str]) -> tuple:
        """Stable, hashable key derived from the instruction mapping."""
        return tuple(sorted(candidate.items()))

    def _get_demos(self, candidate: dict[str, str]) -> dict[str, list[Example]]:
        """
        Return the registered demo set for *candidate*.
        If not yet registered (e.g., the seed candidate), initialise with a
        random draw of k_demos from each predictor's pool.
        """
        key = self._candidate_key(candidate)
        if key not in self._demo_registry:
            demos: dict[str, list[Example]] = {}
            for name in candidate:
                pool_pairs = self.demo_pool.get(name, [])
                k = min(self.k_demos, len(pool_pairs))
                if k > 0:
                    pool_exs = [ex for ex, _ in pool_pairs]
                    pool_ws  = [w  for _, w  in pool_pairs]
                    chosen = self.rng.choices(pool_exs, weights=pool_ws, k=k)
                    demos[name] = chosen
                else:
                    demos[name] = []
            self._demo_registry[key] = demos
        return self._demo_registry[key]

    def _register_demos(
        self, candidate: dict[str, str], demos: dict[str, list[Example]]
    ) -> None:
        self._demo_registry[self._candidate_key(candidate)] = demos

    # ------------------------------------------------------------------
    # Empirical quality tracking
    # ------------------------------------------------------------------

    def evaluate(self, batch, candidate, capture_traces=False):
        """
        Intercepts every non-trace evaluation to update per-demo score tracking.

        After each capture_traces=False call (Pareto scoring path), the mean
        batch score is recorded per candidate and accumulated per demo so that
        _empirical_quality() can replace the fixed 1.0/0.5 quality prior over
        the course of the run.
        """
        if capture_traces:
            self.n_trace_evals += 1
        else:
            self.n_score_evals += 1

        result = super().evaluate(batch, candidate, capture_traces)
        if not capture_traces and result.scores:
            key = self._candidate_key(candidate)
            mean_score = sum(result.scores) / len(result.scores)
            self._score_registry[key] = mean_score
            n_updated = 0
            for pred_demos in self._demo_registry.get(key, {}).values():
                for demo in pred_demos:
                    s, c = self._demo_score_tracker.get(id(demo), (0.0, 0))
                    self._demo_score_tracker[id(demo)] = (s + mean_score, c + 1)
                    n_updated += 1
            logger.debug(
                "score_tracker: batch_size=%d  mean_score=%.3f  "
                "demos_updated=%d  tracker_size=%d",
                len(result.scores), mean_score, n_updated, len(self._demo_score_tracker),
            )
        return result

    def _empirical_quality(self, demo: Example, prior_quality: float, min_obs: int = 3) -> float:
        """
        Pseudo-count blend of the fixed pool prior with observed mean score.

        Returns prior_quality until min_obs evaluations are accumulated, then
        transitions smoothly to the empirical mean.  This prevents early noisy
        observations from immediately overriding the bootstrapped quality signal.
        """
        s, c = self._demo_score_tracker.get(id(demo), (0.0, 0))
        return (prior_quality * min_obs + s) / (min_obs + c)

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def build_program(self, candidate: dict[str, str]) -> Module:
        """Build program with both updated instructions and companion demos."""
        new_prog = super().build_program(candidate)   # sets instructions
        demos = self._get_demos(candidate)
        for name, pred in new_prog.named_predictors():
            if name in demos:
                pred.demos = list(demos[name])
        return new_prog

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Propose new instructions (via parent) and simultaneously mutate demos.
        The mutated demo set is registered under the new candidate's key so
        that the next build_program() call picks it up automatically.
        """
        self.n_mutations += 1

        # 1. Propose new instructions via GEPA's reflection mechanism
        new_instructions = super().propose_new_texts(
            candidate, reflective_dataset, components_to_update
        )

        # 2. Mutate demos for the components being updated; carry others over
        current_demos = self._get_demos(candidate)
        new_demos: dict[str, list[Example]] = {}
        for name, demo_list in current_demos.items():
            if name in components_to_update:
                pool = self.demo_pool.get(name, [])
                new_demos[name] = self._mutate_demos(
                    current=list(demo_list),
                    pool=pool,
                    reflective_examples=reflective_dataset.get(name, []),
                )
            else:
                new_demos[name] = list(demo_list)

        # 3. Register demos for the newly proposed candidate
        self._register_demos(new_instructions, new_demos)
        return new_instructions

    # ------------------------------------------------------------------
    # Demo mutation strategies
    # ------------------------------------------------------------------

    def _mutate_demos(
        self,
        current: list[Example],
        pool: list[tuple[Example, float]],
        reflective_examples: list[dict],
    ) -> list[Example]:
        """Dispatch to the configured mutation strategy."""
        if not pool:
            return current

        if self.mutation_strategy == "random":
            return self._random_mutate(current, pool)
        elif self.mutation_strategy == "metric_based":
            return self._metric_based_mutate(current, pool, reflective_examples)
        else:
            raise ValueError(f"Unknown demo mutation strategy: {self.mutation_strategy!r}")

    def _random_mutate(
        self,
        current: list[Example],
        pool: list[tuple[Example, float]],
    ) -> list[Example]:
        """Uniform random add / remove / swap, respecting k_demos cap."""
        pool_examples = [ex for ex, _ in pool]
        available = [ex for ex in pool_examples if ex not in current]

        ops: list[str] = []
        if len(current) < self.k_demos and available:
            ops.append("add")
        if current:
            ops.append("remove")
        if current and available:
            ops.append("swap")

        if not ops:
            return current

        op = self.rng.choice(ops)
        result = list(current)

        if op == "add":
            result.append(self.rng.choice(available))
        elif op == "remove":
            result.pop(self.rng.randrange(len(result)))
        elif op == "swap":
            idx = self.rng.randrange(len(result))
            result[idx] = self.rng.choice(available)

        return result

    def _failure_targeted_weights(
        self,
        pool: list[tuple[Example, float]],
        reflective_examples: list[dict],
    ) -> list[float]:
        """
        Blend each pool demo's empirical quality score with its token-Jaccard
        relevance to the inputs in the current reflective dataset.

        Weight_i = empirical_quality_i * (1.0 + jaccard_relevance_i)

        empirical_quality_i is a pseudo-count blend of the fixed pool prior
        (1.0 bootstrapped / 0.5 labeled) with the observed mean score across
        all evaluations where demo_i was present — see _empirical_quality().
        Token Jaccard is defined over lowercased whitespace-split word sets.
        When no reflective examples are present, relevance = 0 and weights
        reduce to plain empirical quality scores.
        """
        # Collect text tokens from reflective examples (minibatch inputs)
        failure_token_sets: list[set[str]] = []
        for ex in reflective_examples:
            inputs = ex.get("Inputs", {})
            text = " ".join(str(v) for v in inputs.values())
            failure_token_sets.append(set(text.lower().split()))

        weights: list[float] = []
        for demo, quality in pool:
            q = max(self._empirical_quality(demo, quality), 1e-3)
            if not failure_token_sets:
                weights.append(q)
                continue
            # Token set for this pool demo
            demo_text = " ".join(str(v) for v in demo.values())
            demo_tokens = set(demo_text.lower().split())
            # Mean Jaccard over all failure examples
            jaccard_scores = []
            for ft in failure_token_sets:
                union = demo_tokens | ft
                if union:
                    jaccard_scores.append(len(demo_tokens & ft) / len(union))
                else:
                    jaccard_scores.append(0.0)
            relevance = sum(jaccard_scores) / len(jaccard_scores)
            weights.append(q * (1.0 + relevance))

        return weights

    def _select_operation(
        self,
        ops: list[str],
        current: list[Example],
        available_pairs: list[tuple[Example, float]],
        pool_priors: dict[int, float],
    ) -> str:
        """
        Bias operation type toward the direction of highest improvement potential.

        Uses empirical quality scores (from _demo_score_tracker) to compare the
        best available pool demo against the worst demo currently in the set:

          delta > 0  →  a better demo is available  →  prefer add / swap
          delta < 0  →  no better demo available     →  prefer remove
          delta ≈ 0  →  no clear signal              →  roughly uniform

        Falls back to uniform random until every demo in the current set has
        accumulated at least min_obs=3 observations (matching _empirical_quality's
        pseudo-count threshold), or when only one operation is available.

        pool_priors maps id(demo) → pool quality prior (1.0 for bootstrapped,
        0.5 for labeled) so that unseen bootstrapped demos are valued correctly
        when computing best_avail.
        """
        _MIN_OBS = 3  # must match _empirical_quality's min_obs
        all_observed = all(
            self._demo_score_tracker.get(id(d), (0, 0))[1] >= _MIN_OBS
            for d in current
        )
        if not all_observed or len(ops) == 1:
            chosen = self.rng.choice(ops)
            logger.debug("op_select: insufficient observations — uniform random %s → %s", ops, chosen)
            return chosen

        # best_avail uses the pool's actual quality prior so bootstrapped demos
        # (prior=1.0) are not undervalued against unseen labeled demos (prior=0.5).
        best_avail = max(
            (self._empirical_quality(d, pool_priors.get(id(d), 0.5)) for d, _ in available_pairs),
            default=0.5,
        )
        # worst_curr uses a conservative prior=0.5 — current demos have enough
        # observations (_MIN_OBS guard above) so the prior has limited influence.
        worst_curr = min(
            (self._empirical_quality(d, 0.5) for d in current),
            default=0.5,
        )
        delta = best_avail - worst_curr

        weights = [
            max(0.5 + delta, 1e-2) if op in ("add", "swap") else max(0.5 - delta, 1e-2)
            for op in ops
        ]
        [chosen] = self.rng.choices(ops, weights=weights, k=1)
        logger.debug(
            "op_select: delta=%.3f (best_avail=%.3f worst_curr=%.3f) "
            "ops=%s weights=%s → %s",
            delta, best_avail, worst_curr,
            ops, [f"{w:.2f}" for w in weights], chosen,
        )
        return chosen

    def _metric_based_mutate(
        self,
        current: list[Example],
        pool: list[tuple[Example, float]],
        reflective_examples: list[dict] | None = None,
    ) -> list[Example]:
        """
        Feedback-driven add / remove / swap mutation.

        Selection weights for add/swap blend empirical quality (accumulated from
        _demo_score_tracker via evaluate()) with token-Jaccard relevance to the
        current reflective minibatch — see _failure_targeted_weights().

        The operation type (add / remove / swap) is chosen by _select_operation(),
        which biases toward add/swap when a better demo is available and toward
        remove when no improvement is found in the pool.  Falls back to uniform
        random until sufficient empirical observations accumulate.
        """
        pool_examples = [ex for ex, _ in pool]
        blended_weights = self._failure_targeted_weights(
            pool, reflective_examples or []
        )
        available_pairs = [
            (ex, w) for ex, w in zip(pool_examples, blended_weights)
            if ex not in current
        ]
        # Map id(demo) → pool quality prior for available demos so _select_operation
        # can use the real bootstrapped prior (1.0) rather than a flat 0.5.
        pool_priors = {
            id(ex): prior for ex, prior in pool if ex not in current
        }

        ops: list[str] = []
        if len(current) < self.k_demos and available_pairs:
            ops.append("add")
        if current:
            ops.append("remove")
        if current and available_pairs:
            ops.append("swap")

        if not ops:
            return current

        op = self._select_operation(ops, current, available_pairs, pool_priors)
        result = list(current)
        size_before = len(result)

        if op == "add":
            avail_exs = [ex for ex, _ in available_pairs]
            avail_ws  = [w  for _, w  in available_pairs]
            [chosen] = self.rng.choices(avail_exs, weights=avail_ws, k=1)
            result.append(chosen)
            logger.debug(
                "demo_mutate: add  eq=%.3f  set_size %d→%d",
                self._empirical_quality(chosen, 0.5), size_before, len(result),
            )

        elif op == "remove":
            # Uniform removal — no directional bias within the current set
            idx = self.rng.randrange(len(result))
            removed = result.pop(idx)
            logger.debug(
                "demo_mutate: remove  eq=%.3f  set_size %d→%d",
                self._empirical_quality(removed, 0.5), size_before, len(result),
            )

        elif op == "swap":
            idx = self.rng.randrange(len(result))
            avail_exs = [ex for ex, _ in available_pairs]
            avail_ws  = [w  for _, w  in available_pairs]
            [replacement] = self.rng.choices(avail_exs, weights=avail_ws, k=1)
            evicted = result[idx]
            result[idx] = replacement
            logger.debug(
                "demo_mutate: swap  out_eq=%.3f → in_eq=%.3f  set_size %d",
                self._empirical_quality(evicted, 0.5),
                self._empirical_quality(replacement, 0.5),
                len(result),
            )

        return result


# ---------------------------------------------------------------------------
# GEPAFewShot optimizer
# ---------------------------------------------------------------------------

@experimental(version="3.0.0")
class GEPAFewShot(GEPA):
    """
    Extends GEPA to jointly optimise instructions *and* few-shot demonstrations.

    Each candidate in the Pareto search carries:
      (i)  an instruction string per predictor — evolved via GEPA reflection.
      (ii) a companion demo set of size k_demos — evolved via metric-based mutation.

    Demo pool construction
    ~~~~~~~~~~~~~~~~~~~~~~
    Before the main GEPA loop, a pool of candidate demonstrations is built by
    running BootstrapFewShot on the training set with the provided metric.
    Each pooled demo is tagged with a quality score (1.0 for bootstrapped traces
    that passed the metric, 0.5 for raw labeled examples), which guides
    score-weighted sampling during mutation.

    Metric compatibility
    ~~~~~~~~~~~~~~~~~~~~
    GEPA's metric protocol uses 5 arguments: (gold, pred, trace, pred_name,
    pred_trace).  GEPAFewShot automatically wraps this into the 3-argument
    form expected by BootstrapFewShot for pool construction.

    Args:
        metric:                 GEPA-compatible 5-argument feedback metric.
        k_demos:                Number of demonstrations per candidate (default 3).
        max_bootstrapped_demos: Pool size cap for bootstrapped (augmented) demos.
        max_labeled_demos:      Pool size cap for raw labeled demos.
        demo_mutation_strategy: "metric_based" (default) or "random".
        demo_metric:            Optional separate 5-arg metric for pool scoring;
                                falls back to *metric* if not provided.
        **gepa_kwargs:          All remaining kwargs are forwarded to GEPA.__init__.

    Example::

        optimizer = GEPAFewShot(
            metric=my_metric,
            reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0),
            auto="medium",
            k_demos=3,
            demo_mutation_strategy="metric_based",
        )
        best_prog = optimizer.compile(student, trainset=train, valset=val)
    """

    def __init__(
        self,
        metric: GEPAFeedbackMetric,
        *,
        k_demos: int = 3,
        max_bootstrapped_demos: int = 16,
        max_labeled_demos: int = 4,
        demo_mutation_strategy: Literal["random", "metric_based"] = "metric_based",
        demo_metric: Optional[GEPAFeedbackMetric] = None,
        **gepa_kwargs,
    ):
        # Force use_merge=False — merge is not yet supported in GEPAFewShot.
        # Silently override any inherited default (GEPA defaults to True) so
        # the assert in compile() is never hit in normal usage.
        gepa_kwargs["use_merge"] = False
        super().__init__(metric=metric, **gepa_kwargs)
        self.k_demos = k_demos
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.demo_mutation_strategy = demo_mutation_strategy
        self.demo_metric = demo_metric

    # ------------------------------------------------------------------
    # Demo pool construction
    # ------------------------------------------------------------------

    def _wrap_metric_for_bootstrap(self) -> Callable:
        """
        Produce a 3-argument metric compatible with BootstrapFewShot from
        GEPA's 5-argument metric protocol.
        """
        five_arg = self.demo_metric or self.metric_fn

        def wrapped(gold: Example, pred: Prediction, trace=None) -> float:
            result = five_arg(gold, pred, trace, None, None)
            if isinstance(result, dict) and "score" in result:
                return float(result["score"])
            if hasattr(result, "score"):
                return float(result["score"])
            return float(result)

        return wrapped

    def _bootstrap_demo_pool(
        self,
        student: Module,
        trainset: list[Example],
    ) -> DemoPool:
        """
        Bootstrap a pool of demonstrations with associated quality scores.

        Returns
        -------
        DemoPool
            Mapping predictor_name → list of (Example, quality_score) pairs.
            Bootstrapped (augmented) traces score 1.0; raw labeled examples 0.5.
        """
        from dspy.teleprompt.bootstrap import BootstrapFewShot

        bfs_metric = self._wrap_metric_for_bootstrap()

        bfs = BootstrapFewShot(
            metric=bfs_metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=0,   # we add labeled demos separately below
        )
        # compile() internally calls _bootstrap() + _train() on a deepcopy
        bfs.compile(student.deepcopy(), trainset=trainset)

        pool: DemoPool = {}
        for name, _ in student.named_predictors():
            augmented = bfs.name2traces.get(name, [])
            # Bootstrapped demos: passed metric → quality 1.0
            pool_entries: list[tuple[Example, float]] = [
                (demo, 1.0) for demo in augmented
            ]
            # Raw labeled examples from the unbootstrapped remainder
            # (validation in BootstrapFewShot's terminology)
            labeled_sample = bfs.validation[: self.max_labeled_demos]
            for ex in labeled_sample:
                pool_entries.append((ex, 0.5))

            pool[name] = pool_entries

        pool_summary = {k: len(v) for k, v in pool.items()}
        logger.info(f"GEPAFewShot: demo pool built — {pool_summary}")
        return pool

    # ------------------------------------------------------------------
    # compile() — mirrors GEPA.compile() with GEPAFewShotAdapter injected
    # ------------------------------------------------------------------

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
    ) -> Module:
        """
        Compile the student module with joint instruction + demo optimization.

        The flow is identical to GEPA.compile() except:
          1. A demo pool is bootstrapped before the search starts.
          2. GEPAFewShotAdapter is used in place of DspyAdapter so that
             build_program() and propose_new_texts() handle demonstrations.
        """
        import random as _random

        from gepa import GEPAResult, optimize

        assert trainset is not None and len(trainset) > 0, "Trainset must be non-empty."
        assert teacher is None, "teacher is not yet supported in GEPAFewShot."
        assert not self.use_merge, (
            "use_merge is not yet supported in GEPAFewShot: merged candidates receive a "
            "fresh random demo set unrelated to either parent, producing incoherent "
            "(instruction, demo) pairings.  Set use_merge=False (the default)."
        )

        # ---- Budget resolution (identical to GEPA) ----
        if self.auto is not None:
            self.max_metric_calls = self.auto_budget(
                num_preds=len(student.predictors()),
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
                valset_size=len(valset) if valset is not None else len(trainset),
            )
        elif self.max_full_evals is not None:
            self.max_metric_calls = self.max_full_evals * (
                len(trainset) + (len(valset) if valset is not None else 0)
            )
        else:
            assert self.max_metric_calls is not None

        logger.info(
            f"GEPAFewShot: running for ~{self.max_metric_calls} metric calls "
            f"(k_demos={self.k_demos}, strategy={self.demo_mutation_strategy!r})"
        )

        if valset is None:
            logger.warning(
                "No valset provided; using trainset as valset.  "
                "This causes GEPA to overfit prompts to the trainset — "
                "provide a separate valset for generalisation."
            )
        valset = valset or trainset

        rng = _random.Random(self.seed)

        # ---- Bootstrap demo pool ----
        logger.info("GEPAFewShot: bootstrapping demonstration pool …")
        demo_pool = self._bootstrap_demo_pool(student, trainset)

        # ---- Feedback map (identical to GEPA) ----
        def feedback_fn_creator(
            pred_name: str, predictor
        ) -> PredictorFeedbackFn:
            def feedback_fn(
                predictor_output,
                predictor_inputs,
                module_inputs,
                module_outputs,
                captured_trace,
            ) -> ScoreWithFeedback:
                trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
                o = self.metric_fn(
                    module_inputs,
                    module_outputs,
                    captured_trace,
                    pred_name,
                    trace_for_pred,
                )
                if (isinstance(o, dict) and "feedback" in o) or hasattr(o, "feedback"):
                    if o["feedback"] is None:
                        o["feedback"] = f"This trajectory got a score of {o['score']}."
                    return o
                return dict(score=o, feedback=f"This trajectory got a score of {o}.")

            return feedback_fn

        feedback_map = {k: feedback_fn_creator(k, v) for k, v in student.named_predictors()}

        # ---- Build GEPAFewShotAdapter (key difference from GEPA) ----
        adapter = GEPAFewShotAdapter(
            student_module=student,
            metric_fn=self.metric_fn,
            feedback_map=feedback_map,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=rng,
            reflection_lm=self.reflection_lm,
            custom_instruction_proposer=self.custom_instruction_proposer,
            warn_on_score_mismatch=self.warn_on_score_mismatch,
            reflection_minibatch_size=self.reflection_minibatch_size,
            reflection_prompt_template=self.reflection_prompt_template,
            # Few-shot specific
            demo_pool=demo_pool,
            k_demos=self.k_demos,
            mutation_strategy=self.demo_mutation_strategy,
        )

        # ---- Seed candidate: instructions from student ----
        seed_candidate = {
            name: pred.signature.instructions
            for name, pred in student.named_predictors()
        }

        # ---- Run GEPA optimization ----
        gepa_result: GEPAResult = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=(
                (lambda x: adapter.stripped_lm_call(x)[0])
                if self.reflection_lm is not None
                else None
            ),
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector=self.component_selector,
            perfect_score=self.perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,
            max_metric_calls=self.max_metric_calls,
            logger=LoggerAdapter(logger),
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            use_mlflow=self.use_mlflow,
            track_best_outputs=self.track_best_outputs,
            display_progress_bar=True,
            raise_on_exception=True,
            seed=self.seed,
            **self.gepa_kwargs,
        )

        # ---- Rollout summary ----
        logger.info(
            "GEPAFewShot rollout summary — "
            "mutations (candidate proposals): %d  |  "
            "trace evals (reflection rollouts): %d  |  "
            "score evals (Pareto scoring): %d  |  "
            "unique candidates scored: %d",
            adapter.n_mutations,
            adapter.n_trace_evals,
            adapter.n_score_evals,
            len(adapter._score_registry),
        )

        # ---- Build final program (adapter.build_program applies both
        #      instructions AND demos for the best candidate) ----
        new_prog = adapter.build_program(gepa_result.best_candidate)

        if self.track_stats:
            dspy_gepa_result = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
            new_prog.detailed_results = dspy_gepa_result

        return new_prog
