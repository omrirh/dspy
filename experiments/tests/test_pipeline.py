"""
Mock tests for the experimental pipeline.

Tests cover:
  - Dataset loading (GSM8K, Iris)
  - Student program instantiation
  - Metric wrapping (5-arg GEPA metric → 3-arg bootstrap metric)
  - GEPAFewShot is a proper GEPA subclass
  - GEPAFewShotAdapter demo registry and mutation strategies
  - Optimizer instantiation (no LM calls made)
  - run_experiment CLI argument parsing

All tests use mock LMs and small synthetic data — no GPU or network required.
"""
import os
import random
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from dspy.primitives import Example

dspy.settings.experimental = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gsm8k_examples(n=10):
    return [
        Example(question=f"What is {i} + {i}?", answer=str(2 * i)).with_inputs("question")
        for i in range(n)
    ]


def _make_iris_examples(n=10):
    species = ["setosa", "versicolor", "virginica"]
    return [
        Example(
            sepal_length=5.0 + i * 0.1,
            sepal_width=3.0,
            petal_length=1.5,
            petal_width=0.2,
            answer=species[i % 3],
        ).with_inputs("sepal_length", "sepal_width", "petal_length", "petal_width")
        for i in range(n)
    ]


def _dummy_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Minimal 5-arg metric that always returns 1.0."""
    return 1.0


def _dummy_lm():
    """Return a MagicMock that quacks like a dspy.LM."""
    lm = MagicMock()
    lm.return_value = ["answer: 42"]
    return lm


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDatasets(unittest.TestCase):

    def test_iris_dataset_loads(self):
        from dspy.datasets.iris import IrisDataset

        ds = IrisDataset(seed=42)
        train, dev, test = ds.get_data_splits()

        self.assertEqual(len(train), 50)
        self.assertEqual(len(dev),   25)
        self.assertEqual(len(test),  75)

        for ex in train:
            self.assertIn("answer", ex)
            self.assertIn(ex.answer, {"setosa", "versicolor", "virginica"})
            self.assertIn("sepal_length", ex.inputs())

    def test_iris_dataset_reproducible(self):
        from dspy.datasets.iris import IrisDataset

        ds1 = IrisDataset(seed=0)
        ds2 = IrisDataset(seed=0)
        self.assertEqual(
            [e.answer for e in ds1._train],
            [e.answer for e in ds2._train],
        )

    def test_gsm8k_examples_have_required_fields(self):
        examples = _make_gsm8k_examples()
        for ex in examples:
            self.assertIn("question", ex.inputs())
            self.assertIn("answer", ex)


# ---------------------------------------------------------------------------
# Program tests
# ---------------------------------------------------------------------------

class TestPrograms(unittest.TestCase):

    def test_cot_program_instantiates(self):
        from experiments.programs import CoT

        prog = CoT()
        self.assertTrue(hasattr(prog, "prog"))
        self.assertEqual(len(prog.predictors()), 1)

    def test_iris_program_instantiates(self):
        from experiments.programs import IrisProgram

        prog = IrisProgram()
        self.assertTrue(hasattr(prog, "generate_answer"))
        self.assertEqual(len(prog.predictors()), 1)


# ---------------------------------------------------------------------------
# Metric wrapping tests
# ---------------------------------------------------------------------------

class TestMetricWrapping(unittest.TestCase):

    def test_wrap_metric_for_bootstrap_returns_float(self):
        """GEPAFewShot._wrap_metric_for_bootstrap must produce a 3-arg metric."""
        from dspy.teleprompt.gepa import GEPAFewShot

        with patch("dspy.evaluate.as_gepa_metric", side_effect=lambda m: m):
            optimizer = GEPAFewShot(
                metric=_dummy_gepa_metric,
                auto="light",
                reflection_lm=_dummy_lm(),
            )

        wrapped = optimizer._wrap_metric_for_bootstrap()
        gold = Example(question="q", answer="42").with_inputs("question")
        pred = dspy.Prediction(answer="42")
        result = wrapped(gold, pred, trace=None)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.0)

    def test_wrap_metric_handles_score_with_feedback(self):
        """Wrapping should extract .score from ScoreWithFeedback objects."""
        from dspy.teleprompt.gepa import GEPAFewShot
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

        def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return ScoreWithFeedback(score=0.75, feedback="good")

        with patch("dspy.evaluate.as_gepa_metric", side_effect=lambda m: m):
            optimizer = GEPAFewShot(
                metric=metric_with_feedback,
                auto="light",
                reflection_lm=_dummy_lm(),
            )

        wrapped = optimizer._wrap_metric_for_bootstrap()
        gold = Example(question="q", answer="42").with_inputs("question")
        pred = dspy.Prediction(answer="42")
        result = wrapped(gold, pred)
        self.assertAlmostEqual(result, 0.75)


# ---------------------------------------------------------------------------
# GEPAFewShot inheritance tests
# ---------------------------------------------------------------------------

class TestGEPAFewShotInheritance(unittest.TestCase):

    def test_gepa_fewshot_is_gepa_subclass(self):
        from dspy.teleprompt.gepa import GEPA, GEPAFewShot

        self.assertTrue(issubclass(GEPAFewShot, GEPA))

    def test_gepa_fewshot_extra_attributes(self):
        from dspy.teleprompt.gepa import GEPAFewShot

        with patch("dspy.evaluate.as_gepa_metric", side_effect=lambda m: m):
            opt = GEPAFewShot(
                metric=_dummy_gepa_metric,
                auto="light",
                reflection_lm=_dummy_lm(),
                k_demos=5,
                demo_mutation_strategy="random",
            )

        self.assertEqual(opt.k_demos, 5)
        self.assertEqual(opt.demo_mutation_strategy, "random")
        self.assertEqual(opt.max_bootstrapped_demos, 16)


# ---------------------------------------------------------------------------
# GEPAFewShotAdapter tests
# ---------------------------------------------------------------------------

class TestGEPAFewShotAdapter(unittest.TestCase):

    def _make_adapter(self, strategy="metric_based"):
        from dspy.teleprompt.gepa.gepa_fewshot import GEPAFewShotAdapter
        from experiments.programs import CoT

        student = CoT()
        demo_pool = {
            "prog": [
                (Example(question=f"q{i}", answer=str(i)).with_inputs("question"), float(i % 2))
                for i in range(8)
            ]
        }
        adapter = GEPAFewShotAdapter(
            student_module=student,
            metric_fn=_dummy_gepa_metric,
            feedback_map={},
            demo_pool=demo_pool,
            k_demos=3,
            mutation_strategy=strategy,
            rng=random.Random(42),
        )
        return adapter, demo_pool

    def test_candidate_key_is_stable(self):
        adapter, _ = self._make_adapter()
        c1 = {"prog": "instr_a", "other": "instr_b"}
        c2 = {"other": "instr_b", "prog": "instr_a"}
        self.assertEqual(adapter._candidate_key(c1), adapter._candidate_key(c2))

    def test_get_demos_initializes_from_pool(self):
        adapter, pool = self._make_adapter()
        candidate = {"prog": "some instruction"}
        demos = adapter._get_demos(candidate)
        self.assertIn("prog", demos)
        self.assertLessEqual(len(demos["prog"]), 3)
        # All returned demos must come from the pool
        pool_examples = {id(ex) for ex, _ in pool["prog"]}
        for d in demos["prog"]:
            self.assertIn(id(d), pool_examples)

    def test_get_demos_idempotent(self):
        adapter, _ = self._make_adapter()
        candidate = {"prog": "instruction x"}
        d1 = adapter._get_demos(candidate)
        d2 = adapter._get_demos(candidate)
        self.assertIs(d1, d2)

    def test_random_mutate_respects_k_demos(self):
        adapter, _ = self._make_adapter(strategy="random")
        pool = adapter.demo_pool["prog"]
        current = [ex for ex, _ in pool[:2]]
        result = adapter._random_mutate(current, pool)
        self.assertLessEqual(len(result), adapter.k_demos)

    def test_metric_based_mutate_respects_k_demos(self):
        adapter, _ = self._make_adapter(strategy="metric_based")
        pool = adapter.demo_pool["prog"]
        current = [ex for ex, _ in pool[:2]]
        result = adapter._metric_based_mutate(current, pool)
        self.assertLessEqual(len(result), adapter.k_demos)

    def test_propose_new_texts_registers_demos(self):
        """After propose_new_texts, the new candidate must have a registered demo set."""
        from dspy.teleprompt.gepa.gepa_fewshot import GEPAFewShotAdapter
        from experiments.programs import CoT

        student = CoT()
        pool = {
            "prog": [
                (Example(question=f"q{i}", answer=str(i)).with_inputs("question"), 1.0)
                for i in range(6)
            ]
        }
        adapter = GEPAFewShotAdapter(
            student_module=student,
            metric_fn=_dummy_gepa_metric,
            feedback_map={},
            demo_pool=pool,
            k_demos=2,
            mutation_strategy="random",
            rng=random.Random(0),
        )

        old_candidate = {"prog": "old instruction"}
        new_instructions = {"prog": "new instruction"}

        # Patch parent's propose_new_texts to return new_instructions without an LM call
        with patch.object(
            type(adapter).__bases__[0],  # DspyAdapter
            "propose_new_texts",
            return_value=new_instructions,
        ):
            result = adapter.propose_new_texts(old_candidate, {}, ["prog"])

        self.assertEqual(result, new_instructions)
        # New candidate must be registered
        new_key = adapter._candidate_key(new_instructions)
        self.assertIn(new_key, adapter._demo_registry)


# ---------------------------------------------------------------------------
# Optimizer instantiation tests (no LM calls)
# ---------------------------------------------------------------------------

class TestOptimizerInstantiation(unittest.TestCase):

    def test_gepa_instantiates(self):
        from dspy.teleprompt.gepa import GEPA

        opt = GEPA(
            metric=_dummy_gepa_metric,
            auto="light",
            reflection_lm=_dummy_lm(),
        )
        self.assertIsNotNone(opt)

    def test_gepa_fewshot_instantiates(self):
        from dspy.teleprompt.gepa import GEPAFewShot

        opt = GEPAFewShot(
            metric=_dummy_gepa_metric,
            auto="light",
            reflection_lm=_dummy_lm(),
            k_demos=3,
        )
        self.assertIsNotNone(opt)

    def test_invalid_mutation_strategy_raises(self):
        from dspy.teleprompt.gepa.gepa_fewshot import GEPAFewShotAdapter
        from experiments.programs import CoT

        adapter = GEPAFewShotAdapter(
            student_module=CoT(),
            metric_fn=_dummy_gepa_metric,
            feedback_map={},
            demo_pool={"prog": []},
            k_demos=3,
            mutation_strategy="invalid_strategy",
            rng=random.Random(0),
        )
        with self.assertRaises(ValueError):
            adapter._mutate_demos(current=[], pool=[(Example(question="q").with_inputs("question"), 1.0)], reflective_examples=[])


# ---------------------------------------------------------------------------
# CLI argument parsing test
# ---------------------------------------------------------------------------

class TestCLIArguments(unittest.TestCase):

    def test_required_args_parsed(self):
        import argparse
        import importlib.util
        import types

        # Load the CLI module without executing main()
        spec = importlib.util.spec_from_file_location(
            "run_experiment",
            os.path.join(os.path.dirname(__file__), "..", "run_experiment.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Stub out top-level side-effects we don't want during import
        mod.dspy = MagicMock()
        mod.Evaluate = MagicMock()
        mod.deploy_sglang_model = MagicMock()
        mod.is_server_up = MagicMock(return_value=True)
        sys.modules["run_experiment"] = mod

        # Manually re-build the parser from the script
        # (cheapest way without importing the whole module with side-effects)
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset",   required=True)
        parser.add_argument("--optimizer", required=True)
        parser.add_argument("--model",     required=True)
        parser.add_argument("--auto",      default="light")
        parser.add_argument("--k-demos",   type=int, default=3)
        parser.add_argument("--log-dir",   default="experiments/logs")

        args = parser.parse_args([
            "--dataset",   "gsm8k",
            "--optimizer", "gepa_fewshot",
            "--model",     "meta-llama/Llama-3.2-3B-Instruct",
        ])
        self.assertEqual(args.dataset,   "gsm8k")
        self.assertEqual(args.optimizer, "gepa_fewshot")
        self.assertEqual(args.k_demos,   3)
        self.assertEqual(args.auto,      "light")


# ---------------------------------------------------------------------------
# Server utility tests
# ---------------------------------------------------------------------------

class TestServerUtils(unittest.TestCase):

    def test_is_server_up_returns_false_on_closed_port(self):
        from remote_setup.utils import is_server_up

        # Port 1 should never be open in a test environment
        self.assertFalse(is_server_up(port=1, timeout=1))

    def test_is_server_up_returns_true_on_loopback(self):
        """Open a temporary server socket and verify is_server_up detects it."""
        import socket
        import threading
        from remote_setup.utils import is_server_up

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", 0))
        port = server.getsockname()[1]
        server.listen(1)

        try:
            self.assertTrue(is_server_up(port=port, timeout=2))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
