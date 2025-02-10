import logging
import types
from typing import Any

import pandas as pd
import tqdm

import dspy
from dspy.utils.parallelizer import ParallelExecutor

try:
    from IPython.display import HTML
    from IPython.display import display as display

except ImportError:

    def display(obj: Any):
        """
        Display the specified Python object in the console.

        :param obj: The Python object to display.
        """
        print(obj)

    def HTML(x: str) -> str:
        """
        Obtain the HTML representation of the specified string.
        """
        return x

logger = logging.getLogger(__name__)


class Evaluate:
    def __init__(
        self,
        *,
        devset,
        metric=None,
        num_threads=1,
        display_progress=False,
        display_table=False,
        max_errors=5,
        return_all_scores=False,
        return_outputs=False,
        provide_traceback=False,
        failure_score=0.0,
        **_kwargs,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.return_all_scores = return_all_scores
        self.return_outputs = return_outputs
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score

    def __call__(
        self,
        program,
        metric=None,
        devset=None,
        num_threads=None,
        display_progress=None,
        display_table=None,
        return_all_scores=None,
        return_outputs=None,
    ):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_all_scores = return_all_scores if return_all_scores is not None else self.return_all_scores
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)

            if hasattr(program, "_assert_failures"):
                program._assert_failures += dspy.settings.get("assert_failures")
            if hasattr(program, "_suggest_failures"):
                program._suggest_failures += dspy.settings.get("suggest_failures")

            return prediction, score

        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results)]
        ncorrect, ntotal = sum(score for *_, score in results), len(devset)

        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        return round(100 * ncorrect / ntotal, 2)


class EvaluateWithTraces(Evaluate):
    """
    Extends the Evaluate class to include the option of returning execution traces during evaluation.
    Useful for debugging, interpretability, and understanding model decision-making processes.
    """
    def __init__(self, return_traces=False, **kwargs):
        """
        Initialize the EvaluateWithTraces class.

        :param return_traces: Whether to collect and return traces during evaluation.
        :param kwargs: Additional arguments passed to the parent Evaluate class.
        """
        super().__init__(**kwargs)
        self.return_traces = return_traces

    def __call__(self, program, return_traces=None, **kwargs):
        """
        Run the evaluation with optional trace collection.

        :param program: The program to evaluate.
        :param return_traces: Whether to collect traces during evaluation.
        :param kwargs: Additional evaluation parameters.
        :return: The average score of the evaluation and optional traces.
        """
        return_traces = return_traces if return_traces is not None else self.return_traces

        metric = kwargs.get("metric", self.metric)
        devset = kwargs.get("devset", self.devset)
        num_threads = kwargs.get("num_threads", self.num_threads)
        display_progress = kwargs.get("display_progress", self.display_progress)

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            with dspy.settings.context(trace=[] if return_traces else None):
                prediction = program(**example.inputs())
                score = metric(example, prediction)
                trace = dspy.settings.trace if return_traces else None

                if hasattr(program, "_assert_failures"):
                    program._assert_failures += dspy.settings.get("assert_failures")
                if hasattr(program, "_suggest_failures"):
                    program._suggest_failures += dspy.settings.get("suggest_failures")

                return (prediction, score, trace) if return_traces else (prediction, score)

        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        if return_traces:
            results = [(example, prediction, score, trace) for example, (prediction, score, trace) in zip(devset, results)]
        else:
            results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results)]

        ncorrect = sum(score for *_, score in results)
        ntotal = len(devset)
        avg_score = round(100 * ncorrect / ntotal, 2)

        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({avg_score}%)")

        if return_traces:
            return avg_score, [trace for *_, trace in results]

        return avg_score
