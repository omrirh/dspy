import pytest

import dspy
from dspy import Example
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_numeric_encoder
from dspy.utils.dummies import DummyLM


def numeric_metric(example, prediction, trace=None):
    return example.label == prediction.label


def make_blob_examples(centers, per_blob=4, jitter=0.5):
    """Builds well-separated 2D numeric examples so KMeans/silhouette behave deterministically."""
    examples = []
    for cx, cy in centers:
        for i in range(per_blob):
            offset = (i - per_blob / 2) * jitter
            examples.append(
                Example(x1=cx + offset, x2=cy + offset, label="A").with_inputs("x1", "x2")
            )
    return examples


CENTERS = [(0.0, 0.0), (0.0, 20.0), (20.0, 0.0)]
trainset = make_blob_examples(CENTERS)
valset = make_blob_examples(CENTERS)


class SimpleClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("x1, x2 -> label")

    def forward(self, x1, x2):
        return self.predict(x1=x1, x2=x2)


def test_clusterfewshot_initialization():
    optimizer = ClusterFewshot(
        metric=numeric_metric,
        task_type="classification",
        semantic_encoders=[create_numeric_encoder()],
    )
    assert optimizer.metric == numeric_metric
    assert optimizer.task_type == "classification"
    assert optimizer.sampling_strategies == ["top_n", "best_in_cluster"]


def test_clusterfewshot_unknown_task_type_falls_back_to_defaults():
    optimizer = ClusterFewshot(
        metric=numeric_metric,
        task_type="not-a-real-task-type",
        semantic_encoders=[create_numeric_encoder()],
    )
    assert optimizer.sampling_strategies == ["top_n", "best_in_cluster"]


def test_clusterfewshot_requires_semantic_encoders():
    with pytest.raises(ValueError, match="semantic_encoders"):
        ClusterFewshot(metric=numeric_metric, task_type="classification")


def test_clusterfewshot_compile():
    dspy.configure(lm=DummyLM([{"label": "A"}] * 500, adapter=dspy.ChatAdapter()), adapter=dspy.ChatAdapter())

    optimizer = ClusterFewshot(
        metric=numeric_metric,
        task_type="classification",
        semantic_encoders=[create_numeric_encoder()],
    )

    compiled = optimizer.compile(student=SimpleClassifier(), trainset=trainset, valset=valset)

    assert compiled._compiled
    assert len(compiled.predict.demos) > 0
    assert len(compiled.predict.demos) <= optimizer.N
