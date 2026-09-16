# dspy.ClusterFewshot

`ClusterFewshot` is a task-adaptive few-shot demonstration selector. It clusters training and validation examples in a semantic embedding space, scores each candidate demonstration by its empirical effect as a one-shot example, and then picks the demonstration set that performs best on the validation set among a few candidate sampling strategies. Its Bring-Your-Own-Encoder (BYOE) design lets you supply one or more semantic encoders; `ClusterFewshot` evaluates all of them via grid search and keeps the one that produces the best clustering (highest silhouette score).

!!! note "Requires the `clusterfewshot` extra"
    `dspy.ClusterFewshot` requires scikit-learn, sentence-transformers, and datasets. Install with `pip install dspy[clusterfewshot]`.

<!-- START_API_REF -->
::: dspy.ClusterFewshot
    handler: python
    options:
        members:
            - compile
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

## Example Usage

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import ClusterFewshot
from dspy.teleprompt.clusterfewshot import create_sentence_transformer_encoder

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


dataset = GSM8K()

optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=[create_sentence_transformer_encoder("all-mpnet-base-v2")],
)

optimized = optimizer.compile(
    student=CoT(),
    trainset=dataset.train,
    valset=dataset.dev,
)
```

See `examples/clusterfewshot/quickstart_gsm8k.py` for a runnable version, and `examples/clusterfewshot/quickstart_gsm8k_local.py` for using a locally-hosted model.
