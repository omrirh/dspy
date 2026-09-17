"""Quickstart: ClusterFewshot on real GSM8K with an API model.

Run:
    python examples/clusterfewshot/quickstart_gsm8k.py

This uses a small subset of GSM8K (40 train / 24 val / 20 test) to keep the
call budget and cost low for a first validation run — roughly 500 LM calls
total with gpt-4o-mini, a few cents. To reproduce paper-scale numbers, swap
in the full splits (dataset.train / dataset.dev, 200 / 300 examples) and add
a second encoder to exercise the BYOE grid search, e.g.:

    encoders = [
        create_sentence_transformer_encoder("all-mpnet-base-v2"),
        create_sentence_transformer_encoder("Qwen/Qwen3-Embedding-0.6B"),
    ]
"""

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


dataset = GSM8K()
trainset = dataset.train[:40]
valset = dataset.dev[:24]
testset = dataset.test[:20]

optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=[create_sentence_transformer_encoder("all-mpnet-base-v2")],
)

optimized = optimizer.compile(student=CoT(), trainset=trainset, valset=valset)

evaluate = dspy.Evaluate(devset=testset, metric=gsm8k_metric, display_progress=True)
baseline_score = evaluate(CoT()).score
optimized_score = evaluate(optimized).score

num_demos = sum(len(predictor.demos) for _, predictor in optimized.named_predictors())
print(f"\nDemos selected: {num_demos}")
print(f"Baseline (no demos) test accuracy:  {baseline_score:.1f}%")
print(f"ClusterFewshot test accuracy:       {optimized_score:.1f}%")
