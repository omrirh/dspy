"""Quickstart: ClusterFewshot on real GSM8K with a local OpenAI-compatible model
(e.g. SGLang or vLLM serving Qwen3.5-9B on an A100).

Configure via env vars, matching whatever you deployed:
    export LOCAL_LM_MODEL=Qwen/Qwen3.5-9B      # must match the served model name
    export LOCAL_LM_API_BASE=http://localhost:30000/v1
    export LOCAL_LM_API_KEY=EMPTY              # most local servers don't check this

Run:
    python examples/clusterfewshot/quickstart_gsm8k_local.py

No per-token cost, so free to scale up trainset/valset beyond the 40/24/20
default here — the limit becomes wall-clock time on your GPU, not spend.
"""

import os

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder

model = os.environ.get("LOCAL_LM_MODEL", "Qwen/Qwen3.5-9B")
api_base = os.environ.get("LOCAL_LM_API_BASE", "http://localhost:7501/v1")
api_key = os.environ.get("LOCAL_LM_API_KEY", "EMPTY")

lm = dspy.LM(f"openai/{model}", api_base=api_base, api_key=api_key)
# Open-weight models are generally more reliable with ChatAdapter's
# [[ ## field ## ]] format than strict JSON mode.
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


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

print(f"\nModel: {model} @ {api_base}")
print(f"Demos selected: {len(optimized.predict.demos)}")
print(f"Baseline (no demos) test accuracy:  {baseline_score:.1f}%")
print(f"ClusterFewshot test accuracy:       {optimized_score:.1f}%")
