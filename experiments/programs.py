"""
DSPy program definitions for each task.

  CoT         — Chain-of-Thought for GSM8K (question → answer)
  IrisProgram — Chain-of-Thought for Iris classification
"""
import dspy


class CoT(dspy.Module):
    """Single-hop Chain-of-Thought for math / QA tasks."""

    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


class IrisSignature(dspy.Signature):
    """Given the petal and sepal dimensions in cm, predict the iris species."""

    petal_length: float = dspy.InputField()
    petal_width:  float = dspy.InputField()
    sepal_length: float = dspy.InputField()
    sepal_width:  float = dspy.InputField()
    answer: str = dspy.OutputField(desc="setosa, versicolor, or virginica")


class IrisProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(IrisSignature)

    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        return self.generate_answer(
            petal_length=petal_length,
            petal_width=petal_width,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
        )
