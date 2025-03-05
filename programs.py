import dspy
from dspy.dsp.utils.utils import deduplicate


class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        # TODO: learn how the search_query looks like
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(self.num_hops)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for hop in range(self.num_hops):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)
        answer = self.generate_answer(context=context, question=question).copy(context=context)
        return answer


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


class IrisSignature(dspy.Signature):
    """
    Given the petal and sepal dimensions in cm, predict the iris species.
    """
    petal_length = dspy.InputField()
    petal_width = dspy.InputField()
    sepal_length = dspy.InputField()
    sepal_width = dspy.InputField()
    answer = dspy.OutputField(desc='setosa, versicolor, or virginica')


class IrisProgram(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought(IrisSignature)

    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        return self.generate_answer(
            petal_length=petal_length,
            petal_width=petal_width,
            sepal_length=sepal_length,
            sepal_width=sepal_width
        )
