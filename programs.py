import dspy
import torch
import numpy as np
from dspy.dsp.utils.utils import deduplicate


class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
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
    answer = dspy.OutputField(desc='setosa, versicolor or virginica')


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

# TODO: experiment with this!
# class ClusterFewshotCoT(CoT):
#     def __init__(self, clusterfewshot):
#         super().__init__()
#         self.clusters = clusterfewshot.training_clusters
#         self.centers = {
#             cluster_id: np.mean([clusterfewshot.examples2embeddings[str(ex)] for ex in examples], axis=0)
#             for cluster_id, examples in self.clusters.items()
#         }
#         self.ranked_examples = clusterfewshot.ranked_examples
#
#         self.tokenizer = clusterfewshot.tokenizer
#         self.embedding_model = clusterfewshot.embedding_model
#
#         self.N = 3  # TODO: test this and make it dynamic
#
#     def forward(self, question):
#         # Embed the query using the same method
#         chat_str = self.tokenizer.apply_chat_template(
#             conversation=[
#                 {"role": "user", "content": question},
#                 {"role": "assistant", "content": ""}
#             ],
#             tokenize=False,
#             add_generation_prompt=False,
#         )
#
#         encoding = self.tokenizer(
#             chat_str,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=1024,
#         )
#         input_ids = encoding["input_ids"].to(self.embedding_model.device)
#         if input_ids.dim() == 1:
#             input_ids = input_ids.unsqueeze(0)
#
#         with torch.no_grad():
#             token_embs = self.embedding_model.get_input_embeddings()(input_ids)
#             query_embedding = token_embs.mean(dim=1).squeeze(0).cpu().numpy()
#
#         top_cluster_ids = sorted(
#             list(self.clusters.keys()),
#             key=lambda id: np.linalg.norm(query_embedding - self.centers[id]),
#             reverse=False,
#         )[:self.N]
#
#         # Assign top N performing one-shots from the closest cluster
#         self.prog.predict.demos = [self.clusters[cluster_id][0] for cluster_id in top_cluster_ids] # dynamically assign
#
#         return super().forward(question)
#

# class ClusterFewshotIrisProgram(IrisProgram):
#     def __init__(self, clusterfewshot):
#         super().__init__()
#         self.clusters = clusterfewshot.training_clusters
#         self.centers = {
#             cluster_id: np.mean([clusterfewshot.examples2embeddings[str(ex)] for ex in examples], axis=0)
#             for cluster_id, examples in self.clusters.items()
#         }
#         self.ranked_examples = clusterfewshot.ranked_examples
#
#         self.N = 3  # setosa, versicolor or virginia
#
#     def forward(self, petal_length, petal_width, sepal_length, sepal_width):
#         query_vect = [petal_length, petal_width, sepal_length, sepal_width]
#
#         top_cluster_ids = sorted(
#             list(self.clusters.keys()),
#             key=lambda id: np.linalg.norm(query_vect - self.centers[id]),
#             reverse=False,
#         )[:self.N]
#
#         # Assign top N performing one-shots from the closest cluster
#         self.generate_answer.predict.demos = [self.clusters[cluster_id][0] for cluster_id in top_cluster_ids] # dynamically assign
#
#         return super().forward(petal_length, petal_width, sepal_length, sepal_width)
