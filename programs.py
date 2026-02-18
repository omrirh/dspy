import dspy
import torch
import numpy as np
from typing import TYPE_CHECKING
from dspy.dsp.utils.utils import deduplicate

if TYPE_CHECKING:
    from dspy.teleprompt.retrieval_fewshot import RetrievalFewshot


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


class _RetrievalFewshotMixin:
    """
    Mixin providing per-query embedding and demo assignment for retrieval-driven few-shot programs.
    All retrieval strategy logic lives in the RetrievalClusterFewshot optimizer; this mixin
    only handles query embedding (task-specific) and demo assignment to predictors.
    """

    def _init_retrieval(self, cf_optimizer: "RetrievalFewshot"):
        self._cf_optimizer = cf_optimizer
        self._embedding_model = cf_optimizer.embedding_model
        self._use_target_model = cf_optimizer.use_target_model_embeddings
        if self._use_target_model:
            self._tokenizer = cf_optimizer.tokenizer

    def _embed_query(self, question: str) -> np.ndarray:
        """Embeds a text query using the same model used during compilation."""
        if self._use_target_model:
            return self._embed_with_target_model(question)
        return self._embedding_model.encode([question], convert_to_numpy=True)[0]

    def _embed_with_target_model(self, question: str, max_seq_length: int = 1024) -> np.ndarray:
        """Mean-pooled input embedding via the target LM — mirrors compile-time logic."""
        chat_str = self._tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoding = self._tokenizer(
            chat_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        device = next(self._embedding_model.parameters()).device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            token_embs = self._embedding_model.get_input_embeddings()(input_ids)
            attention_expanded = attention_mask.unsqueeze(-1)
            token_embs = token_embs * attention_expanded
            sum_embs = token_embs.sum(dim=1)
            lengths = attention_expanded.sum(dim=1).clamp(min=1)
            mean_emb = sum_embs / lengths

        return mean_emb.squeeze(0).cpu().numpy()

    def _assign_demos(self, selected_examples: list):
        """Assigns per-predictor demos from the selected bootstrapped example dicts."""
        for name, predictor in self.named_predictors():
            demos = []
            for ex in selected_examples:
                if name in ex:
                    demos.extend(ex[name])
            predictor.demos = demos


class RetrievalFewshotCoT(CoT, _RetrievalFewshotMixin):
    """
    Retrieval-driven CoT for GSM8K (arithmetic tasks).
    Selects few-shot demonstrations dynamically at inference time based on
    semantic similarity between the incoming question and the compiled cluster space.
    """

    def __init__(self, cf_optimizer: "RetrievalFewshot"):
        super().__init__()
        self._init_retrieval(cf_optimizer)

    def forward(self, question):
        query_emb = self._embed_query(question)
        self._assign_demos(self._cf_optimizer.retrieve_demos(query_emb))
        return super().forward(question)


class RetrievalFewshotMH(BasicMH, _RetrievalFewshotMixin):
    """
    Retrieval-driven multi-hop program for HotPotQA.
    Selects few-shot demonstrations dynamically at inference time based on
    semantic similarity between the incoming question and the compiled cluster space.
    """

    def __init__(self, cf_optimizer: "RetrievalFewshot"):
        super().__init__()
        self._init_retrieval(cf_optimizer)

    def forward(self, question):
        query_emb = self._embed_query(question)
        self._assign_demos(self._cf_optimizer.retrieve_demos(query_emb))
        return super().forward(question)


class RetrievalFewshotIrisProgram(IrisProgram, _RetrievalFewshotMixin):
    """
    Retrieval-driven Iris classifier.
    Uses raw feature vectors as the query embedding (no language model required)
    to retrieve the most semantically similar demonstrations at inference time.
    """

    def __init__(self, cf_optimizer: "RetrievalFewshot"):
        super().__init__()
        self._init_retrieval(cf_optimizer)

    def _embed_query(self, petal_length, petal_width, sepal_length, sepal_width) -> np.ndarray:
        """For Iris, the embedding is the raw input feature vector."""
        return np.array([float(petal_length), float(petal_width),
                         float(sepal_length), float(sepal_width)])

    def forward(self, petal_length, petal_width, sepal_length, sepal_width):
        query_emb = self._embed_query(petal_length, petal_width, sepal_length, sepal_width)
        self._assign_demos(self._cf_optimizer.retrieve_demos(query_emb))
        return super().forward(petal_length, petal_width, sepal_length, sepal_width)
