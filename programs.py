import dspy
import torch
import numpy as np
from typing import TYPE_CHECKING, Callable
from dspy.dsp.utils.utils import deduplicate

if TYPE_CHECKING:
    from dspy.teleprompt.retrieval_fewshot import RetrievalFewshot


# ---------------------------------------------------------------------------
# Crop Recommendation — LLM-as-an-Agronomist
# ---------------------------------------------------------------------------

# Feature descriptions for dynamic signature generation
CROP_FEATURE_DESCRIPTIONS = {
    'nitrogen': "Nitrogen (N) content in soil, mg/kg",
    'phosphorous': "Phosphorous (P) content in soil, mg/kg",
    'potassium': "Potassium (K) content in soil, mg/kg",
    'temperature': "Average temperature, °C",
    'humidity': "Relative humidity, %",
    'ph': "Soil pH value",
    'rainfall': "Rainfall, mm",
}


def create_crop_recommender_signature(feature_names: list) -> type:
    """
    Dynamically creates a CropRecommender signature based on selected features.

    Args:
        feature_names: List of feature names to include as input fields

    Returns:
        A dspy.Signature class with the specified input fields
    """
    # Build the signature fields dictionary
    fields = {}
    for feature in feature_names:
        if feature in CROP_FEATURE_DESCRIPTIONS:
            fields[feature] = dspy.InputField(desc=CROP_FEATURE_DESCRIPTIONS[feature])
        else:
            fields[feature] = dspy.InputField()

    # Add the output field
    fields['crop'] = dspy.OutputField(
        desc="The recommended crop (one of: rice, maize, chickpea, kidneybeans, "
             "pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, "
             "banana, mango, grapes, watermelon, muskmelon, apple, orange, "
             "papaya, coconut, cotton, jute, coffee)"
    )

    # Create the signature class dynamically
    signature_class = type(
        'CropRecommenderSignature',
        (dspy.Signature,),
        {
            '__doc__': "You are an expert agronomist advisor. Given key environmental "
                      "conditions for a field, recommend the single most suitable crop to cultivate.",
            **fields
        }
    )

    return signature_class


class CropRecommender(dspy.Module):
    def __init__(self, feature_names: list = None):
        """
        Args:
            feature_names: List of feature names to use. If None, uses the global
                          CROP_INPUT_FIELDS from the dataset module.
        """
        super().__init__()

        # Import here to avoid circular dependency
        if feature_names is None:
            from dspy.datasets.crop_recommendation import CROP_INPUT_FIELDS
            feature_names = CROP_INPUT_FIELDS

        self.feature_names = feature_names
        signature = create_crop_recommender_signature(feature_names)
        self.recommend = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        """
        Dynamically forward based on available features.
        Accepts any subset of: nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall
        """
        # Only pass the features that are expected by the signature
        inputs = {k: v for k, v in kwargs.items() if k in self.feature_names}
        return self.recommend(**inputs)


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


class HotPotQAAgentSignature(dspy.Signature):
    """Answer multi-hop questions by searching Wikipedia for relevant passages.
    Think step-by-step: decompose the question, issue targeted searches to gather
    evidence, and synthesize findings into a concise factual answer."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A short, factual answer (entity name, date, or yes/no)")


class ReactAgentMH(dspy.Module):
    """
    ReAct-based multi-hop agent for HotPotQA backed by ColBERTv2 search.

    This program wraps ``dspy.ReAct`` with a Wikipedia search tool and exposes
    the standard ``question -> answer`` interface expected by ``answer_exact_match``.

    Named predictors (used by ClusterFewshot for demo injection):
        - ``agent.react``    — the step-level Predict module (thought + tool selection)
        - ``agent.extract``  — the final ChainOfThought extraction module

    The ``search_tool`` is a plain Python callable with signature::

        def search(query: str) -> str: ...

    It should return a formatted string of top passages (created by
    ``create_colbert_search_tool`` in ``react_agent_experiment.py``).

    Args:
        search_tool: Callable that accepts a ``query`` string and returns passages.
        max_iters: Maximum ReAct steps before forced extraction. Default 20
                   matches the upstream DSPy ReAct default.
    """

    def __init__(self, search_tool: Callable, max_iters: int = 20):
        super().__init__()
        self.agent = dspy.ReAct(HotPotQAAgentSignature, tools=[search_tool], max_iters=max_iters)

    def forward(self, question: str):
        return self.agent(question=question)


class _RetrievalFewshotMixin:
    """
    Mixin providing per-query embedding and demo assignment for retrieval-driven few-shot programs.
    All retrieval strategy logic lives in the RetrievalClusterFewshot optimizer; this mixin
    only handles query embedding (task-specific) and demo assignment to predictors.
    """

    def _init_retrieval(self, cf_optimizer: "RetrievalFewshot"):
        self._cf_optimizer = cf_optimizer
        # cf_optimizer.embedding_model is the underlying SentenceTransformer (or None for
        # numeric encoders). Set by RetrievalFewshot.compile() after encoder selection.
        self._embedding_model = cf_optimizer.embedding_model

    def _embed_query(self, question: str) -> np.ndarray:
        """Embeds a text query using the same SentenceTransformer used during compilation."""
        return self._embedding_model.encode([question], convert_to_numpy=True)[0]

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
