"""
SemanticEncoder: Modular encoder interface for ClusterFewshot's Bring-Your-Own-Encoder design.

A SemanticEncoder encapsulates an embedding model and its corresponding transform function,
allowing ClusterFewshot to work with various encoding strategies (SentenceTransformers,
task-tuned LLMs, numeric encoders, etc.).
"""

import logging
import numpy as np
from typing import List, Callable, Any, Optional
from dspy.primitives import Example

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Encapsulates an encoder model and its transformation method for semantic embedding.

    The SemanticEncoder provides a unified interface for different types of encoders,
    making it easy to plug in custom example encoding strategies into ClusterFewshot.

    Example usage:
        # Using a SentenceTransformer encoder
        from sentence_transformers import SentenceTransformer

        def sentence_transform(encoder, examples):
            texts = [ex.question for ex in examples]
            return encoder.encode(texts, convert_to_numpy=True)

        model = SentenceTransformer("all-mpnet-base-v2")
        encoder = SemanticEncoder(
            encoder=model,
            transform_fn=sentence_transform,
            name="all-mpnet-base-v2"
        )

        # Using a numeric/identity encoder for classification tasks
        def numeric_transform(encoder, examples):
            return np.array([[val for _, val in ex.inputs().items()] for ex in examples])

        encoder = SemanticEncoder(
            encoder=None,
            transform_fn=numeric_transform,
            name="NumericEncoder"
        )
    """

    def __init__(
        self,
        encoder: Any,
        transform_fn: Callable[[Any, List[Example]], np.ndarray],
        name: Optional[str] = None
    ):
        """
        Initialize a SemanticEncoder.

        Args:
            encoder: The encoding model (e.g., SentenceTransformer, LLM, or None for identity)
            transform_fn: Function that takes (encoder, examples) and returns embeddings array
            name: Optional name for the encoder (used for logging and identification)
        """
        self.encoder = encoder
        self.transform_fn = transform_fn
        self._name = name or self._infer_name()

    def _infer_name(self) -> str:
        """Infer encoder name from the encoder object if not explicitly provided."""
        if self.encoder is None:
            return "CustomEncoder"
        if hasattr(self.encoder, 'model_name'):
            return str(self.encoder.model_name)
        if hasattr(self.encoder, '__class__'):
            return self.encoder.__class__.__name__
        return str(self.encoder)

    def encode(self, examples: List[Example]) -> np.ndarray:
        """
        Encode a list of examples into latent embedding vectors.

        Args:
            examples: List of Example objects to encode

        Returns:
            numpy array of shape (n_examples, embedding_dim) containing the embeddings
        """
        try:
            embeddings = self.transform_fn(encoder=self.encoder, examples=examples)
            logger.debug(f"[{self._name}] Encoded {len(examples)} examples → shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"[{self._name}] Encoding failed: {e}")
            raise

    def name(self) -> str:
        """Return the encoder name."""
        return self._name

    def __str__(self) -> str:
        """String representation showing the encoder name."""
        return self._name

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"SemanticEncoder(name='{self._name}', encoder={type(self.encoder).__name__})"


# ============================================================================
# COMMON TRANSFORM FUNCTIONS
# ============================================================================

def sentence_transformer_transform(encoder, examples: List[Example]) -> np.ndarray:
    """
    Default transform for SentenceTransformer encoders.

    Extracts the 'question' field from examples and encodes them using the encoder.

    Args:
        encoder: SentenceTransformer model
        examples: List of Example objects with 'question' field

    Returns:
        Embeddings array of shape (n_examples, embedding_dim)
    """
    texts = [ex.question for ex in examples]
    return encoder.encode(texts, convert_to_numpy=True)


def numeric_transform(encoder, examples: List[Example]) -> np.ndarray:
    """
    Transform for numeric/classification tasks using input features directly.

    Extracts input features from examples and returns them as embeddings.
    Useful for tasks where inputs are already numeric (e.g., Iris dataset).

    Args:
        encoder: Not used (can be None)
        examples: List of Example objects with numeric inputs

    Returns:
        Feature array of shape (n_examples, n_features)
    """
    return np.array([
        [input_val for _, input_val in dict(example.inputs()).items()]
        for example in examples
    ])


# ============================================================================
# ENCODER FACTORY HELPERS
# ============================================================================

def create_sentence_transformer_encoder(model_name: str, device: str = 'cpu') -> SemanticEncoder:
    """
    Factory function to create a SemanticEncoder from a SentenceTransformer model.

    Args:
        model_name: Name of the SentenceTransformer model (e.g., "all-mpnet-base-v2")
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        SemanticEncoder configured with the SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return SemanticEncoder(
        encoder=model,
        transform_fn=sentence_transformer_transform,
        name=model_name
    )


def create_numeric_encoder() -> SemanticEncoder:
    """
    Factory function to create a numeric/identity encoder for classification tasks.

    Returns:
        SemanticEncoder that uses input features directly as embeddings
    """
    return SemanticEncoder(
        encoder=None,
        transform_fn=numeric_transform,
        name="NumericEncoder"
    )
