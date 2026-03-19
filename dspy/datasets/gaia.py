"""
GAIA (General AI Assistants) dataset — ClusterFewshot agentic experiments.

GAIA is a 2024 benchmark specifically designed to evaluate general-purpose AI
assistants on real-world tasks that require multi-step tool use, grounded
reasoning, and synthesising information across sources.

Reference
---------
Mialon et al., "GAIA: a benchmark for General AI Assistants", 2024.
HuggingFace: https://huggingface.co/datasets/gaia-benchmark/GAIA

Why GAIA for ClusterFewshot?
----------------------------
* Level 1 questions require 1–2 tool calls (calculator, web search, file read),
  making them tractable for 7B–32B models without extensive scaffolding.
* Tasks span diverse domains (maths, trivia, file analysis, web look-up) —
  ideal for demonstrating that ClusterFewshot's semantic clustering can separate
  by task type and provide type-appropriate demonstrations.
* Exact-match scoring on the final answer is directly compatible with
  ``answer_exact_match`` used across all other ClusterFewshot experiments.

Setup — what is needed before using this class
-----------------------------------------------
1. **HuggingFace access**: GAIA is gated. Request access at
   https://huggingface.co/datasets/gaia-benchmark/GAIA and set::

       export HF_TOKEN=<your_token>

2. **Install the datasets library** (already a dependency of this repo)::

       pip install datasets

3. **Tools for Level 1 tasks**:
   Most Level 1 questions need at most one of:
   * ``calculator(expr: str) -> str``  — pure Python eval with math module
   * ``search(query: str) -> str``     — ColBERTv2 / Wikipedia lookup
   * ``read_file(path: str) -> str``   — reads a local file from the GAIA
                                         attachment directory

   Attachments (images, CSVs, PDFs) are distributed with the dataset.
   Level 1 mostly avoids multi-modal inputs — stick to Level 1 for the PoC.

4. **Encoder recommendation**:
   GAIA questions are semantically rich and domain-diverse. Recommended BYOE::

       from dspy.teleprompt.clusterfewshot import create_sentence_transformer_encoder

       encoders = [
           create_sentence_transformer_encoder("sentence-transformers/all-mpnet-base-v2"),
           create_sentence_transformer_encoder("sentence-transformers/multi-qa-mpnet-base-dot-v1"),
       ]

   The grid search will typically select the QA-tuned model given GAIA's
   question-answering structure.

Usage (once implemented)
------------------------
    from dspy.datasets.gaia import GaiaDataset

    dataset = GaiaDataset(level=1)
    trainset, devset, testset = dataset.get_data_splits(
        train_size=200, dev_size=100, test_size=100
    )

    # Each example has fields: question, answer, (optionally) file_path
    print(trainset[0].question)
    print(trainset[0].answer)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GaiaDataset:
    """
    Loader for the GAIA benchmark dataset.

    Downloads from HuggingFace (``gaia-benchmark/GAIA``) on first use.
    Requires a valid HF_TOKEN environment variable for the gated dataset.

    Args:
        level: GAIA difficulty level (1, 2, or 3). Level 1 is recommended
               for ClusterFewshot PoC — questions require at most 1–2 tool
               calls and no multi-modal reasoning.
        seed:  Random seed for train/dev split reproducibility.

    TODO: This class is a stub. Implement ``_load`` and ``get_data_splits``
          once the HF token is available and the agentic PoC on HotPotQA
          is validated end-to-end.
    """

    def __init__(self, level: int = 1, seed: int = 42):
        if level not in (1, 2, 3):
            raise ValueError(f"GAIA level must be 1, 2, or 3 — got {level}")
        self.level = level
        self.seed = seed
        self._train: Optional[list] = None
        self._dev: Optional[list] = None
        self._test: Optional[list] = None

    def _load(self):
        """
        Downloads and parses the GAIA dataset from HuggingFace.

        Each raw example is converted to a ``dspy.Example`` with at minimum:
            - ``question`` (str): the task prompt
            - ``answer``   (str): the expected exact-match answer
            - ``file_path``(str | None): path to an attached file, if any

        TODO: Implement this method.
              The HuggingFace dataset schema has columns:
              task_id, Question, Level, FinalAnswer, Annotator Metadata,
              file_name (for attachments).
        """
        raise NotImplementedError(
            "GaiaDataset._load() is not yet implemented.\n"
            "See the module docstring for setup instructions."
        )

    def get_data_splits(
        self,
        train_size: int = 200,
        dev_size: int = 100,
        test_size: int = 100,
    ):
        """
        Returns (trainset, devset, testset) as lists of ``dspy.Example``.

        The official GAIA dataset has a train split (~165 examples at Level 1)
        and a validation split (~60 examples at Level 1). The test split
        withholds gold answers — use the validation split as test for PoC.

        TODO: Implement once ``_load`` is complete.
        """
        raise NotImplementedError(
            "GaiaDataset.get_data_splits() is not yet implemented.\n"
            "See the module docstring for setup instructions."
        )
