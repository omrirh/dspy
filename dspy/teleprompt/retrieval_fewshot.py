import logging
import numpy as np
from typing import List

from dspy.primitives import Example
from dspy.teleprompt.cluster_fewshot import ClusterFewshot

logger = logging.getLogger(__name__)

RETRIEVAL_STRATEGIES = {"knn", "mmr"}


class RetrievalFewshot(ClusterFewshot):
    """
    RetrievalFewshot (RFS): instance-level adaptive few-shot selection optimizer.

    Pipeline:
        1. Bootstraps training examples to generate successful reasoning paths,
           yielding a candidate demonstration pool.
        2. Embeds the pool into a shared semantic space
           (via SentenceTransformer model selection or target LM input embeddings).
        3. At inference time, retrieves a query-specific few-shot context
           using kNN or MMR over the demonstration pool embeddings.

    Inherits bootstrapping and embedding utilities from ClusterFewshot;
    clustering, ranking, and static subset selection are all skipped.

    Args:
        task_type: str
            Task label (used for classification-specific embedding path).
        retrieval_program_class:
            Class to instantiate as the compiled student (e.g. RetrievalFewshotCoT).
            Must accept a single argument: this optimizer instance.
        metric: Callable
            Evaluation metric forwarded to the bootstrapping step.
        metric_threshold: Optional[float]
            Threshold for metric-based filtering during bootstrapping.
        use_target_model_embeddings: bool
            Use target LM input embeddings instead of a SentenceTransformer.
        n_shots: int
            Number of demonstrations to retrieve per query. Default: 3.
        retrieval_strategy: str
            "knn" — n globally nearest training examples by embedding distance.
            "mmr" — Maximal Marginal Relevance, balances query relevance and diversity.
        mmr_lambda: float
            Relevance/diversity tradeoff for MMR (0 = full diversity, 1 = full relevance).
            Only used when retrieval_strategy="mmr".
    """

    def __init__(
            self,
            task_type: str,
            retrieval_program_class,
            metric=None,
            metric_threshold=None,
            use_target_model_embeddings: bool = False,
            n_shots: int = 3,
            retrieval_strategy: str = "knn",
            mmr_lambda: float = 0.5,
    ):
        super().__init__(
            task_type=task_type,
            metric=metric,
            metric_threshold=metric_threshold,
            soft_select=False,
            use_target_model_embeddings=use_target_model_embeddings,
        )

        if retrieval_strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"Unknown retrieval strategy '{retrieval_strategy}'. "
                f"Choose from: {RETRIEVAL_STRATEGIES}"
            )

        self.retrieval_program_class = retrieval_program_class
        self.n_shots = n_shots
        self.retrieval_strategy = retrieval_strategy
        self.mmr_lambda = mmr_lambda

    def _needs_ranking(self) -> bool:
        """kNN and MMR never use one-shot utility ranks."""
        return False

    def compile(self, student, trainset: List[Example], *, valset):
        """
        Compiles the RetrievalFewshot optimizer.

        1. Bootstraps training examples (generates traced reasoning paths).
        2. Embeds the bootstrapped pool into a semantic space.
        3. Returns a retrieval-enabled student program.

        Clustering, ranking, and static subset selection are skipped entirely.
        """
        self.student = student.deepcopy()
        self.trainset = self.bootstrap_examples(trainset)
        self.valset = valset

        logger.info("Compiling RetrievalFewshot optimizer...")

        # --- Embed the demonstration pool ---
        if self.task_type == "classification":
            # Raw feature vectors (no model needed)
            data = [ex["raw"] for ex in self.trainset]
            embeddings = np.array(
                [[float(v) for _, v in dict(example.inputs()).items()] for example in data]
            )
            self.embedding_model_name = "N/A"
        else:
            if self.use_target_model_embeddings:
                self.embedding_model_name = self.student.named_predictors()[0][1].lm.model
                self.generate_embeddings_func = self.generate_embedding_clusters_with_target_model
            else:
                self.generate_embeddings_func = self.generate_embedding_clusters_with_candidate_models

            data = [ex["raw"] for ex in self.trainset]
            # Model selection runs internally (silhouette-based); cluster labels discarded
            embeddings, _, _ = self.generate_embeddings_func(examples=data)

        self.examples2embeddings = {
            self.get_example_hash(ex): np.array(emb)
            for ex, emb in zip(self.trainset, embeddings)
        }

        logger.info(
            f"Demonstration pool ready: {len(self.trainset)} bootstrapped examples, "
            f"embedding_model={self.embedding_model_name}, strategy={self.retrieval_strategy}."
        )

        retrieval_student = self.retrieval_program_class(self)
        retrieval_student._compiled = True

        logger.info("RetrievalFewshot compilation complete.")

        return retrieval_student

    # ------------------------------------------------------------------
    # Public retrieval entry point (called by student programs at inference)
    # ------------------------------------------------------------------

    def retrieve_demos(self, query_embedding: np.ndarray, n_shots: int = None) -> list:
        """
        Selects few-shot demonstrations for a given query embedding using
        the configured retrieval strategy.

        Args:
            query_embedding: Embedding of the incoming query.
            n_shots: Override the optimizer's default n_shots for this call.

        Returns:
            List of bootstrapped example dicts selected as demonstrations.
        """
        n = n_shots if n_shots is not None else self.n_shots

        if self.retrieval_strategy == "knn":
            return self._retrieve_knn(query_embedding, n)
        else:  # "mmr"
            return self._retrieve_mmr(query_embedding, n)

    # ------------------------------------------------------------------
    # Retrieval strategy implementations
    # ------------------------------------------------------------------

    def _retrieve_knn(self, query_embedding: np.ndarray, n: int) -> list:
        """
        Global k-nearest-neighbor retrieval.
        Returns the n training examples whose embeddings are closest to the query.
        """
        distances = [
            (ex, np.linalg.norm(query_embedding - self.examples2embeddings[self.get_example_hash(ex)]))
            for ex in self.trainset
        ]
        distances.sort(key=lambda t: t[1])
        return [ex for ex, _ in distances[:n]]

    def _retrieve_mmr(self, query_embedding: np.ndarray, n: int) -> list:
        """
        Maximal Marginal Relevance retrieval.
        Iteratively selects examples that balance relevance to the query with
        diversity among already-selected demonstrations.

        Score at each step: λ · sim(query, x) − (1−λ) · max_j sim(x, selected_j)

        Uses cosine similarity. mmr_lambda=1.0 → pure relevance (equivalent to kNN);
        mmr_lambda=0.0 → pure diversity.
        """
        lam = self.mmr_lambda
        candidates = list(self.trainset)

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Precompute query-similarity for all candidates
        query_sims = {
            self.get_example_hash(ex): cosine(query_norm, self.examples2embeddings[self.get_example_hash(ex)])
            for ex in candidates
        }

        selected = []
        selected_embeddings = []

        for _ in range(min(n, len(candidates))):
            best_ex = None
            best_score = -np.inf

            for ex in candidates:
                if ex in selected:
                    continue
                ex_emb = self.examples2embeddings[self.get_example_hash(ex)]
                rel = query_sims[self.get_example_hash(ex)]

                redundancy = (
                    max(cosine(ex_emb, s_emb) for s_emb in selected_embeddings)
                    if selected_embeddings else 0.0
                )

                score = lam * rel - (1 - lam) * redundancy

                if score > best_score:
                    best_score = score
                    best_ex = ex

            if best_ex is not None:
                selected.append(best_ex)
                selected_embeddings.append(self.examples2embeddings[self.get_example_hash(best_ex)])

        return selected
