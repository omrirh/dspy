import logging
import numpy as np
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import dspy
from dspy.teleprompt import Teleprompter
from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)


class ClusterFewShot(Teleprompter):
    def __init__(
            self,
            metric=None,
            metric_threshold=None,
            teacher_settings: Optional[Dict] = None,
            num_fewshot: int = 4,
            valset_ratio: float = 0.1,
            descending: bool = True,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # Default embedding model
    ):
        """
        ClusterFewShot: Optimized few-shot selection using clustering and semantic embeddings.

        Args:
            metric: Callable
                Function to evaluate the model's predictions.
            metric_threshold: Optional[float]
                Threshold for metric-based filtering.
            teacher_settings: Optional[Dict]
                Configuration for the teacher model.
            num_fewshot: int
                Number of few-shot demonstrations (also number of clusters).
            embedding_model: str
                Pretrained model for generating semantic embeddings.
            descending: bool
                Whether to sort demonstrations within each cluster in descending order of impact.
        """
        super().__init__()
        self.valset_ratio = valset_ratio
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = teacher_settings or {}
        self.N = num_fewshot
        self.descending = descending

        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.clusters = None
        self.trainset = None
        self.validation_set = None
        self.sorted_demos = None
        self.fewshot_subset = None  # Final selected demonstrations

    def compile(self, student, *, teacher=None, trainset):
        """
        Compiles the ClusterFewShot optimizer.
        """
        self.trainset = trainset
        self.M = int(len(trainset) * self.valset_ratio)
        self._cluster_trainset()
        self._assemble_validation_set()
        self._sort_demos_per_cluster()  # TODO: would this computationally be too expensive? consider sampling a random subset for each cluster
        self.fewshot_subset = self.collect_fewshot_subset()

        # Update student LM predictors with optimized few-shot subset
        for _, predictor in student.named_predictors():
            predictor.demos = self.fewshot_subset

        self.student = student
        self.student._compiled = True
        return self.student

    def _cluster_trainset(self):
        """
        Uses semantic embeddings to cluster the training set into N groups.
        """
        logger.info("Generating examples embeddings for clustering...")
        texts = [ex.question for ex in self.trainset]  # Should we include `ex.answer` too?
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        logger.info(f"Clustering into {self.N} clusters...")
        kmeans = KMeans(n_clusters=self.N, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        self.clusters = {i: [] for i in range(self.N)}
        for idx, label in enumerate(cluster_labels):
            self.clusters[label].append(self.trainset[idx])

        logger.info(f"Examples clustering by semantic embedding completed (K={self.N}).")

    def _assemble_validation_set(self):
        """
        Selects M/N most central examples from each cluster to form the validation set.
        """
        self.validation_set = []
        samples_per_cluster = max(1, self.M // self.N)  # Ensuring at least one sample per cluster

        for cluster_id, examples in self.clusters.items():
            if len(examples) <= samples_per_cluster:
                selected = examples  # If fewer examples than needed, take all
            else:
                # Sort by proximity to cluster centroids (using embedding distances)
                embeddings = self.embedding_model.encode([ex.question for ex in examples], convert_to_numpy=True)
                cluster_center = np.mean(embeddings, axis=0)
                distances = np.linalg.norm(embeddings - cluster_center, axis=1)
                selected_indices = np.argsort(distances)[:samples_per_cluster]
                selected = [examples[i] for i in selected_indices]

            self.validation_set.extend(selected)

        logger.info(f"Validation set assembled with {len(self.validation_set)} examples.")

    def _sort_demos_per_cluster(self):
        """
        Sorts demonstrations within each cluster based on their impact when used as one-shot examples.
        """
        self.sorted_demos = {}

        evaluator = Evaluate(
            devset=self.validation_set,
            metric=self.metric,
            num_threads=12,
        )

        for cluster_id, cluster_examples in self.clusters.items():
            # TODO: debug the size of cluster demo pool here!
            cluster_demo_pool = [
                example
                for
                example in cluster_examples
                if example not in self.validation_set
            ]
            self.sorted_demos[cluster_id] = sorted(
                cluster_demo_pool,
                key=lambda ex: self._evaluate_demo(ex, evaluator),
                reverse=self.descending
            )
        logger.info(f"Demonstrations sorted within each cluster (descending={self.descending}).")

    def _evaluate_demo(self, example, evaluator):
        """
        Evaluates an example by measuring how well it helps predict the validation set
        when used as a sole demonstration.
        """
        student_copy = self.student.deepcopy()

        for _, predictor in student_copy.named_predictors():
            predictor.demos = [example]  # Test as one-shot demonstration

        student_score, _ = evaluator(program=student_copy)

        example.score = student_score  # Store score for later sorting
        return student_score

    def collect_fewshot_subset(self):
        """
        Collects the final few-shot subset by selecting the top demonstration from each cluster.
        """
        fewshot_subset = [self.sorted_demos[cluster_id][0] for cluster_id in self.sorted_demos if
                          self.sorted_demos[cluster_id]]
        logger.info(f"Collected {len(fewshot_subset)} demonstrations for few-shot optimization.")
        return fewshot_subset
