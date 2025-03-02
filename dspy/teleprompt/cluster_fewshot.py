import logging
import numpy as np
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from dspy.primitives import Program, Example
from sentence_transformers import SentenceTransformer

from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)


class ClusterFewshot(Teleprompter):
    def __init__(
            self,
            metric=None,
            metric_threshold=None,
            teacher_settings: Optional[Dict] = None,
            num_fewshot: int = 4,
            valset_ratio: float = 0.1,
            descending: bool = True,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        ClusterFewshot: Optimized few-shot selection using clustering over semantic embeddings
            and example-as-one-shot evaluation.

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
                Whether to sort examples within each cluster
                    in descending order of impact as one-shot demonstrations.
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
        self.ranked_examples = None
        self.gloabl_sorted_examples = None
        self._sum_of_clusters_mean_rank = None
        self.fewshot_subset = None  # Final selected demonstrations

    def compile(self, student: Program, trainset: List[Example], *, teacher=None, valset=None):
        """
        Compiles the ClusterFewshot optimizer.
        """
        self.student = student.deepcopy()
        self.trainset = trainset
        self.M = int(len(trainset) * self.valset_ratio)

        logger.info("Compiling the student program using ClusteFewshot optimizer...")
        self._cluster_examples()
        self._assemble_validation_set()
        self._sort_examples_as_demos()
        self.fewshot_subset = self.collect_fewshot_subset()

        # Update student LM predictors with optimized few-shot subset
        for _, predictor in student.named_predictors():
            predictor.demos = self.fewshot_subset  # TODO: try with/without in-context sort

        self.student._compiled = True

        logger.info("Student program compiled successfully.")

        return self.student

    def _cluster_examples(self):
        """
        Uses semantic embeddings to cluster the training set into N groups.
        """
        logger.info("Generating examples embeddings for clustering...")
        texts = [ex.question for ex in self.trainset]  # TODO: Try also with ex.answer included (different distribution)

        # maps examples to 384 dimensional dense vector space
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        logger.info(f"Clustering into {self.N} clusters...")
        kmeans = KMeans(n_clusters=self.N, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        self.clusters = {i: [] for i in range(self.N)}
        for idx, label in enumerate(cluster_labels):
            self.clusters[label].append(self.trainset[idx])

        # TODO: add visulization of clusters with matplotlib

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
                # TODO: could there be any valuable demos here we're missing? it will not be used for selecting demonstrations
                # TODO: thought - maybe we should sample "hardest" questions from "selected" by max probability confidence?

            self.validation_set.extend(selected)

        logger.info(f"Validation set assembled with {len(self.validation_set)} examples.")

    def _sort_examples_as_demos(self):
        """
        Sorts examples globally based on their impact when used as one-shot examples.
        """
        raw_trainset = [ex for ex in self.trainset if ex not in self.validation_set]  # TODO: debug number of items in raw trainset

        evaluator = Evaluate(
            devset=self.validation_set,
            metric=self.metric,
            num_threads=12,
        )

        logger.info(f"Sorting examples-as-demos from trainset ({len(raw_trainset)} examples)")
        # TODO: this has high runtime!! consider a clever way of sampling from raw_trainset without randomness.
        self.ranked_examples = {ex: self._evaluate_example_as_demo(ex, evaluator) for ex in raw_trainset}
        self.global_sorted_examples = sorted(
            self.ranked_examples.keys(),
            key=lambda ex: self.ranked_examples[ex],
            reverse=self.descending,
        )

        logger.info(f"Re-ordering clusters by demonstrations ranks")
        self.clusters = {
            cluster_id: sorted(
                [ex for ex in cluster_examples if ex in self.ranked_examples],
                key=lambda ex: self.ranked_examples[ex],  # TODO: verify that ex is a valid key
                reverse=self.descending,
            )
            for cluster_id, cluster_examples in self.clusters
        }

        logger.info(f"Demonstrations are sorted in {'descending' if self.descending else 'ascending'} order.")

    def _evaluate_example_as_demo(self, example, evaluator):
        """
        Evaluates an example by measuring how well it helps predict the validation set
        when used as a sole demonstration.
        """
        student_copy = self.student.deepcopy()

        logger.info(f"ֿ\n\nPredicting the validation set ({len(self.validation_set)} questions) "
                    f"using the following demonstration:\n"
                    f"{example.question} --> {example.answer}")

        for _, predictor in student_copy.named_predictors():
            predictor.demos = [example]  # Test as one-shot demonstration

        student_score = evaluator(program=student_copy)

        return student_score

    def collect_fewshot_subset(self):
        """
        Collects the final few-shot subset using the Strategy design pattern (incorporating several approaches),
        while balancing exploration (diversity) and exploitation (strong demos).
        """
        sampling_method = "cluster_strength"
        logger.info(f"Collecting few-shot subset from clusters using {sampling_method} method")

        fewshot_subset = []

        for cluster_id, _ in self.clusters:
            fewshot_subset.extend(
                self.sample_examples_from_cluster(
                    cluster_id=cluster_id,
                    method=sampling_method  # TODO: run with each of 3 approaches to evaluate
                )
            )

        self.fewshot_subset = fewshot_subset

    def sample_examples_from_cluster(self, cluster_id, method="cluster_strength"):
        """
        Samples examples from a given cluster based on one of 3 approaches:
        1. top N: Collects examples that fall in the top global N potential demos rank.
            If there aren't any in the given cluster, returns an empty list.
        2. Best in cluster: Returns the top-ranked example-as-demo from the given cluster.
        3. Cluster strength: Allocates a proportionate number of slots in few-shot to the given cluster,
            based on cluster strength (i.e., mean example ranking).
        """
        sampled_examples = []

        if cluster_id in self.clusters:
            cluster_examples = self.clusters[cluster_id]
            if not cluster_examples:
                return sampled_examples

            if method == "top_n":
                top_global_n = self.global_sorted_examples[:self.N]
                sampled_examples.extend([ex for ex in cluster_examples if ex in top_global_n])

            elif method == "best_in_cluster":
                sampled_examples.append(cluster_examples[0])

            elif method == "cluster_strength":
                # Compute sum of clusters mean rank **once**
                if not self._sum_of_clusters_mean_rank:
                    self._sum_of_clusters_mean_rank = 0
                    for _, cluster in self.clusters.items():
                        self._sum_of_clusters_mean_rank += np.mean([self.ranked_examples[ex] for ex in cluster])

                # Calculate cluster strength as the mean rank of its examples
                cluster_ranks = [self.ranked_examples[ex] for ex in cluster_examples]
                cluster_strength = np.mean(cluster_ranks)

                # Compute how many slots to allocate for this cluster
                proportion = cluster_strength / self._sum_of_clusters_mean_rank  # TODO: examine edge use-cases for this proportion
                num_slots = int(self.N * proportion)

                # Select top-ranked examples from the cluster based on this proportion
                if num_slots >= 1:
                    sampled_examples = cluster_examples[:num_slots]

        return sampled_examples

