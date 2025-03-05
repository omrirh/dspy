import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
            num_fewshot: int = 4,
            descending: bool = True,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            sampling_strategy: str = "top_n",
    ):
        """
        ClusterFewshot: Optimized few-shot selection using clustering over semantic embeddings
            and example-as-one-shot evaluation.

        Args:
            metric: Callable
                Function to evaluate the model's predictions.
            metric_threshold: Optional[float]
                Threshold for metric-based filtering.
            num_fewshot: int
                Number of few-shot demonstrations (also number of clusters).
            embedding_model: str
                Pretrained model for generating semantic embeddings.
            descending: bool
                Whether to sort examples within each cluster
                    in descending order of impact as one-shot demonstrations.
            sampling_strategy: str
                How ClusterFewshot samples examples as potential demonstrations per-cluster.
        """
        super().__init__()
        self.metric = metric
        self.metric_threshold = metric_threshold

        self.N = num_fewshot
        self.descending = descending
        self.sampling_strategy = sampling_strategy

        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        self.valset = None
        self.student = None
        self.training_clusters = None
        self.validation_clusters = None
        self.trainset = None
        self.ead_set = None  # extracted from validation clusters for example-as-demo testing
        self.ranked_examples = None
        self.global_sorted_examples = None
        self._sum_of_clusters_mean_rank = None

        self.fewshot_subset = None  # Final selected demonstrations

    def compile(self, student: Program, trainset: List[Example], *, valset):
        """
        Compiles the ClusterFewshot optimizer.
        """
        self.student = student.deepcopy()
        self.trainset = trainset
        self.valset = valset

        logger.info("Compiling the student program using ClusteFewshot optimizer...")
        self.training_clusters = self._cluster_examples()
        self.validation_clusters = self._cluster_examples(train=False)
        self._sample_validation_clusters()
        self._sort_examples_as_demos()
        self.collect_fewshot_subset()

        # Update student LM predictors with optimized few-shot subset
        for _, predictor in self.student.named_predictors():
            predictor.demos = self.fewshot_subset  # TODO: try with/without in-context sort

        self.student._compiled = True

        logger.info("Student program compiled successfully.")

        return self.student

    def _cluster_examples(self, train=True):
        """
        Uses semantic embeddings to cluster the training/validation set into N groups.
        """
        logger.info("Generating examples embeddings for clustering...")
        data = self.trainset if train else self.valset
        texts = [ex.question for ex in data]  # TODO: Try also with ex.answer included (different distribution)

        # maps examples to 384 dimensional dense vector space
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        logger.info(f"Clustering into {self.N} clusters...")
        kmeans = KMeans(n_clusters=self.N, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        clusters = {i: [] for i in range(self.N)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.trainset[idx] if train else self.valset[idx])

        self._visualize_clusters(
            embeddings,
            cluster_labels,
            self.N,
            save_path=f"{'training' if train else 'validation'}_clusters.png"
        )

        logger.info(f"{'Training' if train else 'Validation'} examples clustering "
                    f"by semantic embedding completed (K={self.N}).")

        return clusters

    @staticmethod
    def _visualize_clusters(embeddings, cluster_labels, num_clusters, save_path="cluster_plot.png"):
        """
        Visualizes clustered embeddings in 2D using t-SNE and saves the plot to a file.
        """
        logger.info("Performing t-SNE dimensionality reduction for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot clusters
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster Labels")
        plt.title(f"t-SNE of Semantic Embeddings Clusters (K={num_clusters})")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Save instead of showing
        plt.savefig(save_path)
        plt.close()  # Close figure to free memory

        logger.info(f"Cluster visualization saved to {save_path}.")

    def _sample_validation_clusters(self):
        """
        Selects most central examples from each validation cluster to form a set for EAD testing.
        """
        self.ead_set = []
        samples_per_cluster = 3  # TODO: fixed number. consider a dynamic way

        for cluster_id, examples in self.validation_clusters.items():
            if len(examples) <= samples_per_cluster:
                selected = examples  # If fewer examples than needed, take all
            else:
                # Sort by proximity to cluster centroids (using embedding distances)
                embeddings = self.embedding_model.encode([ex.question for ex in examples], convert_to_numpy=True)
                cluster_center = np.mean(embeddings, axis=0)
                distances = np.linalg.norm(embeddings - cluster_center, axis=1)
                selected_indices = np.argsort(distances)[:samples_per_cluster]
                selected = [examples[i] for i in selected_indices]
                # TODO: thought - maybe we should sample "hardest" questions from "selected" by max probability confidence?

            self.ead_set.extend(selected)

        logger.info(f"Validation set assembled with {len(self.ead_set)} examples.")

    def _sort_examples_as_demos(self):
        """
        Sorts examples globally based on their impact when used as one-shot examples.
        """
        evaluator = Evaluate(
            devset=self.ead_set,
            metric=self.metric,
            num_threads=12,
        )

        logger.info(f"Sorting examples-as-demos from trainset ({len(self.trainset)} examples)")
        self.ranked_examples = {ex: self._evaluate_example_as_demo(ex, evaluator) for ex in self.trainset}
        self.global_sorted_examples = sorted(
            self.ranked_examples.keys(),
            key=lambda ex: self.ranked_examples[ex],
            reverse=self.descending,
        )

        logger.info(f"Re-ordering clusters by demonstrations ranks")
        self.training_clusters = {
            cluster_id: sorted(
                [ex for ex in cluster_examples if ex in self.ranked_examples],
                key=lambda ex: self.ranked_examples[ex],  # TODO: verify that ex is a valid key
                reverse=self.descending,
            )
            for cluster_id, cluster_examples in self.training_clusters.items()
        }

        logger.info(f"Demonstrations are sorted in {'descending' if self.descending else 'ascending'} order.")

    def _evaluate_example_as_demo(self, example, evaluator):
        """
        Evaluates an example by measuring how well it helps predict the validation set
        when used as a sole demonstration.
        """
        student_copy = self.student.deepcopy()

        # If current example was answered correctly, collect reasoning from LM
        prediction = student_copy(**example.inputs())

        if self.metric:
            metric_val = self.metric(example, prediction, trace=None)
        if self.metric_threshold:
            success = metric_val >= self.metric_threshold
        else:
            success = metric_val

        if success:
            example.reasoning = prediction.reasoning

        logger.info(f"ֿ\n\nConducting example-as-demo test ({len(self.ead_set)} questions) "
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
        logger.info(f"Collecting few-shot subset from clusters using {self.sampling_strategy} method")

        fewshot_subset = []

        for cluster_id, _ in self.training_clusters.items():
            fewshot_subset.extend(
                self.sample_examples_from_cluster(
                    cluster_id=cluster_id,
                    sampling_strategy=self.sampling_strategy  # TODO: run with each of 3 approaches to evaluate
                )
            )

        self.fewshot_subset = fewshot_subset

    def sample_examples_from_cluster(self, cluster_id, sampling_strategy="cluster_strength"):
        """
        Samples examples from a given cluster based on one of 3 approaches:
        1. top N: Collects examples that fall in the top global N potential demos rank.
            If there aren't any in the given cluster, returns an empty list.
        2. Best in cluster: Returns the top-ranked example-as-demo from the given cluster.
        3. Cluster strength: Allocates a proportionate number of slots in few-shot to the given cluster,
            based on cluster strength (i.e., mean example ranking).
        4. Central: samples most centric examples (i.e most common within a question semantic trend)
        """
        sampled_examples = []

        if cluster_id in self.training_clusters:
            cluster_examples = self.training_clusters[cluster_id]
            if not cluster_examples:
                return sampled_examples

            if sampling_strategy == "top_n":
                top_global_n = self.global_sorted_examples[:self.N]
                sampled_examples.extend([ex for ex in cluster_examples if ex in top_global_n])

            elif sampling_strategy == "best_in_cluster":
                sampled_examples.append(cluster_examples[0])

            elif sampling_strategy == "cluster_strength":
                # Compute sum of clusters mean rank **once**
                if not self._sum_of_clusters_mean_rank:
                    self._sum_of_clusters_mean_rank = 0
                    for _, cluster in self.training_clusters.items():
                        self._sum_of_clusters_mean_rank += np.mean(
                            [self.ranked_examples[ex] for ex in cluster]) if cluster else 0

                # Calculate cluster strength as the mean rank of its examples
                cluster_ranks = [self.ranked_examples[ex] for ex in cluster_examples]
                cluster_strength = np.mean(cluster_ranks)  # TODO: examples with top global ranks might be considered as outliers (ignored)

                # Compute how many slots to allocate for this cluster
                proportion = cluster_strength / self._sum_of_clusters_mean_rank  # TODO: examine edge use-cases for this proportion
                num_slots = int(self.N * proportion)

                # Select top-ranked examples from the cluster based on this proportion
                if num_slots >= 1:
                    sampled_examples = cluster_examples[:num_slots]

            elif sampling_strategy == "central":
                # Sort by proximity to cluster centroids (using embedding distances)
                embeddings = self.embedding_model.encode([ex.question for ex in cluster_examples], convert_to_numpy=True)
                cluster_center = np.mean(embeddings, axis=0)
                distances = np.linalg.norm(embeddings - cluster_center, axis=1)
                selected_indices = np.argsort(distances)[:3]
                sampled_examples = [cluster_examples[i] for i in selected_indices]

        return sampled_examples

