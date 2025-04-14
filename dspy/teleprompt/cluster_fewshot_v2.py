import torch
import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from dspy.primitives import Program, Example
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)

MIN_CLUSTERS: int = 3
MAX_CLUSTERS: int = 4

TASK_2_SAMPLINGS = {
    "arithmetic": ["top_n"],
    "multihop": ["top_n", "central", "cluster_strength"],  # TODO: Still need to figure out compatible sampling.
    "classification": ["best_in_cluster"],
    "general": ["central"],
}

CANDIDATE_EMBEDDING_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/gtr-t5-base"
]


class ClusterFewshotv2(Teleprompter):
    def __init__(
            self,
            task_type: str,
            metric=None,
            metric_threshold=None,
            descending: bool = True,
            use_target_model_embeddings: bool = False
    ):
        """
        ClusterFewshotv2: Task-adaptive few-shot selection using clustering over semantic embeddings
            and example-as-one-shot evaluation.

        Args:
            task_type: str
                Label of the given task type which the final few-shot demonstration set
                will be designed accordingly.
            metric: Callable
                Function to evaluate the model's predictions.
            metric_threshold: Optional[float]
                Threshold for metric-based filtering.
            descending: bool
                Whether to sort examples per-cluster/globally
                    in descending order of impact as one-shot demonstrations.
            use_target_model_embeddings: bool
                Whether to use the target model's input embeddings layer or
                a candidate SentenceTransformer's semantic embeddings.
                This option is relevant for running this teleprompter in the
                BetterTogether optimization pipeline.
        """
        super().__init__()
        self.metric = metric
        self.metric_threshold = metric_threshold

        # Choose sampling strategy based on the given task
        if not task_type in TASK_2_SAMPLINGS:
            raise ValueError(f"{task_type} task is not supported in ClusterFewshotv2.")

        self.task_type = task_type

        self.tokenizer = None
        self.embedding_model = None
        self.embedding_model_name = None
        self.examples2embeddings = {}  # Used for caching example embeddings (reduce computational overhead)
        self.use_target_model_embeddings = use_target_model_embeddings
        self.generate_embeddings_func = None

        self.training_K = None
        self.validation_K = None
        self.descending = descending
        self.valset = None
        self.student = None
        self.training_clusters = None
        self.validation_clusters = None
        self.trainset = None
        self.ead_set = None  # extracted from validation clusters for example-as-demo testing
        self.ranked_examples = None
        self.global_sorted_examples = None
        self._sum_of_clusters_strength = None

        self.candidate_fewshot_subsets = None
        self.final_fewshot_subset = None

    def compile(self, student: Program, trainset: List[Example], *, valset):
        """
        Compiles the ClusterFewshotv2 optimizer.
        """
        self.student = student.deepcopy()
        self.trainset = trainset
        self.valset = valset

        # TODO: this should support init not only for transformers-based models.
        # TODO: this code only supports one predictor per compilation. should be re-visited?
        if self.use_target_model_embeddings:
            model_name = self.student.named_predictors()[0][1].lm.model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name).to("cpu")
            self.embedding_model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
            self.embedding_model_name = model_name
            self.generate_embeddings_func = self.generate_examples_embeddings_from_target_model

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
                self.embedding_model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.generate_embeddings_func = self.generate_example_embeddings_from_candidate_models

        logger.info("Compiling the student program using ClusteFewshotv2 optimizer...")
        self.training_clusters = self._cluster_examples()
        self.validation_clusters = self._cluster_examples(train=False)
        self._sample_validation_clusters()
        self._sort_examples_as_demos()
        self.collect_fewshot_subsets()
        self.pick_best_fewshot_subset()

        # Update student LM predictors with optimized few-shot subset
        for _, predictor in self.student.named_predictors():
            predictor.demos = self.final_fewshot_subset

        self.student._compiled = True

        logger.info("Student program compiled successfully.")

        return self.student

    def _cluster_examples(self, train=True):
        """
        Clustering the given examples into semantic groups.
        It performs semantic embeddings & K search to find clusters that maximizes the silhouette metric.
        """
        data = self.trainset if train else self.valset
        data_type = 'training' if train else 'validation'
        embeddings = []

        # TODO: think of a better way to generalize that (only supports Iris for now)
        if self.task_type == "classification":
            for example in data:
                example_embeddings = np.array(
                    [example.sepal_length, example.sepal_width, example.petal_length, example.petal_width])

                embeddings.append(example_embeddings)
                self.examples2embeddings.update({
                    example: example_embeddings,  # Use the embeddings as the example itself (for Iris)
                })

            self.embedding_model_name = "N/A"  # No model was used to create embeddings
            embeddings = np.array(embeddings)

            # setosa, versicolor or virginica
            iris_kinds = sorted(set(ex.answer for ex in data))
            k = len(iris_kinds)
            cluster_labels = [iris_kinds.index(example.answer) for example in data]
        else:
            logger.info(f"Generating {len(data)} {data_type} examples embeddings")
            examples_embeddings, cluster_labels, k = self.generate_embeddings_func(
                examples=data
            )
            self.examples2embeddings.update({
                example: example_embeddings
                for example, example_embeddings
                in zip(data, examples_embeddings)
            })

        if train:
            self.training_K = k
            self.N = k  # Used as hyperparameter for few-shot sampling
        else:
            self.validation_K = k

        clusters = {i: [] for i in range(k)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.trainset[idx] if train else self.valset[idx])

        self._visualize_clusters(
            embeddings,
            self.embedding_model_name,
            cluster_labels,
            k,
            data_type,
            save_path=f"{data_type}_clusters.png"
        )

        logger.info(f"{data_type} clustering completed with K={k}.")

        return clusters

    def _visualize_clusters(
            self, embeddings, embedding_model, cluster_labels, num_clusters, data_type, save_path
    ):
        """
        Visualizes clustered embeddings in 2D using t-SNE.
        """
        logger.info("Performing t-SNE dimensionality reduction for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(2, min(50, len(embeddings) // 3)))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot clusters
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster Labels")
        plt.title(f"t-SNE of {data_type} Semantic Embedding Clusters\n"
                  f"K={num_clusters}\n"
                  f"size={len(embeddings)}\n"
                  f"embedding model={embedding_model if embedding_model else 'not used'}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        plt.savefig(save_path)
        plt.close()

        logger.info(f"Cluster visualization saved to {save_path}.")

    def _sample_validation_clusters(self):
        """
        Selects most central examples (optionally hardest among them) from each validation cluster to form a set for EAD testing.
        """
        self.ead_set = []

        for cluster_id, examples in self.validation_clusters.items():
            sample_size = min(3, len(examples))

            logger.info(f"Sampling {sample_size} most central validation examples from cluster no. {cluster_id + 1}")
            selected = self.get_central_examples(examples=examples, sample_size=sample_size)
            self.ead_set.extend(selected)

            logger.info(f"Collected {len(selected)} questions from cluster {cluster_id + 1}")

        logger.info(f"EAD evaluation set assembled with {len(self.ead_set)} questions.")

    def _sort_examples_as_demos(self):
        """
        Sorts examples globally based on their impact when used as one-shot examples.
        """
        evaluator = Evaluate(
            devset=self.ead_set,
            metric=self.metric,
            num_threads=len(self.ead_set),
            display_progress=True,
        )
        student_copy = self.student.deepcopy()

        logger.info(f"Sorting examples-as-demos from trainset ({len(self.trainset)} examples)")
        self.ranked_examples = {}
        trainset_size = len(self.trainset)

        for idx, ex in enumerate(self.trainset):
            logger.info(f"\n\nEvaluating example {idx + 1}/{trainset_size}")
            self.ranked_examples[ex] = self._evaluate_example_as_demo(ex, evaluator, student_copy)

        self.global_sorted_examples = sorted(
            self.ranked_examples.keys(),
            key=lambda ex: self.ranked_examples[ex],
            reverse=self.descending,
        )

        logger.info(f"Ordering clusters by demonstrations ranks")
        self.training_clusters = {
            cluster_id: sorted(
                [ex for ex in cluster_examples if ex in self.ranked_examples],
                key=lambda ex: self.ranked_examples[ex],
                reverse=self.descending,
            )
            for cluster_id, cluster_examples in self.training_clusters.items()
        }

        logger.info(f"Demonstrations are sorted in {'descending' if self.descending else 'ascending'} order.")

    def _evaluate_example_as_demo(self, example, evaluator, student):
        """
        Evaluates an example by measuring how well it demonstrate the student
        to predict correctly when used as a sole demonstration.
        """
        # If current example was answered correctly by the student, collect reasoning from LM
        pred, success = self.answer(student, example)

        if success:
            example.reasoning = pred.reasoning

        if self.task_type == "classification":
            example_visual = (f"sepal_length: {example.sepal_length}, "
                              f"sepal_width: {example.sepal_width}, "
                              f"petal_length: {example.petal_length}, "
                              f"petal_width: {example.petal_width} "
                              f"--> {example.answer}")
        else:
            example_visual = f"{example.question} --> {example.answer}"

        logger.info(
            f"Conducting example-as-demo test ({len(self.ead_set)} questions) "
            f"using the following demonstration "
            f"(reasoning info {'is not' if not success else 'is'} included):\n"
            f"{example_visual}"
        )

        cached_demos = [pred.demos for _, pred in student.named_predictors()]

        for _, predictor in student.named_predictors():
            predictor.demos = [example]  # Test as one-shot demonstration

        student_score = evaluator(program=student)

        for (_, predictor), demos in zip(student.named_predictors(), cached_demos):
            predictor.demos = demos

        return student_score

    def collect_fewshot_subsets(self):
        """
        Collects the potential few-shot demonstrations using task-adaptive sampling strategies,
        while balancing exploration (diversity) and exploitation (strong demos).
        """
        sampling_strategies = TASK_2_SAMPLINGS[self.task_type]
        adaptive_fewshot_subsets = {
            sampling_strategy: []
            for
            sampling_strategy
            in sampling_strategies
        }

        for sampling_strategy in sampling_strategies:
            logger.info(
                f"Collecting candidate few-shot using '{sampling_strategy}' "
                f"sampling strategy ({self.task_type} task optimization)"
            )
            for cluster_id, _ in self.training_clusters.items():
                adaptive_fewshot_subsets[sampling_strategy].extend(
                    self.sample_examples_from_cluster(
                        cluster_id=cluster_id,
                        sampling_strategy=sampling_strategy
                    )
                )

        self.candidate_fewshot_subsets = adaptive_fewshot_subsets

    def sample_examples_from_cluster(self, cluster_id, sampling_strategy):
        """
        Samples examples from a given cluster based on one of 4 approaches:
        1. top N: Collects examples that fall in the top global N potential demos rank.
            If there aren't any in the given cluster, returns an empty list.
        2. Best in cluster: Returns the top-ranked example-as-demo from the given cluster.
        3. Cluster strength: Allocates a proportionate number of slots in the few-shot subset
            for the given cluster top-ranked examples, based on cluster's performance consistency.
        4. Central: Samples most centric example (i.e most common within a semantic demonstrations trend)
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
                if not self._sum_of_clusters_strength:
                    self._sum_of_clusters_strength = 0
                    for _, cluster in self.training_clusters.items():
                        if cluster:
                            cluster_ranks = [self.ranked_examples[ex] for ex in cluster][:self.N]
                            self._sum_of_clusters_strength += np.mean(cluster_ranks) - 0.3 * np.std(cluster_ranks)
                            # TODO: Should dynamically set std scaler param

                # Calculate cluster strength as the consistency
                # rate of performance Over a semantic group
                # TODO: need to re-visit implementation
                cluster_ranks = [self.ranked_examples[ex] for ex in cluster_examples][:self.N]
                cluster_strength = np.mean(cluster_ranks) - 0.3 * np.std(cluster_ranks)

                # Compute how many slots to allocate for this cluster
                proportion = cluster_strength / self._sum_of_clusters_strength
                num_slots = round(self.N * proportion)

                # Select top-ranked examples from the cluster based on this proportion
                if num_slots >= 1:
                    sampled_examples = cluster_examples[:num_slots]

            elif sampling_strategy == "central":
                sampled_examples = self.get_central_examples(examples=cluster_examples, sample_size=1)

        logger.info(
            f"{len(sampled_examples)}/{self.N} slots given to cluster {cluster_id + 1} "
            f"in the few-shot subset according to {sampling_strategy} sampling strategy."
        )

        return sampled_examples

    def pick_best_fewshot_subset(self):
        sampling_strategies = TASK_2_SAMPLINGS[self.task_type]

        if len(sampling_strategies) == 1:
            sampling_strategy = sampling_strategies[0]
            self.final_fewshot_subset = self.candidate_fewshot_subsets[sampling_strategy]
            return

        ranked_sampling_strategies = {}
        evaluator = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=12,
            display_progress=True,
        )
        student = self.student.deepcopy()

        for sampling_strategy in sampling_strategies:
            logger.info(f"\n\nTesting '{sampling_strategy}' sampled few-shot subset")
            fewshot_subset = self.candidate_fewshot_subsets[sampling_strategy]

            for _, predictor in student.named_predictors():
                predictor.demos = fewshot_subset

            fewshot_subset_score = evaluator(student)
            ranked_sampling_strategies[sampling_strategy] = fewshot_subset_score
            logger.info(f"'{sampling_strategy}' few-shot subset scored {fewshot_subset_score:.2f}% "
                        f"on the validation set with {len(fewshot_subset)} demonstrations.")

        best_strategy = max(ranked_sampling_strategies, key=ranked_sampling_strategies.get)
        self.final_fewshot_subset = self.candidate_fewshot_subsets[best_strategy]

        logger.info(
            f"Best few-shot subset sampled according to '{best_strategy}' strategy "
            f"({ranked_sampling_strategies[best_strategy]}% accuracy on the validation set)")

    def get_central_examples(self, examples, sample_size):
        embeddings = [self.examples2embeddings[ex] for ex in examples]
        cluster_center = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - cluster_center, axis=1)
        selected_indices = np.argsort(distances)[:sample_size]
        sampled_examples = [examples[i] for i in selected_indices]

        return sampled_examples

    def answer(self, student, example):
        prediction = student(**example.inputs())

        if self.metric:
            metric_val = self.metric(example, prediction, trace=None)
        if self.metric_threshold:
            success = metric_val >= self.metric_threshold
        else:
            success = metric_val

        return prediction, success

    def generate_examples_embeddings_from_target_model(
            self,
            examples: List[Example],
            max_seq_length: int = 1024,
    ) -> Tuple[np.ndarray, List[int], int]:
        """
        Generates a mean-pooled input embedding for a DSPy `Example`,
        aligning with the model's chat-based fine-tuning input format.
        """
        embeddings = []

        logger.info(f"Encoding examples with model: {self.embedding_model_name}")

        for example in examples:
            # TODO: this should be generic for all I/O type fields (i.e. now only QA)
            conversation_format = [
                {"role": "user", "content": example.question},
                {"role": "assistant", "content": example.answer},
            ]

            chat_str = self.tokenizer.apply_chat_template(
                conversation=conversation_format,
                tokenize=False,
                add_generation_prompt=False,
            )

            encoding = self.tokenizer(
                chat_str,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
            )

            input_ids = encoding["input_ids"].to(self.embedding_model.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            input_ids = input_ids.to(self.embedding_model.device)

            # TODO: re-visit the logic of this to understand the embeddings better
            with torch.no_grad():
                token_embs = self.embedding_model.get_input_embeddings()(input_ids)  # (1, seq_len, hidden_dim)
                mean_emb = token_embs.mean(dim=1).squeeze(0)  # (hidden_dim,)

            embeddings.append(mean_emb.cpu().numpy())

        best_score = -np.inf
        best_k = None
        best_labels = None
        embeddings = np.array(embeddings)

        logger.info(f"Searching best K for {self.embedding_model_name} model embeddings")

        for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=400, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            logger.info(f"K={k}, Silhouette Score={score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        logger.info(f"Selected K={best_k} with silhouette score={best_score:.3f}")

        k = best_k
        cluster_labels = best_labels

        return np.array(embeddings), cluster_labels, k

    def generate_example_embeddings_from_candidate_models(self, examples):
        best_k = None
        best_score = None
        best_labels = None
        best_embeddings = None
        best_embedding_model = None
        best_embedding_model_name = None

        if not self.embedding_model:
            for model_id in CANDIDATE_EMBEDDING_MODELS:
                logger.info(f"Encoding examples with model: {model_id}")

                embedding_model = SentenceTransformer(model_id, device='cpu')
                embeddings = embedding_model.encode(examples, convert_to_numpy=True)

                for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
                    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=500, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    logger.info(f"K={k}, Silhouette Score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_embedding_model_name = model_id
                        best_embedding_model = embedding_model
                        best_labels = labels
                        best_embeddings = embeddings

            self.embedding_model = best_embedding_model
            self.embedding_model_name = best_embedding_model_name
        else:
            embeddings = self.embedding_model.encode(examples, convert_to_numpy=True)

            for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=400, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                score = silhouette_score(embeddings, labels)

                logger.info(f"K={k}, Silhouette Score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            best_embeddings = embeddings

        logger.info(f"Selected model: {self.embedding_model_name} with K={best_k} (silhouette={best_score:.3f})")

        return best_embeddings, best_labels, best_k
