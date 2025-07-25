import json
import torch
import random
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datasets.fingerprint import Hasher
from dspy.primitives import Program, Example
from sklearn.metrics import silhouette_score
from dspy.utils.parallelizer import ParallelExecutor
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from programs import (
    BasicMH,
    CoT
)
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)

MIN_CLUSTERS: int = 3
MAX_CLUSTERS: int = 3

TASK_2_SAMPLINGS = {
    "arithmetic": ["top_n", "best_in_cluster"],
    "multihop": ["top_n", "best_in_cluster"],
    "classification": ["best_in_cluster"],
}

CANDIDATE_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
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
            soft_select: bool = False,
            use_target_model_embeddings: bool = False
    ):
        """
        ClusterFewshotv2: Task-adaptive few-shot selection using clustering over examples embeddings
            and example-as-one-shot evaluation.

        Args:
            task_type: str
                Label of the given task type which the candidate few-shot demonstration sets
                will be designed accordingly.
            metric: Callable
                Function to evaluate the model's predictions.
            metric_threshold: Optional[float]
                Threshold for metric-based filtering.
            descending: bool
                Whether to sort examples per-cluster/globally
                in descending order of impact as one-shot demonstrations.
            soft_select: bool
                Whether to sample the final few-shot based on differentiable "soft" selection process
                that learns a few-shot subset by balancing strong one-shot examples with semantic diversity,
                using an SGD-trained probability distribution over candidates.
            use_target_model_embeddings: bool
                Whether to use the target model layers embedding or
                a candidate SentenceTransformer's semantic embeddings.
                This option is recommended when running this teleprompter in the
                BetterTogether optimization pipeline.
        """
        super().__init__()
        self.metric = metric
        self.metric_threshold = metric_threshold

        # Choose sampling strategy based on the given task
        if task_type not in TASK_2_SAMPLINGS:
            raise ValueError(
                f"'{task_type}' task is not supported in ClusterFewshotv2. Currently supported tasks:\n{list(TASK_2_SAMPLINGS.keys())}")

        self.task_type = task_type
        self._soft_select = soft_select

        self.tokenizer = None
        self.embedding_model = None
        self.embedding_model_name = None
        self.examples2embeddings = {}  # Used for caching example embeddings (reduce computational overhead)
        self.embeddings2examples = {}
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
        self.os_test = None  # extracted from validation clusters for example-as-demo testing
        self.ranked_examples = None
        self.global_sorted_examples = None
        self._sum_of_clusters_strength = None
        self.trainset_by_hash = {}

        self.candidate_fewshot_subsets = None
        self.final_fewshot_subset = None

    def compile(self, student: Program, trainset: List[Example], *, valset):
        """
        Compiles the ClusterFewshotv2 optimizer.
        """
        self.student = student.deepcopy()
        self.trainset = self.bootstrap_examples(trainset)
        self.valset = valset

        # This code assumes the base model LM is shared across all predictors
        if self.use_target_model_embeddings:
            self.embedding_model_name = self.student.named_predictors()[0][1].lm.model
            self.generate_embeddings_func = self.generate_embedding_clusters_with_target_model
        else:
            self.generate_embeddings_func = self.generate_embedding_clusters_with_candidate_models

        logger.info("Compiling the student program using ClusteFewshotv2 optimizer...")
        self.training_clusters = self._cluster_examples()
        self.validation_clusters = self._cluster_examples(train=False)

        # # Log clusters for debug
        # if self.task_type != "classification":
        #     logger.info("--- TRAINING CLUSTERS ---\n")
        #     for cid, examples in self.training_clusters.items():
        #         examples_str = "\n".join([f"{ex.question} --> {ex.answer}" for ex in examples[:3]])
        #         logger.info(f"Cluster {cid + 1}:\n{examples_str}")

        #     logger.info("--- VALIDATION CLUSTERS ---\n")
        #     for cid, examples in self.validation_clusters.items():
        #         examples_str = "\n".join([f"{ex.question} --> {ex.answer}" for ex in examples[:3]])
        #         logger.info(f"Cluster {cid + 1}:\n{examples_str}")

        # exit(0)

        self._sample_one_shot_evaluation_set()
        self._sort_examples_as_demos()

        if self._soft_select:
            self.soft_select(N=self.N)
        else:
            self.collect_fewshot_subsets()
            self.pick_best_fewshot_subset()

        # Update student LM predictors with optimized few-shot subset
        for name, predictor in self.student.named_predictors():
            predictor.demos = [
                ex for demo in self.final_fewshot_subset for ex in demo[name]
            ]

        self.student._compiled = True

        logger.info("Student program compiled successfully.")

        return self.student

    def _cluster_examples(self, train=True):
        """
        Clustering the given examples into semantic groups.
        It performs semantic embeddings & K search to find clusters that maximizes the silhouette metric.
        """
        data = [ex['raw'] for ex in self.trainset] if train else self.valset
        data_type = 'training' if train else 'validation'
        examples_embeddings = None

        # TODO: think of a better way to generalize that (only supports Iris for now)
        if self.task_type == "classification":
            examples_embeddings = np.array(
                [
                    [input_val for _, input_val in dict(example.inputs()).items()]
                    for example in data
                ]
            )

            self.embedding_model_name = "N/A"  # No model was used to create embeddings

            kinds = sorted(set(ex.answer for ex in data))
            k = len(kinds)
            cluster_labels = [kinds.index(example.answer) for example in data]
        else:
            logger.info(f"Generating {len(data)} {data_type} examples embeddings")
            examples_embeddings, cluster_labels, k = self.generate_embeddings_func(examples=data)

        self.examples2embeddings.update({
            self.get_example_hash(example): np.array(example_embeddings)
            for example, example_embeddings
            in zip(self.trainset if train else data, examples_embeddings)
            # TODO: assuming self.trainset and data are ordered the same.
        })
        self.embeddings2examples.update({
            str(example_embeddings): example
            for example_embeddings, example
            in zip(examples_embeddings, self.trainset if train else data)
        })

        if train:
            self.N = k  # Used as hyperparameter for few-shot sampling

        clusters = {i: [] for i in range(k)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.trainset[idx] if train else self.valset[idx])

        self._visualize_examples(
            embeddings=examples_embeddings,
            embedding_model=self.embedding_model_name,
            cluster_labels=cluster_labels,
            num_clusters=k,
            data_type=data_type,
            save_path=f"{data_type}_clusters.png",
            silhouette=silhouette_score(examples_embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        )

        logger.info(f"{data_type} clustering completed with K={k}.")

        return clusters

    def _visualize_examples(
            self,
            embeddings,
            embedding_model,
            cluster_labels=None,
            num_clusters=None,
            data_type=None,
            save_path=None,
            silhouette=None,
            show_ranks=False,
    ):
        """
        Visualizes clustered embeddings / ranked examples in 2D using PCA.
        """
        logger.info("Performing PCA dimensionality reduction for visualization...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        if show_ranks:
            examples = [self.embeddings2examples[str(emb)] for emb in embeddings]
            scores = [self.ranked_examples.get(self.get_example_hash(ex), 0) for ex in examples]
            np_scores = np.array(scores)

            color_values = np_scores
            color_label = "One-shot Score"

            # Choose a colormap that works well with continuous values
            cmap = "coolwarm" if len(set(np_scores)) <= 5 else "viridis"
        else:
            color_values = np.array(cluster_labels)
            color_label = "Cluster Labels"
            cmap = "tab10"

        # Ensure color_values is valid
        if color_values is None or len(color_values) == 0:
            raise ValueError("Color values are empty, cannot plot scatter with cmap.")

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=color_values,
            cmap=cmap,
            alpha=0.8,
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label(color_label)

        if show_ranks:
            logger.info(f"Unique one-shot scores: {set(scores)}")
            plt.title(f"PCA of {data_type.title()} One-shot Ranks\n"
                      f"Rank mean={np.mean(np_scores):.2f}\n"
                      f"Rank std={np.std(np_scores):.2f}")
        else:
            plt.title(f"PCA of {data_type.title()} Embeddings Clusters\n"
                      f"K={num_clusters}\n"
                      f"Size={len(embeddings)}\n"
                      f"Silhouette={silhouette:.3f}\n"
                      f"Embedding Model={embedding_model}\n"
                      f"Dataset={'GSM8K' if isinstance(self.student, CoT) else 'HotPotQA' if isinstance(self.student, BasicMH) else 'Iris'}")

        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

        logger.info(f"Cluster visualization saved to {save_path}.")

    def visualize_one_shot_scores_distribution(self, save_path="one_shot_scores_distribution.png"):
        """
        Visualizes the distribution of one-shot evaluation scores (one-shot ranks).
        Each score reflects how well an example performed when used as a one-shot demonstration.
        """
        from collections import Counter

        if not self.ranked_examples:
            logger.warning("No ranked examples found. Skipping one-shot score visualization.")
            return

        score_counts = Counter(self.ranked_examples.values())

        sorted_scores = sorted(score_counts.items())
        scores, counts = zip(*sorted_scores)

        plt.figure(figsize=(8, 5))
        plt.bar(scores, counts, color='skyblue', edgecolor='black')
        plt.xlabel("One-shot Evaluation Score")
        plt.ylabel("Score frequency")
        plt.title("Distribution of One-shot Scores")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"One-shot scores distribution saved to {save_path}")

    def visualize_reasoning_distribution(self):
        # Extract ranks based on whether the example has reasoning
        ranks_with_reasoning = [self.ranked_examples[ex] for ex in self.trainset if hasattr(ex, "reasoning")]
        ranks_without_reasoning = [self.ranked_examples[ex] for ex in self.trainset if not hasattr(ex, "reasoning")]

        # Plot the distributions
        plt.figure(figsize=(10, 6))
        if ranks_with_reasoning:
            plt.hist(ranks_with_reasoning, bins=len(set(ranks_with_reasoning)), alpha=0.7, label="With reasoning",
                     edgecolor='black')
        plt.hist(ranks_without_reasoning, bins=len(set(ranks_without_reasoning)), alpha=0.5, label="Without reasoning",
                 edgecolor='black')

        plt.title("Distribution of One-shot Ranks With vs Without Reasoning")
        plt.xlabel("One-shot Evaluation Score (Rank)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reasoning_ranks_distribution.png")

    def _sample_one_shot_evaluation_set(self):
        """
        Selects examples from each validation cluster to form the one-shot testing set.
        """
        self.os_test = []
        samples_per_cluster = 3

        for cluster_id, examples in self.validation_clusters.items():
            sample_size = min(samples_per_cluster, len(examples))
            selected = self.get_central_examples(examples=examples, sample_size=sample_size)

            logger.info(
                f"Sampling {sample_size} questions from cluster {cluster_id + 1} (size={len(examples)})")

            self.os_test.extend(selected)

        self.visualize_os_test()
        logger.info(f"One-shot evaluation set assembled with {len(self.os_test)} questions.")

    def _sort_examples_as_demos(self):
        """
        Sorts examples based on their impact when used as one-shot demonstrations.
        """
        evaluator = Evaluate(
            devset=self.os_test,
            metric=self.metric,
            num_threads=min(12, len(self.os_test)),
            display_progress=True,
        )
        student_copy = self.student.deepcopy()

        logger.info(f"Sorting examples-as-demos from training set ({len(self.trainset)} examples)")
        self.ranked_examples = {}
        trainset_size = len(self.trainset)

        for idx, ex in enumerate(self.trainset):
            logger.info(f"\n\nEvaluating example {idx + 1}/{trainset_size}")
            self.ranked_examples[self.get_example_hash(ex)] = self._evaluate_example_as_demo(ex, evaluator,
                                                                                             student_copy)

        logger.info(f"Ordering {len(self.ranked_examples)} demonstrations "
                    f"by {len(set(self.ranked_examples.values()))} different ranks...")

        self.global_sorted_examples = [
            self.trainset_by_hash[ex_hash]
            for ex_hash in sorted(
                self.ranked_examples,
                key=lambda h: self.ranked_examples[h],
                reverse=self.descending,
            )
        ]

        self.training_clusters = {
            cluster_id: sorted(
                cluster_examples,
                key=lambda ex: self.ranked_examples[self.get_example_hash(ex)],
                reverse=self.descending,
            )
            for cluster_id, cluster_examples in self.training_clusters.items()
        }

        self.visualize_one_shot_scores_distribution()
        self._visualize_examples(
            embeddings=[self.examples2embeddings[self.get_example_hash(ex)] for ex in self.trainset],
            embedding_model=self.embedding_model_name,
            save_path=f"embeddings_to_one_shot_ranks.png",
            data_type="training",
            show_ranks=True,
        )

        logger.info(f"Demonstrations are sorted in {'descending' if self.descending else 'ascending'} order.")

    def _evaluate_example_as_demo(self, example, evaluator, student):
        """
        Evaluates an example by measuring how well it demonstrates the student
        to predict correctly when used as a sole demonstration.
        """
        example_visual = f"{', '.join([f'{input_key}: {input_val}' for input_key, input_val in dict(example['raw'].inputs()).items()])} --> {example['raw'].answer}"

        logger.info(
            f"Conducting example-as-demo test ({len(self.os_test)} questions) "
            f"using the following demonstration:\n"
            f"{example_visual}"
        )

        cached_demos = [pred.demos for _, pred in student.named_predictors()]

        for name, predictor in student.named_predictors():
            predictor.demos = example[name]  # Test as one-shot demonstration

        student_score = evaluator(program=student)

        for (_, predictor), demos in zip(student.named_predictors(), cached_demos):
            predictor.demos = demos

        return student_score

    def collect_fewshot_subsets(self):
        """
        Collects the potential few-shot candidate subsets using task-adaptive sampling strategies.
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
        1. top N: Collects examples that fall in the top global N demonstrations set.
            If there aren't any in the given cluster, returns an empty list.
        2. Best in cluster: Returns the top-ranked example-as-demo from the given cluster.
        3. Popularity: Allocates a proportionate number of slots in the few-shot subset
            for the given cluster top-ranked examples, based on cluster size (i.e. semantic popularity)
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

            elif sampling_strategy == "popularity":
                total_examples = len(self.trainset)
                proportion = len(cluster_examples) / total_examples
                sample_size = min(len(cluster_examples), round(proportion * self.N))
                sampled_examples = cluster_examples[:sample_size]

            elif sampling_strategy == "central":
                sampled_examples = self.get_central_examples(examples=cluster_examples, sample_size=1)

        logger.info(
            f"{len(sampled_examples)}/{self.N} slots given to cluster {cluster_id + 1} (size={len(self.training_clusters[cluster_id])})"
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
            num_threads=9,
            display_progress=True,
        )
        student = self.student.deepcopy()

        for sampling_strategy in sampling_strategies:
            logger.info(f"\n\nTesting '{sampling_strategy}' sampled few-shot subset")
            fewshot_subset = self.candidate_fewshot_subsets[sampling_strategy]

            for name, predictor in student.named_predictors():
                predictor.demos = [
                    ex for demo in fewshot_subset for ex in demo[name]
                ]

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
        embeddings = [self.examples2embeddings[self.get_example_hash(ex)] for ex in examples]
        cluster_center = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - cluster_center, axis=1)
        selected_indices = np.argsort(distances)[:sample_size]
        sampled_examples = [examples[i] for i in selected_indices]

        return sampled_examples

    def ask(self, student, question):
        prediction = student(**question.inputs())

        if self.metric:
            metric_val = self.metric(question, prediction, trace=None)
        if self.metric_threshold:
            success = metric_val >= self.metric_threshold
        else:
            success = metric_val

        return prediction, success

    def generate_embedding_clusters_with_target_model(
            self,
            examples: List[Example],
            max_seq_length: int = 1024,
            device: str = "cpu",
    ) -> Tuple[np.ndarray, List[int], int]:
        """
        1. Generates a mean-pooled input embedding of the give examples,
            aligning with the model's chat-based fine-tuning input format.
        2. Searches for such K that optimizes the Silhouette score of
            the embedding clusters using K-means
        """

        # ----- NOTE -----
        # This operation requires additional compute resources
        # to encode examples using the LM model input embeddings layer.
        model_name = self.embedding_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
            self.embedding_model.resize_token_embeddings(len(self.tokenizer))

        embeddings = []
        examples_size = len(examples)

        for i in range(examples_size):
            # Generate chat-formatted string
            chat_str = self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": examples[i].question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            encoding = self.tokenizer(
                chat_str,
                return_tensors="pt",
                padding="max_length",  # keep consistent sequence length
                truncation=True,
                max_length=max_seq_length,
            )

            input_ids = encoding["input_ids"].to(self.embedding_model.device)
            attention_mask = encoding["attention_mask"].to(self.embedding_model.device)

            with torch.no_grad():
                # Get token embeddings from input layer
                token_embs = self.embedding_model.get_input_embeddings()(input_ids)

                attention_expanded = attention_mask.unsqueeze(-1)
                token_embs = token_embs * attention_expanded  # zero out pads

                sum_embs = token_embs.sum(dim=1)
                lengths = attention_expanded.sum(dim=1).clamp(min=1)
                mean_emb = sum_embs / lengths

            embeddings.append(mean_emb.squeeze(0).cpu().numpy())

        best_score = -np.inf
        best_k = None
        best_labels = None
        embeddings = np.array(embeddings)

        logger.info(f"Searching best K for {self.embedding_model_name} model embeddings")

        for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            logger.info(f"K={k}, Silhouette Score={score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        logger.info(f"Selected K={best_k} with silhouette score={best_score:.3f}")

        return np.array(embeddings), best_labels, best_k

    def generate_embedding_clusters_with_candidate_models(self, examples: List[Example], device: str = 'cpu'):
        best_k = None
        best_score = -np.inf
        best_labels = None
        best_embeddings = None
        best_embedding_model = None
        best_embedding_model_name = None

        examples_texts = [ex.question for ex in examples]

        if not self.embedding_model:
            for model_id in CANDIDATE_EMBEDDING_MODELS:
                logger.info(f"Encoding examples with model: {model_id}")

                embedding_model = SentenceTransformer(model_id, device=device)
                embeddings = embedding_model.encode(examples_texts, convert_to_numpy=True)

                for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    logger.info(f"K={k}, Silhouette Score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        best_embedding_model_name = model_id
                        best_embedding_model = embedding_model
                        best_embeddings = embeddings

            self.embedding_model = best_embedding_model
            self.embedding_model_name = best_embedding_model_name
        else:
            embeddings = self.embedding_model.encode(examples_texts, convert_to_numpy=True)

            for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
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

    def soft_select(
        self,
        N: int,
        steps: int = 1000,
        log_step: int = 100,
        lr: float = 1e-1,
        device: str = "cpu",
        verbose: bool = True,
        min_lambda: float = 10,
        max_lambda: float = np.inf,
    ):
        """
        Differentiable “soft” selection of a final N-shot subset.
        Balances one-shot impact (One-shot ranks) against semantic redundancy.
        """
        import math

        trainset = self.trainset
        M = len(trainset)

        one_shot_scores = torch.tensor(
            [score for _, score in self.ranked_examples.items()],
            dtype=torch.float32,
            device=device
        )  # shape (M,)

        # Build embedding matrix
        embs = torch.stack([
            torch.tensor(self.examples2embeddings[self.get_example_hash(ex)], device=device, dtype=torch.float32)
            for ex, _ in self.ranked_examples.items()
        ]).to(device)  # shape (M, D)

        # Build cosine-similarity matrix
        embs = embs / (embs.norm(dim=1, keepdim=True).clamp(min=1e-8))
        S = embs @ embs.t()  # shape (M, M)

        # Initialize learnable logits and diversity log‐lambda
        logits = torch.zeros(M, device=device, requires_grad=True)
        log_lambda = torch.zeros(1, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([logits, log_lambda], lr=lr)

        logger.info(f"Learning {N} potential few-shot candidates from {M} examples")

        for step in range(steps):
            p = F.softmax(logits, dim=0)
            lambda_ = log_lambda.exp()

            # Loss: negative impact + diversity penalty
            loss = - (p * one_shot_scores).sum() + lambda_ * (p @ S @ p)
            loss_val = loss.item()
            lambda_val = lambda_.item()

            if verbose and step % log_step == 0:
                logger.info(f"loss: {loss_val:.4f}, λ: {lambda_val:.4f}, step: {step}")

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                # clip λ to [min, max] for diversity control
                log_lambda.clamp_(math.log(min_lambda), math.log(max_lambda))
            optimizer.step()

        # Pick top‐N examples by final p probabilities
        soft_scores = F.softmax(logits, dim=0)
        topn = torch.topk(soft_scores, N).indices.tolist()
        candidate_examples = list(self.ranked_examples)
        self.final_fewshot_subset = [candidate_examples[i] for i in topn]

        self.visualize_soft_selection(div_lambda=log_lambda.exp().item())

    def visualize_soft_selection(self, div_lambda, save_path="soft_selection_pca.png"):
        """
        PCA plot of all training embeddings, highlighting selected few-shot examples.
        """
        all_examples = list(self.ranked_examples)
        embs = np.stack([
            self.examples2embeddings[self.get_example_hash(ex)]
            for ex in all_examples
        ])
        selected_set = set(self.final_fewshot_subset)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embs)

        colors = ['red' if ex in selected_set else 'gray' for ex in all_examples]

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, edgecolor='k')
        plt.title(f"PCA of Training Embeddings with Selected Few-shot (Red)\nDiversity λ={div_lambda:.3f}")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Soft selection PCA plot saved to {save_path}")

    def visualize_os_test(self, save_path="one_shot_test.png"):
        """
        PCA plot of all validation embeddings, highlighting selected few-shot examples.
        """
        all_examples = list(self.valset)
        embs = np.stack([
            self.examples2embeddings[self.get_example_hash(ex)]
            for ex in all_examples
        ])
        selected_set = set(self.os_test)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embs)

        colors = ['red' if ex in selected_set else 'gray' for ex in all_examples]

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, edgecolor='k')
        plt.title(f"PCA of Validation Embeddings with Selected One-shot test questions (Red)")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"One-shot test set PCA plot saved to {save_path}")

    def bootstrap_examples(self, examples):
        import dspy

        student = self.student
        predictor2name = {
            predictor: name for name, predictor in self.student.named_predictors()
        }

        logger.info(f"Bootstrapping {len(examples)} examples")

        def process_example(example):
            predictor_cache = {}
            name2traces = {}

            try:
                with dspy.settings.context(trace=[]):
                    with dspy.settings.context():
                        for name, predictor in student.named_predictors():
                            predictor_cache[name] = predictor.demos
                            predictor.demos = [x for x in predictor.demos if x != example]

                        prediction = student(**example.inputs())
                        trace = dspy.settings.trace

                        for name, predictor in student.named_predictors():
                            predictor.demos = predictor_cache[name]

                        if self.metric:
                            metric_val = self.metric(example, prediction, trace)
                            if self.metric_threshold:
                                success = metric_val >= self.metric_threshold
                            else:
                                success = metric_val
                        else:
                            success = True
            except Exception as e:
                # Handling as failed bootstrapping attempt (ignored example)
                return None

            if success:
                for step in trace:
                    predictor, inputs, outputs = step
                    demo = dspy.Example(augmented=True, **inputs, **outputs)
                    name2traces.setdefault(predictor2name[predictor], []).append(demo)

                for name, demos in name2traces.items():
                    if len(demos) > 1:
                        rng = random.Random(Hasher.hash(tuple(demos)))
                        if rng.random() < 0.5:
                            demos = [rng.choice(demos[:-1])]
                        else:
                            demos = [demos[-1]]
                    name2traces[name] = demos

                bootstrapped = {'raw': example}
                bootstrapped.update(name2traces)
                return bootstrapped

            return None  # Misleading bootstrapped example considered as non useful

        # Use the same settings as Evaluate
        executor = ParallelExecutor(
            num_threads=min(12, len(examples)),
            disable_progress_bar=False,
            max_errors=0,
            provide_traceback=True,
            compare_results=False,
        )

        bootstrapped_results = executor.execute(process_example, examples)

        bootstrapped_examples = []
        for bootstrapped in bootstrapped_results:
            if bootstrapped:
                self.trainset_by_hash[self.get_example_hash(bootstrapped)] = bootstrapped
                bootstrapped_examples.append(bootstrapped)

        logger.info(f"{len(bootstrapped_examples)}/{len(examples)} remaining after bootstrapping")

        return bootstrapped_examples

    def _normalize_example(self, obj):
        if isinstance(obj, Example):
            # Convert to dict using both inputs and outputs
            return dict(obj)
        elif isinstance(obj, list):
            return [dict(obj_item) for obj_item in obj]
        elif isinstance(obj, dict):
            return {k: self._normalize_example(v) for k, v in obj.items()}
        else:
            return obj  # Return primitives or other serializables as-is

    def get_example_hash(self, example_obj):
        """
        Computes a stable string hash (via JSON dump) for an example object.
        Supports:
        - validation examples: `Example`
        - training examples: `{'raw': Example, name1: Example, ...}`
        """
        normalized = self._normalize_example(example_obj)
        return json.dumps(normalized, sort_keys=True)
