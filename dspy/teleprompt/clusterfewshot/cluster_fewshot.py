import logging
from typing import List, Callable, Optional

from dspy.primitives import Program, Example
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.evaluate import Evaluate
from .semantic_encoder import SemanticEncoder
from .clusterfewshot_utils import (
    cluster_examples,
    sample_one_shot_evaluation_set,
    sort_examples_as_demos,
    sample_examples_from_cluster,
    soft_select_examples,
    bootstrap_examples,
    get_example_hash,
    visualize_os_test,
    generate_embedding_clusters_with_semantic_encoders,
)

logger = logging.getLogger(__name__)

TASK_2_SAMPLINGS = {
    "arithmetic": ["top_n", "best_in_cluster"],
    "multihop": ["top_n", "best_in_cluster"],
    "classification": ["best_in_cluster"],
}


class ClusterFewshot(Teleprompter):
    def __init__(
            self,
            task_type: str,
            metric: Optional[Callable] = None,
            metric_threshold: Optional[float] = None,
            soft_select: bool = False,
            apply_visuals: bool = True,
            semantic_encoders: Optional[List[SemanticEncoder]] = None
    ):
        """
        ClusterFewshot: Task-adaptive few-shot selection with Bring-Your-Own-Encoder support.

        This teleprompter optimizes few-shot demonstrations by:
        1. Clustering training examples based on semantic embeddings from custom encoders
        2. Evaluating each example's effectiveness as a one-shot demonstration
        3. Selecting the best few-shot subset using task-specific sampling strategies

        The Bring-Your-Own-Encoder (BYOE) design allows you to provide custom semantic
        encoders tailored to your task. ClusterFewshot will evaluate all encoders via
        grid search and select the one that produces the best clustering (highest silhouette score).

        Args:
            task_type: str
                Type of task being optimized. Determines the sampling strategy.
                Supported tasks: 'arithmetic', 'multihop', 'classification'
            metric: Optional[Callable]
                Evaluation function to measure prediction quality.
            metric_threshold: Optional[float]
                Minimum metric value for filtering examples during bootstrapping.
            soft_select: bool
                If True, uses differentiable soft selection to balance one-shot impact
                with semantic diversity via gradient descent optimization.
                Default: False
            apply_visuals: bool
                If True, generates matplotlib visualizations (PCA plots, score distributions)
                throughout the optimization process.
                Default: True
            semantic_encoders: Optional[List[SemanticEncoder]]
                List of SemanticEncoder instances to use for embedding examples.
                ClusterFewshot will evaluate each encoder and select the best one.
                If None, you must provide encoders - no defaults are assumed.

        Example:
            # Using SentenceTransformer encoders for QA tasks
            from dspy.teleprompt.clusterfewshot import (
                ClusterFewshot,
                create_sentence_transformer_encoder
            )

            encoders = [
                create_sentence_transformer_encoder("all-mpnet-base-v2"),
                create_sentence_transformer_encoder("gtr-t5-base"),
            ]

            optimizer = ClusterFewshot(
                task_type="arithmetic",
                metric=gsm8k_metric,
                semantic_encoders=encoders
            )

            # Using numeric encoder for classification
            from dspy.teleprompt.clusterfewshot import create_numeric_encoder

            optimizer = ClusterFewshot(
                task_type="classification",
                metric=dspy.evaluate.answer_exact_match,
                semantic_encoders=[create_numeric_encoder()]
            )
        """
        super().__init__()
        self.metric = metric
        self.metric_threshold = metric_threshold

        # Validate task type
        if task_type not in TASK_2_SAMPLINGS:
            raise ValueError(
                f"'{task_type}' task is not supported in ClusterFewshot. Currently supported tasks:\n{list(TASK_2_SAMPLINGS.keys())}")

        self.task_type = task_type
        self._soft_select = soft_select
        self.apply_visuals = apply_visuals
        self.semantic_encoders = semantic_encoders

        if semantic_encoders is None:
            raise ValueError(
                "semantic_encoders parameter is required. Please provide a list of SemanticEncoder instances.\n"
                "Example: semantic_encoders=[create_sentence_transformer_encoder('all-mpnet-base-v2')]"
            )

        self.selected_encoder = None  # Best encoder selected after grid search
        self.examples2embeddings = {}  # Cache for example embeddings
        self.embeddings2examples = {}

        self.N = None  # Number of clusters (used as few-shot size hyperparameter)
        self.training_K = None
        self.validation_K = None
        self.valset = None
        self.student = None
        self.training_clusters = None
        self.validation_clusters = None
        self.trainset = None
        self.os_test = None  # Examples for one-shot evaluation
        self.ranked_examples = None  # Examples ranked by one-shot impact
        self.global_sorted_examples = None
        self._sum_of_clusters_strength = None
        self.trainset_by_hash = {}
        self.pca_2d = None

        self.candidate_fewshot_subsets = None
        self.final_fewshot_subset = None

    def compile(self, student: Program, trainset: List[Example], *, valset):
        """
        Compiles the ClusterFewshot optimizer to produce an optimized student program.

        This method executes the following optimization pipeline:
        1. Bootstraps training examples to retain traces from solvable examples
        2. Clusters training and validation examples using semantic embeddings
        3. Samples representative examples from validation clusters for one-shot evaluation
        4. Ranks training examples by their effectiveness as one-shot demonstrations
        5. Selects the optimal few-shot subset using task-specific sampling strategies
        6. Updates the student program with the selected demonstrations

        Args:
            student: Program
                The student program to optimize
            trainset: List[Example]
                Training examples for bootstrapping and clustering
            valset: List[Example]
                Validation examples for evaluation and clustering

        Returns:
            Program: The optimized student program with selected few-shot demonstrations
        """
        self.student = student.deepcopy()

        logger.info("Compiling the student program using ClusterFewshot optimizer...")
        self.trainset = bootstrap_examples(
            examples=trainset,
            student=self.student,
            metric=self.metric,
            metric_threshold=self.metric_threshold,
            trainset_by_hash=self.trainset_by_hash
        )
        self.valset = valset

        self.training_clusters = self._cluster_examples(train=True)
        self.validation_clusters = self._cluster_examples(train=False)

        self.os_test = sample_one_shot_evaluation_set(
            validation_clusters=self.validation_clusters,
            examples2embeddings=self.examples2embeddings
        )

        if self.apply_visuals:
            visualize_os_test(
                valset=self.valset,
                os_test=self.os_test,
                examples2embeddings=self.examples2embeddings
            )

        self.ranked_examples, self.global_sorted_examples, self.pca_2d = sort_examples_as_demos(
            trainset=self.trainset,
            os_test=self.os_test,
            student=self.student,
            metric=self.metric,
            trainset_by_hash=self.trainset_by_hash,
            examples2embeddings=self.examples2embeddings,
            embedding_model_name=str(self.selected_encoder),
            pca_2d=self.pca_2d,
            apply_visuals=self.apply_visuals
        )

        # Sort training clusters by ranked examples (descending order)
        self.training_clusters = {
            cluster_id: sorted(
                cluster_examples,
                key=lambda ex: self.ranked_examples[get_example_hash(ex)],
                reverse=True,
            )
            for cluster_id, cluster_examples in self.training_clusters.items()
        }

        if self._soft_select:
            self.final_fewshot_subset = soft_select_examples(
                trainset=self.trainset,
                ranked_examples=self.ranked_examples,
                examples2embeddings=self.examples2embeddings,
                N=self.N,
                apply_visuals=self.apply_visuals
            )
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
        Clusters examples into semantic groups using semantic encoders.

        Performs K-means clustering and selects optimal K based on silhouette score.
        On first call (training), evaluates all semantic encoders and selects the best one.
        On subsequent calls (validation), reuses the selected encoder for consistency.

        Args:
            train: bool
                If True, clusters training examples. If False, clusters validation examples.

        Returns:
            Dict[int, List]: Dictionary mapping cluster IDs to lists of examples
        """
        data = [ex["raw"] for ex in self.trainset] if train else self.valset
        data_type = "training" if train else "validation"

        # Generate embeddings using semantic encoders (or reuse selected encoder)
        embeddings, labels, k, self.selected_encoder = \
            generate_embedding_clusters_with_semantic_encoders(
                examples=data,
                semantic_encoders=self.semantic_encoders if train else [self.selected_encoder],
                selected_encoder=None if train else self.selected_encoder
            )

        encoder_name = str(self.selected_encoder)

        clusters, encoder_name, N = cluster_examples(
            data=data,
            task_type=self.task_type,
            trainset=self.trainset,
            examples2embeddings=self.examples2embeddings,
            embeddings2examples=self.embeddings2examples,
            embedding_model_name=encoder_name,
            pca_2d=self.pca_2d,
            student=self.student,
            embeddings=embeddings,
            cluster_labels=labels,
            k=k,
            data_type=data_type,
            train=train,
            apply_visuals=self.apply_visuals
        )

        if train and N is not None:
            self.N = N  # Used as hyperparameter for few-shot sampling

        return clusters

    def collect_fewshot_subsets(self):
        """
        Collects candidate few-shot subsets using task-adaptive sampling strategies.

        Uses strategies defined in TASK_2_SAMPLINGS for the current task type:
        - 'top_n': Selects globally top-ranked examples present in each cluster
        - 'best_in_cluster': Selects the highest-ranked example from each cluster
        - 'popularity': Allocates examples proportionally to cluster sizes
        - 'central': Selects the most semantically central example in each cluster
        """
        sampling_strategies = TASK_2_SAMPLINGS[self.task_type]
        adaptive_fewshot_subsets = {
            sampling_strategy: []
            for sampling_strategy in sampling_strategies
        }

        for sampling_strategy in sampling_strategies:
            logger.info(
                f"Collecting candidate few-shot using '{sampling_strategy}' "
                f"sampling strategy ({self.task_type} task optimization)"
            )
            for cluster_id, _ in self.training_clusters.items():
                adaptive_fewshot_subsets[sampling_strategy].extend(
                    sample_examples_from_cluster(
                        cluster_id=cluster_id,
                        training_clusters=self.training_clusters,
                        sampling_strategy=sampling_strategy,
                        N=self.N,
                        global_sorted_examples=self.global_sorted_examples,
                        trainset=self.trainset,
                        examples2embeddings=self.examples2embeddings
                    )
                )

        self.candidate_fewshot_subsets = adaptive_fewshot_subsets

    def pick_best_fewshot_subset(self):
        """
        Evaluates candidate few-shot subsets and selects the best performing one.

        If only one sampling strategy is applicable, uses it directly.
        Otherwise, evaluates each strategy on the validation set and selects
        the one with the highest accuracy.
        """
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
