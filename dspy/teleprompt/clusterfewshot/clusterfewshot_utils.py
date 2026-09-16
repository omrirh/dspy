import json
import logging
import random
from typing import Any

import numpy as np
from datasets.fingerprint import Hasher
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dspy.evaluate import Evaluate
from dspy.primitives import Example
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

MIN_CLUSTERS: int = 3
MAX_CLUSTERS: int = 4


# ============================================================================
# CLUSTERING UTILITIES
# ============================================================================

def cluster_examples(
    data: list[Example],
    task_type: str,
    trainset: list[dict],
    examples2embeddings: dict,
    embeddings2examples: dict,
    embedding_model_name: str,
    embeddings: np.ndarray,
    cluster_labels: list[int],
    k: int,
    data_type: str = "training",
    train: bool = True,
) -> tuple[dict[int, list], str]:
    """
    Clusters examples into semantic groups using pre-computed embeddings and labels.

    Args:
        data: List of Example objects to cluster
        task_type: Type of task (e.g., "classification")
        trainset: List of training examples (dictionaries)
        examples2embeddings: Dictionary mapping example hashes to embeddings
        embeddings2examples: Dictionary mapping embedding strings to examples
        embedding_model_name: Name of the embedding model used
        embeddings: Pre-computed embeddings for the examples
        cluster_labels: Pre-computed cluster labels for each example
        k: Number of clusters
        data_type: Type of data being clustered (e.g., "training", "validation")
        train: Whether this is training data

    Returns:
        Tuple of (clusters dictionary mapping cluster IDs to examples,
                  embedding model name,
                  N value for sampling)
    """
    examples_embeddings = embeddings

    for ex, emb in zip(trainset if train else data, examples_embeddings, strict=False):
        examples2embeddings[get_example_hash(ex)] = np.array(emb)

    for emb, ex in zip(examples_embeddings, trainset if train else data, strict=False):
        embeddings2examples[str(emb)] = ex

    n = k if train else None

    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(trainset[idx] if train else data[idx])

    logger.info(f"{data_type} clustering completed with K={k}.")

    return clusters, embedding_model_name, n


def generate_embedding_clusters_with_semantic_encoders(
    examples: list[Example],
    semantic_encoders: list,
    selected_encoder: Any | None = None
):
    """
    Generates embeddings and finds optimal clusters using semantic encoders.

    This function implements a Bring-Your-Own-Encoder design that allows users
    to provide custom SemanticEncoder implementations. The function either:
    1. Uses a pre-selected encoder (for validation, reusing the training encoder)
    2. Evaluates multiple encoders to find the best one based on silhouette score

    For each encoder, it tests different values of K (number of clusters) to find
    the optimal configuration that produces the most coherent semantic groupings.

    Args:
        examples: List of Example objects to embed and cluster
        semantic_encoders: List of SemanticEncoder instances to evaluate
        selected_encoder: Optional pre-selected encoder (if provided, skip search)

    Returns:
        Tuple of (best_embeddings array, best_cluster_labels, optimal K, best_encoder)
    """
    best_k = None
    best_score = -np.inf
    best_labels = None
    best_embeddings = None
    best_encoder = None

    if selected_encoder is not None:
        # Validation case: reuse the training encoder
        logger.info(f"Using pre-selected encoder: {selected_encoder.name()}")
        embeddings = selected_encoder.encode(examples)

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
        best_encoder = selected_encoder
    else:
        # Training case: search for the best encoder
        for encoder in semantic_encoders:
            encoder_name = encoder.name()
            logger.info(f"Encoding examples with encoder: {encoder_name}")

            embeddings = encoder.encode(examples)

            for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)

                logger.info(f"K={k}, Silhouette Score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_encoder = encoder
                    best_embeddings = embeddings

    logger.info(f"Selected encoder: {best_encoder.name()} with K={best_k} (silhouette={best_score:.3f})")
    return best_embeddings, best_labels, best_k, best_encoder


def get_central_examples(examples: list, examples2embeddings: dict, sample_size: int):
    """
    Selects the most central examples from a cluster based on their proximity to the cluster center.

    Central examples are those closest to the mean embedding vector of all examples in the cluster.
    This helps identify representative examples that capture the core semantic meaning of a cluster.

    Args:
        examples: List of examples to select from
        examples2embeddings: Dictionary mapping example hashes to their embeddings
        sample_size: Number of central examples to return

    Returns:
        List of the most central examples
    """
    embeddings = [examples2embeddings[get_example_hash(ex)] for ex in examples]
    cluster_center = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - cluster_center, axis=1)
    selected_indices = np.argsort(distances)[:sample_size]
    sampled_examples = [examples[i] for i in selected_indices]

    return sampled_examples



# ============================================================================
# SAMPLING AND EVALUATION UTILITIES
# ============================================================================

def sample_one_shot_evaluation_set(
    validation_clusters: dict[int, list],
    examples2embeddings: dict
) -> list[Example]:
    """
    Creates a balanced one-shot evaluation set by sampling from each validation cluster.

    Selects the most central examples from each cluster to form a representative
    test set that covers all semantic regions of the validation data. This ensures
    that one-shot evaluation is performed across diverse example types.

    Args:
        validation_clusters: Dictionary mapping cluster IDs to lists of validation examples
        examples2embeddings: Dictionary mapping example hashes to embeddings

    Returns:
        List of examples selected for one-shot evaluation
    """
    os_test = []
    samples_per_cluster = 3

    for cluster_id, examples in validation_clusters.items():
        sample_size = min(samples_per_cluster, len(examples))
        selected = get_central_examples(
            examples=examples,
            sample_size=sample_size,
            examples2embeddings=examples2embeddings
        )

        logger.info(
            f"Sampling {sample_size} questions from cluster {cluster_id + 1} (size={len(examples)})")

        os_test.extend(selected)

    logger.info(f"One-shot evaluation set assembled with {len(os_test)} questions.")

    return os_test


def sort_examples_as_demos(
    trainset: list[dict],
    os_test: list[Example],
    student,
    metric,
    trainset_by_hash: dict,
) -> tuple[dict, list]:
    """
    Ranks training examples by their effectiveness as one-shot demonstrations.

    Evaluates each training example by using it as a single demonstration and
    measuring performance on the one-shot test set. Examples that lead to better
    performance when used as demonstrations receive higher scores.

    Args:
        trainset: List of training examples to evaluate
        os_test: One-shot evaluation test set
        student: The student model being trained
        metric: Evaluation metric function
        trainset_by_hash: Dictionary mapping example hashes to training examples

    Returns:
        Tuple of (ranked_examples dictionary mapping hashes to scores,
                  globally sorted examples list)
    """
    evaluator = Evaluate(
        devset=os_test,
        metric=metric,
        num_threads=min(12, len(os_test)),
        display_progress=True,
    )
    student_copy = student.deepcopy()

    logger.info(f"Sorting examples-as-demos from training set ({len(trainset)} examples)")
    ranked_examples = {}
    trainset_size = len(trainset)

    for idx, ex in enumerate(trainset):
        logger.info(f"\n\nEvaluating example {idx + 1}/{trainset_size}")
        ranked_examples[get_example_hash(ex)] = evaluate_example_as_demo(
            ex, evaluator, student_copy, os_test
        )

    logger.info(f"Ordering {len(ranked_examples)} demonstrations "
                f"by {len(set(ranked_examples.values()))} different ranks...")

    global_sorted_examples = [
        trainset_by_hash[ex_hash]
        for ex_hash in sorted(
            ranked_examples,
            key=lambda h: ranked_examples[h],
            reverse=True,
        )
    ]

    logger.info("Demonstrations are sorted in descending order of empirical contribution.")

    return ranked_examples, global_sorted_examples


def evaluate_example_as_demo(example: dict, evaluator, student, os_test: list[Example]) -> float:
    """
    Evaluates a single example's quality as a demonstration.

    Temporarily sets the example as the sole demonstration for the student model
    and measures performance on the one-shot test set. Higher scores indicate
    the example is more effective at teaching the student.

    Args:
        example: Training example to evaluate as a demonstration
        evaluator: Evaluation object for scoring predictions
        student: The student model being trained
        os_test: One-shot evaluation test set

    Returns:
        Score indicating demonstration quality (higher is better)
    """
    raw = example["raw"]
    inputs_str = ", ".join(f"{k}: {v}" for k, v in dict(raw.inputs()).items())
    labels_str = ", ".join(f"{k}: {v}" for k, v in dict(raw.labels()).items())
    example_visual = f"{inputs_str} --> {labels_str}"

    logger.info(
        f"Conducting example-as-demo test ({len(os_test)} questions) "
        f"using the following demonstration:\n"
        f"{example_visual}"
    )

    cached_demos = [pred.demos for _, pred in student.named_predictors()]

    for name, predictor in student.named_predictors():
        predictor.demos = example[name]  # Test as one-shot demonstration

    student_score = evaluator(program=student).score

    for (_, predictor), demos in zip(student.named_predictors(), cached_demos, strict=False):
        predictor.demos = demos

    return student_score


def sample_examples_from_cluster(
    cluster_id: int,
    training_clusters: dict[int, list],
    sampling_strategy: str,
    n: int,
    global_sorted_examples: list,
    trainset: list,
    examples2embeddings: dict
) -> list:
    """
    Samples examples from a specific cluster using one of several strategies.

    Supports four sampling strategies:
    1. top_n: Selects examples from this cluster that are in the global top-N
    2. best_in_cluster: Selects the highest-ranked example from this cluster
    3. popularity: Allocates slots proportional to cluster size
    4. central: Selects the most central (representative) example

    Args:
        cluster_id: ID of the cluster to sample from
        training_clusters: Dictionary mapping cluster IDs to example lists
        sampling_strategy: Strategy to use ("top_n", "best_in_cluster", "popularity", "central")
        n: Target number of examples to select
        global_sorted_examples: All examples sorted by one-shot score
        trainset: Complete training dataset
        examples2embeddings: Dictionary mapping example hashes to embeddings

    Returns:
        List of sampled examples from the cluster
    """
    sampled_examples = []

    if cluster_id in training_clusters:
        cluster_examples = training_clusters[cluster_id]
        if not cluster_examples:
            return sampled_examples

        if sampling_strategy == "top_n":
            top_global_n = global_sorted_examples[:n]
            sampled_examples.extend([ex for ex in cluster_examples if ex in top_global_n])

        elif sampling_strategy == "best_in_cluster":
            sampled_examples.append(cluster_examples[0])

        elif sampling_strategy == "popularity":
            total_examples = len(trainset)
            proportion = len(cluster_examples) / total_examples
            sample_size = min(len(cluster_examples), round(proportion * n))
            sampled_examples = cluster_examples[:sample_size]

        elif sampling_strategy == "central":
            sampled_examples = get_central_examples(
                examples=cluster_examples,
                sample_size=1,
                examples2embeddings=examples2embeddings
            )

    logger.info(
        f"{len(sampled_examples)}/{n} slots given to cluster {cluster_id + 1} (size={len(training_clusters[cluster_id])})"
    )

    return sampled_examples


# ============================================================================
# BOOTSTRAPPING
# ============================================================================

def bootstrap_examples(
    examples: list[Example],
    student,
    metric,
    metric_threshold,
    trainset_by_hash: dict
) -> list[dict]:
    """
    Bootstraps training examples by generating predictions and filtering by quality.

    For each example, runs the student model to generate predictions, then evaluates
    those predictions against the metric. Only examples that meet the metric threshold
    are kept as high-quality training demonstrations.

    Args:
        examples: List of examples to bootstrap
        student: The student model to generate predictions
        metric: Evaluation metric function
        metric_threshold: Minimum metric value to accept an example
        trainset_by_hash: Dictionary to store bootstrapped examples by hash

    Returns:
        List of bootstrapped training examples that passed the quality threshold
    """
    import dspy

    predictor2name = {
        predictor: name for name, predictor in student.named_predictors()
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

                    if metric:
                        metric_val = metric(example, prediction, trace)
                        if metric_threshold:
                            success = metric_val >= metric_threshold
                        else:
                            success = metric_val
                    else:
                        success = True
        except Exception as e:
            logger.warning(f"Bootstrapping failed for an example: {type(e).__name__}: {e}")
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

            bootstrapped = {"raw": example}
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
            trainset_by_hash[get_example_hash(bootstrapped)] = bootstrapped
            bootstrapped_examples.append(bootstrapped)

    logger.info(f"{len(bootstrapped_examples)}/{len(examples)} remaining after bootstrapping")

    return bootstrapped_examples


# ============================================================================
# HASH UTILITIES
# ============================================================================

def normalize_example(obj: Any) -> Any:
    """
    Normalizes an example object to ensure consistent hashing.

    Recursively processes Example objects, dictionaries, and lists to produce
    a canonical representation that will hash consistently regardless of
    ordering or type variations.

    Args:
        obj: Object to normalize (Example, dict, list, or primitive)

    Returns:
        Normalized version of the object suitable for stable hashing
    """
    if isinstance(obj, Example):
        # stable: convert to plain dict and sort nested structures
        return {k: normalize_example(v) for k, v in dict(obj).items()}

    if isinstance(obj, dict):
        # stable: sort keys
        return {k: normalize_example(obj[k]) for k in sorted(obj.keys())}

    if isinstance(obj, list):
        # stable: keep list order (semantic), but normalize each item
        return [normalize_example(x) for x in obj]

    if isinstance(obj, set):
        # stable: sort for deterministic ordering (sets are unordered)
        return sorted([normalize_example(x) for x in obj])

    return obj


def get_example_hash(example_obj: Any) -> str:
    """
    Computes a stable, deterministic hash string for an example object.

    Uses JSON serialization of normalized examples to create consistent hashes
    that can be used as dictionary keys for tracking examples across different
    data structures.

    Args:
        example_obj: Example object to hash. Can be:
            - A validation Example object
            - A training example dictionary with 'raw' and predictor keys

    Returns:
        JSON string hash that uniquely and consistently identifies the example
    """
    normalized = normalize_example(example_obj)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
