import json
import torch
import random
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datasets.fingerprint import Hasher
from dspy.primitives import Example
from sklearn.metrics import silhouette_score
from dspy.utils.parallelizer import ParallelExecutor

from dspy.evaluate import Evaluate

logger = logging.getLogger(__name__)

MIN_CLUSTERS: int = 3
MAX_CLUSTERS: int = 10


# ============================================================================
# CLUSTERING UTILITIES
# ============================================================================

def cluster_examples(
    data: List[Example],
    task_type: str,
    trainset: List[Dict],
    examples2embeddings: Dict,
    embeddings2examples: Dict,
    embedding_model_name: str,
    pca_2d,
    student,
    embeddings: np.ndarray,
    cluster_labels: List[int],
    k: int,
    data_type: str = "training",
    train: bool = True,
    apply_visuals: bool = True
) -> Tuple[Dict[int, List], str]:
    """
    Clusters examples into semantic groups using pre-computed embeddings and labels.

    Args:
        data: List of Example objects to cluster
        task_type: Type of task (e.g., "classification")
        trainset: List of training examples (dictionaries)
        examples2embeddings: Dictionary mapping example hashes to embeddings
        embeddings2examples: Dictionary mapping embedding strings to examples
        embedding_model_name: Name of the embedding model used
        pca_2d: Pre-fitted PCA model for 2D visualization (or None)
        student: The student model being trained
        embeddings: Pre-computed embeddings for the examples
        cluster_labels: Pre-computed cluster labels for each example
        k: Number of clusters
        data_type: Type of data being clustered (e.g., "training", "validation")
        train: Whether this is training data
        apply_visuals: Whether to generate and save visualizations

    Returns:
        Tuple of (clusters dictionary mapping cluster IDs to examples,
                  embedding model name,
                  N value for sampling)
    """
    examples_embeddings = embeddings

    for ex, emb in zip(trainset if train else data, examples_embeddings):
        examples2embeddings[get_example_hash(ex)] = np.array(emb)

    for emb, ex in zip(examples_embeddings, trainset if train else data):
        embeddings2examples[str(emb)] = ex

    N = k if train else None

    clusters = {i: [] for i in range(k)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(trainset[idx] if train else data[idx])

    if apply_visuals:
        visualize_examples(
            embeddings=examples_embeddings,
            embedding_model=embedding_model_name,
            cluster_labels=cluster_labels,
            num_clusters=k,
            data_type=data_type,
            save_path=f"{data_type}_clusters.png",
            silhouette=silhouette_score(examples_embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else 0,
            pca_2d=pca_2d,
            student=student,
        )

    logger.info(f"{data_type} clustering completed with K={k}.")

    return clusters, embedding_model_name, N


def generate_embedding_clusters_with_semantic_encoders(
    examples: List[Example],
    semantic_encoders: List,
    selected_encoder: Optional[Any] = None
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


def get_central_examples(examples: List, examples2embeddings: Dict, sample_size: int):
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
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_examples(
    embeddings,
    embedding_model,
    cluster_labels=None,
    num_clusters=None,
    data_type=None,
    save_path=None,
    silhouette=None,
    show_ranks=False,
    examples=None,
    ranked_examples=None,
    pca_2d=None,
    student=None,
):
    """
    Visualizes high-dimensional embeddings in 2D space using PCA projection.

    Creates scatter plots of embeddings colored either by cluster labels or by
    one-shot performance scores. Useful for understanding the semantic structure
    of the data and the relationship between embedding space and performance.

    Args:
        embeddings: Array of embedding vectors to visualize
        embedding_model: Name of the embedding model used
        cluster_labels: Cluster assignment for each embedding (for cluster visualization)
        num_clusters: Total number of clusters
        data_type: Type of data being visualized (e.g., "training", "validation")
        save_path: Path to save the visualization image
        silhouette: Silhouette score for cluster quality assessment
        show_ranks: If True, color by one-shot scores instead of clusters
        examples: List of Example objects aligned with embeddings
        ranked_examples: Dictionary mapping example hashes to one-shot scores
        pca_2d: Pre-fitted PCA model (or None to fit a new one)
        student: The student model being trained

    Returns:
        The PCA model used for dimensionality reduction
    """
    logger.info("Performing PCA dimensionality reduction for visualization...")
    if not pca_2d:
        pca_2d = PCA(n_components=2)
        embeddings_2d = pca_2d.fit_transform(embeddings)
    else:
        embeddings_2d = pca_2d.transform(embeddings)

    if show_ranks:
        if examples is None:
            raise ValueError(
                "When show_ranks=True, you must pass `examples` aligned with `embeddings`."
            )
        if len(examples) != len(embeddings):
            raise ValueError(
                f"`examples` and `embeddings` must have the same length "
                f"(got {len(examples)} vs {len(embeddings)})."
            )

        scores = [ranked_examples.get(get_example_hash(ex), 0) for ex in examples]
        np_scores = np.array(scores, dtype=np.float32)

        color_values = np_scores
        color_label = "One-shot Score"

        # Choose a colormap that works well with continuous values
        cmap = "coolwarm" if len(set(scores)) <= 5 else "viridis"
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
        s=10,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label)

    if show_ranks:
        logger.info(f"Unique one-shot scores: {set(scores)}")
        plt.title(
            f"PCA of {data_type.title()} One-shot Ranks\n"
            f"Rank mean={np.mean(np_scores):.2f}\n"
            f"Rank std={np.std(np_scores):.2f}"
        )
    else:
        plt.title(
            f"PCA of {data_type.title()} Embeddings Clusters\n"
            f"K={num_clusters}\n"
            f"Size={len(embeddings)}\n"
            f"Silhouette={silhouette:.3f}\n"
            f"Embedding Model={embedding_model}\n"
            # TODO: derive dataset name without coupling to specific program classes
            f"Dataset={type(student).__name__}"
        )

    plt.xlabel("PCA Dimension 1", fontsize=12, labelpad=8)
    plt.ylabel("PCA Dimension 2", fontsize=12, labelpad=8)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    logger.info(f"Cluster visualization saved to {save_path}.")

    return pca_2d


def visualize_one_shot_scores_distribution(ranked_examples: Dict, save_path="one_shot_scores_distribution.png"):
    """
    Creates a bar chart showing the distribution of one-shot evaluation scores.

    Visualizes how frequently each score value appears across all evaluated examples.
    This helps understand the overall quality distribution of potential demonstrations
    and identify clusters of similar-performing examples.

    Args:
        ranked_examples: Dictionary mapping example hashes to one-shot scores
        save_path: Path to save the distribution plot
    """
    from collections import Counter

    if not ranked_examples:
        logger.warning("No ranked examples found. Skipping one-shot score visualization.")
        return

    score_counts = Counter(ranked_examples.values())

    sorted_scores = sorted(score_counts.items())
    scores, counts = zip(*sorted_scores)

    plt.figure(figsize=(10, 7))
    plt.bar(scores, counts, color='skyblue', edgecolor='black')
    plt.xlabel("One-shot Evaluation Score")
    plt.ylabel("Score frequency")
    plt.title("Distribution of One-shot Scores")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"One-shot scores distribution saved to {save_path}")


def visualize_soft_selection(
    ranked_examples: Dict,
    examples2embeddings: Dict,
    final_fewshot_subset: List,
    div_lambda: float,
    save_path="soft_selection_pca.png"
):
    """
    Visualizes the final few-shot example selection in embedding space.

    Creates a 2D PCA plot where selected examples are highlighted in red against
    all candidate examples in gray. This helps visualize the diversity and coverage
    of the selected few-shot subset across the semantic space.

    Args:
        ranked_examples: Dictionary mapping example hashes to one-shot scores
        examples2embeddings: Dictionary mapping example hashes to embeddings
        final_fewshot_subset: List of examples selected for the few-shot set
        div_lambda: Diversity penalty coefficient used during selection
        save_path: Path to save the visualization
    """
    all_examples = list(ranked_examples)
    embs = np.stack([
        examples2embeddings[get_example_hash(ex)]
        for ex in all_examples
    ])
    selected_set = set(final_fewshot_subset)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embs)

    colors = ['red' if ex in selected_set else 'gray' for ex in all_examples]

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, edgecolor='k')
    plt.title(f"PCA of Training Embeddings with Selected Few-shot (Red)\nDiversity λ={div_lambda:.3f}")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Soft selection PCA plot saved to {save_path}")


def visualize_os_test(
    valset: List[Example],
    os_test: List[Example],
    examples2embeddings: Dict,
    save_path="one_shot_test.png"
):
    """
    Visualizes the one-shot test set selection from validation data.

    Creates a 2D PCA plot highlighting which validation examples were selected
    for one-shot evaluation. Selected examples are shown in red against all
    validation examples in gray.

    Args:
        valset: Complete validation dataset
        os_test: Subset of validation examples selected for one-shot testing
        examples2embeddings: Dictionary mapping example hashes to embeddings
        save_path: Path to save the visualization
    """
    all_examples = list(valset)
    embs = np.stack([
        examples2embeddings[get_example_hash(ex)]
        for ex in all_examples
    ])
    selected_set = set(os_test)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embs)

    colors = ['red' if ex in selected_set else 'gray' for ex in all_examples]

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, edgecolor='k')
    plt.title(f"PCA of Validation Embeddings with Selected One-shot test questions (Red)")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"One-shot test set PCA plot saved to {save_path}")


# ============================================================================
# SAMPLING AND EVALUATION UTILITIES
# ============================================================================

def sample_one_shot_evaluation_set(
    validation_clusters: Dict[int, List],
    examples2embeddings: Dict
) -> List[Example]:
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
    trainset: List[Dict],
    os_test: List[Example],
    student,
    metric,
    trainset_by_hash: Dict,
    examples2embeddings: Dict,
    embedding_model_name: str,
    pca_2d,
    apply_visuals: bool = True
) -> Tuple[Dict, List, Dict]:
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
        examples2embeddings: Dictionary mapping example hashes to embeddings
        embedding_model_name: Name of the embedding model used
        pca_2d: Pre-fitted PCA model for visualization
        apply_visuals: Whether to generate and save visualizations

    Returns:
        Tuple of (ranked_examples dictionary mapping hashes to scores,
                  globally sorted examples list,
                  updated PCA model)
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

    if apply_visuals:
        visualize_one_shot_scores_distribution(ranked_examples)
        pca_2d = visualize_examples(
            embeddings=[examples2embeddings[get_example_hash(ex)] for ex in trainset],
            examples=list(trainset),
            embedding_model=embedding_model_name,
            save_path="embeddings_to_one_shot_ranks.png",
            data_type="training",
            show_ranks=True,
            ranked_examples=ranked_examples,
            pca_2d=pca_2d,
            student=student,
        )

    logger.info("Demonstrations are sorted in descending order of empirical contribution.")

    return ranked_examples, global_sorted_examples, pca_2d


def evaluate_example_as_demo(example: Dict, evaluator, student, os_test: List[Example]) -> float:
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
    raw = example['raw']
    inputs_str = ', '.join(f'{k}: {v}' for k, v in dict(raw.inputs()).items())
    labels_str = ', '.join(f'{k}: {v}' for k, v in dict(raw.labels()).items())
    example_visual = f"{inputs_str} --> {labels_str}"

    logger.info(
        f"Conducting example-as-demo test ({len(os_test)} questions) "
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


def sample_examples_from_cluster(
    cluster_id: int,
    training_clusters: Dict[int, List],
    sampling_strategy: str,
    N: int,
    global_sorted_examples: List,
    trainset: List,
    examples2embeddings: Dict
) -> List:
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
        N: Target number of examples to select
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
            top_global_n = global_sorted_examples[:N]
            sampled_examples.extend([ex for ex in cluster_examples if ex in top_global_n])

        elif sampling_strategy == "best_in_cluster":
            sampled_examples.append(cluster_examples[0])

        elif sampling_strategy == "popularity":
            total_examples = len(trainset)
            proportion = len(cluster_examples) / total_examples
            sample_size = min(len(cluster_examples), round(proportion * N))
            sampled_examples = cluster_examples[:sample_size]

        elif sampling_strategy == "central":
            sampled_examples = get_central_examples(
                examples=cluster_examples,
                sample_size=1,
                examples2embeddings=examples2embeddings
            )

    logger.info(
        f"{len(sampled_examples)}/{N} slots given to cluster {cluster_id + 1} (size={len(training_clusters[cluster_id])})"
    )

    return sampled_examples


# ============================================================================
# SOFT SELECTION
# ============================================================================

def soft_select_examples(
    trainset: List,
    ranked_examples: Dict,
    examples2embeddings: Dict,
    N: int,
    steps: int = 1000,
    log_step: int = 100,
    lr: float = 1e-1,
    device: str = "cpu",
    verbose: bool = True,
    min_lambda: float = 10,
    max_lambda: float = np.inf,
    apply_visuals: bool = True
) -> List:
    """
    Performs differentiable soft selection of few-shot examples.

    Uses gradient-based optimization to select N examples that balance two objectives:
    1. High one-shot demonstration quality (maximize impact)
    2. Low semantic redundancy (maximize diversity)

    The diversity penalty is controlled by a learnable lambda parameter that is
    optimized jointly with the selection probabilities.

    Args:
        trainset: List of all training examples
        ranked_examples: Dictionary mapping example hashes to one-shot scores
        examples2embeddings: Dictionary mapping example hashes to embeddings
        N: Number of examples to select
        steps: Number of optimization steps
        log_step: How often to log progress
        lr: Learning rate for optimization
        device: Device to run optimization on ('cpu' or 'cuda')
        verbose: Whether to log detailed progress
        min_lambda: Minimum diversity penalty coefficient
        max_lambda: Maximum diversity penalty coefficient
        apply_visuals: Whether to generate and save visualizations

    Returns:
        List of N selected examples optimized for both quality and diversity
    """
    import math

    M = len(trainset)

    one_shot_scores = torch.tensor(
        [score for _, score in ranked_examples.items()],
        dtype=torch.float32,
        device=device
    )  # shape (M,)

    # Build embedding matrix
    embs = torch.stack([
        torch.tensor(examples2embeddings[get_example_hash(ex)], device=device, dtype=torch.float32)
        for ex, _ in ranked_examples.items()
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
    candidate_examples = list(ranked_examples)
    final_fewshot_subset = [candidate_examples[i] for i in topn]

    div_lambda = log_lambda.exp().item()

    if apply_visuals:
        visualize_soft_selection(
            ranked_examples=ranked_examples,
            examples2embeddings=examples2embeddings,
            final_fewshot_subset=final_fewshot_subset,
            div_lambda=div_lambda
        )

    return final_fewshot_subset


# ============================================================================
# BOOTSTRAPPING
# ============================================================================

def bootstrap_examples(
    examples: List[Example],
    student,
    metric,
    metric_threshold,
    trainset_by_hash: Dict
) -> List[Dict]:
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
