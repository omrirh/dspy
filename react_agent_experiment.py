"""
HotPotQA ReAct Agent experiment — prompt-optimizer comparison PoC.

Benchmarks ClusterFewshot, MIPROv2, and BFRS as few-shot/prompt optimizers
for ``dspy.ReAct`` on HotPotQA multi-hop QA. The agent calls a ColBERTv2
Wikipedia search tool iteratively to gather evidence before producing a final
answer.

Key design decisions
--------------------
* **Outcome-quality bootstrapping**: ``answer_exact_match`` is a binary metric,
  so ``bootstrap_examples`` retains only episodes where the agent produced the
  correct answer. All demonstration steps therefore originate from *successful*
  reasoning chains — the agentic equivalent of "examples of good problem-solving".

* **Diversity-first demo selection (ClusterFewshot)**: task_type="agentic" uses
  ``best_in_cluster`` before ``top_n``, ensuring demonstrations span semantically
  distinct question types (bridge, comparison, temporal, geographic, ...) rather
  than collapsing onto the highest-scoring single topic cluster.

* **Two-encoder BYOE grid search (ClusterFewshot)**: ``all-mpnet-base-v2``
  (general) and ``multi-qa-mpnet-base-dot-v1`` (QA-tuned) are evaluated in
  parallel; the encoder with the highest silhouette score is selected automatically.

Usage — ClusterFewshot (default):
    python react_agent_experiment.py \\
        --model gemini/gemini-2.5-flash \\
        --colbert-url http://localhost:8894/api/search

Usage — MIPROv2:
    python react_agent_experiment.py \\
        --model gemini/gemini-2.5-flash \\
        --optimizer miprov2

Usage — BFRS:
    python react_agent_experiment.py \\
        --model gemini/gemini-2.5-flash \\
        --optimizer bfrs

Usage — baseline only (no optimization):
    python react_agent_experiment.py \\
        --model gemini/gemini-2.5-flash \\
        --baseline

Flags
-----
--model             LM to use (see react_agent_experiment_driver.sh for full list)
--optimizer         Prompt optimizer: clusterfs | miprov2 | bfrs (default: clusterfs)
--colbert-url       ColBERTv2 endpoint (default: http://localhost:8894/api/search)
--sglang-port       If set, connects to a local sglang server on this port
--train-size        Number of training examples (default: 500)
--dev-size          Number of validation examples (default: 200)
--test-size         Number of test examples (default: 500)
--max-iters         Max ReAct steps per question (default: 20)
--encoder-device    Device for SentenceTransformer encoders (default: cpu)
--baseline          Skip optimization; evaluate zero-shot agent only
--no-visuals        Disable matplotlib cluster plots
--sample-trajectory Print a qualitative trajectory comparison after evaluation
--seed              Random seed for reproducibility (default: derived from time)
--results-dir       Base directory for structured JSON output (default: results)
"""

import json
import os
import random
import subprocess
import time
import logging
import argparse
from collections import Counter
from datetime import datetime, timezone

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets import HotPotQA
from programs import ReactAgentMH
from dspy.teleprompt.clusterfewshot import (
    ClusterFewshot,
    create_hotpotqa_question_encoder,
)
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Questions filtered from evaluation sets (content-safety)
_HOTPOTQA_EXCLUDE = [
    "beat, torture, and sexually assault",
    "Anti-pedophile activism advocates for victims",
    "hosting a video of the murder of an international student",
    "contemporary scholars likens to Ilminism",
    "Joseph Druce murder John Geoghan",
    "George Pell first sexually assault a 12 year old boy",
    "The Gay Nigger Association of America",
    "insertion and thrusting of the erect penis into a person's anus",
]

COLBERT_DEFAULT_URL = "http://localhost:8894/api/search"


# ---------------------------------------------------------------------------
# Parse failure detection (TODO 6)
# ---------------------------------------------------------------------------

def count_parse_failures(prediction, max_iters=20):
    """Inspect a ReAct prediction's trajectory to detect parse failures.

    Returns (n_failures, failure_type, n_steps, finished_via_tool) where:
      - n_failures: 0 or 1 (did a parse failure cause early termination?)
      - failure_type: None | "A" (empty/no thought) | "B" (no tool_name) | "C" (no tool_args)
      - n_steps: number of complete steps (all 4 fields present)
      - finished_via_tool: True if the last tool_name was "finish"
    """
    trajectory = getattr(prediction, "trajectory", None) or {}

    if not trajectory:
        return 1, "A", 0, False

    # Count complete steps and find the highest step index
    max_idx = -1
    complete_steps = 0
    for key in trajectory:
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            max_idx = max(max_idx, idx)

    if max_idx < 0:
        return 1, "A", 0, False

    for i in range(max_idx + 1):
        has_all = all(
            f"{field}_{i}" in trajectory
            for field in ("thought", "tool_name", "tool_args", "observation")
        )
        if has_all:
            complete_steps += 1

    # Check if the agent cleanly finished via the "finish" tool
    last_complete = complete_steps - 1
    finished_via_tool = (
        last_complete >= 0
        and trajectory.get(f"tool_name_{last_complete}") == "finish"
    )

    if finished_via_tool:
        return 0, None, complete_steps, True

    # Check if max_iters was exhausted (all steps complete, no finish)
    if complete_steps == max_iters:
        return 0, None, complete_steps, False

    # Parse failure on the step after the last complete one
    fail_idx = complete_steps
    has_thought = f"thought_{fail_idx}" in trajectory
    has_tool_name = f"tool_name_{fail_idx}" in trajectory
    has_tool_args = f"tool_args_{fail_idx}" in trajectory

    if not has_thought:
        return 1, "A", complete_steps, False
    if not has_tool_name:
        return 1, "B", complete_steps, False
    if not has_tool_args:
        return 1, "C", complete_steps, False

    # tool_args present but no observation — shouldn't happen (tool exec always sets it)
    return 0, None, complete_steps, False


def aggregate_parse_failures(per_example_results):
    """Aggregate parse failure stats from a list of per-example result dicts.

    Returns dict with keys: total, rate, types, exhausted_max_iters.
    """
    n = len(per_example_results)
    total_failures = sum(r["parse_failures"] for r in per_example_results)
    type_counts = Counter(
        r["parse_failure_type"] for r in per_example_results if r["parse_failure_type"]
    )
    exhausted = sum(
        1 for r in per_example_results
        if r["parse_failures"] == 0 and not r["finished_via_tool"] and r["trajectory_steps"] > 0
    )
    return {
        "total": total_failures,
        "rate": round(total_failures / n, 4) if n else 0.0,
        "types": dict(type_counts),
        "exhausted_max_iters": exhausted,
    }


def compute_compliant_accuracy(per_example_results):
    """Accuracy over examples with zero parse failures.

    Returns (accuracy_pct, n_compliant).
    """
    compliant = [r for r in per_example_results if r["parse_failures"] == 0]
    if not compliant:
        return 0.0, 0
    correct = sum(1 for r in compliant if r["score"])
    return round(100 * correct / len(compliant), 2), len(compliant)


# ---------------------------------------------------------------------------
# Results helpers (TODO 5)
# ---------------------------------------------------------------------------

def _build_per_example_results(eval_results, max_iters):
    """Convert Evaluate's (example, prediction, score) triples to result dicts."""
    results = []
    for idx, (example, prediction, score) in enumerate(eval_results):
        n_failures, failure_type, n_steps, finished = count_parse_failures(prediction, max_iters)
        trajectory = getattr(prediction, "trajectory", None) or {}
        search_queries = [
            trajectory.get(f"tool_args_{i}")
            for i in range(n_steps)
            if str(trajectory.get(f"tool_name_{i}", "")).lower() == "search"
        ]
        results.append({
            "idx": idx,
            "question": example.get("question", ""),
            "gold": example.get("answer", ""),
            "predicted": getattr(prediction, "answer", None),
            "score": float(score) if score is not None else 0.0,
            "parse_failures": n_failures,
            "parse_failure_type": failure_type,
            "trajectory_steps": n_steps,
            "finished_via_tool": finished,
            "search_queries": search_queries,
        })
    return results


def _collect_optimizer_meta(optimizer_name, optimizer_obj, train_size):
    """Extract post-compile metadata from the optimizer object."""
    meta = {}
    if optimizer_name == "clusterfs":
        meta["selected_encoder"] = str(getattr(optimizer_obj, "selected_encoder", "unknown"))
        meta["n_clusters"] = getattr(optimizer_obj, "N", None)
        trainset = getattr(optimizer_obj, "trainset", None)
        meta["bootstrap_yield"] = len(trainset) if trainset else None
        meta["bootstrap_attempts"] = train_size
    elif optimizer_name == "miprov2":
        meta["num_candidates"] = getattr(optimizer_obj, "num_candidates", None)
        meta["prompt_model_total_calls"] = getattr(optimizer_obj, "prompt_model_total_calls", None)
        meta["total_calls"] = getattr(optimizer_obj, "total_calls", None)
    elif optimizer_name == "bfrs":
        meta["num_candidate_sets"] = getattr(optimizer_obj, "num_candidate_programs", None)
        meta["max_num_samples"] = getattr(optimizer_obj, "max_num_samples", None)
        meta["max_labeled_demos"] = getattr(optimizer_obj, "max_labeled_demos", None)
    return {optimizer_name: meta}


def _get_git_sha():
    """Return short git SHA or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Search tool factory
# ---------------------------------------------------------------------------

def create_colbert_search_tool(colbert_url: str, k: int = 3):
    """
    Creates a plain Python callable that wraps ColBERTv2 retrieval.

    The returned function is passed directly to ``dspy.ReAct`` as a tool.
    It retrieves the top-k Wikipedia passages for a query and returns them
    as a numbered string so the agent can parse the evidence easily.

    Configures ``dspy.settings.rm`` as a side effect — this must happen
    before any call to ``dspy.Retrieve``.

    Args:
        colbert_url: Full ColBERTv2 API endpoint, e.g.
                     "http://localhost:8894/api/search"
        k:           Number of passages to retrieve per query.

    Returns:
        Callable with signature ``(query: str) -> str``
    """
    retriever = dspy.ColBERTv2(url=colbert_url)
    dspy.configure(rm=retriever)
    retrieve = dspy.Retrieve(k=k)

    def search(query: str) -> str:
        """Search Wikipedia for relevant passages. Returns the top passages as text."""
        try:
            passages = retrieve(query).passages
            return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        except Exception as e:
            return f"Search failed: {e}"

    return search


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_hotpotqa_splits(train_size: int, dev_size: int, test_size: int):
    """
    Loads HotPotQA (hard examples only) and applies content-safety filtering.

    Returns (trainset, devset, testset) as lists of ``dspy.Example`` objects
    with ``question`` as the only active input field.
    """
    logger.info("Loading HotPotQA dataset (hard examples only)...")
    dataset = HotPotQA(only_hard_examples=True)

    def _filter(examples):
        return [
            x.with_inputs("question")
            for x in examples
            if not any(excl in x.question for excl in _HOTPOTQA_EXCLUDE)
        ]

    trainset = _filter(dataset.train)[:train_size]
    devset = _filter(dataset.dev)[:dev_size]
    testset = _filter(dataset.test)[:test_size]

    logger.info(f"Split sizes — train: {len(trainset)}, dev: {len(devset)}, test: {len(testset)}")
    return trainset, devset, testset


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

# Max tokens for generation per model class.
# ReAct steps (thought + tool_name + tool_args) rarely exceed 300 tokens;
# the extract step is ~100 tokens. With max_iters=20, 2048 is a safe ceiling.
# 32B+ gets 2048 — the larger context window means longer thoughts are common.
_LOCAL_MODEL_MAX_TOKENS: dict[str, int] = {
    "Qwen2.5-32B-Instruct": 2048,
    "Qwen3-32B": 2048,
    "Llama-3.3-70B-Instruct": 2048,
}
_LOCAL_MODEL_MAX_TOKENS_DEFAULT = 1024


def configure_lm(model: str, sglang_port: int | None, devset, metric):
    """
    Configures the DSPy language model.

    Local HuggingFace models (in ``_HF_MODELS``) connect to a running sglang
    server. All other model strings (Gemini, OpenAI, etc.) use the LiteLLM
    API path with the appropriate environment variable key.
    """
    import dspy.clients.huggingface as hf_client

    if sglang_port or model in hf_client._HF_MODELS:
        from remote_setup.utils import assign_local_lm
        from dspy.clients.huggingface import HFProvider

        port = sglang_port or 7501
        sglang_url = f"http://localhost:{port}/v1"
        model_basename = model.split("/")[-1]
        max_tokens = next(
            (v for k, v in _LOCAL_MODEL_MAX_TOKENS.items() if k in model_basename),
            _LOCAL_MODEL_MAX_TOKENS_DEFAULT,
        )
        logger.info(f"Connecting to local sglang server at {sglang_url} (model: {model}, max_tokens: {max_tokens})")
        lm = assign_local_lm(
            model=model,
            api_base=sglang_url,
            provider=HFProvider(validation_set=devset, validation_metric=metric),
            max_tokens=max_tokens,
        )
    elif model.startswith("gemini/"):
        lm = dspy.LM(model, api_key=os.getenv("GEMINI_API_KEY"), max_tokens=4096)
        dspy.configure(lm=lm)
    else:
        lm = dspy.LM(model, max_tokens=4096)
        dspy.configure(lm=lm)

    return lm


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def _run_clusterfs(student, trainset, devset, metric, encoder_device, apply_visuals):
    """
    Runs ClusterFewshot optimization.

    Encoder notes:
      - all-mpnet-base-v2          : general sentence similarity; captures broad
                                     topic proximity across question subjects.
      - multi-qa-mpnet-base-dot-v1 : fine-tuned for QA semantic matching; better
                                     separates questions by reasoning type (bridge
                                     vs. comparison vs. temporal multi-hop).

    Grid search evaluates both encoders across K ∈ [3, 10] and selects the
    (encoder, K) pair with the highest silhouette score. metric_threshold=None
    because answer_exact_match is binary, so every retained demo is correct.
    """
    semantic_encoders = [
        create_hotpotqa_question_encoder(
            "sentence-transformers/all-mpnet-base-v2", device=encoder_device
        ),
        create_hotpotqa_question_encoder(
            "sentence-transformers/multi-qa-mpnet-base-dot-v1", device=encoder_device
        ),
    ]
    opt = ClusterFewshot(
        metric=metric,
        metric_threshold=None,
        task_type="agentic",
        semantic_encoders=semantic_encoders,
        apply_visuals=apply_visuals,
    )
    program = opt.compile(student=student, trainset=trainset, valset=devset)
    return program, opt


def _run_miprov2(student, trainset, devset, metric):
    """
    Runs MIPROv2 optimization with medium-auto settings suitable for agentic tasks.

    Uses minibatch evaluation to keep wall-clock time manageable and passes devset
    as valset so MIPROv2 can do held-out Bayesian trial selection. max_labeled_demos=0
    aligns with the official DSPy agents tutorial (bootstrapped demos only).
    """
    opt = MIPROv2(
        metric=metric,
        auto="medium",
        max_bootstrapped_demos=4,
        max_labeled_demos=0,
        num_threads=6,
    )
    program = opt.compile(
        student=student,
        trainset=trainset,
        valset=devset,
        minibatch=True,
        minibatch_size=25,
        minibatch_full_eval_steps=10,
        requires_permission_to_run=False,
    )
    return program, opt


def _run_bfrs(student, trainset, devset, metric):
    """
    Runs BootstrapFewShotWithRandomSearch (BFRS) optimization.

    Searches over 6 candidate programs with up to 3 bootstrapped demonstrations
    (no labeled demos, matching MIPROv2 config). valset=devset ensures candidate
    selection uses held-out data, consistent with MIPROv2 and ClusterFewshot.
    """
    opt = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=0,
        num_candidate_programs=6,
        num_threads=6,
    )
    program = opt.compile(student=student, trainset=trainset, valset=devset)
    return program, opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model: str,
    optimizer: str,
    colbert_url: str,
    sglang_port: int | None,
    train_size: int,
    dev_size: int,
    test_size: int,
    max_iters: int,
    encoder_device: str,
    baseline: bool,
    apply_visuals: bool,
    sample_trajectory: bool,
    seed: int,
    results_dir: str,
):
    script_start = time.time()
    random.seed(seed)
    model_basename = model.split("/")[-1]

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    trainset, devset, testset = load_hotpotqa_splits(train_size, dev_size, test_size)
    metric = dspy.evaluate.answer_exact_match

    # -----------------------------------------------------------------------
    # 2. LM
    # -----------------------------------------------------------------------
    configure_lm(model, sglang_port, devset, metric)

    # -----------------------------------------------------------------------
    # 3. Search tool + student program
    # -----------------------------------------------------------------------
    logger.info(f"Creating ColBERTv2 search tool ({colbert_url})")
    search_tool = create_colbert_search_tool(colbert_url, k=3)

    student = ReactAgentMH(search_tool=search_tool, max_iters=max_iters)

    # -----------------------------------------------------------------------
    # 4. Baseline evaluation (always run; used as reference)
    # -----------------------------------------------------------------------
    evaluate = Evaluate(
        devset=testset,
        metric=metric,
        num_threads=8,
        display_progress=True,
        display_table=False,
    )

    logger.info("Evaluating zero-shot baseline (no demonstrations)...")
    baseline_start = time.time()
    baseline_score, baseline_results = evaluate(student, return_outputs=True)
    baseline_runtime = time.time() - baseline_start
    logger.info(f"Baseline score: {baseline_score:.2f}%  ({baseline_runtime:.1f}s)")

    baseline_per_example = _build_per_example_results(baseline_results, max_iters)
    baseline_parse_summary = aggregate_parse_failures(baseline_per_example)
    baseline_compliant_acc, baseline_compliant_n = compute_compliant_accuracy(baseline_per_example)

    if baseline:
        total_runtime = time.time() - script_start
        out_dir = os.path.join(results_dir, model_basename, "baseline")
        os.makedirs(out_dir, exist_ok=True)
        result_json = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "dspy_version": getattr(dspy, "__version__", "unknown"),
            "config": {
                "model": model,
                "optimizer": "baseline",
                "seed": seed,
                "train_size": train_size,
                "dev_size": dev_size,
                "test_size": test_size,
                "max_iters": max_iters,
                "colbert_url": colbert_url,
                "encoder_device": encoder_device,
            },
            "scores": {
                "baseline": baseline_score,
                "optimized": None,
                "delta_pp": None,
                "compliant_accuracy": baseline_compliant_acc,
                "compliant_n": baseline_compliant_n,
            },
            "timing_seconds": {
                "baseline_eval": round(baseline_runtime, 2),
                "compile_total": None,
                "optimized_eval": None,
                "total": round(total_runtime, 2),
            },
            "optimizer_meta": None,
            "parse_failures": {
                "baseline_eval": baseline_parse_summary,
                "optimized_eval": None,
            },
            "per_example_results": baseline_per_example,
            "program_state_path": None,
        }
        json_path = os.path.join(out_dir, f"{seed}.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)

        print(f"\n[BASELINE | {model}]")
        print(f"  Score              : {baseline_score:.2f}%")
        print(f"  Compliant accuracy : {baseline_compliant_acc:.2f}% ({baseline_compliant_n}/{len(baseline_per_example)})")
        print(f"  Parse failures     : {baseline_parse_summary['total']} ({baseline_parse_summary['rate']:.1%})")
        print(f"  Runtime            : {baseline_runtime:.1f}s")
        print(f"  Results saved to   : {json_path}")
        return

    # -----------------------------------------------------------------------
    # 5. Prompt optimization
    # -----------------------------------------------------------------------
    logger.info(f"Starting {optimizer.upper()} compilation...")
    compile_start = time.time()

    if optimizer == "clusterfs":
        optimized_program, optimizer_obj = _run_clusterfs(
            student, trainset, devset, metric, encoder_device, apply_visuals
        )
    elif optimizer == "miprov2":
        optimized_program, optimizer_obj = _run_miprov2(
            student, trainset, devset, metric
        )
    elif optimizer == "bfrs":
        optimized_program, optimizer_obj = _run_bfrs(
            student, trainset, devset, metric
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    compile_runtime = time.time() - compile_start
    logger.info(f"Compilation finished in {compile_runtime:.1f}s")

    # Optimizer-specific summary header
    print(f"\n[{optimizer.upper()} | {model}]")
    if optimizer == "clusterfs":
        print(f"  Selected encoder : {optimizer_obj.selected_encoder}")
        print(f"  Clusters (N)     : {optimizer_obj.N}")
    print(f"  Compile time     : {compile_runtime:.1f}s\n")
    print("  Demonstrations per predictor:")
    for name, predictor in optimized_program.named_predictors():
        print(f"    '{name}' — {len(predictor.demos)} demo(s)")

    # -----------------------------------------------------------------------
    # 6. Optimized evaluation
    # -----------------------------------------------------------------------
    logger.info("Evaluating optimized agent...")
    optimized_start = time.time()
    optimized_score, optimized_results = evaluate(optimized_program, return_outputs=True)
    optimized_runtime = time.time() - optimized_start

    optimized_per_example = _build_per_example_results(optimized_results, max_iters)
    optimized_parse_summary = aggregate_parse_failures(optimized_per_example)
    optimized_compliant_acc, optimized_compliant_n = compute_compliant_accuracy(optimized_per_example)

    # -----------------------------------------------------------------------
    # 7. Results summary
    # -----------------------------------------------------------------------
    delta = optimized_score - baseline_score
    sign = "+" if delta >= 0 else ""

    print(f"\n{'=' * 55}")
    print(f"  Results — HotPotQA ReAct | {optimizer.upper()} | {model}")
    print(f"{'=' * 55}")
    print(f"  Baseline  (0-shot) : {baseline_score:.2f}%")
    print(f"  Optimized (N-shot) : {optimized_score:.2f}%  ({sign}{delta:.2f}pp)")
    print(f"  Compliant accuracy : {optimized_compliant_acc:.2f}% ({optimized_compliant_n}/{len(optimized_per_example)})")
    print(f"  Parse failures     : baseline {baseline_parse_summary['total']}, optimized {optimized_parse_summary['total']}")
    print(f"  Compile runtime    : {compile_runtime:.1f}s")
    print(f"  Eval runtime       : {optimized_runtime:.1f}s")

    # -----------------------------------------------------------------------
    # 8. Save structured results JSON + program state
    # -----------------------------------------------------------------------
    total_runtime = time.time() - script_start
    out_dir = os.path.join(results_dir, model_basename, optimizer)
    os.makedirs(out_dir, exist_ok=True)

    # Save program state (may fail for ReAct agents with tool closures)
    program_state_path = None
    try:
        program_path = os.path.join(out_dir, f"{seed}_program.json")
        optimized_program.save(program_path)
        program_state_path = program_path
    except Exception as e:
        logger.warning(f"Could not save program state: {e}")

    result_json = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "dspy_version": getattr(dspy, "__version__", "unknown"),
        "config": {
            "model": model,
            "optimizer": optimizer,
            "seed": seed,
            "train_size": train_size,
            "dev_size": dev_size,
            "test_size": test_size,
            "max_iters": max_iters,
            "colbert_url": colbert_url,
            "encoder_device": encoder_device,
        },
        "scores": {
            "baseline": baseline_score,
            "optimized": optimized_score,
            "delta_pp": round(delta, 2),
            "compliant_accuracy": optimized_compliant_acc,
            "compliant_n": optimized_compliant_n,
        },
        "timing_seconds": {
            "baseline_eval": round(baseline_runtime, 2),
            "compile_total": round(compile_runtime, 2),
            "optimized_eval": round(optimized_runtime, 2),
            "total": round(total_runtime, 2),
        },
        "optimizer_meta": _collect_optimizer_meta(optimizer, optimizer_obj, train_size),
        "parse_failures": {
            "baseline_eval": baseline_parse_summary,
            "optimized_eval": optimized_parse_summary,
        },
        "per_example_results": optimized_per_example,
        "program_state_path": program_state_path,
    }
    json_path = os.path.join(out_dir, f"{seed}.json")
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"  Results saved to   : {json_path}")
    print(f"{'=' * 55}\n")

    # -----------------------------------------------------------------------
    # 9. Optional: qualitative trajectory comparison
    # -----------------------------------------------------------------------
    if sample_trajectory:
        _print_trajectory_comparison(student, optimized_program, optimizer, testset)


def _print_trajectory_comparison(baseline_agent, optimized_agent, optimizer_name, testset):
    """
    Runs both agents on a single sampled test question and prints their
    full trajectories side-by-side for qualitative inspection.
    """
    example = random.choice(testset[:50])
    question = example.question
    gold = example.answer

    print(f"\n{'=' * 55}")
    print("  Trajectory Comparison")
    print(f"{'=' * 55}")
    print(f"  Question : {question}")
    print(f"  Gold     : {gold}\n")

    for label, agent in [("BASELINE (0-shot)", baseline_agent), (optimizer_name.upper(), optimized_agent)]:
        print(f"  --- {label} ---")
        try:
            pred = agent(question=question)
            traj = pred.trajectory
            for k, v in traj.items():
                print(f"    [{k}] {str(v)[:200]}")
            print(f"  => Predicted answer: {pred.answer}\n")
        except Exception as e:
            print(f"  => Error: {e}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HotPotQA ReAct — prompt-optimizer comparison (ClusterFewshot / MIPROv2 / BFRS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="LM model name (e.g. gemini/gemini-2.5-flash, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--optimizer", type=str, default="clusterfs",
                        choices=["clusterfs", "miprov2", "bfrs"],
                        help="Prompt optimizer to use")
    parser.add_argument("--colbert-url", type=str, default=COLBERT_DEFAULT_URL,
                        help="ColBERTv2 API endpoint")
    parser.add_argument("--sglang-port", type=int, default=None,
                        help="sglang server port for local HF models (e.g. 7501)")
    parser.add_argument("--train-size", type=int, default=500,
                        help="Number of training examples")
    parser.add_argument("--dev-size", type=int, default=200,
                        help="Number of validation examples")
    parser.add_argument("--test-size", type=int, default=500,
                        help="Number of test examples")
    parser.add_argument("--max-iters", type=int, default=20,
                        help="Maximum ReAct steps per question")
    parser.add_argument("--encoder-device", type=str, default="cpu",
                        help="Device for SentenceTransformer encoders (cpu / cuda / cuda:0)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline evaluation only (skip optimization)")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Disable matplotlib cluster visualizations (ClusterFewshot only)")
    parser.add_argument("--sample-trajectory", action="store_true",
                        help="Print a qualitative trajectory comparison after evaluation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: derived from current time)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base directory for structured JSON result files")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())

    main(
        model=args.model,
        optimizer=args.optimizer,
        colbert_url=args.colbert_url,
        sglang_port=args.sglang_port,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        max_iters=args.max_iters,
        encoder_device=args.encoder_device,
        baseline=args.baseline,
        apply_visuals=not args.no_visuals,
        sample_trajectory=args.sample_trajectory,
        seed=seed,
        results_dir=args.results_dir,
    )
