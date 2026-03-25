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
--max-iters         Max ReAct steps per question (default: 5)
--encoder-device    Device for SentenceTransformer encoders (default: cpu)
--baseline          Skip optimization; evaluate zero-shot agent only
--no-visuals        Disable matplotlib cluster plots
--sample-trajectory Print a qualitative trajectory comparison after evaluation
"""

import os
import time
import logging
import argparse

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

dspy.settings.experimental = True

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
    devset = _filter(dataset.dev)[train_size: train_size + dev_size]
    testset = _filter(dataset.test)[:test_size]

    logger.info(f"Split sizes — train: {len(trainset)}, dev: {len(devset)}, test: {len(testset)}")
    return trainset, devset, testset


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

# Max tokens for generation per model class.
# ReAct steps (thought + tool_name + tool_args) rarely exceed 300 tokens;
# the extract step is ~100 tokens. 1024 is a safe ceiling for all sizes.
# 32B gets 2048 — the larger context window means longer thoughts are common.
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
    as valset so MIPROv2 can do held-out Bayesian trial selection.
    """
    opt = MIPROv2(
        metric=metric,
        auto="medium",
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_threads=6,
    )
    program = opt.compile(
        student=student,
        trainset=trainset,
        valset=devset,
        minibatch=True,
        minibatch_size=25,
        minibatch_full_eval_steps=10,
    )
    return program, opt


def _run_bfrs(student, trainset, metric):
    """
    Runs BootstrapFewShotWithRandomSearch (BFRS) optimization.

    Searches over 6 candidate programs with up to 3 bootstrapped and 3 labeled
    demonstrations, matching the settings used in the BetterTogether experiments
    for a fair comparison.
    """
    opt = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=6,
        num_threads=6,
    )
    program = opt.compile(student=student, trainset=trainset)
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
):
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
    baseline_score = evaluate(student)
    baseline_runtime = time.time() - baseline_start
    logger.info(f"Baseline score: {baseline_score:.2f}%  ({baseline_runtime:.1f}s)")

    if baseline:
        print(f"\n[BASELINE | {model}]")
        print(f"  Score   : {baseline_score:.2f}%")
        print(f"  Runtime : {baseline_runtime:.1f}s")
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
            student, trainset, metric
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
    optimized_score = evaluate(optimized_program)
    optimized_runtime = time.time() - optimized_start

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
    print(f"  Compile runtime    : {compile_runtime:.1f}s")
    print(f"  Eval runtime       : {optimized_runtime:.1f}s")
    print(f"{'=' * 55}\n")

    # -----------------------------------------------------------------------
    # 8. Optional: qualitative trajectory comparison
    # -----------------------------------------------------------------------
    if sample_trajectory:
        _print_trajectory_comparison(student, optimized_program, optimizer, testset)


def _print_trajectory_comparison(baseline_agent, optimized_agent, optimizer_name, testset):
    """
    Runs both agents on a single sampled test question and prints their
    full trajectories side-by-side for qualitative inspection.
    """
    import random
    example = random.choice(testset[:50])
    question = example.question
    gold = example.answer

    print(f"\n{'=' * 55}")
    print(f"  Trajectory Comparison")
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
    parser.add_argument("--max-iters", type=int, default=5,
                        help="Maximum ReAct steps per question")
    parser.add_argument("--encoder-device", type=str, default="cpu",
                        help="Device for SentenceTransformer encoders (cpu / cuda / cuda:0)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline evaluation only (skip optimization)")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Disable matplotlib cluster visualizations (ClusterFewshot only)")
    parser.add_argument("--sample-trajectory", action="store_true",
                        help="Print a qualitative trajectory comparison after evaluation")
    args = parser.parse_args()

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
    )
