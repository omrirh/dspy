"""
Prompt optimization experiment runner.

Supported optimizers : gepa | gepa_fewshot | miprov2
Supported datasets   : gsm8k | iris

Usage
-----
python experiments/run_experiment.py \\
    --dataset gsm8k \\
    --optimizer gepa_fewshot \\
    --model meta-llama/Llama-3.2-3B-Instruct

Run `python experiments/run_experiment.py --help` for full options.

Log structure
-------------
experiments/logs/
  <dataset>__<optimizer>__<model>__<auto>__<date>/
    run.log                  — full console log
    config.json              — all CLI args + seed for reproducibility
    results.json             — test score, runtimes, demo counts, instructions
    optimized_program.json   — saved DSPy module state
    gepa/                    — GEPA internal checkpoints and candidate logs
"""
import argparse
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from dspy.evaluate import Evaluate
from experiments.metrics import (
    BETTER_REFLECTION_PROMPT,
    gsm8k_gepa_metric,
    iris_gepa_metric,
)
from remote_setup.utils import deploy_sglang_model, is_server_up

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_experiment")

dspy.settings.experimental = True
# RANDOM_SEED is set in main() after CLI parsing so --seed is respected.
RANDOM_SEED: int = 0


# ---------------------------------------------------------------------------
# SGLang pre-flight
# ---------------------------------------------------------------------------

def ensure_sglang_server(model: str, api_base: str):
    """
    Parse host:port from *api_base* and verify the SGLang server is live.
    If the server is not responding, attempt to deploy *model* automatically.
    """
    from urllib.parse import urlparse

    parsed = urlparse(api_base)
    host = parsed.hostname or "localhost"
    port = parsed.port or 30000

    if is_server_up(host=host, port=port):
        logger.info(f"SGLang server already running at {api_base}.")
        return

    logger.warning(
        f"SGLang server not detected at {api_base}. "
        f"Attempting to deploy '{model}' automatically ..."
    )
    deploy_sglang_model(
        model_path=model,
        log_file=f"sglang_{os.path.basename(model)}.log",
        port=port,
    )
    logger.info(f"SGLang server for '{model}' is now ready.")


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------

def build_dataset(dataset_name: str, train_size: int, val_size: int, test_size: int):
    """
    Returns (trainset, valset, testset, metric, gepa_metric).

    *metric*      — 2/3-arg DSPy metric used for test-set Evaluate().
    *gepa_metric* — 5-arg GEPA metric used during optimization (rich feedback).
    """
    if dataset_name == "gsm8k":
        from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

        dataset = GSM8K()
        trainset = [x.with_inputs("question") for x in dataset.train][:train_size]
        valset   = [x.with_inputs("question") for x in dataset.dev][:val_size]
        testset  = [x.with_inputs("question") for x in dataset.test][:test_size]
        metric      = gsm8k_metric
        gepa_metric = gsm8k_gepa_metric

    elif dataset_name == "iris":
        from dspy.datasets.iris import IrisDataset

        dataset  = IrisDataset(seed=0)  # fixed shuffle: cross-seed variance reflects optimizer randomness only
        trainset, valset, testset = dataset.get_data_splits()
        trainset = trainset[:train_size]
        valset   = valset[:val_size]
        testset  = testset[:test_size]
        metric      = dspy.evaluate.answer_exact_match
        gepa_metric = iris_gepa_metric

    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    logger.info(
        f"Dataset '{dataset_name}': "
        f"{len(trainset)} train / {len(valset)} val / {len(testset)} test"
    )
    return trainset, valset, testset, metric, gepa_metric


# ---------------------------------------------------------------------------
# Program setup
# ---------------------------------------------------------------------------

def build_student(dataset_name: str):
    from experiments.programs import CoT, IrisProgram

    if dataset_name == "gsm8k":
        return CoT()
    elif dataset_name == "iris":
        return IrisProgram()
    else:
        raise ValueError(f"No program defined for dataset: {dataset_name!r}")


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def build_optimizer(optimizer_name: str, metric, gepa_metric, args, gepa_log_dir: str):
    """
    Instantiate the chosen prompt optimizer.

    *metric*      — 2/3-arg metric for MIPROv2 (unchanged).
    *gepa_metric* — 5-arg GEPA metric with rich feedback (used by GEPA / GEPAFewShot).
    """
    # Reflection LM defaults to the task model itself (self-improving).
    reflection_model = args.reflection_model or args.model
    reflection_lm_kwargs = {}
    if "Qwen3" in reflection_model:
        reflection_lm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.max_tokens is not None:
        reflection_lm_kwargs["max_tokens"] = args.max_tokens
    reflection_lm = dspy.LM(
        reflection_model,
        api_base=args.api_base,
        api_key=args.api_key,
        **reflection_lm_kwargs,
    )

    if optimizer_name in ("gepa", "gepa_merge"):
        from dspy.teleprompt.gepa import GEPA

        # gepa       = Vanilla GEPA, merge disabled (cleaner ablation baseline)
        # gepa_merge = GEPA with merge enabled (upstream default behaviour)
        use_merge = optimizer_name == "gepa_merge"
        return GEPA(
            metric=gepa_metric,
            auto=args.auto,
            reflection_lm=reflection_lm,
            num_threads=args.num_threads,
            seed=RANDOM_SEED,
            log_dir=gepa_log_dir,
            track_stats=True,
            use_merge=use_merge,
            reflection_minibatch_size=args.reflection_minibatch_size,
            reflection_prompt_template=BETTER_REFLECTION_PROMPT,
        )

    elif optimizer_name == "gepa_fewshot":
        from dspy.teleprompt.gepa import GEPAFewShot

        return GEPAFewShot(
            metric=gepa_metric,
            auto=args.auto,
            reflection_lm=reflection_lm,
            num_threads=args.num_threads,
            seed=RANDOM_SEED,
            log_dir=gepa_log_dir,
            track_stats=True,
            k_demos=args.k_demos,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            demo_mutation_strategy=args.demo_mutation_strategy,
            reflection_minibatch_size=args.reflection_minibatch_size,
            reflection_prompt_template=BETTER_REFLECTION_PROMPT,
        )

    elif optimizer_name == "miprov2":
        from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

        return MIPROv2(
            metric=metric,
            auto=args.auto,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            num_threads=args.num_threads,
            seed=RANDOM_SEED,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- Seed — set globally so all random draws in this process are reproducible ----
    global RANDOM_SEED
    RANDOM_SEED = args.seed if args.seed is not None else int(time.time())
    random.seed(RANDOM_SEED)
    try:
        import numpy as np
        np.random.seed(RANDOM_SEED % (2**31))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
    except ImportError:
        pass

    seed_tag = f"seed{RANDOM_SEED}"
    run_tag = (
        f"{args.dataset}__{args.optimizer}__"
        f"{os.path.basename(args.model)}__{args.auto or 'custom'}__"
        f"{seed_tag}__{time.strftime('%Y-%m-%d_%H-%M')}"
    )
    log_dir      = os.path.join(args.log_dir, run_tag)
    gepa_log_dir = os.path.join(log_dir, "gepa")
    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    fh.setLevel(getattr(logging, args.log_level))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run: {run_tag} | seed: {RANDOM_SEED}")
    logger.info(f"Args: {vars(args)}")

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "seed": RANDOM_SEED}, f, indent=2)

    # ---- SGLang pre-flight: ensure server is up before building LM ----
    ensure_sglang_server(args.model, args.api_base)

    # ---- Task LM ----
    # Qwen3 family models are hybrid reasoning models that generate <think> chains by default.
    # Disable thinking for prompt optimization runs — it adds 1000-2000 tokens per call with
    # no benefit for structured tasks like classification.
    lm_kwargs = {}
    if "Qwen3" in args.model:
        lm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.max_tokens is not None:
        lm_kwargs["max_tokens"] = args.max_tokens
    lm = dspy.LM(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        **lm_kwargs,
    )
    dspy.configure(lm=lm)

    # ---- Data & program ----
    trainset, valset, testset, metric, gepa_metric = build_dataset(
        args.dataset, args.train_size, args.val_size, args.test_size
    )
    student = build_student(args.dataset)

    # ---- Optimize (or skip for baseline) ----
    if args.optimizer == "baseline":
        # Baseline: evaluate the unoptimized student program directly.
        logger.info("Optimizer: BASELINE — skipping optimization, evaluating zero-shot.")
        optimized    = student
        runtime_opt  = 0.0
    else:
        optimizer = build_optimizer(args.optimizer, metric, gepa_metric, args, gepa_log_dir)
        logger.info(f"Starting optimization with {args.optimizer.upper()} ...")
        t0 = time.time()
        optimized = optimizer.compile(student, trainset=trainset, valset=valset)
        runtime_opt = time.time() - t0
        logger.info(f"Optimization done in {runtime_opt:.1f}s")

    # ---- Report optimized instructions ----
    logger.info("--- Optimized instructions ---")
    instructions_report = {}
    for name, pred in optimized.named_predictors():
        instr = pred.signature.instructions
        instructions_report[name] = instr
        logger.info(f"  [{name}]\n{instr}\n")

    # ---- Report demos ----
    logger.info("--- Demos per predictor ---")
    demos_report = {}
    for name, pred in optimized.named_predictors():
        n = len(pred.demos) if hasattr(pred, "demos") else 0
        demos_report[name] = n
        logger.info(f"  {name}: {n} demos")

    # ---- Evaluate on test set ----
    evaluate_test = Evaluate(
        devset=testset,
        metric=metric,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=False,
    )
    t1 = time.time()
    test_result  = evaluate_test(optimized)
    runtime_eval = time.time() - t1
    test_score   = test_result.score if hasattr(test_result, "score") else float(test_result)

    logger.info(f"Test accuracy : {test_score:.4f}")
    logger.info(f"Eval runtime  : {runtime_eval:.1f}s")

    # ---- Persist results ----
    # n_demos_total: total demo slots filled across all predictors in the final program.
    n_demos_total = sum(demos_report.values())

    # total_metric_calls: available on optimized.detailed_results when track_stats=True
    # (set for all GEPA / GEPAFewShot runs).  None for baseline and MIPROv2.
    total_metric_calls = None
    detailed = getattr(optimized, "detailed_results", None)
    if detailed is not None:
        total_metric_calls = getattr(detailed, "total_metric_calls", None)
    if total_metric_calls is not None:
        logger.info(f"Total metric calls (optimization): {total_metric_calls}")

    results = {
        # Identification
        "run_tag":    run_tag,
        "seed":       RANDOM_SEED,
        "dataset":    args.dataset,
        "optimizer":  args.optimizer,
        "model":      os.path.basename(args.model),
        "auto":       args.auto,
        # Performance
        "test_score":           test_score,
        "runtime_opt_s":        runtime_opt,
        "runtime_eval_s":       runtime_eval,
        # Optimization cost (GEPA only; None for baseline / MIPROv2)
        "total_metric_calls":   total_metric_calls,
        # Program config
        "optimized_instructions": instructions_report,
        "demos_per_predictor":    demos_report,
        "n_demos_total":          n_demos_total,
        # Hyperparams for traceability
        "train_size":  len(trainset),
        "val_size":    len(valset),
        "test_size":   len(testset),   # actual size, not requested (may differ if dataset is smaller)
        "k_demos":     args.k_demos if args.optimizer == "gepa_fewshot" else 0,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.optimizer != "baseline":
        optimized.save(os.path.join(log_dir, "optimized_program.json"))

    print(f"\n{'='*60}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Optimizer  : {args.optimizer}")
    print(f"  Model      : {args.model}")
    print(f"  Budget     : {args.auto}")
    print(f"  Seed       : {RANDOM_SEED}")
    print(f"  Test acc   : {test_score:.4f}")
    print(f"  Opt time   : {runtime_opt:.1f}s")
    print(f"  Log dir    : {log_dir}")
    print(f"{'='*60}\n")

    return test_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone prompt optimization runner — GEPA / GEPAFewShot / MIPROv2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--dataset",   required=True, choices=["gsm8k", "iris"],
                        help="Benchmark dataset")
    parser.add_argument("--optimizer", required=True,
                        choices=["baseline", "gepa", "gepa_merge", "gepa_fewshot", "miprov2"],
                        help="Prompt optimizer. "
                             "baseline=zero-shot eval; "
                             "gepa=Vanilla GEPA (use_merge=False); "
                             "gepa_merge=GEPA with merge enabled; "
                             "gepa_fewshot=GEPA+FewShot; "
                             "miprov2=MIPROv2.")
    parser.add_argument("--model",     required=True,
                        help="Task LM (e.g. meta-llama/Llama-3.2-3B-Instruct)")

    # LM connection (SGLang local server)
    parser.add_argument("--api-base", default="http://localhost:30000/v1",
                        help="OpenAI-compatible endpoint for the task model")
    parser.add_argument("--api-key",  default="local")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per LM response. Defaults to None (model decides). "
                             "Set explicitly for models prone to verbose output (e.g. 500 for Phi-4-mini-instruct on GSM8K).")

    # Reflection LM (GEPA / GEPAFewShot)
    parser.add_argument("--reflection-model", default=None,
                        help="LM for GEPA reflection proposals. Defaults to --model (self-improving).")

    # Budget
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"],
                        help="Budget preset passed to the optimizer")

    # Few-shot (GEPAFewShot / MIPROv2)
    parser.add_argument("--k-demos",                type=int, default=3,
                        help="Demonstrations per candidate (GEPAFewShot only)")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=16)
    parser.add_argument("--max-labeled-demos",      type=int, default=4)
    parser.add_argument("--demo-mutation-strategy", default="metric_based",
                        choices=["random", "metric_based"])
    parser.add_argument("--reflection-minibatch-size", type=int, default=10,
                        help="Minibatch size for GEPA reflection step")

    # Dataset sizes
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size",   type=int, default=100)
    parser.add_argument("--test-size",  type=int, default=300)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. "
                             "Defaults to None (time-based). "
                             "Pass an explicit value for matrix runs.")

    # Misc
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--log-dir",     default="experiments/logs",
                        help="Root directory for run logs and artifacts")
    parser.add_argument("--log-level",   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity. Use DEBUG to see per-iteration "
                             "demo mutation decisions (op_select / demo_mutate / score_tracker).")

    args = parser.parse_args()
    # Target only the GEPAFewShot mutation logger — setting root to DEBUG would
    # flood output with litellm / httpx / DSPy internals noise.
    logging.getLogger("dspy.teleprompt.gepa.gepa_fewshot").setLevel(
        getattr(logging, args.log_level)
    )
    main(args)
