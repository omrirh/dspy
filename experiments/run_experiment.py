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
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from dspy.evaluate import Evaluate
from remote_setup.utils import deploy_sglang_model, is_server_up

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_experiment")

dspy.settings.experimental = True
RANDOM_SEED = int(time.time())


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
    if dataset_name == "gsm8k":
        from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

        dataset = GSM8K()
        trainset = [x.with_inputs("question") for x in dataset.train][:train_size]
        valset   = [x.with_inputs("question") for x in dataset.dev][:val_size]
        testset  = [x.with_inputs("question") for x in dataset.test][:test_size]
        metric   = gsm8k_metric

    elif dataset_name == "iris":
        from dspy.datasets.iris import IrisDataset

        dataset  = IrisDataset(seed=0)
        trainset, valset, testset = dataset.get_data_splits()
        trainset = trainset[:train_size]
        valset   = valset[:val_size]
        testset  = testset[:test_size]
        metric   = dspy.evaluate.answer_exact_match

    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    logger.info(
        f"Dataset '{dataset_name}': "
        f"{len(trainset)} train / {len(valset)} val / {len(testset)} test"
    )
    return trainset, valset, testset, metric


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

def build_optimizer(optimizer_name: str, metric, args, gepa_log_dir: str):
    """Instantiate the chosen prompt optimizer."""

    # Reflection LM defaults to the task model itself (self-improving).
    reflection_model = args.reflection_model or args.model
    reflection_lm = dspy.LM(
        reflection_model,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    if optimizer_name == "gepa":
        from dspy.teleprompt.gepa import GEPA

        return GEPA(
            metric=dspy.evaluate.as_gepa_metric(metric),
            auto=args.auto,
            reflection_lm=reflection_lm,
            num_threads=args.num_threads,
            seed=RANDOM_SEED,
            log_dir=gepa_log_dir,
            track_stats=True,
        )

    elif optimizer_name == "gepa_fewshot":
        from dspy.teleprompt.gepa import GEPAFewShot

        return GEPAFewShot(
            metric=dspy.evaluate.as_gepa_metric(metric),
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
    run_tag = (
        f"{args.dataset}__{args.optimizer}__"
        f"{os.path.basename(args.model)}__{args.auto or 'custom'}__"
        f"{time.strftime('%Y-%m-%d_%H-%M')}"
    )
    log_dir      = os.path.join(args.log_dir, run_tag)
    gepa_log_dir = os.path.join(log_dir, "gepa")
    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run: {run_tag} | seed: {RANDOM_SEED}")
    logger.info(f"Args: {vars(args)}")

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "seed": RANDOM_SEED}, f, indent=2)

    # ---- SGLang pre-flight: ensure server is up before building LM ----
    ensure_sglang_server(args.model, args.api_base)

    # ---- Task LM ----
    lm = dspy.LM(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    dspy.configure(lm=lm)

    # ---- Data & program ----
    trainset, valset, testset, metric = build_dataset(
        args.dataset, args.train_size, args.val_size, args.test_size
    )
    student = build_student(args.dataset)

    # ---- Optimize ----
    optimizer = build_optimizer(args.optimizer, metric, args, gepa_log_dir)

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
    results = {
        "run_tag":              run_tag,
        "seed":                 RANDOM_SEED,
        "test_score":           test_score,
        "runtime_opt_s":        runtime_opt,
        "runtime_eval_s":       runtime_eval,
        "optimized_instructions": instructions_report,
        "demos_per_predictor":  demos_report,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    optimized.save(os.path.join(log_dir, "optimized_program.json"))

    print(f"\n{'='*60}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Optimizer  : {args.optimizer}")
    print(f"  Model      : {args.model}")
    print(f"  Budget     : {args.auto}")
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
                        choices=["gepa", "gepa_fewshot", "miprov2"],
                        help="Prompt optimizer")
    parser.add_argument("--model",     required=True,
                        help="Task LM (e.g. meta-llama/Llama-3.2-3B-Instruct)")

    # LM connection (SGLang local server)
    parser.add_argument("--api-base", default="http://localhost:30000/v1",
                        help="OpenAI-compatible endpoint for the task model")
    parser.add_argument("--api-key",  default="local")

    # Reflection LM (GEPA / GEPAFewShot)
    parser.add_argument("--reflection-model", default=None,
                        help="LM for GEPA reflection proposals. Defaults to --model (self-improving).")

    # Budget
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"],
                        help="Budget preset passed to the optimizer")

    # Few-shot (GEPAFewShot / MIPROv2)
    parser.add_argument("--k-demos",                type=int, default=3,
                        help="Demonstrations per candidate (GEPAFewShot only)")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=16)
    parser.add_argument("--max-labeled-demos",      type=int, default=4)
    parser.add_argument("--demo-mutation-strategy", default="metric_based",
                        choices=["random", "metric_based"])

    # Dataset sizes
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size",   type=int, default=100)
    parser.add_argument("--test-size",  type=int, default=300)

    # Misc
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--log-dir",     default="experiments/logs",
                        help="Root directory for run logs and artifacts")

    args = parser.parse_args()
    main(args)
