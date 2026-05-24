import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.evaluate import Evaluate
from programs import (
    CoT,
    BasicMH,
    IrisProgram,
    CropRecommender,
    RetrievalFewshotCoT,
    RetrievalFewshotMH,
    RetrievalFewshotIrisProgram,
)
from dspy.datasets import HotPotQA, IrisDataset
from dspy.datasets.crop_recommendation import (
    CropRecommendationDataset,
    crop_recommendation_metric,
    create_crop_numeric_encoder,
)
from remote_setup.utils import assign_local_lm
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.clusterfewshot import (
    ClusterFewshot,
    create_sentence_transformer_encoder,
    create_numeric_encoder,
)
from dspy.teleprompt.retrieval_fewshot import RetrievalFewshot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch

import logging

logger = logging.getLogger(__name__)

dspy.settings.experimental = True
QA_DATASETS = ["gsm8k", "hotpotqa"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sum_tokens(history_slice):
    """Sum prompt/completion/total tokens across a GLOBAL_HISTORY slice."""
    prompt, completion = 0, 0
    for entry in history_slice:
        usage = entry.get("usage", {})
        prompt += usage.get("prompt_tokens", 0) or 0
        completion += usage.get("completion_tokens", 0) or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _print_token_budget(label, counts):
    p, c, t = counts["prompt_tokens"], counts["completion_tokens"], counts["total_tokens"]
    print(f"  {label:<28} prompt={p:>10,}  completion={c:>8,}  total={t:>10,}")


def _build_per_example_results(eval_results):
    results = []
    for idx, (example, prediction, score) in enumerate(eval_results):
        gold = example.get("answer", example.get("variety", ""))
        results.append({
            "idx": idx,
            "gold": gold,
            "predicted": getattr(prediction, "answer", None),
            "score": float(score) if score is not None else 0.0,
        })
    return results


def _load_baseline_score(results_dir, dataset_name, model_basename, canonical_seed=100):
    """Load baseline score from the canonical baseline JSON if it exists."""
    if not results_dir:
        return None
    bl_path = os.path.join(results_dir, dataset_name, model_basename, "baseline", f"{canonical_seed}.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            return json.load(f)["scores"].get("baseline")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dataset, prompt_optimizer, strategy, model, baseline=False, count_total_tokens=False, seed=None, results_dir=None):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    model_basename = model.split("/")[-1]

    test_size = 0
    train_size = 1000
    dev_size = 500
    metric = None
    student = None
    devset = None
    task_type = None
    exclude_examples = []

    dataset_name = dataset
    if dataset_name == "gsm8k":
        dataset = GSM8K()
        exclude_examples = ["Jack is mad at his neighbors", "John plans to sell all his toys", "Sandy's goal is to drink"]
        devset = [x.with_inputs('question') for x in dataset.dev if not any(ex in x.question for ex in exclude_examples)][train_size:train_size + dev_size]
        test_size = 1319  # According to BetterTogether report
        metric = gsm8k_metric
        task_type = "arithmetic"
        student = CoT()

    elif dataset_name == "hotpotqa":
        dataset = HotPotQA(only_hard_examples=True)
        exclude_examples = [
            "beat, torture, and sexually assault",
            "Anti-pedophile activism advocates for victims",
            "hosting a video of the murder of an international student",
            "contemporary scholars likens to Ilminism",
            "Joseph Druce murder John Geoghan",
            "George Pell first sexually assault a 12 year old boy",
            "The Gay Nigger Association of America",
            "insertion and thrusting of the erect penis into a person's anus",
        ]
        devset = [x.with_inputs('question') for x in dataset.dev if not any(ex in x.question for ex in exclude_examples)][train_size:train_size + dev_size]
        test_size = 1500  # According to BetterTogether report
        metric = dspy.evaluate.answer_exact_match
        task_type = "multihop"
        student = BasicMH()

    elif dataset_name == "iris":
        dataset = IrisDataset()
        metric = dspy.evaluate.answer_exact_match
        task_type = "classification"
        student = IrisProgram()
        trainset, devset, testset = dataset.get_data_splits()

    elif dataset_name == "crop_recommendation":
        dataset = CropRecommendationDataset(csv_path="Crop_recommendation_5features.csv")
        metric = crop_recommendation_metric
        task_type = "classification"
        student = CropRecommender()
        trainset, devset, testset = dataset.get_data_splits()

    if dataset_name in QA_DATASETS:
        trainset = [x.with_inputs('question') for x in dataset.train if
                    not any(ex in x.question for ex in exclude_examples)][:train_size]
        testset = [x.with_inputs('question') for x in dataset.test if
                   not any(ex in x.question for ex in exclude_examples)][:test_size]

    if model in dspy.clients.huggingface._HF_MODELS:
        sglang_port = 7501
        sglang_url = f"http://localhost:{sglang_port}/v1"
        lm = assign_local_lm(
            model=model,
            api_base=sglang_url,
            provider=HFProvider(validation_set=devset, validation_metric=metric)
        )
    else:  # Currently supports Gemini via API
        lm = dspy.LM(model, api_key=os.getenv("GEMINI_API_KEY"))
        dspy.configure(lm=lm)

    # Set up the metric and evaluation tool
    evaluate_test = Evaluate(
        devset=testset,
        metric=metric,
        num_threads=8,
        display_progress=True,
        display_table=False
    )

    # Retriever model as local ColBERTv2
    COLBERT_V2_ENDPOINT = "http://localhost:8894/api/search"
    retriever = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)
    dspy.configure(rm=retriever)

    # Initialize the BetterTogether class with optimizers
    train_kwargs = {}
    adapter = dspy.ChatAdapter()

    weight_optimizer = BootstrapFinetune(
        metric=metric,
        multitask=True,
        train_kwargs=train_kwargs,
        adapter=adapter,
    )

    prompt_optimizer_name = prompt_optimizer
    if prompt_optimizer_name == "bfrs":
        prompt_optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            num_candidate_programs=6,
            num_threads=6
        )

    if prompt_optimizer_name in ("clusterfs", "retrievalfs"):
        # Initialize semantic encoders based on dataset/task type
        if dataset_name == "crop_recommendation":
            semantic_encoders = [create_crop_numeric_encoder()]
        elif task_type == "classification":
            # Generic numeric encoder for other classification tasks (e.g., Iris)
            semantic_encoders = [create_numeric_encoder()]
        else:
            # SentenceTransformer encoders for text-based tasks (QA, arithmetic, etc.)
            semantic_encoders = [
                create_sentence_transformer_encoder("Qwen/Qwen3-Embedding-0.6B"),
                create_sentence_transformer_encoder("sentence-transformers/all-mpnet-base-v2"),
                create_sentence_transformer_encoder("sentence-transformers/gtr-t5-base"),
                create_sentence_transformer_encoder("BAAI/bge-large-en-v1.5"),
            ]

    if prompt_optimizer_name == "clusterfs":
        prompt_optimizer = ClusterFewshot(
            metric=metric,
            task_type=task_type,
            semantic_encoders=semantic_encoders,
            apply_visuals=True,
        )

    if prompt_optimizer_name == "retrievalfs":
        retrieval_class_map = {
            "arithmetic": RetrievalFewshotCoT,
            "multihop": RetrievalFewshotMH,
            "classification": RetrievalFewshotIrisProgram,
        }
        prompt_optimizer = RetrievalFewshot(
            metric=metric,
            task_type=task_type,
            retrieval_program_class=retrieval_class_map[task_type],
            semantic_encoders=semantic_encoders,
            n_shots=3,
            retrieval_strategy="mmr",
            mmr_lambda=0.8,
        )

    if prompt_optimizer_name == "miprov2":
        prompt_optimizer = MIPROv2(
            metric=metric,
            auto="medium",
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            num_threads=6,
        )

    # -----------------------------------------------------------------------
    # Baseline mode — evaluate student directly, save results, return
    # -----------------------------------------------------------------------
    if baseline:
        experiment_header = f"[BASELINE x {dataset_name} x {model}]"
        print(f"{experiment_header}\nRunning baseline evaluation (no optimization)...")
        eval_start = time.time()
        history_before_eval = len(GLOBAL_HISTORY)
        accuracy_test, eval_results = evaluate_test(student, return_outputs=True)
        eval_runtime = time.time() - eval_start
        history_after_eval = len(GLOBAL_HISTORY)

        per_example = _build_per_example_results(eval_results)

        print(f"\nScore:\t{accuracy_test}\nRuntime:\t{eval_runtime:.2f}")

        if count_total_tokens:
            eval_tok = _sum_tokens(GLOBAL_HISTORY[history_before_eval:history_after_eval])
            print(f"\nToken budget  [{experiment_header}]")
            print(f"  {'Phase':<28} {'prompt':>15}  {'completion':>13}  {'total':>15}")
            print(f"  {'-' * 68}")
            _print_token_budget("Baseline evaluation", eval_tok)
            print(f"  {'-' * 68}")

        if results_dir:
            out_dir = os.path.join(results_dir, dataset_name, model_basename, "baseline")
            os.makedirs(out_dir, exist_ok=True)
            eval_tok = _sum_tokens(GLOBAL_HISTORY[history_before_eval:history_after_eval])
            result_json = {
                "schema_version": "2.0",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_sha": _get_git_sha(),
                "dspy_version": getattr(dspy, "__version__", "unknown"),
                "config": {
                    "model": model,
                    "dataset": dataset_name,
                    "optimizer": "baseline",
                    "strategy": strategy,
                    "seed": seed,
                    "train_size": len(trainset),
                    "dev_size": len(devset),
                    "test_size": len(testset),
                },
                "scores": {"baseline": accuracy_test, "optimized": None, "delta_pp": None},
                "timing_seconds": {
                    "compile_total": None,
                    "eval": round(eval_runtime, 2),
                    "total": round(eval_runtime, 2),
                },
                "token_usage": {"eval": eval_tok},
                "per_example_results": per_example,
            }
            json_path = os.path.join(out_dir, f"{seed}.json")
            with open(json_path, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"Results saved to: {json_path}")
        return

    # -----------------------------------------------------------------------
    # Standard BetterTogether optimization
    # -----------------------------------------------------------------------
    better_together = BetterTogether(
        metric=metric,
        weight_optimizer=weight_optimizer,
        prompt_optimizer=prompt_optimizer,
        seed=seed
    )

    history_before_compile = len(GLOBAL_HISTORY)
    compile_start = time.time()
    with dspy.context(lm=lm, rm=retriever):
        optimized_program = better_together.compile(
            student=student,
            trainset=trainset,
            strategy=strategy,
            valset_ratio=0.1
        )
    compile_runtime = time.time() - compile_start
    history_after_compile = len(GLOBAL_HISTORY)

    experiment_header = f"[BetterTogether x {dataset_name} x {model} x {strategy} x {prompt_optimizer_name.upper()}]"

    # Report collected demonstrations
    final_fewshot_size = len(optimized_program.named_predictors()[0][1].demos)
    num_predictors = len(optimized_program.named_predictors())
    print(f"{experiment_header}\nDemonstrations collected ({final_fewshot_size} in total for {num_predictors} predictors):\n")
    for name, predictor in optimized_program.named_predictors():
        print(f"'{name}' predictor demos: {predictor.demos}\n")

    # Evaluate optimized program
    print(f"{experiment_header}\nCalculating experiment program results...")
    eval_start = time.time()
    history_before_eval = len(GLOBAL_HISTORY)
    accuracy_test, eval_results = evaluate_test(optimized_program, return_outputs=True)
    eval_runtime = time.time() - eval_start
    history_after_eval = len(GLOBAL_HISTORY)
    total_runtime = compile_runtime + eval_runtime

    per_example = _build_per_example_results(eval_results)

    # Load baseline score for delta (canonical seed 100)
    baseline_score = _load_baseline_score(results_dir, dataset_name, model_basename)
    delta = round(accuracy_test - baseline_score, 2) if baseline_score is not None else None
    sign = "+" if delta is not None and delta >= 0 else ""

    print(f"\nScore:\t{accuracy_test}\nRuntime:\t{total_runtime:.2f}")
    if delta is not None:
        print(f"Delta vs baseline:\t{sign}{delta:.2f}pp")

    if count_total_tokens:
        opt_tok = _sum_tokens(GLOBAL_HISTORY[history_before_compile:history_after_compile])
        eval_tok = _sum_tokens(GLOBAL_HISTORY[history_before_eval:history_after_eval])
        all_tok = _sum_tokens(GLOBAL_HISTORY[history_before_compile:history_after_eval])
        print(f"\nToken budget  [{experiment_header}]")
        print(f"  {'Phase':<28} {'prompt':>15}  {'completion':>13}  {'total':>15}")
        print(f"  {'-' * 68}")
        _print_token_budget("Optimization", opt_tok)
        _print_token_budget("Final evaluation", eval_tok)
        _print_token_budget("Total e2e", all_tok)
        print(f"  {'-' * 68}")

    if results_dir:
        out_dir = os.path.join(results_dir, dataset_name, model_basename, prompt_optimizer_name)
        os.makedirs(out_dir, exist_ok=True)
        compile_tok = _sum_tokens(GLOBAL_HISTORY[history_before_compile:history_after_compile])
        eval_tok = _sum_tokens(GLOBAL_HISTORY[history_before_eval:history_after_eval])
        result_json = {
            "schema_version": "2.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "dspy_version": getattr(dspy, "__version__", "unknown"),
            "config": {
                "model": model,
                "dataset": dataset_name,
                "optimizer": prompt_optimizer_name,
                "strategy": strategy,
                "seed": seed,
                "train_size": len(trainset),
                "dev_size": len(devset),
                "test_size": len(testset),
            },
            "scores": {
                "baseline": baseline_score,
                "optimized": accuracy_test,
                "delta_pp": delta,
            },
            "timing_seconds": {
                "compile_total": round(compile_runtime, 2),
                "eval": round(eval_runtime, 2),
                "total": round(total_runtime, 2),
            },
            "token_usage": {"compile": compile_tok, "eval": eval_tok},
            "per_example_results": per_example,
        }
        json_path = os.path.join(out_dir, f"{seed}.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BetterTogether experiment argument parser")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--prompt-optimizer", type=str, required=True, help="Name of the prompt optimizer")
    parser.add_argument("--strategy", type=str, required=True, help="Desired optimization strategy (e.g. p -> w -> p)")
    parser.add_argument("--model", type=str, required=True, help="Name of Language Model")
    parser.add_argument("--baseline", action="store_true", help="Run in baseline mode (skip optimization, evaluate student program directly)")
    parser.add_argument("--count-total-tokens", action="store_true", help="Print a per-phase token budget summary at the end of the run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: derived from current time)")
    parser.add_argument("--results-dir", type=str, default=None, help="Base directory for structured JSON result files (e.g. results_bt)")
    args = parser.parse_args()

    main(args.dataset, args.prompt_optimizer, args.strategy, args.model, args.baseline, args.count_total_tokens, args.seed, args.results_dir)
