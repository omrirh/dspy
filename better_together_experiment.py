import time
import dspy
from dspy.evaluate import Evaluate
from programs import CoT, BasicMH, IrisProgram
from dspy.datasets import HotPotQA, IrisDataset
from remote_setup.utils import assign_local_lm
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.cluster_fewshot import ClusterFewshot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch

import logging

logger = logging.getLogger(__name__)

dspy.settings.experimental = True
RANDOM_SEED = int(time.time())
QA_DATASETS = ["gsm8k", "hotpotqa"]


def main(dataset, prompt_optimizer, strategy, model):
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
        import os

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

    if prompt_optimizer_name == "clusterfs":
        prompt_optimizer = ClusterFewshot(
            metric=metric,
            task_type=task_type,
            use_target_model_embeddings=("w -> p" in strategy),
        )

    if prompt_optimizer_name == "miprov2":
        prompt_optimizer = MIPROv2(
            metric=metric,
            auto="medium",
            # TODO: should we add num_threads=6?
        )

    if prompt_optimizer_name == "gepa":
        prompt_optimizer = dspy.GEPA(
            metric=metric,
            auto="medium",
            reflection_lm=lm,
            instruction_proposer=None,
            # TODO: should we add num_threads=6?
        )

    better_together = BetterTogether(
        metric=metric,
        weight_optimizer=weight_optimizer,
        prompt_optimizer=prompt_optimizer,
        seed=RANDOM_SEED
    )

    # Run the BetterTogether optimization
    start_time = time.time()
    with dspy.context(lm=lm, rm=retriever):
        optimized_program = better_together.compile(
            student=student,
            trainset=trainset,
            strategy=strategy,
            valset_ratio=0.1
        )

    end_time = time.time()
    runtime = end_time - start_time

    experiment_header = f"[BetterTogether x {dataset_name} x {model} x {strategy} x {prompt_optimizer_name.upper()}]"

    # Report collected demonstrations
    final_fewshot_size = len(optimized_program.named_predictors()[0][1].demos)
    num_predictors = len(optimized_program.named_predictors())
    print(f"{experiment_header}\nDemonstrations collected ({final_fewshot_size} in total for {num_predictors} predictors):\n")
    for name, predictor in optimized_program.named_predictors():
        print(f"'{name}' predictor demos: {predictor.demos}\n")

    # Evaluate accuracy and output the results
    print(f"{experiment_header}\nCalculating experiment program results...")
    accuracy_test = evaluate_test(optimized_program).score  # DSPy 3.0 supports only EvaluationResult objects with score attribute.
    print(f"\nScore:\t{accuracy_test}\n"
          f"Runtime:\t{runtime:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BetterTogether experiment argument parser")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--prompt-optimizer", type=str, required=True, help="Name of the prompt optimizer")
    parser.add_argument("--strategy", type=str, required=True, help="Desired optimization strategy (e.g. p -> w -> p)")
    parser.add_argument("--model", type=str, required=True, help="Name of Language Model")
    args = parser.parse_args()

    main(args.dataset, args.prompt_optimizer, args.strategy, args.model)

    # # for debugging
    # dataset = "gsm8k"
    # prompt_optimizer = "clusterfs"
    # strategy = "p"
    # model = "meta-llama/Llama-3.2-3B-Instruct"

    # main(dataset, prompt_optimizer, strategy, model)
