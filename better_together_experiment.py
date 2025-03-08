import time
import dspy
from programs import CoT, BasicMH
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from remote_setup.utils import assign_local_lm
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.cluster_fewshot import ClusterFewshot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch

dspy.settings.experimental = True
RANDOM_SEED = int(time.time())


def main(dataset, prompt_optimizer, strategy, model):
    test_size = 0
    train_size = 1000
    dev_size = 500
    metric = None
    student = None

    dataset_name = dataset
    if dataset_name == "gsm8k":
        dataset = GSM8K()
        test_size = 1319  # According to BetterTogether report
        metric = gsm8k_metric
        student = CoT()

    elif dataset_name == "hotpotqa":
        dataset = HotPotQA(only_hard_examples=True)
        test_size = 1500  # According to BetterTogether report
        metric = dspy.evaluate.answer_exact_match
        student = BasicMH()

    trainset = [x.with_inputs('question') for x in dataset.train][:train_size]
    devset = [x.with_inputs('question') for x in dataset.dev][train_size:train_size+dev_size]
    testset = [x.with_inputs('question') for x in dataset.test][:test_size]

    # TODO: add support for iris dataset in the future

    sglang_port = 7501
    sglang_url = f"http://localhost:{sglang_port}/v1"
    lm = assign_local_lm(
        model=model,
        api_base=sglang_url,
        provider=HFProvider(validation_set=devset, validation_metric=gsm8k_metric)
    )

    # Set up the metric and evaluation tool
    evaluate_test = Evaluate(
        devset=testset,
        metric=metric,
        num_threads=12,
        display_progress=True,
        display_table=False
    )

    # Retriever model as ColBERTv2
    COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
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
        exclude_demos=True,
        num_threads=1
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
            num_fewshot=3,
            sampling_strategy="central"
        )

    better_together = BetterTogether(
        metric=metric,
        weight_optimizer=weight_optimizer,
        prompt_optimizer=prompt_optimizer,
        seed=RANDOM_SEED
    )

    # Run the BetterTogether optimization
    with dspy.context(lm=lm, rm=retriever):
        optimized_program = better_together.compile(
            student=student,
            trainset=trainset,
            strategy=strategy,
            valset_ratio=0.1
        )

    # Evaluate accuracy and output the results
    print(f"[BetterTogether x {dataset_name} x {model} x {strategy} x {prompt_optimizer_name.upper()}]\n"
          "Calculating experiment program results...")
    accuracy_test = evaluate_test(optimized_program)
    print(f"\nScore:\t{accuracy_test}")


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
    # model = "meta-llama/Meta-Llama-3-8B-Instruct"

    # main(dataset, prompt_optimizer, strategy, model)
