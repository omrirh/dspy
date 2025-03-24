import time
import dspy
from programs import CoT, BasicMH, IrisProgram
from dspy.datasets import HotPotQA, IrisDataset
from dspy.evaluate import Evaluate
from remote_setup.utils import assign_local_lm
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
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
    exclude_examples = []

    dataset_name = dataset
    if dataset_name == "gsm8k":
        dataset = GSM8K()
        exclude_examples = ["Jack is mad at his neighbors", "John plans to sell all his toys", "Sandy's goal is to drink"]
        devset = [x.with_inputs('question') for x in dataset.dev if not any(ex in x.question for ex in exclude_examples)][train_size:train_size + dev_size]
        test_size = 1319  # According to BetterTogether report
        metric = gsm8k_metric
        student = CoT()

    elif dataset_name == "hotpotqa":
        dataset = HotPotQA(only_hard_examples=True)
        devset = [x.with_inputs('question') for x in dataset.dev if not any(ex in x.question for ex in exclude_examples)][:dev_size]
        test_size = 1500  # According to BetterTogether report
        metric = dspy.evaluate.answer_exact_match
        student = BasicMH()

    elif dataset_name == "iris":
        dataset = IrisDataset()
        metric = dspy.evaluate.answer_exact_match
        student = IrisProgram()
        trainset, devset, testset = dataset.get_data_splits()

    trainset = [x.with_inputs('question') for x in dataset.train if not any(ex in x.question for ex in exclude_examples)][:train_size]
    testset = [x.with_inputs('question') for x in dataset.test if not any(ex in x.question for ex in exclude_examples)][:test_size]

    if dataset_name in QA_DATASETS:
        trainset = [x.with_inputs('question') for x in dataset.train][:train_size]
        testset = [x.with_inputs('question') for x in dataset.test][:test_size]

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
    # TODO: Retriever issue reported at https://github.com/stanfordnlp/dspy/issues/7966
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
    """
    demonstrations set by ClusterFewshot achieving 82% accuracy on GSM8K (standalone mode):
    --------------------------------------------------------------------
    optimized_demos = [dspy.Example({'question': 'Wanda has 62 crayons. Dina has 28 and Jacob has two fewer crayons than Dina. How many crayons do they have in total?',
            'gold_reasoning': 'Jacob has 28 - 2 = <<28-2=26>>26 crayons. You can find the total number of crayons by adding the number of crayons each person has: 26 crayons + 62 crayons + 28 crayons = <<26+62+28=116>>116 crayons',
            'answer': '116', 'reasoning': "Let's start by finding the number of crayons Jacob has. Since Jacob has two fewer crayons than Dina, and Dina has 28 crayons, Jacob has 28 - 2 = 26 crayons.\n\nTo find the total number of crayons they have, we add the number of crayons each of them has: Wanda has 62, Dina has 28, and Jacob has 26. We add these numbers together to get the total:\n\n62 + 28 + 26 = 116"}, input_keys={'question'}),
    dspy.Example({'question': 'There are three times as many girls as boys in the Biology class. The Physics class has 200 students. If the Biology class has half as many students as the Physics class, how many boys are in the Biology class?',
            'gold_reasoning': 'The Biology class has 200/2=<<200/2=100>>100 students. The boys in the Biology class are 1/4*100=<<1/4*100=25>>25 students.',
            'answer': '25',
            'reasoning': "Let's start by finding the number of students in the Biology class. Since it has half as many students as the Physics class, and the Physics class has 200 students, the Biology class has 200 / 2 = 100 students.\n\nSince there are three times as many girls as boys in the Biology class, let's say the number of boys is x. Then, the number of girls is 3x. The total number of students in the Biology class is the sum of boys and girls, which is x + 3x = 4x. Since the total number of students is 100, we can set up the equation 4x = 100 and solve for x.\n\n4x = 100\nx = 25\n\nSo, there are 25 boys in the Biology class."}, input_keys={'question'}),
    dspy.Example({'question': "Macy's is selling shirts that have been reduced to $6.  This price is at 25% of the original price.  What was the original price?",
            'gold_reasoning': 'The original price is x. The discount is 100% - 25% remaining = 75% discount. The discount is 75%, so the original price is x - .75x = .25x. The sale price is $6, so $6 = .25x, which is the same as 6/.25 = x. X = $<<24=24>>24.', 'answer': '24', 'reasoning': 'Let x be the original price. Since the price is reduced to 25% of the original price, we can set up the equation: 6 = 0.25x. To solve for x, we can divide both sides by 0.25: x = 6 / 0.25 = 24.'}, input_keys={'question'})]

    optimized_program = student.deepcopy()
    for _, predictor in optimized_program.named_predictors():
        predictor.demos = optimized_demos
    """

    experiment_header = f"[BetterTogether x {dataset_name} x {model} x {strategy} x {prompt_optimizer_name.upper()}]"

    # Report collected demonstrations
    print(f"{experiment_header}\nDemonstrations collected:\n{optimized_program.named_predictors()[0][1].demos}\n\n")

    # Evaluate accuracy and output the results
    print(f"{experiment_header}\nCalculating experiment program results...")
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
    # strategy = "w"
    # model = "meta-llama/Meta-Llama-3-8B-Instruct"

    # main(dataset, prompt_optimizer, strategy, model)
