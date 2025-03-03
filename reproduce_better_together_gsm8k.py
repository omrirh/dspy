import time
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.cluster_fewshot import ClusterFewshot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
# from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.clients.huggingface import HFProvider
from programs import CoT

dspy.settings.experimental = True
RANDOM_SEED = int(time.time())

# Prepare the GSM8K dataset
dataset = GSM8K()
TRAINSET_SIZE = 1000
DEVSET_SIZE = 500
TESTSET_SIZE = 1319
AVOID_INPUT_TRAIN = 'Jack is mad at his neighbors'
AVOID_INPUT_TEST = 'Michael is racing his horse'
trainset = [x.with_inputs('question') for x in dataset.train if AVOID_INPUT_TRAIN not in x.question][:TRAINSET_SIZE]
devset = [x.with_inputs('question') for x in dataset.dev if AVOID_INPUT_TRAIN not in x.question][TRAINSET_SIZE:TRAINSET_SIZE+DEVSET_SIZE]
testset = [x.with_inputs('question') for x in dataset.test if AVOID_INPUT_TEST not in x.question][:TESTSET_SIZE]

# Define local Llama model endpoint for training
sglang_port = 7501
sglang_url = f"http://localhost:{sglang_port}/v1"
model_name = "Meta-Llama-3-8B-Instruct"
lm = dspy.LM(
    model=f"meta-llama/{model_name}",
    api_base=sglang_url,
    api_key="local",
    provider=HFProvider(validation_set=devset, validation_metric=gsm8k_metric),
)
dspy.configure(lm=lm)

# Set up the metric and evaluation tool
NUM_THREADS = 12
metric = gsm8k_metric
evaluate_test = Evaluate(
    devset=testset,
    metric=metric,
    num_threads=NUM_THREADS,
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

# prompt_optimizer = BootstrapFewShotWithRandomSearch(
#     metric=metric,
#     max_bootstrapped_demos=3,
#     max_labeled_demos=3,
#     num_candidate_programs=6,
#     num_threads=6
# )
prompt_optimizer = ClusterFewshot(
    metric=metric,
    num_fewshot=3,
    valset_ratio=0.1
)

better_together = BetterTogether(
    metric=metric,
    weight_optimizer=weight_optimizer,
    prompt_optimizer=prompt_optimizer,
    seed=RANDOM_SEED
)

# Sample a smaller dataset for quick testing
small_trainset = trainset[:100]

# Run the BetterTogether optimization
optimization_strategy = "p"
with dspy.context(lm=lm, rm=retriever):
    optimized_program = better_together.compile(
        student=CoT(),
        trainset=small_trainset,
        strategy=optimization_strategy,
        valset_ratio=0.1
    )

# Evaluate accuracy and output the results
print(f"[BetterTogether x GSM8K x {model_name} x {optimization_strategy}] Calculating experiment program results...")
accuracy_test = evaluate_test(optimized_program)
print(f"Experiment Accuracy:\n"
      f"Test set:\t{accuracy_test}")
