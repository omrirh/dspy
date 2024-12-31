import time
import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.clients.huggingface import HFProvider
from programs import BasicMH

dspy.settings.experimental = True

# Define local Llama model endpoint for training
sglang_port = 7501
sglang_url = f"http://localhost:{sglang_port}/v1"
lm = dspy.LM(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    api_base=sglang_url,
    api_key="local",
    provider=HFProvider(),
)
dspy.configure(lm=lm)

# Prepare the HotPotQA dataset
TRAIN_SIZE = 1000
DEV_SIZE = 500
TESTSET_SIZE = 1500
dataset = HotPotQA(test_size=TESTSET_SIZE, only_hard_examples=True)
AVOID_INPUT_TEST = "The Gay Nigger Association of America"  # Intercepted as inappropriate example
trainset = [x.with_inputs('question') for x in dataset.train][:TRAIN_SIZE]
devset = [x.with_inputs('question') for x in dataset.dev][TRAIN_SIZE:DEV_SIZE+TRAIN_SIZE]
testset = [x.with_inputs('question') for x in dataset.test if AVOID_INPUT_TEST not in x.question][:TESTSET_SIZE]

# Set up the metric and evaluation tool
NUM_THREADS = 12
metric = dspy.evaluate.answer_exact_match
evaluate_dev = Evaluate(devset=devset, metric=metric, num_threads=NUM_THREADS, display_progress=True,
                        provide_traceback=True)
evaluate_test = Evaluate(devset=testset, metric=metric, num_threads=NUM_THREADS, display_progress=True,
                         provide_traceback=True)

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

prompt_optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
    num_candidate_programs=6,
    num_threads=6
)

better_together = BetterTogether(
    metric=metric,
    weight_optimizer=weight_optimizer,
    prompt_optimizer=prompt_optimizer,
    seed=RANDOM_SEED
)

# Sample a smaller dataset for quick testing
# TODO: Use full trainset after getting a stable run with results.
small_trainset = trainset[:10]

# Run the BetterTogether optimization
with dspy.context(lm=lm, rm=retriever):
    optimized_program = better_together.compile(
        student=BasicMH(),
        trainset=small_trainset,
        strategy="w -> p",
        valset_ratio=0.1
    )

# Evaluate accuracy and output the results
print("[BetterTogether x HotPotQA x w -> p] Calculating experiment program results...")
accuracy_dev = evaluate_dev(optimized_program)
accuracy_test = evaluate_test(optimized_program)
print(f"Experiment Accuracy:\nValidation set:\t{accuracy_dev}\nTest set:\t{accuracy_test}")
