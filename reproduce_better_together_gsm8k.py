import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.clients.huggingface import HFProvider
from programs import CoT

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

# Prepare the HotPotQA dataset (devset max size = 300, trainset max size = 200)
dataset = GSM8K()
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# Set up the metric and evaluation tool
NUM_THREADS = 12
metric = gsm8k_metric
evaluate = Evaluate(
    devset=devset[:],
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
    seed=2023
)

# Sample a smaller dataset for quick testing
# TODO: Use full trainset after getting a stable run with results.
# small_trainset = trainset[:50]

# Run the BetterTogether optimization
with dspy.context(lm=lm, rm=retriever):
    optimized_program = better_together.compile(
        student=CoT(),
        trainset=trainset,
        strategy="p -> w",
        valset_ratio=0.1
    )

# Evaluate accuracy on validation (dev) set and output the results
accuracy = evaluate(optimized_program)
print(f"Experiment Accuracy: {accuracy}%")