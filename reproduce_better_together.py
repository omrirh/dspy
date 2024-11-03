# Import necessary libraries and set environment variables
import os
import time
import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dsp.utils.utils import deduplicate

# Set environment variables (Update these with actual paths/keys if needed)
# os.environ["DSPY_CACHEDIR"] = "<your-cache-dir>"
# os.environ["OPENAI_API_KEY"] = "<your-api-key>"

# Experimental feature flag
dspy.settings.experimental = True


# Define the program for multi-hop QA
class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(self.num_hops)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for hop in range(self.num_hops):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)
        return self.generate_answer(context=context, question=question).copy(context=context)


# Prepare the HotPotQA dataset
TRAIN_SIZE = 1000
DEV_SIZE = 500
dataset = HotPotQA(train_seed=1, eval_seed=2023, test_size=0, only_hard_examples=True)
trainset = [x.with_inputs('question') for x in dataset.train][:TRAIN_SIZE]
devset = [x.with_inputs('question') for x in dataset.dev][:DEV_SIZE]

# Set up the metric and evaluation tool
NUM_THREADS = 12
metric = dspy.evaluate.answer_exact_match
evaluate = Evaluate(devset=devset, metric=metric, num_threads=NUM_THREADS, display_progress=True)

# Retrieve model endpoint (Update with actual ColBERT endpoint URL)
COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
retriever = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)

# Initialize the BetterTogether class with optimizers
train_kwargs = {"n_epochs": 1}
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

# Sample a smaller dataset for testing
lm = dspy.LM('llama-2-7b-chat')
small_trainset = trainset[:50]  # Sample 50 examples for quick testing

# Run the BetterTogether optimization
with dspy.context(lm=lm, rm=retriever):
    optimized_program = better_together.compile(
        student=BasicMH(),
        trainset=small_trainset,
        strategy="p -> w -> p",
        valset_ratio=0.1
    )

# Evaluate accuracy on devset and output the results
accuracy = evaluate(optimized_program, devset=devset, metric=metric)
print(f"Experiment Accuracy: {accuracy}%")

# Output the fine-tuned models
for predictor in optimized_program.predictors():
    print(predictor.lm.model)
