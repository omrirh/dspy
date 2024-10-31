import argparse
import torch
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.utils.utils import deduplicate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Parse command-line arguments to get the local model path
parser = argparse.ArgumentParser(description="Run BetterTogether experiment with specified Llama model path.")
parser.add_argument('--llama-model-path', type=str, required=True, help="Path to the Llama model weights")
args = parser.parse_args()

# Step 2: Setup CUDA device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_map = {"": device}

# Step 3: Load local model and tokenizer with Transformers
llama_model_path = args.llama_model_path
tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
model = AutoModelForCausalLM.from_pretrained(llama_model_path).to(device)

# Step 4: Configure the LM and Retriever with the local model
llamaChat = dspy.LM(
    model=llama_model_path,
    hf_device_map=device_map,
    launch_kwargs={"model": model, "tokenizer": tokenizer}
)
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(rm=colbertv2, lm=llamaChat, provider='local')  # Specify 'local' provider to prevent API calls

# Step 5: Load HotPotQA data
dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=1000, test_size=0)
question_str = "question"
answer_str = "answer"
trainset = [x.with_inputs(question_str, answer_str) for x in dataset.train]
devset = [x.with_inputs(question_str, answer_str) for x in dataset.dev]

print("Dataset sizes:\n"
      f"Training set: {len(trainset)}\n"
      f"Validation set: {len(devset)}\n")


# Step 6: Define the multi-hop reasoning program
class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question, answer):
        context = []
        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)
        return self.generate_answer(context=context, question=question).copy(context=context)


# Step 7: Compile with BFRS optimization
NUM_THREADS = 24
metric_EM = dspy.evaluate.answer_exact_match
tp = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
basicmh_bs = tp.compile(BasicMH(), trainset=trainset[:50], valset=trainset[50:200])

ensemble = [prog for *_, prog in basicmh_bs.candidate_programs[:4]]
for idx, prog in enumerate(ensemble):
    prog.save(f'checkpoints/multihop_llama_7b_chat_{idx}.json')

# Step 8: Evaluate the program
llama_program = ensemble[0]
evaluate_hotpot = Evaluate(devset=devset[:1000], metric=metric_EM, num_threads=NUM_THREADS, display_progress=True)
score_llama = evaluate_hotpot(llama_program)
print(f"Llama Program Average Metric: {score_llama * 100:.2f}%")

# Step 9: Fine-tune with T5-Large
config = dict(target='t5-large', epochs=2, bf16=True, bsize=4, accumsteps=1, lr=5e-5)
tp = BootstrapFinetune(metric=metric_EM)
t5_program = tp.compile(BasicMH(), teacher=ensemble, trainset=trainset[:200], **config)

# Disable chain-of-thought prompting for faster predictions
for p in t5_program.predictors():
    p.activated = False

# Step 10: Evaluate the T5-Large program
score_t5 = evaluate_hotpot(t5_program)
print(f"T5 Program Average Metric: {score_t5 * 100:.2f}%")
