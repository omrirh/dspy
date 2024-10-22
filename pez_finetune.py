import dspy
import argparse
from datasets import load_dataset
from dspy.teleprompt.pez import BootstrapFewShotWithPEZ
from dspy.teleprompt.finetune import BootstrapFinetune
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.retrieve import Retrieve
from dsp.utils.utils import deduplicate

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run PEZ finetuning with Llama model')
parser.add_argument('--llama-model-path', type=str, required=True, help='Path to the Llama model weights')
args = parser.parse_args()

# Load the HotPotQA dataset
dataset = load_dataset("hotpot_qa", "fullwiki")
trainset = dataset['train']


# Define evaluation metric for PEZ optimization
def pez_metric(gold, prediction):
    is_correct = gold['answer'] == prediction['answer']
    return 1.0 if is_correct else 0.0


# Load the Llama 2 model using HFModel from DSPy
llama_model_path = args.llama_model_path
teacher_model = dspy.HFModel(model=llama_model_path)


# Define the HotPotQA program for multi-hop reasoning
class HotPotQAProgram(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.retrieve = Retrieve(k=passages_per_hop)
        self.generate_query = [ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            passages = self.retrieve(search_query).passages
            context = deduplicate(context + passages)
        return self.generate_answer(context=context, question=question).copy(context=context)


# Instantiate the HotPotQA program
hotpotqa_program = HotPotQAProgram()

# No. of programs to select optimal prompts from
num_candidate_programs = 8

# Initialize the PEZ-based few-shot optimizer
fewshot_optimizer = BootstrapFewShotWithPEZ(
    metric=pez_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_candidate_programs=num_candidate_programs,
    prompt_len=5,
    opt_iters=500,
    lr=5e-5,
    weight_decay=1e-4,
    print_step=50,
    loss_weight=1.0
)

# Compile the HotPotQA program with PEZ optimization
compiled_program = fewshot_optimizer.compile(
    student=hotpotqa_program,
    teacher=teacher_model,  # Use Llama 2 model loaded via HFModel
    trainset=trainset,
    restrict=[seed for seed in range(0, num_candidate_programs)]
)

# Fine-tune the student model after PEZ optimization
finetune_optimizer = BootstrapFinetune(metric=pez_metric)
finetuned_program = finetune_optimizer.compile(
    student=compiled_program,
    teacher=teacher_model,
    trainset=trainset,
    valset=None,
    target=llama_model_path,
    bsize=16,
    accumsteps=2,
    lr=5e-5,
    epochs=3
)

print("Few-shot optimization and finetuning completed.")
