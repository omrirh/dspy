import dspy
from datasets import load_dataset
from dspy.teleprompt.pez import BootstrapFewShotWithPEZ
from dspy.teleprompt.finetune import BootstrapFinetune
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.retrieve import Retrieve
from dsp.utils.utils import deduplicate

# Load the HotPotQA dataset
dataset = load_dataset("hotpot_qa", "fullwiki")
trainset = dataset['train']


def pez_metric(gold, prediction):
    """
    Evaluates the performance of the model given the optimized prompt and prediction.
    """
    is_correct = gold['answer'] == prediction['answer']
    return 1.0 if is_correct else 0.0


# Load teacher model (RoBERTa-large for prompt optimization and finetuning)
teacher_model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)


# Define a DSPy program for multi-hop reasoning
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

# Step 1: Compile the HotPotQA program with few-shot optimization via PEZ
compiled_program = fewshot_optimizer.compile(
    student=hotpotqa_program,  # Pass the HotPotQA program instance for prompt optimization
    teacher=teacher_model,  # Teacher model (e.g., RoBERTa-large)
    trainset=trainset,  # HotPotQA dataset
    restrict=[seed for seed in range(0, num_candidate_programs)]
)

# Step 2: Finetune the student model after prompt optimization
finetune_optimizer = BootstrapFinetune(
    metric=pez_metric,  # Same evaluation metric
)

# Fine-tune using the optimized prompts from PEZ step
finetuned_program = finetune_optimizer.compile(
    student=compiled_program,  # Use the optimized program from Step 1
    teacher=teacher_model,  # Same teacher model
    trainset=trainset,  # Use the same training set
    valset=None,  # Validation set (can be added separately if available)
    target=teacher_model_name,  # Model name for finetuning
    bsize=16,  # Batch size for finetuning
    accumsteps=2,  # Accumulation steps
    lr=5e-5,  # Learning rate for finetuning
    epochs=3  # Number of finetuning epochs
)

print("Few-shot optimization and finetuning completed.")
