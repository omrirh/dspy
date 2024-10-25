import argparse
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.utils.utils import deduplicate

# Parse command-line arguments (local Llama model path)
parser = argparse.ArgumentParser(description="Run BetterTogether experiment with specified Llama model path.")
parser.add_argument('--llama-model-path', type=str, required=True, help="Path to the Llama model weights")
args = parser.parse_args()

# Step 1: Configure the LM and Retriever with specified Llama model path
llamaChat = dspy.HFModel(model=args.llama_model_path, hf_device_map="auto")
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(rm=colbertv2, lm=llamaChat)

# Step 2: Load a sample of labeled HotPotQA data for BetterTogether fine-tuning
dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=1000, test_size=0)
trainset = [x.with_inputs('question', 'answer') for x in dataset.train]
devset = [x.with_inputs('question', 'answer') for x in dataset.dev]
testset = [x.with_inputs('question', 'answer') for x in dataset.test]

print("Dataset sizes:", len(trainset), len(devset), len(testset))


# Step 3: Define the multi-hop reasoning program
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


# Step 4: Compile the program using Llama2-13b-chat and apply BFRS optimization
RECOMPILE_INTO_LLAMA_FROM_SCRATCH = True
NUM_THREADS = 24
metric_EM = dspy.evaluate.answer_exact_match

if RECOMPILE_INTO_LLAMA_FROM_SCRATCH:
    tp = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
    basicmh_bs = tp.compile(BasicMH(), trainset=trainset[:50], valset=trainset[50:200])

    ensemble = [prog for *_, prog in basicmh_bs.candidate_programs[:4]]

    for idx, prog in enumerate(ensemble):
        prog.save(f'checkpoints/multihop_llama213b_{idx}.json')
else:
    ensemble = []
    for idx in range(4):
        prog = BasicMH()
        prog.load(f'checkpoints/multihop_llama213b_{idx}.json')
        ensemble.append(prog)

# Step 5: Evaluate the Llama program on the dev set
llama_program = ensemble[0]
evaluate_hotpot = Evaluate(devset=devset[:1000], metric=metric_EM, num_threads=NUM_THREADS, display_progress=True)
score_llama = evaluate_hotpot(llama_program)
print(f"Llama Program Average Metric: {score_llama * 100:.2f}%")

# Step 6: Prepare labeled data for T5-Large finetuning (BetterTogether approach)
RECOMPILE_INTO_T5_FROM_SCRATCH = True

if RECOMPILE_INTO_T5_FROM_SCRATCH:
    config = dict(target='t5-large', epochs=2, bf16=True, bsize=6, accumsteps=2, lr=5e-5)
    tp = BootstrapFinetune(metric=metric_EM)
    t5_program = tp.compile(BasicMH(), teacher=ensemble, trainset=trainset[:200], **config)

    # Disable chain-of-thought prompting for faster predictions
    for p in t5_program.predictors():
        p.activated = False
else:
    t5_program = BasicMH()
    ckpt_path = "colbert-ir/dspy-Oct11-T5-Large-MH-3k-v1"
    LM = dspy.HFModel(checkpoint=ckpt_path, model='t5-large')

    for p in t5_program.predictors():
        p.lm = LM
        p.activated = False

# Step 7: Evaluate the T5-Large multihop program on the dev set
score_t5 = evaluate_hotpot(t5_program)
print(f"T5 Program Average Metric: {score_t5 * 100:.2f}%")
