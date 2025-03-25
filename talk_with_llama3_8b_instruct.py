import dspy
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.experimental = True

# Prepare the GSM8K dataset
dataset = GSM8K()
TRAINSET_SIZE = 1000
DEVSET_SIZE = 500
TESTSET_SIZE = 1319
AVOID_INPUT = ['Jack is mad at his neighbors', "John plans to sell all his toys", "Sandy's goal is to drink"]
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
testset = [x.with_inputs('question') for x in dataset.test]

exclude_exampels = []

for ex in trainset:
    if any(exclude_ex in ex.question for exclude_ex in AVOID_INPUT):
        exclude_exampels.append(ex)

for ex in testset:
    if any(exclude_ex in ex.question for exclude_ex in AVOID_INPUT):
        exclude_exampels.append(ex)

if exclude_exampels:
    print(f"Examples to exclude:\n{exclude_exampels}")

# Define local Llama model endpoint for training
sglang_port = 7501
sglang_url = f"http://localhost:{sglang_port}/v1"
lm = dspy.LM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_base=sglang_url,
    api_key="local",
    provider=HFProvider(validation_set=devset, validation_metric=gsm8k_metric),
)
dspy.configure(lm=lm)


def main():
    while True:
        prompt = str(input("How can I assist you?\n#> "))
        output = lm(prompt)
        print(output[0])
        print("\n\n")


if __name__ == '__main__':
    main()
