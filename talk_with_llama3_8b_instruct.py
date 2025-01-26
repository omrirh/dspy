import dspy
from dspy.clients.huggingface import HFProvider
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.experimental = True

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


