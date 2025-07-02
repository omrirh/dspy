import dspy
from dspy.clients.huggingface import HFProvider

dspy.settings.experimental = True


def main(model):
    sglang_port = 7501
    sglang_url = f"http://localhost:{sglang_port}/v1"
    lm = dspy.LM(
        model=model,
        api_base=sglang_url,
        api_key="local",
        provider=HFProvider(validation_set=None, validation_metric=None),
    )
    dspy.configure(lm=lm)

    while True:
        prompt = str(input(f"\n[{model}]\nHow can I assist you?\n#> "))

        if prompt == "exit":
            exit(0)
        output = lm(prompt)
        print(output[0])
        print("\n\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SGLang LM chatbot application")
    parser.add_argument("--model", type=str, required=True, help="Name of Language Model")
    args = parser.parse_args()

    main(args.model)
