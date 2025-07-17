import os
import dspy

dspy.settings.experimental = True


def main(model):
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
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
