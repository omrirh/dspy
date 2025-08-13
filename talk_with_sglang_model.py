import dspy
from dspy.datasets.gsm8k import GSM8K
from dspy.datasets.hotpotqa import HotPotQA
from dspy.clients.huggingface import HFProvider
from programs import CoTQuestionClassifier

dspy.settings.experimental = True


def question_type_classifier(dataset, task_type):
    task_type_category_str = f"type_of_{task_type}_skills_labels"
    questions_classifier = CoTQuestionClassifier(task_type=task_type)
    val_questions = [x.with_inputs('question') for x in dataset.train][:10]

    for q in val_questions:
        prediction = questions_classifier(**q.inputs())
        question_category = prediction[task_type_category_str]
        print(f"Question: {q.question}\nClassified type: {question_category}\n\n")

        q[task_type_category_str] = prediction[task_type_category_str]

    print("now here")


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

    dataset = HotPotQA()
    task_type = "multihop"

    question_type_classifier(dataset, task_type)

    # while True:
    #     prompt = str(input(f"\n[{model}]\nHow can I assist you?\n#> "))

    #     if prompt == "exit":
    #         exit(0)
    #     output = lm(prompt)
    #     print(output[0])
    #     print("\n\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SGLang LM chatbot application")
    parser.add_argument("--model", type=str, required=True, help="Name of Language Model")
    args = parser.parse_args()

    main(args.model)
    # main("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
