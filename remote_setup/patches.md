## Required patches (third party)
### LiteLLLM
- Provide `text` instead of `inputs` in [huggingface_restapi.py](https://github.com/BerriAI/litellm/blob/ae385cfcdcc891b23cd99a10387635f705193752/litellm/llms/huggingface_restapi.py#L558)
- Provide `text` instead of `generated_text` in [huggingface_restapi.py](https://github.com/BerriAI/litellm/blob/ae385cfcdcc891b23cd99a10387635f705193752/litellm/llms/huggingface_restapi.py#L356)
- This examples fails training in HotPotQA using MultiHop program: `Example({'question': 'Some performance of Take a Bow took place near the bank of which river?'}) (input_keys={'question'})`