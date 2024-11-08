## Required patches (third party)
### LiteLLLM
- Provide `text` instead of `inputs` in [huggingface_restapi.py](https://github.com/BerriAI/litellm/blob/ae385cfcdcc891b23cd99a10387635f705193752/litellm/llms/huggingface_restapi.py#L558)
- Provide `text` instead of `generated_text` in [huggingface_restapi.py](https://github.com/BerriAI/litellm/blob/ae385cfcdcc891b23cd99a10387635f705193752/litellm/llms/huggingface_restapi.py#L356)
