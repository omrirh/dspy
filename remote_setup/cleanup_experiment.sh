#!/bin/bash

SGLANG_PIDS=$(pgrep -f sglang)

if [ -z "$SGLANG_PIDS" ]; then
    echo "No sglang processes found."
else
    MODEL_NAME=$(ps -ef | grep sglang | grep -v grep | awk '{print $12}' | head -n1)

    if [[ "$MODEL_NAME" == *-trained ]]; then
        echo "Trained model '$MODEL_NAME' is running. Killing all sglang processes..."
        echo "$SGLANG_PIDS" | xargs kill
        echo "All sglang processes for '$MODEL_NAME' have been killed."
    else
        echo "Model '$MODEL_NAME' is not a trained model. No processes killed."
    fi
fi

# Remove experiment cache and model directory
rm -rf ~/.dspy_cache
rm -rf meta-llama/
rm -rf mistralai/
rm -rf Qwen/
rm -rf deepseek-ai/
rm -rf google/

echo "Cleanup DSPy experiment cache is done."
