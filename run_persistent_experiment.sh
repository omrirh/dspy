#!/bin/bash

EXPERIMENT_LOG_FILE="better_together_experiment_run_$(date +'%Y-%m-%d').log"

# Run BetterTogether experiment persistently
nohup python3 reproduce_better_together.py 2>&1 | tee "$EXPERIMENT_LOG_FILE" &

# run "tail -f $EXPERIMENT_LOG_FILE" or "ps -ef | grep reproduce_better_together.py"
# To make sure experiment job is still alive in case ssh sessions drops.