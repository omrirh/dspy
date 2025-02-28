#!/bin/bash

# Enable session variables and env
source ../vm_vars.env
source ../dspy_venv/bin/activate

# Default program
PROGRAM="gsm8k"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --program) PROGRAM="$2"; shift ;; # Accepts the --program flag and sets the program variable
        -h|--help)
            echo "Usage: $0 [--program <program name>]"
            echo "  --program  Specify the program to run. Options: hotpotqa, gsm8k"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$PROGRAM" != "hotpotqa" && "$PROGRAM" != "gsm8k" ]]; then
    echo "Invalid program: $PROGRAM"
    echo "Supported programs: hotpotqa, gsm8k"
    exit 1
fi

# Set log file
EXPERIMENT_LOG_FILE="better_together_experiment_run_${PROGRAM}_$(date +'%Y-%m-%d').log"

# Run BetterTogether experiment persistently
if [[ "$PROGRAM" == "hotpotqa" ]]; then
    nohup python3.11 reproduce_better_together_hotpotqa.py 2>&1 | tee "$EXPERIMENT_LOG_FILE" &
elif [[ "$PROGRAM" == "gsm8k" ]]; then
    nohup python3.11 reproduce_better_together_gsm8k.py 2>&1 | tee "$EXPERIMENT_LOG_FILE" &
fi

echo "Experiment for $PROGRAM started. Log file: $EXPERIMENT_LOG_FILE"
echo "Run 'tail -f $EXPERIMENT_LOG_FILE' or 'ps -ef | grep reproduce_better_together_${PROGRAM}.py'"
echo "To ensure the experiment job is still alive if SSH session drops."
