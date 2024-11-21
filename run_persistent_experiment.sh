#!/bin/bash

# Ensure script is run with sudo privileges
if [[ $EUID -ne 0 ]]; then
   echo "Please run this script as root or using sudo."
   exit 1
fi

# Check if nohup is installed, install if not
if ! command -v nohup &> /dev/null; then
    echo "nohup not found. Installing coreutils package..."
    apt-get update && apt-get install -y coreutils
else
    echo "nohup is already installed."
fi

# Define the experiment log file name with the current date
EXPERIMENT_LOG_FILE="better_together_experiment_run_$(date +'%Y-%m-%d').log"

# Run the Python program persistently with real-time log output
echo "Starting the Python program with nohup..."
nohup python3 reproduce_better_together.py 2>&1 | tee "$EXPERIMENT_LOG_FILE" &

# Provide feedback to the user
echo "Python program is running persistently. Logs are being written to $EXPERIMENT_LOG_FILE and shown in the terminal."


# run "tail -f $EXPERIMENT_LOG_FILE" or "ps -ef | grep reproduce_better_together.py"
# To make sure experiment job is still alive in case ssh sessions drops.