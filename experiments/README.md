# Experiments — GEPA + FewShot

Standalone prompt optimization experiments comparing **GEPA**, **GEPAFewShot**, and **MIPROv2** on GSM8K and Iris.

## Directory structure

```
experiments/
  run_experiment.py          CLI runner (core entry point)
  run_experiment_driver.sh   Shell driver for background runs / nohup
  programs.py                DSPy program definitions (CoT, IrisProgram)
  analyze_results.py         Results aggregation and plots
  experiment_notebook.ipynb  End-to-end walkthrough notebook
  logs/                      Auto-created; one sub-dir per run
  plots/                     Auto-created by analyze_results.py
  tests/
    test_pipeline.py         Mock unit tests (no GPU required)

remote_setup/
  utils.py                   SGLang server management helpers
  run_sglang_model.sh        Manual SGLang launcher
  prepare_virtualenv.sh      One-time venv setup on remote instance
  install_nvidia_drivers.sh  NVIDIA driver + CUDA 12.4 setup
  requirements.txt           Python dependencies

vm_vars.env.template         Copy → vm_vars.env and fill in HF_TOKEN
```

## Quick start (remote GPU machine)

```bash
# 1. Clone repo and enter directory
git clone <repo> dspy && cd dspy
git checkout gepa-fewshot

# 2. Install NVIDIA drivers + CUDA (once per machine)
bash remote_setup/install_nvidia_drivers.sh
source ~/.bashrc

# 3. Create virtualenv
bash remote_setup/prepare_virtualenv.sh
source dspy_venv/bin/activate

# 4. Set environment variables
cp vm_vars.env.template vm_vars.env
# Edit vm_vars.env: fill in HF_TOKEN

# 5. Start SGLang model server
bash remote_setup/run_sglang_model.sh --model-name meta-llama/Llama-3.2-3B-Instruct

# 6. Run an experiment (background, logs to file)
bash experiments/run_experiment_driver.sh \
    --dataset gsm8k \
    --optimizer gepa_fewshot \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --auto light
```

## Running experiments on a remote machine

Use the shell driver for remote runs — it activates the venv, loads env vars, raises the file-descriptor limit, and launches the process under `nohup` so it survives SSH disconnects.

```bash
# Vanilla GEPA baseline
bash experiments/run_experiment_driver.sh \
    --dataset gsm8k --optimizer gepa \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light

# GEPA + FewShot (our extension)
bash experiments/run_experiment_driver.sh \
    --dataset gsm8k --optimizer gepa_fewshot \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light \
    --k-demos 3 --demo-mutation-strategy metric_based

# MIPROv2 baseline
bash experiments/run_experiment_driver.sh \
    --dataset iris --optimizer miprov2 \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light
```

Logs are written to `experiments/logs/experiment__<dataset>__<optimizer>__<model>__<auto>__<date>.log` and streamed to stdout via `tee`.

## Running experiments directly (local / interactive)

For local development with a venv already active and env vars set, you can invoke the Python script directly for foreground output.

```bash
# Vanilla GEPA baseline
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer gepa \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light

# GEPA + FewShot (our extension)
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer gepa_fewshot \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light \
    --k-demos 3 --demo-mutation-strategy metric_based

# MIPROv2 baseline
python experiments/run_experiment.py \
    --dataset iris --optimizer miprov2 \
    --model meta-llama/Llama-3.2-3B-Instruct --auto light
```

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | — | `gsm8k` or `iris` |
| `--optimizer` | — | `gepa`, `gepa_fewshot`, `miprov2` |
| `--model` | — | HuggingFace model ID |
| `--auto` | `light` | Budget preset: `light / medium / heavy` |
| `--reflection-model` | same as `--model` | LM for GEPA reflection (self-improving by default) |
| `--k-demos` | `3` | Demonstrations per candidate *(GEPAFewShot only)* |
| `--demo-mutation-strategy` | `metric_based` | `random` or `metric_based` |
| `--train-size / --val-size / --test-size` | 200/100/300 | Dataset split sizes |
| `--num-threads` | `4` | Parallel evaluation threads |
| `--log-dir` | `experiments/logs` | Root directory for run artifacts |

## Log structure

Each run writes to `experiments/logs/<run_tag>/`:

```
run.log                  Full console output
config.json              All CLI arguments + random seed
results.json             Test score, runtimes, demo counts, optimized instructions
optimized_program.json   Saved DSPy module (loadable with module.load())
gepa/                    GEPA internal checkpoints and candidate logs
```

## Analyzing results

```bash
# Print summary table of all completed runs
python experiments/analyze_results.py

# Filter and save plots
python experiments/analyze_results.py \
    --dataset gsm8k \
    --plot-dir experiments/plots \
    --show-instructions
```

## Running tests

```bash
python -m pytest experiments/tests/test_pipeline.py -v
```

Tests are fully mocked — no GPU or network access required.

## Optimizers overview

| Optimizer | Optimizes | Search strategy |
|---|---|---|
| `gepa` | Instructions only | Pareto-based reflective evolution |
| `gepa_fewshot` | Instructions + demos | Pareto + metric-based demo mutation |
| `miprov2` | Instructions + demos | Bayesian optimization (Optuna) |
