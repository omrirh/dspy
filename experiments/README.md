# Experiments — GEPA + FewShot

Standalone prompt optimization experiments comparing **GEPA**, **GEPAFewShot**,
**MIPROv2**, and a zero-shot **baseline** on GSM8K and Iris.

## Directory structure

```
experiments/
  run_experiment.py          Single-run CLI entry point
  run_experiment_driver.sh   Shell wrapper for background / nohup runs
  run_matrix.py              Scalable matrix driver (all combos, --dry-run, --resume)
  programs.py                DSPy program definitions (CoT, IrisProgram)
  metrics.py                 GEPA-compatible metrics and BETTER_REFLECTION_PROMPT
  analyze_results.py         Results aggregation, statistical summary, plots
  insights.md                Empirical findings and hypothesis tracking
  todos.md                   Experiments to run, implementation tasks, known bugs
  DESIGN.md                  Implementational design notes (GEPAFewShot + matrix)
  experiment_notebook.ipynb  End-to-end walkthrough notebook
  results_v1/                Matrix run artifacts (created by run_matrix.py)
  logs/                      Ad-hoc single-run artifacts (created by run_experiment.py)
  plots/                     Figures (created by analyze_results.py --plot-dir)
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

# 6. Single run (background, logs to file)
bash experiments/run_experiment_driver.sh \
    --dataset gsm8k \
    --optimizer gepa_fewshot \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --auto medium \
    --seed 42
```

## Running the full experiment matrix

The matrix driver covers all hypothesis-relevant combinations in a single command:

```bash
# Preview all 100 commands without executing
python experiments/run_matrix.py --dry-run

# Run the full matrix (100 runs: 2 datasets × 2 models × 5 optimizers × 5 seeds)
python experiments/run_matrix.py

# Resume an interrupted run (skips cells whose results.json already exists)
python experiments/run_matrix.py --resume

# Subset: quick smoke test (1 dataset, 2 optimizers, 1 seed)
python experiments/run_matrix.py \
    --datasets gsm8k \
    --optimizers baseline gepa_fewshot \
    --seeds 42 \
    --models meta-llama/Llama-3.2-3B-Instruct
```

Matrix results are written to `experiments/results_v1/`.  A `matrix_summary.json`
with cross-seed statistics is written automatically when the matrix completes.

For remote runs that must survive SSH disconnects, use the shell wrapper instead:

```bash
# Full matrix under nohup
bash experiments/run_matrix_driver.sh

# All run_matrix.py flags are passed through verbatim
bash experiments/run_matrix_driver.sh --resume
bash experiments/run_matrix_driver.sh --dry-run
bash experiments/run_matrix_driver.sh \
    --datasets gsm8k \
    --optimizers baseline gepa_fewshot \
    --seeds 42

# Monitor progress
tail -f experiments/logs/matrix_<timestamp>.log
```

### Matrix configuration (encoded in `run_matrix.py`)

| Dimension | Values |
|---|---|
| Datasets | `gsm8k`, `iris` |
| Models | `Llama-3.2-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |
| Optimizers | `baseline`, `miprov2`, `gepa`, `gepa_merge`, `gepa_fewshot` |
| Seeds | `42, 123, 456, 789, 1337` |
| Budget | `medium` (all optimizers) |

**Per-dataset splits:**

| Dataset | Train | Val | Test |
|---|---|---|---|
| GSM8K | 100 | 250 | all (~1318) |
| Iris | 15 | 35 | 50 |

## Running single experiments (interactive)

```bash
# Zero-shot baseline
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer baseline \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train-size 100 --val-size 250 --test-size 1500

# Vanilla GEPA (no merge)
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer gepa \
    --model meta-llama/Llama-3.2-3B-Instruct --auto medium --seed 42

# Vanilla GEPA with merge
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer gepa_merge \
    --model meta-llama/Llama-3.2-3B-Instruct --auto medium --seed 42

# GEPA + FewShot (our extension)
python experiments/run_experiment.py \
    --dataset gsm8k --optimizer gepa_fewshot \
    --model meta-llama/Llama-3.2-3B-Instruct --auto medium --seed 42 \
    --k-demos 3 --demo-mutation-strategy metric_based

# MIPROv2 baseline
python experiments/run_experiment.py \
    --dataset iris --optimizer miprov2 \
    --model meta-llama/Llama-3.2-3B-Instruct --auto medium --seed 42
```

## Key CLI flags (`run_experiment.py`)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | — | `gsm8k` or `iris` |
| `--optimizer` | — | `baseline`, `gepa`, `gepa_merge`, `gepa_fewshot`, `miprov2` |
| `--model` | — | HuggingFace model ID |
| `--auto` | `medium` | Budget preset: `light / medium / heavy` |
| `--seed` | *time-based* | Random seed for reproducibility |
| `--reflection-model` | same as `--model` | LM for GEPA reflection (self-improving by default) |
| `--k-demos` | `3` | Demonstrations per candidate *(GEPAFewShot only)* |
| `--demo-mutation-strategy` | `metric_based` | `random` or `metric_based` *(GEPAFewShot only)* |
| `--train-size / --val-size / --test-size` | 200/100/300 | Dataset split sizes |
| `--num-threads` | `4` | Parallel evaluation threads |
| `--log-dir` | `experiments/logs` | Root directory for run artifacts |
| `--max-tokens` | *model default* | Max tokens per LM call |

## Log / result structure

Each run writes to `{log-dir}/{run_tag}/`:

```
run.log                  Full console output
config.json              All CLI arguments + seed
results.json             Test score, runtimes, metric calls, demo counts, instructions
optimized_program.json   Saved DSPy module (loadable with module.load())
gepa/                    GEPA internal checkpoints (GEPA runs only)
```

`results.json` includes `total_metric_calls` for GEPA/GEPAFewShot runs, enabling
sample-efficiency comparisons alongside accuracy.

## Analyzing results

```bash
# Summary table of all single runs in experiments/logs/
python experiments/analyze_results.py

# Statistical aggregation over seeds (results_v1/)
python experiments/analyze_results.py \
    --log-dir experiments/results_v1 \
    --aggregate

# Aggregation + comparison plots
python experiments/analyze_results.py \
    --log-dir experiments/results_v1 \
    --aggregate \
    --plot-dir experiments/plots/results_v1

# Recompute matrix_summary.json only
python experiments/run_matrix.py --summary-only --results-dir experiments/results_v1
```

The aggregate table reports mean accuracy, std, 95% CI, and optimization time
(median and mean in minutes) per `(dataset, model, optimizer)` group.

## Running tests

```bash
python -m pytest experiments/tests/test_pipeline.py -v
```

Tests are fully mocked — no GPU or network access required.

## Optimizers overview

| Optimizer | Optimizes | Key mechanism |
|---|---|---|
| `baseline` | Nothing | Zero-shot eval, no optimization |
| `gepa` | Instructions only | Pareto-based reflective evolution, merge disabled |
| `gepa_merge` | Instructions only | Pareto + merge-based candidate combination |
| `gepa_fewshot` | Instructions + demos | Pareto + metric-based demo mutation |
| `miprov2` | Instructions + demos | Bayesian optimization (Optuna) |

## Further reading

- [DESIGN.md](DESIGN.md) — implementation design of GEPAFewShot and the matrix runner
- [insights.md](insights.md) — empirical findings and hypothesis tracking
- [todos.md](todos.md) — pending experiments, implementation tasks, known issues
