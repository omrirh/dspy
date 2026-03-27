# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSPy is a framework for programming (not prompting) language models. Users write compositional Python code and DSPy optimizes it. This repo is on the `cluster-few-shot-agentic` branch extending DSPy with ClusterFewshot and RetrievalFewshot optimizers (MSc project by Omri at TAU).

## Common Commands

### Setup
```bash
# Install nvidia drivers (optional)
bash dspy/remote_setup/install_nvidia_drivers.sh

# Prepare environment dependencies                                                                  
bash dspy/remote_setup/prepare_virtualenv.sh   

# Deploy local model via SGLang                                                                         
bash dspy/remote_setup/run_sglang_model.sh --model-name <model_id> 
```

### Testing
```bash
pytest tests/                              # all tests
pytest tests/predict/test_predict.py -v    # single file
pytest tests/predict/test_predict.py::test_name -v  # single test
pytest tests/ --reliability                # opt-in stress tests (skipped by default)
```

### Linting & Formatting
```bash
ruff check dspy/ tests/           # lint check
ruff check --fix dspy/ tests/     # lint with auto-fix
ruff format dspy/ tests/          # format
pre-commit run --all-files        # run all hooks
```

## Code Style

- Line length: 120 characters
- Double quotes
- No relative imports in `dspy/` (TID252 rule); relative imports OK in tests
- isort with `--profile=black` (via pre-commit hook)
- Ruff lint rules: `F` (Pyflakes), `E` (pycodestyle), `TID252` (absolute imports). Tests and `__init__.py` are excluded from lint.
- Pre-commit hooks auto-fix on commit (install with `pre-commit install`)

## Architecture

### Core Abstractions (in dependency order)

1. **Signature** (`dspy/signatures/`) — Declarative input/output spec. Created via string shorthand (`"question -> answer"`) or Pydantic subclass with `InputField`/`OutputField`.

2. **Module** (`dspy/primitives/module.py`) — Base class for all DSPy programs. Subclasses implement `forward()`. Has `ProgramMeta` metaclass. Supports state save/load, callbacks, history.

3. **Predict** (`dspy/predict/predict.py`) — Base predictor module. Takes a Signature, formats prompt with demos, calls LM, parses response. Extended by `ChainOfThought`, `ReAct`, `CodeAct`, `BestOfN`, etc.

4. **Adapter** (`dspy/adapters/`) — Bridges Signatures to LM interfaces. Transforms inputs → formatted prompts → LM call → parsed response. Variants: `ChatAdapter`, `JSONAdapter`.

5. **Teleprompter** (`dspy/teleprompt/`) — Optimizers. `compile(student, trainset, ...)` returns an optimized Module. Key optimizers: `BootstrapFewShot`, `MIPROv2`, `COPRO`, `BetterTogether`, `ClusterFewshot`, `RetrievalFewshot`.

6. **LM Client** (`dspy/clients/`) — Language model interface. Configured via `dspy.configure(lm=dspy.LM("openai/gpt-4"))`. Uses LiteLLM under the hood.

### ClusterFewshot (`dspy/teleprompt/clusterfewshot/`)

ClusterFewshot is a semantically informed few-shot prompt optimizer. It replaces the random sampling / metric-based ranking used by BFRS and MIPROv2 with a two-stage process: (1) embed and cluster training examples to capture the task's latent semantic structure, then (2) score each candidate demonstration via one-shot evaluation on a held-out validation subset constructed from cluster centroids. The final demo set is selected from candidate strategies (Global Top-k, Cluster Representatives) evaluated on the full validation set.

**Compilation pipeline** (`compile()` in `cluster_fewshot.py`):
1. **Bootstrap** — execute the student program on `trainset`, retain only successful traces (`bootstrap_examples`)
2. **Embed & cluster** — encode examples via BYOE encoders, K-means with silhouette-score grid search over encoders × K values (`generate_embedding_clusters_with_semantic_encoders`, `cluster_examples`)
3. **One-shot evaluation set** — cluster `valset`, pick m=3 examples nearest each centroid (`sample_one_shot_evaluation_set`)
4. **One-shot scoring** — evaluate every bootstrapped example as a sole demonstration on the evaluation set (`sort_examples_as_demos`)
5. **Candidate construction** — build demo subsets via task-adaptive sampling strategies (`collect_fewshot_subsets`)
6. **Final selection** — evaluate candidates on full `valset`, pick the best (`pick_best_fewshot_subset`)

**Bring-Your-Own-Encoder (BYOE)** (`semantic_encoder.py`):
- `SemanticEncoder(encoder, transform_fn, name)` — wraps any embedding model with a `transform_fn(encoder, examples) → np.ndarray` contract. The `encode(examples)` method delegates to the transform function.
- Factory helpers: `create_sentence_transformer_encoder(model_name)`, `create_numeric_encoder()`, `create_hotpotqa_question_encoder(model_name)`.
- Multiple encoders can be passed to `ClusterFewshot(semantic_encoders=[...])`. The optimizer runs a grid search across all encoders × cluster counts and selects the `(encoder, K)` pair with the highest silhouette score.
- Custom encoders are created by writing a `transform_fn` that extracts the relevant field(s) from `Example` objects and returns an embedding matrix, then wrapping it in `SemanticEncoder`.

**Task-type sampling strategies** (`TASK_2_SAMPLINGS` in `cluster_fewshot.py`):
- `"arithmetic"`, `"multihop"`, `"classification"`: `["top_n", "best_in_cluster"]` — global ranking first
- `"agentic"`: `["best_in_cluster", "top_n"]` — diversity-first ordering so each semantic cluster contributes at least one demo, preventing over-fitting to the most common question pattern

### RetrievalFewshot (`dspy/teleprompt/retrieval_fewshot.py`)

Subclasses `ClusterFewshot` for instance-level adaptive few-shot selection. Skips clustering, ranking, and static subset selection entirely. Pipeline: bootstrap demos → embed the pool via BYOE encoders → at inference, retrieve query-specific demonstrations using kNN or MMR (Maximal Marginal Relevance, balancing relevance and diversity via `mmr_lambda`). Returns a `retrieval_program_class` instance that calls `retrieve_demos(query_embedding)` at forward time.

### Current Experiment: Agentic AI (`react_agent_experiment.py`)

The active experiment benchmarks ClusterFewshot, MIPROv2, and BFRS as prompt optimizers for `dspy.ReAct` agents on HotPotQA multi-hop QA. The agent uses a ColBERTv2 Wikipedia search tool. Key design: `task_type="agentic"` uses diversity-first demo selection (`best_in_cluster` before `top_n`); two-encoder BYOE grid search (`all-mpnet-base-v2` general + `multi-qa-mpnet-base-dot-v1` QA-tuned) selects the best encoder automatically. Run via `react_agent_experiment_driver.sh`.

## Test Configuration

- pytest marker `reliability` is **skipped by default** — pass `--reliability` to enable
- `clear_settings` autouse fixture resets `dspy.settings.configure()` after each test

## Commit Message Convention

Enforced by pre-commit hook: `<type>(<scope>): <description>`

Valid types: `break`, `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`, `ops`, `hotfix`, `release`, `maint`, `init`, `enh`, `revert`
