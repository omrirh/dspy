# CLAUDE.md — GEPA+FewShot Research Project

*NLP MSc Project — Omri Bar Haim, Roy Zemah, Yaniv Cohen (TAU, 2025–2026)*
*Branch: `gepa-fewshot`*

---

## What this project is

An extension of the [DSPy](https://github.com/stanfordnlp/dspy) framework with a new prompt
optimizer, **GEPAFewShot**, that combines GEPA's reflective instruction evolution with
bootstrapped few-shot demonstration mutation.  The research question is whether demonstrative
context outperforms instructional context for small language models, and whether this advantage
inverts with model scale.

The `experiments/` directory contains a full matrix experiment infrastructure for statistically
grounded evaluation across models, datasets, optimizers, and seeds.

---

## Re-dive protocol

When starting a new session:

1. Read this file.
2. Read each context file listed below, in the order shown.
3. State your understanding of the current project status and the next actions on the table,
   then wait for the user's call.

---

## Context files

Context is split into two kinds:

### Static (repo) — building blocks, ground truth

These files are manually maintained and define the facts of the project.
Read them to understand what was built and why.

| File | What it contains |
|---|---|
| [experiments/DESIGN.md](experiments/DESIGN.md) | Architecture of GEPAFewShot and the experiment matrix runner — the authoritative implementation reference |
| [experiments/README.md](experiments/README.md) | Practical workflow guide: directory layout, CLI flags, entry points, result structure |

### Live (repo) — current project state

These files evolve with the project and reflect the most up-to-date status.
Always check them before proposing changes or next steps.

| File | What it contains |
|---|---|
| [experiments/todos.md](experiments/todos.md) | Active task list: experiments to run, implementation work, known bugs |
| [experiments/insights.md](experiments/insights.md) | Empirical findings and hypothesis validation status as results come in |

### Auto-memory (self-evolving, written by Claude across sessions)

These files are maintained by Claude and capture context that is not derivable from the code
or the repo — decisions made, hypotheses formed, findings observed, collaboration preferences.
They live outside the repo and are loaded automatically at session start via `MEMORY.md`.

| File | What it captures |
|---|---|
| [gepa_fewshot_hypothesis.md](~/.claude/projects/-home-obarhaim-dspy/memory/gepa_fewshot_hypothesis.md) | The academic hypothesis, mechanism, and scaling findings — updated as results accumulate |
| [experiment_matrix_setup.md](~/.claude/projects/-home-obarhaim-dspy/memory/experiment_matrix_setup.md) | Decisions made when building the `results_v1` infrastructure |
| [project_state.md](~/.claude/projects/-home-obarhaim-dspy/memory/project_state.md) | Point-in-time snapshot of done / pending / blocked — verify against live files |
| [experiment_automation_vision.md](~/.claude/projects/-home-obarhaim-dspy/memory/experiment_automation_vision.md) | Mental model of the full experiment pipeline and its automation potential |
| [collaboration_prefs.md](~/.claude/projects/-home-obarhaim-dspy/memory/collaboration_prefs.md) | How this user wants Claude to behave on this project |
| [sglang_stack.md](~/.claude/projects/-home-obarhaim-dspy/memory/sglang_stack.md) | Validated GPU stack (SGLang + CUDA) — version pins and known pitfalls |

---

## What evolves and what does not

**This file (`CLAUDE.md`) does not self-evolve.** It is the static lobby — edit it only when the
project's structure or context map changes meaningfully.

**Auto-memory files evolve.** As results come in, hypotheses are refined, and new decisions are
made, Claude should update the relevant memory files to reflect the current state of knowledge.
`todos.md` and `insights.md` in the repo are the human-maintained counterparts to this.
