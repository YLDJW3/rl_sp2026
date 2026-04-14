# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UC Berkeley CS 285/185 Deep Reinforcement Learning (Spring 2026) homework and final project repository. Each subdirectory (`hw1`–`hw5`, `final_project_llm_rl`, `final_project_offline_online`) is an independent Python project with its own `pyproject.toml`, `uv.lock`, and `.python-version`.

## Package Manager

All projects use **`uv`** (Astral). Never use `python` or `pip` directly.

```bash
uv sync              # install dependencies (run from within each hw/project directory)
uv run <script>      # run any script
uv add <package>     # add a dependency
```

## Running Experiments

Always `cd` into the specific homework/project directory first.

```bash
# Local run (hw2 example)
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole

# Modal (cloud GPU) run
uv run modal run src/scripts/modal_run.py -- <args>

# Download Modal results
uv run modal volume get <volume-name> exp/
```

hw1 and hw2 have `makefile` targets for batch experiments (e.g., `make reward`, `make baseline`, `make gae_sweep`). hw3 uses YAML configs in `experiments/` passed via `-cfg`.

## Experiment Tracking

All assignments use **Weights & Biases**. Login once per environment:

```bash
uv run wandb login    # or: uvx wandb login
```

## Per-Assignment Details

| HW | Topic | Entry Script | Python |
|----|-------|-------------|--------|
| hw1 | Imitation Learning | `src/hw1_imitation/train.py` | 3.11 |
| hw2 | Policy Gradients | `src/scripts/run.py` | 3.10 |
| hw3 | DQN / SAC | `src/scripts/run_dqn.py`, `src/scripts/run_sac.py` | 3.10 |
| hw4 | LLM RL (single-GPU) | Modal-based (`hw4/` package) | 3.12 |
| hw5 | Offline RL | `src/scripts/run.py` | 3.11 |
| final_project_llm_rl | RLHF | `scripts/modal_train.py` | 3.12 |

## Code Architecture (hw2–hw5)

These assignments share a common structure under `src/`:

- `agents/` — Algorithm implementations (PG, DQN, SAC)
- `networks/` — Neural network modules (policies, critics/Q-networks)
- `infrastructure/` — Replay buffers, logging, PyTorch utilities
- `configs/` — Configuration factories (hw3+)
- `scripts/` — Training entry points and Modal launchers

## Final Project (LLM RL)

Package: `llm_rl_final_proj/`. Base model: `Qwen/Qwen2.5-1.5B-Instruct`. Dataset: WildChat 5k preference pairs.

Key subpackages:
- `offline/` — DPO, IPO, AOT preference optimization
- `online/` — GRPO, DrGRPO, GSPO
- `reward_model/` — Bradley-Terry reward model
- `rollout/` — Sampling and rollout buffer
- `models/` — Model loading/inference
- `data/` — Dataset utilities

Runs on Modal with H100 GPUs. Evaluated by GPT-5.4 head-to-head win rate vs frozen base.

## System Dependencies

`swig` and `cmake` are required for `gym[box2d]` (hw2, hw3):

```bash
brew install swig cmake   # macOS
```

## Key Constraints

- `numpy<2.0` across all projects
- `gym==0.25.2` (pinned, not latest)
- hw4 and final project split deps: `modal` locally, heavy ML deps only in `[remote]` optional group
