# Repository Guidelines

## Project Structure & Module Organization
This repository is a course monorepo. Main work lives in `hw1/` through `hw5/`, plus `final_project_llm_rl/` and `final_project_offline_online/problem/`. Most assignments use a `src/` layout:

- `hw1/src/hw1_imitation/`: imitation learning package
- `hw2/src/`, `hw3/src/`, `hw5/src/`: agents, networks, infrastructure, and runnable scripts
- `hw4/hw4/` and `final_project_llm_rl/llm_rl_final_proj/`: LLM RL training, rollout, and evaluation code
- `experiments/`, `dataset/`, `public_eval/`, `scripts/`: configs, data, evaluation assets, and Modal entrypoints

Treat `.venv/`, `wandb/`, `runs/`, `logs/`, and generated bundles as local artifacts.

## Build, Test, and Development Commands
Use `uv` inside the target project directory.

- `uv sync`: install that project’s dependencies from `pyproject.toml` and `uv.lock`
- `uv run <script> --help`: inspect script flags, for example `uv run src/scripts/run.py --help`
- `uv run src/scripts/run.py ...`: run local training for HW2/HW3/HW5-style projects
- `uv run modal run ...`: launch remote jobs for HW1, HW3, HW4, HW5, and the final project
- `uv run python student_autograder/run_local_autograder.py ...`: run the final project’s local evaluation

Python versions differ by folder (`hw2`/`hw3`: 3.10, `hw1`/`hw5`: 3.11, `hw4`/`final_project_llm_rl`: 3.12). Respect each subproject’s `.python-version` or `pyproject.toml`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and small reusable helpers under `infrastructure/`, `utils/`, or `networks/`. Keep new scripts beside existing entrypoints under `src/scripts/` or `scripts/`. No repo-wide formatter config is checked in, so match surrounding code.

## Testing Guidelines
There is no single repo-wide test suite. Validate changes with the smallest relevant command for the affected assignment: a local training run, the matching Modal entrypoint, or the final-project autograder. If you add tests, keep them project-local and use `test_*.py`.

## Commit & Pull Request Guidelines
Recent commits use short imperative prefixes such as `feat: hw4 impl` and `feat: variational inference`. Keep that format (`feat:`, `fix:`, `docs:`) and scope the message to one assignment or feature. PRs should state the touched subproject, the commands you ran, and any dataset or Modal assumptions. Include screenshots only when a change affects logged metrics or generated reports.

## Security & Configuration Tips
Do not commit secrets such as `OPENAI_API_KEY`, WandB credentials, or Modal tokens. Keep large datasets and run outputs in `dataset/`, Modal volumes, or ignored local artifact folders instead of adding ad hoc dumps to source directories.
