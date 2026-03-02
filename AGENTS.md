# Repository Guidelines

## Project Structure & Module Organization
Core Isaac Lab extension code lives under `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/`, with `tasks/` for environments, `controllers/` for control logic, `assets/` for robot configs and USD/URDF data, and `plotting_tools/` for analysis helpers. Training and evaluation entry points are under `scripts/`, split by library (`scripts/skrl`, `scripts/rsl_rl`) and small testing helpers in `scripts/MARL_mav_carry/testing_scripts`. The `skrl/` directory is a forked RL library with its own docs and tests. Generated artifacts land in `logs/`, `outputs/`, and `ablation_study_csv/`.

## Build, Test, and Development Commands
Install prerequisites by setting up Isaac Lab, then install the forked RL library and extension:
`cd skrl && pip install -e .`
`python -m pip install -e exts/MARL_mav_carry_ext`

Train and play (example with SKRL):
`python3 scripts/skrl/train.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 --headless --num_envs=4096 --seed=-1 --algorithm="MAPPO"`
`python3 scripts/skrl/play.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 --headless --video --video_length=2000 --algorithm="MAPPO" --control_mode="ACCBR" --save_plots --checkpoint=$(PATH_TO_PT_FILE)`

Formatting is managed via pre-commit:
`pre-commit run --all-files`

## Coding Style & Naming Conventions
Python 3.10 is the baseline. Use 4-space indentation and keep lines within 120 characters (isort uses the Black profile with `line_length=120`). Prefer `snake_case` for modules/functions and `CamelCase` for classes. Environment and config files follow the Isaac Lab pattern, e.g., `*_env.py` and `*_env_cfg.py`. Run pre-commit before pushing changes.

## Testing Guidelines
There are no dedicated tests for the extension itself; validation is usually via running training or play scripts in Isaac Lab. The forked `skrl/` project uses `pytest` with tests under `skrl/tests/`:
`pytest skrl/tests`
GPU-dependent tests skip when unavailable. No explicit coverage target is defined.

## Commit & Pull Request Guidelines
Recent history shows short, single-line subjects (English or Chinese), often imperative and sometimes generic (e.g., “Update README.md”) with occasional merge commits. Follow that convention: keep the subject concise, add a scope if it improves clarity, and avoid noisy formatting in the body. For PRs, include a brief summary, the task/config used, and relevant artifacts (logs, plots, or videos) when behavior changes.
