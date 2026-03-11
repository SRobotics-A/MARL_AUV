# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Decentralized multi-agent reinforcement learning for aerial manipulation of cable-suspended loads. Agents are quadrotor drones (Falcon/FlyCrane) trained in NVIDIA Isaac Lab using the SKRL library.

**Paper:** arXiv:2508.01522

## Development Commands

### Installation
```bash
# Install forked SKRL library (required, not the upstream version)
cd skrl && pip install -e .

# Install the Isaac Lab extension
cd exts/MARL_mav_carry_ext && python -m pip install -e .
```

### Training
```bash
# MAPPO (decentralized multi-agent, primary algorithm)
python3 scripts/skrl/train.py \
  --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
  --headless --num_envs=4096 --seed=-1 --algorithm="MAPPO"

# Fly-follow task (current development focus)
python3 scripts/skrl/train.py \
  --task=Isaac-marl-flyfollow-v0 \
  --headless --num_envs=2048 --algorithm="MAPPO"
```

### Inference / Playback
```bash
python3 scripts/skrl/play.py \
  --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
  --headless --video --video_length=2000 \
  --algorithm="MAPPO" --save_plots \
  --checkpoint=/path/to/checkpoint.pt
```

### Code Formatting
```bash
pre-commit run --all-files
```

## Architecture

### Extension Structure
The core code lives in `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/`:
- **`tasks/directMARL/`** — Active decentralized environments (use these)
  - `flyfollow/` — Follow moving targets (current development focus)
  - `hover/` — Hover in formation (baseline)
  - `hover_flycart/`, `hover_flypent/` — Load-carrying variants
- **`tasks/managerbased/`** — Deprecated centralized environments, do not use
- **`assets/`** — Robot configs (`ArticulationCfg`) for Falcon, FlyCrane, FlyCrane-Rod, etc.
- **`controllers/`** — Physics-aware controllers: `geometric.py` (DFBC), `indi.py` (INDI), `motor_model.py`

### Environment Pattern
Every task has two files:
- `*_env.py` — Environment class extending `DirectMARLEnv` from Isaac Lab
- `*_env_cfg.py` — Dataclass config with reward weights, physics params, observation/action specs

The environment class handles vectorized simulation across all parallel environments simultaneously using PyTorch tensors. All per-agent state uses shape `(num_envs, num_agents, ...)`.

### Multi-Agent Design
- **Decentralized execution:** Each drone has its own policy and observation space
- **Target assignment:** `_assign_targets_for_envs()` runs at episode reset to assign drones to targets (Hungarian-like matching)
- **Per-drone buffers:** History buffers and timers are indexed by `(env_idx, drone_idx)`
- **Reward:** Computed per agent and returned as a dict keyed by agent name

### SKRL Integration
The `skrl/` directory is a forked SKRL library with custom modifications for this project. Always use this fork, not the upstream PyPI package. Training entry point is `scripts/skrl/train.py`.

## Coding Style
- Python 3.10, 4-space indent, 120 character line length
- `snake_case` for functions/variables, `CamelCase` for classes
- isort with Black profile (configured in `pyproject.toml`)
- Comments may be in Chinese or English

## Key File Paths
| Component | Path |
|-----------|------|
| Flyfollow env | `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env.py` |
| Flyfollow cfg | `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/flyfollow/marl_flyfollow_env_cfg.py` |
| Hover env | `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hover/marl_hover_env.py` |
| Controllers | `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/controllers/` |
| Assets | `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/` |
| Training logs | `logs/skrl/` |
| Reward design notes | `REWARD_DESIGN.md` |
