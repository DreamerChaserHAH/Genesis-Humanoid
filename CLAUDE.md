# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Genesis-Humanoid is a humanoid robot learning framework for RL-based locomotion and teleoperation. The current branch (`extremcontrol`) implements **ExtremControl** — low-latency humanoid teleoperation with direct extremity control for the Unitree G1 robot. It uses the Genesis physics engine for simulation, PPO for RL training, and behavioral cloning for policy distillation.

## Common Commands

### Setup
```bash
uv sync --package gs-env       # Install dependencies via uv
source .venv/bin/activate       # Activate virtual environment
```

### Training
```bash
# RL teacher policy
python examples/run_ppo_motion.py --exp_name NAME --env_name g1_motion_teacher --env.motion_file assets/motion/motion.yaml

# BC distillation
python examples/run_bc_motion.py --exp_name NAME --env_name g1_motion --teacher_exp_name TEACHER_NAME --env.motion_file assets/motion/motion.yaml

# RL finetune / resume
python examples/run_ppo_motion.py --exp_name NAME --env_name g1_motion --resume True --use_stored_config False
```

### Evaluation
```bash
python examples/run_ppo_motion.py --exp_name NAME --eval True --show_viewer True
```

### Deployment
```bash
uv pip install redis
redis-server                    # Required for teleoperation
python deploy/g1_teleop.py --exp_name NAME                          # Sim
python deploy/g1_teleop.py --exp_name NAME --sim False --action_scale 0.1  # Real (start small)
```

### Linting & Type Checking
```bash
ruff check .                    # Lint (config in pyproject.toml)
ruff format .                   # Format
pyright                         # Type check
pytest                          # Tests (mark GPU tests with @pytest.mark.gpu)
```

## Architecture

### Workspace Structure (uv workspace with 3 packages)

- **`gs-schemas`** (`src/schemas/`) — Shared Pydantic dataclasses and type definitions. All configs use `genesis_pydantic_config()` from `gs_schemas.base_types` with `extra="forbid"` and `validate_assignment=True`.
- **`gs-agent`** (`src/agent/`) — RL algorithms and training pipeline (PPO, BC), experience buffers (GAE), neural network modules (policies, critics), and the `OnPolicyRunner`. Depends on `gs-schemas`.
- **`gs-env`** (`src/env/`) — Simulation and real-world environments. Depends on both `gs-agent` and `gs-schemas`.

Dependencies flow: `gs-schemas` ← `gs-agent` ← `gs-env`.

### Key Patterns

- **Registry pattern**: Configs are registered in `config/registry.py` files (e.g., `EnvArgsRegistry`, `PPO_MOTION_MLP`, `RUNNER_MOTION_MLP`) and looked up by name.
- **Base classes**: All major components inherit from abstract bases in `gs_agent/bases/` (algo, buffer, critic, policy, runner, env_wrapper) and `gs_env/common/bases/` (env, robot, object, scene, sensor).
- **Config override via CLI**: Training scripts use `fire` for CLI args. Nested config fields are overridden with dot notation (e.g., `--env.motion_file`, `--runner.freeze_actor_iterations`).
- **Environment wrapping**: Genesis envs are wrapped in `GenesisEnvWrapper` before being passed to RL algorithms.

### Simulation Environments (`gs_env/sim/`)

Locomotion environments for the Unitree G1 humanoid: `walking_env.py`, `motion_env.py`, and the base `leggedrobot_env.py`. Robot definitions are in `sim/robots/leggedrobots.py`.

### Real-World Stack (`gs_env/real/` and `deploy/`)

- Real robot interface via `leggedrobot_env.py` in `real/`
- Motion capture: OptiTrack (`optitrack/`) and SteamVR (`steamvr/`) integrations
- Deployment scripts in `deploy/`: `g1_teleop.py`, `g1_motion.py`, and motion publishers
- Teleoperation uses Redis for inter-process communication

### Motion Data Pipeline

Convert motion datasets before training:
- `examples/convert_lafan.py` (LAFAN1), `convert_hub.py` (HuB), `convert_amass.py` (AMASS), `convert_optitrack.py` (recorded MoCap)
- Processed motions go to `assets/motion/`
- Robot assets in `assets/robot/unitree_g1/`

## Code Style

- Line length: 100 (ruff)
- Target Python: 3.10+
- Type annotations enforced via pyright (strict settings in pyproject.toml)
- Ruff rules include annotation requirements (ANN), bugbear (B), isort (I), and pyupgrade (UP); E501 is ignored
