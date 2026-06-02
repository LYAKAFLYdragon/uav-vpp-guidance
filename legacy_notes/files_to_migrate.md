# Files to Migrate — Priority Order

## Priority 1: Core Dynamics & Environment

These are required for the minimal closed-loop simulation.

- [ ] `envs/JSBSim/__init__.py`
  - **New target**: `src/uav_vpp_guidance/envs/jsbsim_env.py`
  - **Scope**: JSBSim initialization, aircraft loading, property binding, step/reset.
  - **Refactor needed**: Remove multi-aircraft combat and missile logic; keep single-ownship + target wrapper.

- [ ] `envs/JSBSim/core/simulatior.py`
  - **New target**: `src/uav_vpp_guidance/envs/jsbsim_env.py`
  - **Scope**: Low-level simulation stepping, state query.
  - **Refactor needed**: Wrap as internal helper, not top-level API.

- [ ] `envs/JSBSim/core/catalog.py`
  - **New target**: `src/uav_vpp_guidance/envs/jsbsim_env.py` (internal)
  - **Scope**: JSBSim property catalog for state extraction.
  - **Refactor needed**: May not need full catalog; extract only properties used for F-16 state.

- [ ] `envs/JSBSim/envs/env_base.py`
  - **New target**: `src/uav_vpp_guidance/envs/tracking_env.py`, `jsbsim_env.py`
  - **Scope**: Base environment state management.
  - **Refactor needed**: Split into high-level tracking env and low-level JSBSim wrapper.

## Priority 2: Task Logic — Reward, Termination, Scenario

- [ ] `envs/JSBSim/tasks/singlecombat_task.py`
  - **New target**: `src/uav_vpp_guidance/envs/reward.py`, `termination.py`, `scenario_sampler.py`
  - **Scope**: Reward shaping, termination conditions, scenario initialization.
  - **Refactor needed**: Remove missile and weapon logic; focus on close-range tracking geometry.

- [ ] `envs/JSBSim/reward_functions/*.py`
  - **New target**: `src/uav_vpp_guidance/envs/reward.py`
  - **Scope**: Individual reward terms (distance, altitude, velocity, heading, etc.).
  - **Refactor needed**: Consolidate into `RewardCalculator.compute()` with configurable weights.

- [ ] `envs/JSBSim/termination_conditions/*.py`
  - **New target**: `src/uav_vpp_guidance/envs/termination.py`
  - **Scope**: Crash, success, timeout conditions.
  - **Refactor needed**: Consolidate into `TerminationChecker.check()`.

- [ ] `envs/env_wrappers.py`
  - **New target**: `src/uav_vpp_guidance/envs/observation.py`, `tracking_env.py`
  - **Scope**: Observation construction, normalization, feature engineering.
  - **Refactor needed**: Extract relative geometry computation; design new observation that includes gain vector.

- [ ] `config.py`
  - **New target**: `config/*.yaml`
  - **Scope**: Global hyperparameters.
  - **Refactor needed**: Convert to structured YAML; separate concerns (env, algorithm, guidance, reward).

## Priority 3: Control & Guidance

- [ ] `envs/JSBSim/core/NDI_controller.py`
  - **New target**: `src/uav_vpp_guidance/flight_control/low_level_controller.py`, `guidance/los_rate_guidance.py`
  - **Scope**: NDI-based low-level controller and guidance command generation.
  - **Refactor needed**: Split into guidance law (LOS-rate -> nz/roll_rate) and low-level controller (nz/roll_rate -> actuator).

- [ ] `envs/JSBSim/model/baseline_actor.py`, `baseline_model.pt`
  - **New target**: `src/uav_vpp_guidance/agents/policy_network.py` (reference only)
  - **Scope**: Baseline network architecture.
  - **Refactor needed**: Use as architectural reference; do not copy weights unless specifically needed.

## Priority 4: RL Algorithms

- [ ] `algorithms/ppo/ppo_trainer.py`
  - **New target**: `src/uav_vpp_guidance/agents/ppo_agent.py`
  - **Scope**: PPO training loop.
  - **Refactor needed**: Modularize into agent class; support checkpointing and logging hooks.

- [ ] `algorithms/ppo/ppo_actor.py`, `ppo_critic.py`
  - **New target**: `src/uav_vpp_guidance/agents/policy_network.py`
  - **Scope**: Network definitions.
  - **Refactor needed**: Unify into configurable MLP with separate actor/critic classes.

- [ ] `algorithms/utils/buffer.py`
  - **New target**: `src/uav_vpp_guidance/agents/replay_buffer.py` (or internal PPO buffer)
  - **Scope**: Rollout buffer.
  - **Refactor needed**: Adapt to new observation/action shapes.

## Priority 5: Evaluation & Recording

- [ ] `envs/JSBSim/core/render_tacview.py`
  - **New target**: `src/uav_vpp_guidance/utils/recorder.py`
  - **Scope**: ACMI file generation.
  - **Refactor needed**: Extract ACMI formatting; keep recorder decoupled from training loop.

- [ ] `runner/jsbsim_runner.py`
  - **New target**: `src/uav_vpp_guidance/training/`, `evaluation/`
  - **Scope**: Training loop, evaluation loop, logging.
  - **Refactor needed**: Decompose into train_fixed_gain, train_bilevel, evaluate_policy, monte_carlo.

## Do Not Migrate

- `baseline_results_*` — Historical experiment outputs.
- `sensitivity_results/*` — Old sensitivity analysis outputs.
- `*.acmi` — Historical flight recordings.
- `*.png`, `*.jpg`, `*.gif` — Historical figures.
- `*.csv`, `*.mat` — Historical data files.
- `1111.py`, `python_try.py`, `jindou.py`, `w2.py` — Temporary scripts.
- `check_sqlite.py`, `check_swanlab*.py` — Infrastructure checks.
- `generate_1m_scripts.py`, `run_*.py` (ad-hoc) — Replaced by modular scripts.
- `get_reward_figure/*.py` (most) — Ad-hoc plotting; only extract general utilities if any.
- `renders/train_singlecombat*.py` — Training scripts superseded by new training module.
- `scripts/train/train_gym.py`, `train_jsbsim.py` — Legacy training entry points.
- `algorithms/mappo/`, `algorithms/mhppo/` — Not needed unless multi-agent required.
- `sko/*.py` — Generic optimization library; only migrate if CEM/PBT insufficient.

## Post-Migration Checklist

- [ ] All Priority 1 files migrated and unit-tested.
- [ ] F-16 can be initialized, stepped, and state queried.
- [ ] Reward and termination logic validated against legacy behavior.
- [ ] PPO agent trains on fixed-gain baseline.
- [ ] ACMI recording produces valid TacView output.
