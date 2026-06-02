# Legacy Mapping

This document maps legacy files from `E:\CloseAirCombat_control` to new modules in `uav-vpp-guidance`.

## Migration Status Legend

- **pending**: Not yet migrated, needs manual review.
- **migrated**: Core logic has been moved.
- **ignored**: Will not be migrated (results, temporary scripts, etc.).
- **refactored**: Significant restructuring required during migration.

## File Mapping Table

| Legacy File | New Module | Status | Notes |
|---|---|---|---|
| `config.py` | `config/*.yaml` | pending | Extract only stable parameters; split into env/ppo/guidance/gain_space/reward |
| `envs/JSBSim/__init__.py` | `envs/jsbsim_env.py` | **migrated P1** | Minimal JSBSim wrapper initialized; rendering/missile code removed |
| `envs/JSBSim/core/simulatior.py` | `envs/jsbsim_env.py` | **migrated P1** | AircraftSimulator step/reset/state-extraction logic migrated to `_JSBSimAircraft` |
| `envs/JSBSim/utils/utils.py` | `envs/jsbsim_env.py` | **migrated P1** | `LLA2NEU` / `NEU2LLA` migrated as `lla2neu` / `neu2lla`; root_dir helpers replaced by config |
| `envs/JSBSim/core/NDI_controller.py` | `flight_control/low_level_controller.py`, `guidance/los_rate_guidance.py` | pending | Split NDI controller into low-level controller and guidance law |
| `envs/JSBSim/envs/singlecombat_env.py` | `envs/tracking_env.py` | **migrated P1** | 1v1 reset pattern and two-aircraft setup migrated; missile/task logic removed |
| `envs/JSBSim/envs/env_base.py` | `envs/tracking_env.py`, `envs/jsbsim_env.py` | **migrated P1** | BaseEnv multi-aircraft management simplified into `JSBSimEnv` and `CloseRangeTrackingEnv` |
| `envs/JSBSim/tasks/singlecombat_task.py` | `envs/reward.py`, `envs/termination.py` | pending | Extract reward and termination logic from task definitions |
| `envs/JSBSim/tasks/task_base.py` | `envs/tracking_env.py` | pending | Base task interface |
| `envs/JSBSim/reward_functions/*.py` | `envs/reward.py` | pending | Consolidate reward terms into RewardCalculator |
| `envs/JSBSim/termination_conditions/*.py` | `envs/termination.py` | pending | Consolidate termination conditions into TerminationChecker |
| `envs/env_wrappers.py` | `envs/observation.py`, `envs/tracking_env.py` | pending | Observation wrapper and feature engineering |
| `runner/jsbsim_runner.py` | `training/train_fixed_gain.py`, `training/train_bilevel.py`, `evaluation/monte_carlo.py` | pending | Split monolithic runner into training and evaluation modules |
| `runner/base_runner.py` | `training/` | pending | Base runner pattern; adapt to new modular structure |
| `algorithms/ppo/ppo_trainer.py` | `agents/ppo_agent.py` | pending | Migrate PPO training logic |
| `algorithms/ppo/ppo_actor.py` | `agents/policy_network.py` | pending | Actor network architecture |
| `algorithms/ppo/ppo_critic.py` | `agents/policy_network.py` | pending | Critic network architecture |
| `algorithms/utils/buffer.py` | `agents/replay_buffer.py` | pending | Rollout buffer for PPO |
| `algorithms/utils/mlp.py` | `agents/policy_network.py` | pending | MLP building blocks |
| `envs/JSBSim/core/render_tacview.py` | `utils/recorder.py` | pending | ACMI/TacView recording logic |
| `renders/render_1v1.py` | `utils/recorder.py`, `evaluation/` | pending | Visualization and rendering logic |
| `test_env.py` | `tests/test_*.py` | pending | Convert ad-hoc tests to pytest |
| `tests/test_jsbsim.py` | `tests/test_*.py` | pending | Legacy JSBSim tests |
| `tests/test_ppo.py` | `tests/test_*.py` | pending | Legacy PPO tests |
| `get_reward_figure/*.py` | `utils/plotting.py` | pending | Extract general plotting utilities; discard ad-hoc figure scripts |
| `GA.py`, `PID.py` | `gain_optimizer/`, `flight_control/` | pending | Review if generic GA/PID logic is reusable |
| `sko/*.py` | `gain_optimizer/` | pending | Optimization algorithms (GA, PSO, etc.); evaluate if needed |
| `generate_1m_scripts.py` | `scripts/*.ps1` | ignored | Replaced by PowerShell scripts |
| `run_*.py` | `scripts/*.ps1`, `training/` | ignored | Ad-hoc run scripts replaced by modular training entry points |
| `1111.py`, `python_try.py`, `jindou.py`, `w2.py` | — | ignored | Temporary/test scripts with no unique core logic |
| `check_sqlite.py`, `check_swanlab*.py` | — | ignored | Logging infrastructure checks |
| `compile_report.py`, `plot_cross_algo_comparison*.py` | `evaluation/` | pending | Extract evaluation report generation if reusable |
| `analyze_rewards.py` | `evaluation/metrics.py`, `utils/plotting.py` | pending | Reward analysis utilities |
| `*.acmi` | `outputs/acmi/` | ignored | Do not migrate historical ACMI files |
| `baseline_results_*` | `outputs/` or ignored | ignored | Do not migrate old result folders |
| `sensitivity_results/*` | `outputs/` or ignored | ignored | Do not migrate old sensitivity analysis results |
| `*.png`, `*.jpg`, `*.gif` | `outputs/figures/` | ignored | Do not migrate historical images |
| `*.csv`, `*.mat` | `outputs/tables/` | ignored | Do not migrate historical data files |
| `renders/*.py` | `evaluation/`, `utils/recorder.py` | pending | Review rendering scripts for reusable recording logic |
| `scripts/train/*.py` | `training/` | pending | Legacy training scripts; extract core logic only |
| `scripts/render/*.py` | `utils/recorder.py` | pending | Legacy rendering scripts |
| ` Discussion.md`, `parameter_tuning_log.md`, `revision_experiment_report.md` | `docs/` | pending | Extract methodological notes |
| `expected_sensitivity_analysis.md`, `experiment_record_conversation.md` | `docs/` | pending | Extract experiment design notes |

## Additional Notes

### JSBSim Data Files
The `envs/JSBSim/data/` directory contains the full JSBSim source and aircraft definitions. In the new project, we should:
- **Not copy** the entire `data/` tree into git.
- Reference the legacy JSBSim installation path via `legacy_project_root` in `config/env.yaml`.
- If a standalone JSBSim package is available via pip, prefer that.

### Model Checkpoints
Legacy `.pt` files (`baseline_model.pt`, `dodge_missile_model.pt`) are **not migrated**.
New checkpoints will be saved to `experiments/*/checkpoints/` (gitignored).
