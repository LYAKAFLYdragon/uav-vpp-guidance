# Migration Log

## Phase 1: Framework Creation (Current)

- Created clean project structure under `E:\uav-vpp-guidance`.
- Initialized git repository with `.gitignore` excluding outputs and large files.
- Created `pyproject.toml` for editable installation.
- Created YAML configuration system (`config/*.yaml`).
- Created core Python module skeletons with clear interfaces and TODOs.
- Created unit test skeletons (`tests/test_*.py`).
- Created PowerShell launch scripts (`scripts/*.ps1`).
- Created documentation (`README.md`, `docs/legacy_mapping.md`, etc.).
- Scanned legacy project structure and documented mapping.

## Phase 2: P1 Core Migration (Completed)

### Migrated Modules
- `src/uav_vpp_guidance/envs/jsbsim_env.py`
  - `_JSBSimAircraft`: single-aircraft JSBSim wrapper (from `simulatior.py`)
  - `JSBSimEnv`: multi-aircraft environment manager (from `env_base.py`)
  - `lla2neu` / `neu2lla`: coordinate conversion (from `utils.py`)
  - Supports: init, reload (reset), step, state extraction
- `src/uav_vpp_guidance/envs/tracking_env.py`
  - `CloseRangeTrackingEnv`: high-level env with own + target aircraft
  - `reset()`: initializes both aircraft with default ICs
  - `step()`: runs multiple sim steps per decision step
  - Observation: basic relative geometry (position, velocity, distance)
  - Reward / termination: placeholder (P2)

### Validation
- JSBSim F-16 model loads and steps successfully
- `test_jsbsim_env_p1.py`: 5 passed, 2 skipped (pymap3d optional)
- Full test suite: 24 passed, 2 skipped

## Next Steps (Phase 3)

1. Migrate reward functions and termination conditions (P2).
2. Implement fixed-gain pursuit baseline.
3. Implement virtual pursuit point generator.
4. Implement LOS-rate guidance law.
5. Run fixed-gain VPP policy training.
6. Implement gain-only CEM optimization.
7. Implement strategy-gain bilevel training.
