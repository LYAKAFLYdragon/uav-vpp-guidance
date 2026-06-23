# Agent Guidance: UAV VPP Guidance

This file documents the **structured contracts** that keep the code, the
experiments, and the paper narrative consistent. Any change that touches
observation schema, backend selection, or config overrides must update the
corresponding contract and tests.

## 1. Observation schema contract

`src/uav_vpp_guidance/envs/observation.py::build_observation` appends features
in a fixed order:

| Segment | Keys | Count | Config switch |
|---|---|---|---|
| Base | `range_m`, `range_rate_mps`, `altitude_diff_m`, `speed_diff_mps`, `los_azimuth_sin/cos`, `los_elevation_sin/cos`, `ata_sin/cos`, `aa_sin/cos`, `own_speed`, `target_speed`, `own_altitude`, `target_altitude` | 16 | always |
| Gains | `gain_k_los`, `gain_k_pos` | +2 | `observation.include_gains` |
| Guidance state | `vp_error_x/y/z` | +3 | `observation.include_guidance_state` |
| Saturation | `nz_saturated`, `roll_rate_saturated`, `throttle_saturated` | +3 | `observation.include_saturation` |
| Prediction | `pred_rel_x/y/z`, `pred_disp_x/y/z`, `pred_vel_x/y/z`, `pred_var_x/y/z`, `pred_valid`, `pred_fallback` | +14 | `trajectory_prediction` integration flags |

`CloseRangeTrackingEnv._get_observation` returns an `observation_schema` dict
with boolean flags, `dim`, and `feature_names`. The schema is also embedded in
`provenance`.

**Rule**: do not change the base-feature order without bumping a schema version
and updating all downstream checkpoints / tests.

## 2. Backend contract

Backend resolution in `CloseRangeTrackingEnv.__init__`:

1. `config["backend"]` if present.
2. Otherwise legacy `config["env"]["use_jsbsim"]` (`True` -> `jsbsim`).
3. If JSBSim init fails:
   - `strict_backend=True` -> raise `RuntimeError(..."strict_backend=True"...)`
   - `strict_backend=False` -> fallback to `simple`

The env records:
- `info["backend"]` and `provenance["backend"]` = final active backend
- `info["backend_fallback_occurred"]` and `provenance["backend_fallback_occurred"]`
- `info["backend_fallback_reason"]` and `provenance["backend_fallback_reason"]`

**Rule**: training/evaluation scripts must not silently force a backend. They may
fill in a safe default with `setdefault`, but any actual override must be
recorded via `record_config_override`.

## 3. Config override contract

Any script that mutates the loaded YAML config must record the mutation so it
appears in `provenance["config_overrides"]`.

Use:

```python
from uav_vpp_guidance.common.provenance import record_config_override

record_config_override(
    config,
    key="guidance.mode_switch.enabled",
    new_value=False,
    old_value=old_value,
    source="my_script.py:reason",
)
```

This applies to hard-coded overrides such as:
- Disabling `mode_switch` for gain-only / bilevel training
- Setting backend defaults
- CLI flags that write into config (`--seed`, `--backend`, etc.)

**Rule**: if a script changes a config value loaded from YAML, it must call
`record_config_override`.

## 4. Testing contract

The following are required and already covered by tests:

- `tests/test_tracking_env_no_prediction.py` checks `observation_schema` flags,
  dimensions, `feature_names`, saturation flags, backend fallback, and
  provenance.
- `tests/test_common_provenance.py` checks `record_config_override`.

When adding a new observation extension:
1. Add it to `build_observation` with a deterministic name.
2. Add an `include_*` config flag in `CloseRangeTrackingEnv.__init__`.
3. Update `observation_schema` with the flag and `feature_names`.
4. Add a test asserting `dim` change and flag semantics.
5. Document it here and in `README.md`.

When adding a new CLI flag that writes to config:
1. Record the override via `record_config_override`.
2. Add a test verifying the override appears in `provenance["config_overrides"]`.

## 5. Portable artifact pipeline contract

The pipeline layer lives in `src/uav_vpp_guidance/pipeline/` and is designed to
make Stage 6G/6H result production reproducible across machines.

### 5.1 Layered design

| Layer | Module | Responsibility |
|---|---|---|
| Common utilities | `common.git`, `common.hash`, `common.manifest`, `common.artifact_contract`, `common.provenance` | Unified git info, canonical hashing, manifest dataclass, artifact contract, config override recording |
| Pipeline core | `pipeline.stage_runner`, `pipeline.artifact_bundle` | Base `PipelineStage`, config snapshotting, manifest writing, artifact validation, portable bundle creation/verification |
| Concrete stages | `pipeline.stages.stage6g_guidance_limitation`, `pipeline.stages.stage6h_gain_only`, `pipeline.stages.stage6h_bilevel` | Wrap existing Stage 6G/6H scripts, declare contracts, hash inputs |
| Orchestrator | `scripts/run_stage6g6h_pipeline.py` | Run one or more stages, produce top-level pipeline manifest, create bundles |
| Verification | `scripts/verify_artifact_bundle.py` | Verify bundle integrity against stored hashes and contract |

### 5.2 Manifest schema (`RunManifest`)

Every stage writes a `run_manifest.json` with:

- `schema_version`, `stage_name`, `stage_version`
- `start_time`, `end_time`, `status`
- `command_line`, `working_directory`, `output_dir`
- `config_path`, `config_hash`, `resolved_config`
- `git_info`: full SHA, short SHA, branch, dirty flag, description
- `python_version`, `platform`
- `inputs`: checkpoint/config file info with SHA-256, size, mtime
- `outputs`: produced file info with SHA-256
- `artifacts_present`: boolean map
- `config_overrides`: list of recorded config mutations
- `paper_safe` and `invalid_for_paper_reasons`

### 5.3 Artifact contract

Each stage declares a required/optional artifact list. After running, the stage:
1. Checks that every required file/directory exists.
2. Checks manifest consistency.
3. Writes `artifact_contract.json`.
4. Fails validation (and marks not paper-safe) if anything is missing.

### 5.4 Portable bundle

`ArtifactBundle` copies `run_manifest.json`, `artifact_contract.json`,
`resolved_config.yaml`, and all declared outputs into a single directory with a
`bundle_info.json` containing SHA-256 hashes. The bundle can be archived,
transferred, and verified on another host with
`scripts/verify_artifact_bundle.py`.

### 5.5 Adding a new pipeline stage

1. Subclass `PipelineStage`.
2. Set `stage_name`, `stage_version`, and `artifact_contract`.
3. Implement `run()`. Record inputs with `manifest.record_input_file()` and
   outputs with `manifest.record_output_file()`.
4. Call `self.snapshot_config()` and `self.finalize()`.
5. Add the stage to `scripts/run_stage6g6h_pipeline.py`.
6. Add tests in `tests/test_pipeline_core.py` or a new
   `tests/test_pipeline_stages.py`.

### 5.6 Existing scripts

The existing standalone scripts (`scripts/run_stage6g_guidance_limitation_probe.py`,
`scripts/run_gain_only_cem.py`, `scripts/train_bilevel.py`) remain usable. The
pipeline wrappers call them as subprocesses and layer the manifest/contract on
top. Over time, new work should prefer the pipeline entry points.
