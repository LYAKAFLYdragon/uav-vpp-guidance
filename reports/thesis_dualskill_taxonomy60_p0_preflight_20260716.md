# Taxonomy60 P0 Preflight Record

**Checklist:** `thesis_master_evidence_closure_checklist_v2_20260716_zh.md`  
**Scope:** development preflight only; no formal Source ID, JSBSim comparison, policy training, or result output was created.  
**Decision:** `PASS - P1 manifest development may begin; P2 formal smoke remains required before any Source execution.`

## Frozen Development Baseline

| Item | Verified value |
| --- | --- |
| Development worktree | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1` |
| Clean commit | `55f45cac650ebc697badb0555c652c11b2be9287` at inspection time |
| Formal source budget | No `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` or `-R1` output root exists |
| Formal interpreter candidate | `D:/Anaconda3/envs/msrl/python.exe` |
| Interpreter contract | Python 3.9.21, Torch 2.6.0+cpu, NumPy 1.24.4, PyYAML 6.0.2, JSBSim 1.2.3 |
| Targeted test command | `python -m pytest tests/test_thesis_taxonomy_ablation30_manifest.py tests/test_thesis_taxonomy_ablation30_configs.py tests/test_analyze_thesis_taxonomy_frozen_ablation30.py tests/test_hierarchical_commander_policy.py -q --tb=short` |
| Targeted test result | `54 passed in 8.61s` |

`D:/Anaconda3/envs/jsbenv/python.exe` remains usable for its CUDA/JSBSim assets but is Python 3.8.0; two existing analyzer tests use `dict | dict`, so that interpreter fails before testing research behavior. It is therefore not the formal Taxonomy60 interpreter unless compatibility is separately repaired and revalidated. Base Anaconda is also excluded because its NumPy package metadata and imported version disagree.

## Asset Hash Verification

All asset files exist under `E:/uav-vpp-guidance-thesis-five-state-v1-assets/` and match the frozen shared-intent v1 manifest exactly.

| Asset | SHA-256 match | Contract |
| --- | --- | --- |
| Canonical PPO commander | pass | 19-D observation, 2 discrete modes |
| Head-on specialist | pass | 19-D observation, 3-D VPP action |
| Crossing specialist | pass | 18-D observation, 3-D VPP action |
| Trajectory predictor | pass | frozen predicted-target VPP input |
| End-to-end opponent | pass | 16-D observation, 3-D action |
| Independent PPO/VPP opponent | pass | 16-D observation, 3-D action |

The independent PPO/VPP opponent is eligible only for its documented limited preparation envelope. Taxonomy60 must report it as a separate provenance-limited opponent condition, not as an Elo-ranked or universally stronger adversary.

## Capacity and Reuse Audit

| Observation | Value | Consequence |
| --- | ---: | --- |
| E-drive free space | 311,362,375,680 bytes, about 290.0 GiB | Sufficient for a conservative Taxonomy60 raw-output reserve |
| Taxonomy30 expert output | 2.413 GiB for 120 raw episodes | Historical size reference |
| Taxonomy30 end-to-end output | 1.894 GiB for 120 raw episodes | Historical size reference |
| Linear Taxonomy60 estimate | about 16.2 GiB for 900 raw episodes | Reserve at least 50 GiB before formal execution |

The existing `run_jsbsim_hrl_comparison.py` resolves arbitrary registry stages and can instantiate a neural opponent from a checkpoint/config entry. The current Taxonomy30 config contains only `expert` and `end_to_end` entries, four methods, and a 30-scene manifest. Adding the independent opponent, M4 unlocked PPO, and the Heldout60 contract is therefore P1/P2 implementation work, not an unverified claim of existing support.

## P0 Boundary and Next Step

- P0 does **not** approve the formal Source, a training run, a new policy, or any performance conclusion.
- P1 must create a new 60-scene manifest and prove its geometry, mirror balance, and full-signature disjointness against all frozen evidence families.
- P2 must add the five-method/three-opponent config and verifier, then complete the 15-episode engineering smoke using the frozen Python 3.9.21 interpreter.
