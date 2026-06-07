# Reproduction Guide

## Environment Setup

1. Install Python 3.9+
2. Install JSBSim (ensure `<JSBSIM_ROOT>` contains a working JSBSim installation)
3. Run `pip install -e .`

## Running Experiments

Each experiment uses a YAML config in `config/experiment/`.

Results are written to `experiments/<id>_<name>/`.

## Expected Runtime

- Fixed-gain VPP training: ~4-8 hours (5M steps)
- Gain-only CEM: ~2-4 hours
- Bilevel training: ~20-40 hours (multiple policy iterations)

## Random Seeds

Default seed is 0. For statistical rigor, run with seeds 0, 1, 2 and report mean ± std.
