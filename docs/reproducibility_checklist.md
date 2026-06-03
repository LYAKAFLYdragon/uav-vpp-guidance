# Reproducibility Checklist

Use this checklist before running any experiment intended for paper inclusion.

## Environment

- [ ] Python version matches `requires-python` in `pyproject.toml` (≥3.9).
- [ ] Dependencies installed from `requirements.txt` or `pyproject.toml`.
- [ ] Editable install active: `pip install -e .`.
- [ ] JSBSim data directory accessible (if using JSBSim backend).
- [ ] GPU / CUDA versions recorded (if training with GPU).

## Code State

- [ ] Working on a clean branch (no uncommitted changes).
- [ ] Git commit hash recorded in run metadata (`run_metadata.json`).
- [ ] Branch name and dirty state logged.
- [ ] Feature branch (if applicable) noted in experiment notes.

## Configuration

- [ ] Config file is version-controlled (not edited locally post-commit).
- [ ] Backend explicitly set (`simple` or `jsbsim`).
- [ ] Random seed explicitly set and documented.
- [ ] Episode count and seed count documented.
- [ ] Output directory does not overwrite a previous run (use timestamps).

## Execution

- [ ] Smoke test passes: `python -m ... --smoke`.
- [ ] Full test suite passes: `python -m pytest tests/`.
- [ ] Lint clean: `ruff check .`.
- [ ] Format clean: `black --check .`.

## Outputs

- [ ] `run_metadata.json` present in output directory.
- [ ] `prediction_metrics.json` / `.csv` present.
- [ ] `summary.md` present and human-readable.
- [ ] Trajectories saved (if required for analysis).
- [ ] All metrics fields populated (no unexpected NaN).

## Post-Experiment

- [ ] Results copied to a version-controlled `results/` or `experiments/` directory.
- [ ] Figures generated with explicit seed/method labels.
- [ ] Statistical comparison includes confidence intervals or paired deltas.
- [ ] Any anomalies documented in experiment notes.
