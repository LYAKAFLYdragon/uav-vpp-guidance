from pathlib import Path
import importlib.util

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiment/mild_disadvantage_handoff_source_traces_isolated_v1.yaml"
RUNNER = ROOT / "scripts/run_mild_disadvantage_handoff_source_traces_isolated_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("isolated_source_trace_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_contract_is_independent_and_12_episodes():
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert spec["experiment"]["source_id"] == "THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904"
    assert spec["protocol"]["one_fresh_process_per_episode"] is True
    assert spec["protocol"]["source_handoff_hash_required"] is True
    runner = load_runner()
    assert len(runner.episode_jobs(spec)) == 12


def test_seed_schedule_matches_historical_protocol():
    runner = load_runner()
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    jobs = runner.episode_jobs(spec)
    assert jobs[0]["seed"] == 981300
    assert jobs[2]["seed"] == 981302
    assert jobs[3]["seed"] == 981400
    assert jobs[6]["seed"] == 981350
