"""
Numeric-diagnostic tests for the control-method rationality audit.

These are closed-form math facts — they must PASS without any harness module.
Each pure function is implemented inline so these tests are entirely self-contained.

Validates: Requirements 1.3, 1.4, 2.1, 2.2, 3.1, 6.2, 7.2, 9.4
"""

import json
import math
from pathlib import Path
import re
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Pure diagnostic functions (inline — no external harness dependency)
# ---------------------------------------------------------------------------


def filter_time_constant(alpha: float, dt: float) -> float:
    """Effective first-order filter time constant: τ = -dt / ln(1 - α)."""
    if alpha == 1.0:
        return 0.0
    return -dt / math.log(1.0 - alpha)


def offset_angular_deviation(offset_m: float, range_m: float) -> float:
    """Angular deviation (degrees) for a metric offset at a given range."""
    return math.degrees(math.atan2(offset_m, range_m))


def cem_elite_count(candidates: int, elite_ratio: float) -> int:
    """CEM elite selection: max(1, int(n * ratio))."""
    return max(1, int(candidates * elite_ratio))


def sincos_injective_witness() -> dict:
    """Return sin/cos pairs for 30° and 330° showing (sin,cos) is injective."""
    a_rad = math.radians(30.0)
    b_rad = math.radians(330.0)
    return {
        "30deg": (math.sin(a_rad), math.cos(a_rad)),
        "330deg": (math.sin(b_rad), math.cos(b_rad)),
    }


def dwell_seconds(steps: int, dt: float) -> float:
    """Total dwell time in seconds: steps * dt."""
    return steps * dt


def range_travel(closing_mps: float, seconds: float) -> float:
    """Range closed during a dwell period at constant closing speed."""
    return closing_mps * seconds


def roll_rate_limit_deg_s(limit_rad_s: float) -> float:
    """Convert roll-rate limit from rad/s to deg/s."""
    return math.degrees(limit_rad_s)


# ---------------------------------------------------------------------------
# Test 1: filter_time_constant (F03 / F06 / F11 phase-lag claim)
# ---------------------------------------------------------------------------


class TestFilterTimeConstant:
    """Validates: Requirements 1.3, 2.2 — filter phase-lag claim."""

    def test_alpha_0p3_dt_0p2(self):
        """τ at α=0.3, dt=0.2 s should be ≈ 0.5601 s."""
        tau = filter_time_constant(0.3, 0.2)
        assert tau == pytest.approx(0.5601, abs=0.001)

    @pytest.mark.parametrize(
        "alpha_lo,alpha_hi",
        [
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.5),
            (0.5, 0.7),
            (0.7, 0.9),
        ],
    )
    def test_tau_decreases_monotonically_with_alpha(self, alpha_lo, alpha_hi):
        """τ must decrease as α increases (faster filter → shorter time constant)."""
        dt = 0.2
        tau_lo = filter_time_constant(alpha_lo, dt)
        tau_hi = filter_time_constant(alpha_hi, dt)
        assert tau_lo > tau_hi

    def test_alpha_1p0_has_zero_time_constant(self):
        """A fully refreshed filter has no memory and therefore τ = 0 s."""
        assert filter_time_constant(1.0, 0.2) == 0.0
# ---------------------------------------------------------------------------


class TestOffsetAngularDeviation:
    """Validates: Requirements 2.1 — VPP metric offset semantics drift."""

    def test_500m_at_2500m(self):
        """500 m lateral offset at 2500 m range → ≈ 11.31°."""
        angle = offset_angular_deviation(500.0, 2500.0)
        assert angle == pytest.approx(11.31, abs=0.01)

    @pytest.mark.parametrize(
        "offset_m,far_range,near_range",
        [
            (500.0, 2500.0, 1000.0),
            (500.0, 5000.0, 2500.0),
            (800.0, 3000.0, 1500.0),
        ],
    )
    def test_same_offset_larger_angle_at_shorter_range(
        self, offset_m, far_range, near_range
    ):
        """The SAME metric offset yields a strictly LARGER angle at shorter range."""
        angle_far = offset_angular_deviation(offset_m, far_range)
        angle_near = offset_angular_deviation(offset_m, near_range)
        assert angle_near > angle_far


# ---------------------------------------------------------------------------
# Test 3: cem_elite_count (F08 optimizer credibility)
# ---------------------------------------------------------------------------


class TestCemEliteCount:
    """Validates: Requirements 3.1 — CEM elite selection."""

    @pytest.mark.parametrize(
        "candidates,elite_ratio,expected",
        [
            (12, 0.25, 3),
            (70, 0.25, 17),
        ],
    )
    def test_known_values(self, candidates, elite_ratio, expected):
        """Concrete elite-count checks from design doc."""
        assert cem_elite_count(candidates, elite_ratio) == expected

    @pytest.mark.parametrize("candidates", [1, 2, 3])
    def test_max1_floor_for_tiny_populations(self, candidates):
        """max(1, ...) floor guarantees at least 1 elite even for tiny populations."""
        # With elite_ratio=0.25, int(1*0.25)=0, int(2*0.25)=0, int(3*0.25)=0
        count = cem_elite_count(candidates, 0.25)
        assert count >= 1


# ---------------------------------------------------------------------------
# Test 4: sincos_injective_witness (F15 disproof)
# ---------------------------------------------------------------------------


class TestSincosInjectiveWitness:
    """Validates: Requirements 6.2 — executable disproof of F15."""

    def test_30_and_330_share_cos_differ_in_sin(self):
        """30° and 330° share cos (≈0.866) but differ in sin (+0.5 vs -0.5)."""
        witness = sincos_injective_witness()
        sin_30, cos_30 = witness["30deg"]
        sin_330, cos_330 = witness["330deg"]

        # cos values are the same
        assert cos_30 == pytest.approx(cos_330, abs=1e-10)
        assert cos_30 == pytest.approx(0.866, abs=0.001)

        # sin values differ (proving injectivity of the pair)
        assert sin_30 == pytest.approx(0.5, abs=1e-10)
        assert sin_330 == pytest.approx(-0.5, abs=1e-10)
        assert sin_30 != pytest.approx(sin_330, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 5: dwell_seconds and range_travel (F16/F17 dwell adequacy)
# ---------------------------------------------------------------------------


class TestDwellAndRangeTravel:
    """Validates: Requirements 7.2 — hybrid dwell adequacy."""

    def test_dwell_3_steps_at_0p2(self):
        """3 steps × 0.2 s = 0.6 s dwell."""
        assert dwell_seconds(3, 0.2) == pytest.approx(0.6)

    @pytest.mark.parametrize(
        "closing_mps,dwell_s,expected_travel_m",
        [
            # Representative head-on closing speed ~300 m/s (each at ~150 m/s)
            (300.0, 0.6, 180.0),
            # Tail-chase closing speed ~50 m/s
            (50.0, 0.6, 30.0),
        ],
    )
    def test_range_travel_at_closing_speed(
        self, closing_mps, dwell_s, expected_travel_m
    ):
        """Range closed during the 0.6 s dwell period at stated closing speed."""
        assert range_travel(closing_mps, dwell_s) == pytest.approx(expected_travel_m)


# ---------------------------------------------------------------------------
# Test 6: roll_rate_limit_deg_s (F04 comparison to F-16 reference)
# ---------------------------------------------------------------------------


class TestRollRateLimitDegS:
    """Validates: Requirements 1.4 — saturation limits vs F-16-class reference."""

    def test_1p5_rad_s(self):
        """1.5 rad/s ≈ 85.94°/s — well below the ~270°/s F-16-class capability."""
        assert roll_rate_limit_deg_s(1.5) == pytest.approx(85.94, abs=0.01)


# ---------------------------------------------------------------------------
# Test 7: audit report output (Phase 4 verification)
# ---------------------------------------------------------------------------


def test_audit_emits_triaged_findings(tmp_path):
    """The audit CLI emits a complete, source-backed triage report.

    Validates: Requirements 9.1, 9.2, 9.3, 1.5, 2.4, 3.3, 4.3, 5.3, 6.3,
    7.3, 8.2
    """
    repo_root = Path(__file__).resolve().parents[1]
    out_md = tmp_path / "control_method_rationality_audit_findings_zh.md"
    out_json = tmp_path / "control_method_rationality_audit_findings.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_control_method_rationality.py",
            "--guidance-config",
            "config/guidance.yaml",
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ],
        cwd=repo_root,
        check=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    findings = report["findings"]
    expected_ids = [f"F{index:02d}" for index in range(1, 19)]
    controlled_verdicts = {
        "confirmed_defect",
        "confirmed_risk",
        "by_design_ok",
        "premise_incorrect",
    }

    assert len(findings) == 18
    assert [finding["id"] for finding in findings] == expected_ids
    assert len({finding["id"] for finding in findings}) == 18

    for finding in findings:
        assert finding["verdict"] in controlled_verdicts
        assert finding["evidence_paths"]
        assert finding["failure_mode"]
        assert finding["recommendation"]
        assert "priority" in finding
        assert finding["confidence"]
        if finding["verdict"] == "premise_incorrect":
            assert finding["priority"] == 0
        if finding["verdict"].startswith("confirmed_"):
            assert finding["priority"] >= 1

    f15 = next(finding for finding in findings if finding["id"] == "F15")
    assert f15["verdict"] == "premise_incorrect"
    assert f15["priority"] == 0

    assert report["source_facts"]
    assert report["diagnostics"]
    summary = report["summary"]
    for verdict in controlled_verdicts:
        assert verdict in summary
    assert "remediation_backlog" in summary

    provenance = report["provenance"]
    for field in ("git_commit", "git_branch", "python_version", "platform"):
        assert provenance[field]

    assert "F18" in report["audit_scope"]["analytic_only_findings"]


# ---------------------------------------------------------------------------
# Test 8: source facts from real instances and configuration (Phase 3 task 4.3)
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OBSERVATION_FEATURE_NAMES = [
    "range_m",
    "range_rate_mps",
    "altitude_diff_m",
    "speed_diff_mps",
    "los_azimuth_sin",
    "los_azimuth_cos",
    "los_elevation_sin",
    "los_elevation_cos",
    "ata_sin",
    "ata_cos",
    "aa_sin",
    "aa_cos",
    "own_speed",
    "target_speed",
    "own_altitude",
    "target_altitude",
]


def _read_yaml(relative_path: str) -> dict:
    """Load a committed configuration without mutating it."""
    import yaml

    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _load_real_source_facts():
    """Load the audit snapshot using explicit real-repository input paths."""
    from uav_vpp_guidance.evaluation.control_method_audit import load_source_facts

    return load_source_facts(
        {
            "guidance_config": REPO_ROOT / "config/guidance.yaml",
            "gain_space_config": REPO_ROOT / "config/gain_space.yaml",
            "source_root": REPO_ROOT / "src/uav_vpp_guidance",
        }
    )


class TestAuditSourceFacts:
    """Validates: Requirements 1.1, 1.2, 1.4, 2.3, 3.1, 5.1, 5.2, 6.1, 7.1, 7.2."""

    def test_los_instance_limits_and_audit_snapshot_follow_real_config(self, tmp_path):
        """LOS limits/filter come from a real configured instance, not audit literals."""
        import yaml

        from uav_vpp_guidance.evaluation.control_method_audit import load_source_facts
        from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance

        config = _read_yaml("config/guidance.yaml")
        los = LOSRateGuidance(config["guidance"])
        assert (los.nz_min, los.nz_max) == (-2.0, 7.0)
        assert (los.roll_rate_min, los.roll_rate_max) == (-1.5, 1.5)
        assert (los.throttle_min, los.throttle_max) == (0.4, 0.9)
        assert los.alpha_filter == pytest.approx(0.3)

        facts = _load_real_source_facts()
        assert facts.guidance_limits == {
            "nz_min": los.nz_min,
            "nz_max": los.nz_max,
            "roll_rate_min": los.roll_rate_min,
            "roll_rate_max": los.roll_rate_max,
            "throttle_min": los.throttle_min,
            "throttle_max": los.throttle_max,
        }
        assert facts.los_gains["alpha_filter"] == los.alpha_filter

        # Change only an in-memory-derived temporary config.  The audit must
        # mirror the newly constructed instance rather than retained literals.
        config["guidance"]["gains"]["alpha_filter"] = 0.45
        config["guidance"]["limits"]["nz_max"] = 6.5
        alternate_config = tmp_path / "guidance.yaml"
        alternate_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        alternate_los = LOSRateGuidance(config["guidance"])
        alternate_facts = load_source_facts(
            {
                "guidance_config": alternate_config,
                "gain_space_config": REPO_ROOT / "config/gain_space.yaml",
                "source_root": REPO_ROOT / "src/uav_vpp_guidance",
            }
        )
        assert alternate_facts.los_gains["alpha_filter"] == alternate_los.alpha_filter
        assert alternate_facts.guidance_limits["nz_max"] == alternate_los.nz_max

    def test_cem_instance_config_and_elite_count_are_source_derived(self):
        """The real configured CEM instance supplies its population and elite ratio."""
        from uav_vpp_guidance.evaluation.control_method_audit import cem_elite_count
        from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
        from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace

        config = _read_yaml("config/gain_space.yaml")
        gain_space = GainSpace(config["gain_space"])
        optimizer = CEMGainOptimizer(gain_space, config["gain_optimizer"])
        assert optimizer.candidates == 12
        assert optimizer.elite_ratio == pytest.approx(0.25)
        assert cem_elite_count(optimizer.candidates, optimizer.elite_ratio) == 3

        facts = _load_real_source_facts()
        assert facts.cem_config["candidates"] == optimizer.candidates
        assert facts.cem_config["elite_ratio"] == optimizer.elite_ratio
        assert facts.cem_config["dimension"] == len(gain_space.names)
        assert cem_elite_count(
            facts.cem_config["candidates"], facts.cem_config["elite_ratio"]
        ) == 3

        # The committed config currently has five gain dimensions, not the
        # seven-dimensional premise recorded in the audit plan.
        assert len(gain_space.names) == 5

    def test_base_observation_is_real_16d_agents_schema_order(self):
        """A minimal real state pair emits exactly the fixed 16-feature base segment."""
        from uav_vpp_guidance.envs.observation import build_observation

        own_state = {
            "position_neu": [0.0, 0.0, 5000.0],
            "velocity": [250.0, 0.0, 0.0],
            "altitude_m": 5000.0,
        }
        target_state = {
            "position_neu": [1000.0, 0.0, 5000.0],
            "velocity": [250.0, 0.0, 0.0],
            "altitude_m": 5000.0,
        }
        observation, feature_names = build_observation(
            own_state, target_state, return_feature_names=True
        )
        assert len(observation) == 16
        assert feature_names == BASE_OBSERVATION_FEATURE_NAMES

        facts = _load_real_source_facts()
        assert facts.observation_base_dim == len(observation)
        assert facts.observation_feature_names == feature_names

    def test_config_and_constructor_defaults_expose_both_mismatches(self):
        """The audit records the real mode-switch and VPP config/default mismatches."""
        from uav_vpp_guidance.guidance.hybrid_guidance import HybridGuidance
        from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator

        config = _read_yaml("config/guidance.yaml")
        guidance = config["guidance"]
        configured_vpp = VirtualPointGenerator(config["virtual_point"])
        default_vpp = VirtualPointGenerator({})
        hybrid = HybridGuidance(guidance)

        assert guidance["mode_switch"]["aspect_threshold_deg"] == 25.0
        assert configured_vpp.action_dim == 3
        assert default_vpp.action_dim == 5
        assert configured_vpp.action_dim != default_vpp.action_dim

        facts = _load_real_source_facts()
        assert facts.mode_switch_config["aspect_threshold_deg"] == 25.0
        assert facts.mode_switch_code_default == 15.0
        assert facts.vpp_config["action_dim"] == configured_vpp.action_dim
        assert facts.vpp_generator_default_action_dim == default_vpp.action_dim
        assert facts.hybrid_params["hysteresis_m"] == hybrid.hysteresis_m == 500.0
        assert facts.hybrid_params["min_dwell_steps"] == hybrid.min_dwell_steps == 3

    def test_reward_defaults_are_real_calculator_values(self):
        """The audit reports reward weights and terminals from RewardCalculator defaults."""
        from uav_vpp_guidance.envs.reward import RewardCalculator

        reward = RewardCalculator({})
        assert reward.w_safety == 2.0
        assert reward.w_angle == 0.8
        assert reward.terminal_success == 200.0
        assert reward.terminal_crash == -300.0

        facts = _load_real_source_facts()
        assert facts.reward_weights["w_safety"] == reward.w_safety
        assert facts.reward_weights["w_angle"] == reward.w_angle
        assert facts.reward_weights["terminal_success"] == reward.terminal_success
        assert facts.reward_weights["terminal_crash"] == reward.terminal_crash


# ---------------------------------------------------------------------------
# Test 9: parametrized triage rules and generic self-check invariants (Task 4.4)
# ---------------------------------------------------------------------------


from dataclasses import replace


def _synthetic_triage_inputs(
    *,
    premise_disproved: bool,
    wrong_relation: bool,
    quantified_failure_mode: bool,
):
    """Build isolated inputs that exercise only the ordered F15 triage rules."""
    from uav_vpp_guidance.evaluation.control_method_audit import (
        DiagnosticResults,
        SourceFacts,
    )

    return (
        SourceFacts(
            evidence_paths={"F15": ["synthetic://F15"]},
            source_checks={
                "F15_wrong_relation": wrong_relation,
                "F15_quantified": quantified_failure_mode,
            },
        ),
        DiagnosticResults(
            sincos_witness={"synthetic": (0.5, 0.866)} if premise_disproved else {}
        ),
    )


def _synthetic_records():
    """Create a complete valid record set without loading real audit facts."""
    from uav_vpp_guidance.evaluation.control_method_audit import FindingRecord

    records = []
    for index in range(1, 19):
        finding_id = f"F{index:02d}"
        premise_incorrect = finding_id == "F15"
        records.append(
            FindingRecord(
                id=finding_id,
                dimension="synthetic",
                title_zh=f"Synthetic {finding_id}",
                verdict="premise_incorrect" if premise_incorrect else "by_design_ok",
                evidence_paths=[f"synthetic://{finding_id}"],
                diagnostic_ref=[],
                observed_fact="synthetic fact",
                failure_mode="synthetic failure mode",
                recommendation="synthetic recommendation",
                priority=0 if premise_incorrect else 1,
                confidence="high",
            )
        )
    return records


class TestAuditTriageRules:
    """Validates: Requirements 1.5, 2.4, 3.3, 4.3, 5.3, 6.3, 7.3, 8.2, 9.2."""

    @pytest.mark.parametrize(
        (
            "premise_disproved,wrong_relation,quantified_failure_mode,"
            "expected_verdict,expected_priority"
        ),
        [
            (True, True, True, "premise_incorrect", 0),
            (False, True, True, "confirmed_defect", None),
            (False, False, True, "confirmed_risk", None),
            (False, False, False, "by_design_ok", None),
        ],
    )
    def test_triage_applies_ordered_rules_to_synthetic_inputs(
        self,
        premise_disproved,
        wrong_relation,
        quantified_failure_mode,
        expected_verdict,
        expected_priority,
    ):
        """Disproof takes precedence over defect, which precedes risk and design-ok."""
        from uav_vpp_guidance.evaluation.control_method_audit import triage

        facts, diagnostics = _synthetic_triage_inputs(
            premise_disproved=premise_disproved,
            wrong_relation=wrong_relation,
            quantified_failure_mode=quantified_failure_mode,
        )
        record = next(record for record in triage(facts, diagnostics) if record.id == "F15")

        assert record.verdict == expected_verdict
        if expected_priority is not None:
            assert record.priority == expected_priority
        else:
            assert record.priority >= 1

    def test_triage_is_deterministic_for_identical_synthetic_inputs(self):
        """Triage is a pure function of the supplied facts and diagnostics."""
        from uav_vpp_guidance.evaluation.control_method_audit import triage

        facts, diagnostics = _synthetic_triage_inputs(
            premise_disproved=False,
            wrong_relation=False,
            quantified_failure_mode=True,
        )

        assert triage(facts, diagnostics) == triage(facts, diagnostics)


class TestAuditSelfCheckInvariants:
    """Validate each generic self_check invariant using synthetic records only."""

    @pytest.mark.parametrize(
        "name,mutate,error_pattern",
        [
            (
                "wrong record count",
                lambda records: records[:-1],
                "records must contain F01..F18 exactly once",
            ),
            (
                "duplicate id",
                lambda records: records.__setitem__(1, replace(records[1], id="F01"))
                or records,
                "records must contain F01..F18 exactly once",
            ),
            (
                "unknown verdict label",
                lambda records: records.__setitem__(
                    0, replace(records[0], verdict="unrecognized")
                )
                or records,
                "unknown verdict for F01",
            ),
            (
                "missing evidence path",
                lambda records: records.__setitem__(
                    0, replace(records[0], evidence_paths=[])
                )
                or records,
                "missing evidence path for F01",
            ),
            (
                "premise incorrect with nonzero priority",
                lambda records: records.__setitem__(
                    14, replace(records[14], priority=1)
                )
                or records,
                "premise_incorrect finding F15 must have priority 0",
            ),
            (
                "confirmed finding with zero priority",
                lambda records: records.__setitem__(
                    0,
                    replace(records[0], verdict="confirmed_risk", priority=0),
                )
                or records,
                "confirmed finding F01 must have positive priority",
            ),
            (
                "zero premise incorrect findings",
                lambda records: records.__setitem__(
                    14, replace(records[14], verdict="by_design_ok", priority=1)
                )
                or records,
                "at least one premise_incorrect record is required",
            ),
        ],
        ids=lambda case: case,
    )
    def test_self_check_rejects_each_required_invariant_violation(
        self, name, mutate, error_pattern
    ):
        """Each invariant must reject the corresponding malformed synthetic report."""
        from uav_vpp_guidance.evaluation.control_method_audit import self_check

        with pytest.raises(ValueError, match=error_pattern):
            self_check(mutate(_synthetic_records()))


# ---------------------------------------------------------------------------
# Test 10: concrete preservation checks (Phase 4 task 6)
# ---------------------------------------------------------------------------


PRESERVATION_BASELINE_PATH = (
    REPO_ROOT
    / ".kiro"
    / "specs"
    / "control-method-rationality-audit"
    / "preservation_baseline.sha256.json"
)
PROTECTED_DIRECTORIES = (
    "src/uav_vpp_guidance/guidance",
    "src/uav_vpp_guidance/flight_control",
    "src/uav_vpp_guidance/virtual_point",
    "src/uav_vpp_guidance/gain_optimizer",
    "config",
)
PROTECTED_FILES = (
    "src/uav_vpp_guidance/envs/reward.py",
    "src/uav_vpp_guidance/envs/observation.py",
    "src/uav_vpp_guidance/envs/tracking_env.py",
)
ALLOWED_CHANGED_PATHS = {
    "scripts/audit_control_method_rationality.py",
    "src/uav_vpp_guidance/evaluation/control_method_audit.py",
    "tests/test_control_method_audit.py",
    "reports/control_method_rationality_audit_findings_20260725_zh.md",
    "reports/control_method_rationality_audit_findings_20260725.json",
}
ALLOWED_CHANGED_PREFIXES = (
    ".kiro/specs/control-method-rationality-audit/",
)


def _sha256(path: Path) -> str:
    """Return the byte-level SHA256 digest for one protected file."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_sha256_manifest() -> dict[str, str]:
    """Recompute the complete protected-set manifest deterministically."""
    manifest = {}
    for relative_directory in PROTECTED_DIRECTORIES:
        directory = REPO_ROOT / relative_directory
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                manifest[path.relative_to(REPO_ROOT).as_posix()] = _sha256(path)
    for relative_file in PROTECTED_FILES:
        path = REPO_ROOT / relative_file
        manifest[relative_file] = _sha256(path)
    return dict(sorted(manifest.items()))


def _git_lines(*args: str) -> set[str]:
    """Return non-empty, repository-relative path lines from a git command."""
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.replace("\\", "/") for line in completed.stdout.splitlines() if line}


def _is_allowlisted_audit_change(relative_path: str) -> bool:
    """Whether a changed path is an audit artifact permitted by Property 2."""
    return relative_path in ALLOWED_CHANGED_PATHS or relative_path.startswith(
        ALLOWED_CHANGED_PREFIXES
    )


def test_protected_control_and_config_paths_match_phase1_sha256_manifest():
    """Every protected source/config byte matches the persisted Phase 1 baseline.

    Validates: Requirements 10.1, 10.2, 10.3, 10.4
    """
    baseline = json.loads(PRESERVATION_BASELINE_PATH.read_text(encoding="utf-8"))
    actual = _protected_sha256_manifest()

    assert actual.keys() == baseline.keys(), (
        "protected-set membership changed; missing="
        f"{sorted(set(baseline) - set(actual))}, extra={sorted(set(actual) - set(baseline))}"
    )
    assert actual == baseline, "protected source/config SHA256 manifest differs from Phase 1"
    assert actual["config/guidance.yaml"] == baseline["config/guidance.yaml"], (
        "config/guidance.yaml must remain byte-identical; no audit may fix gains, "
        "limits, or mode-switch thresholds"
    )


def test_git_changed_and_untracked_paths_are_audit_allowlisted():
    """Tracked and untracked changes must remain within the audit allowlist.

    `git diff --name-only` is checked explicitly; staged paths and untracked additions
    are included so a protected change cannot be hidden by the index or by being new.

    Validates: Requirements 10.1, 10.2
    """
    changed_paths = _git_lines("diff", "--name-only")
    changed_paths |= _git_lines("diff", "--cached", "--name-only")
    changed_paths |= _git_lines("ls-files", "--others", "--exclude-standard")

    disallowed_paths = sorted(
        path for path in changed_paths if not _is_allowlisted_audit_change(path)
    )
    assert not disallowed_paths, (
        "changed paths outside the control-method audit allowlist: "
        f"{disallowed_paths}"
    )


# ---------------------------------------------------------------------------
# Test 11: real-repository CLI end-to-end integration (Task 7)
# ---------------------------------------------------------------------------


def test_audit_cli_end_to_end_external_outputs_agree_across_formats(tmp_path):
    """The real CLI writes self-consistent audit artifacts outside the repository.

    Validates: Requirements 9.1, 9.2, 9.3, 9.4
    """
    from uav_vpp_guidance.evaluation.control_method_audit import FindingRecord, self_check

    out_md = tmp_path / "audit" / "control_method_rationality_audit.md"
    out_json = tmp_path / "audit" / "control_method_rationality_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_control_method_rationality.py",
            "--guidance-config",
            "config/guidance.yaml",
            "--source-root",
            "src/uav_vpp_guidance",
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_md.is_file()
    assert out_json.is_file()

    report = json.loads(out_json.read_text(encoding="utf-8"))
    findings = report["findings"]
    expected_ids = [f"F{index:02d}" for index in range(1, 19)]
    assert len(findings) == 18
    assert [finding["id"] for finding in findings] == expected_ids
    self_check([FindingRecord(**finding) for finding in findings])

    f15 = next(finding for finding in findings if finding["id"] == "F15")
    assert f15["verdict"] == "premise_incorrect"

    backlog = report["summary"]["remediation_backlog"]
    priorities = {finding["id"]: finding["priority"] for finding in findings}
    assert backlog
    assert all(finding_id in priorities for finding_id in backlog)
    assert all(priorities[finding_id] > 0 for finding_id in backlog)
    assert [priorities[finding_id] for finding_id in backlog] == sorted(
        priorities[finding_id] for finding_id in backlog
    )

    markdown = out_md.read_text(encoding="utf-8")
    markdown_verdicts = {
        finding_id: verdict
        for finding_id, verdict in re.findall(
            r"^### (F\d{2}) .*?\n- 判定：`([^`]+)`", markdown, flags=re.MULTILINE
        )
    }
    assert markdown_verdicts == {
        finding["id"]: finding["verdict"] for finding in findings
    }
