# AST Submission-Readiness Checklist

**Purpose:** convert the current VPP hierarchical-guidance evidence into an
auditable, AST-ready submission package without reopening unlimited policy or
specialist tuning.

## 0. What "80% confidence" can mean

No checklist can guarantee an 80% acceptance probability: editor fit, reviewer
assignment, and competing submissions are external variables. This checklist
instead defines an **80/100 internal readiness gate**. Reaching it means the
paper has closed the controllable scientific, statistical, reproducibility,
and packaging risks expected for a credible AST submission.

**Rule:** do not submit when any hard gate below fails, even if the readiness
score is at least 80.

## 1. Evidence Roles Are Frozen

### 1.1 Headline canonical evidence

Use only the canonical PPO-vs-oracle formal evidence for headline performance
claims:

- Tasks: `head_on`, `crossing_feasible`.
- Opponents: `expert`, `end_to_end`.
- Backend: JSBSim.
- Crossing attack-zone angle gate: 60 deg.
- Safe claim boundary:
  - `expert/head_on`: learned high-level policy is no longer weaker than the
    static oracle task gate.
  - `crossing_feasible`: canonical performance is preserved with the explicit
    `N=4` caveat. Supplementary Table S7 and Figure S7 add a separate,
    24-scenario crossing envelope and three valid PPO seeds that meet a
    practical preservation screen against a fixed-VPP reference.
  - `end_to_end/head_on`: learned policy remains slightly below oracle; the
    residual is localized and must remain disclosed.

**Hard gate H1 - Claim discipline**

- [ ] The manuscript makes no claim of uniform oracle parity or universal
  superiority.
- [ ] It labels the static oracle as a privileged task-label reference.
- [ ] It labels historical DQN, early PPO, SAC, self-play, and exploratory
  lanes as `historical/non-canonical`.
- [ ] Every `crossing_feasible` result states `N=4` in text or caption.
- [ ] Any expanded RQ1 crossing claim identifies the fixed-VPP comparator, the
  finite 24-scenario envelope, and the `-0.05` practical margin; it must not be
  presented as superiority or universal generalization.
- [ ] No sentence claims certified airframe safety or a validated F-16 flight
  envelope.

**Pass:** all boxes checked.  
**Fail action:** narrow or correct prose; do not run another training job to
mask an unsupported claim.

### 1.1a RQ1 crossing provenance and robustness record

- [x] Supplementary Table S7 reports the G4 counterexample, both independent
  RQ1 opponent rows, all six valid seed-by-opponent cells, and cross-seed mean
  plus population SD.
- [x] Figure S7 plots paired PPO-minus-fixed differences with deterministic
  paired-bootstrap intervals and marks the `-0.05` practical margin.
- [x] Only seeds `701`, `702`, and `703` from the `_cleanimport` cohort are
  included; the invalid external-import cohort is explicitly excluded.
- [x] The manuscript states that the margin is a preregistered practical
  preservation gate, not a statistical superiority test.

**Recorded status (2026-07-12): MET.** The evidence supports practical
crossing preservation within the explicit RQ1 envelope and the two tested
opponents. It does not support broad generalization, superiority, or invariant
behavior across PPO training seeds.

### 1.2 Auxiliary paper-safe evidence

The telemetry comparison is a **supplementary execution diagnostic**, not a
replacement for headline performance evidence.

- Simulation SHA: `cc478847ed279504d737df8ec67ffc2c5733bc81`.
- Analysis correction SHA: `c9c94b3`.
- Run: `flight_envelope_manifest60_expert_clean_cc47884_20260711`.
- Scope: expert opponent, manifest60 (`60 head_on + 4 crossing_feasible`),
  JSBSim, AoA60.
- Methods: static oracle gate, canonical PPO high-level policy, fixed head-on
  VPP specialist without high-level routing.

**Hard gate H2 - Evidence separation**

- [ ] The telemetry figure is labelled `execution and protection diagnostic`.
- [ ] It uses true aerodynamic `alpha_deg`, never `ego_attack_aoa_deg`.
- [ ] It reports `Ego terminal failure`, not a mixed `Crash/OOB` rate.
- [ ] It states that max-range OOB means pair separation exceeds 12 km, not
  that an aircraft left a Cartesian map.
- [ ] It states that command limits are not certified aircraft limits.

**Pass:** all boxes checked.  
**Fail action:** retain the diagnostic only in internal material until its
caption and Supplementary Note are corrected.

## 2. Priority 1: Three-Method Causal Ablation

### Goal

Show whether high-level routing adds value beyond a frozen low-level VPP policy
that never routes. This is the most valuable remaining causal test because it
does not require new training.

### Inputs

- Formal summary and raw data from the auxiliary paper-safe run above.
- `oracle_task_gate` (privileged reference).
- `commander_post_merge_recovery3` (PPO high-level routing).
- `head_on_vpp_specialist_no_routing` (fixed low-level reference).

### Required analyses

- [ ] Build a task-separated outcome table for all three methods: wins,
  losses, draws/unresolved, resolved denominator, win rate, and Wilson 95% CI.
- [ ] Build a paired scenario table: PPO-only win, fixed-only win, both win,
  both loss, and discordant outcome examples.
- [ ] Report mode-selection behavior for PPO; do not infer routing value only
  from aggregate win rate.
- [ ] Keep `head_on` and `crossing_feasible` separate; do not pool them into a
  single performance score.

### Acceptance gate G1

- [ ] On `crossing_feasible`, PPO is not weaker than the fixed head-on policy,
  or a contrary outcome is explicitly reported and the routing claim is
  narrowed.
- [ ] On `head_on`, PPO is not materially weaker than the fixed policy. A
  practical screen is an absolute resolved win-rate gap no worse than 5 pp;
  paired discordant outcomes must also be inspected.
- [ ] The conclusion is causal and bounded: routing is compared against a
  no-routing reference, while oracle remains privileged.

**If G1 passes:** include the three-method table in the main text or main
Supplement.  
**If G1 fails:** do not hide the fixed-policy result. Retain only the
two-method oracle comparison as headline evidence and state that the present
ablation does not establish a routing gain. Do not tune the canonical PPO
family in response.

## 3. Priority 2: Statistics and Termination Semantics

### Required tables

- [x] Main table has `N_total`, `N_resolved`, wins, losses, win rate, and
  Wilson 95% CI for every headline task/opponent/method cell.
- [x] A note defines `win_rate = wins / (wins + losses)` and explains every
  excluded unresolved episode.
- [x] Paired oracle-vs-PPO scenario deltas are reported for `head_on` instead
  of relying on overlapping CIs alone.
- [x] No p-value is presented unless its paired test, null hypothesis, and
  multiplicity treatment are specified in advance.

### Required termination terminology

- [x] `ego_crash_or_out_of_bounds` is separated into low-altitude crash and
  max-range OOB using the configured `500 m` and `12 km` thresholds.
- [x] `target_crash_or_out_of_bounds` is reported, if needed, as an ego win,
  never as an ego safety failure.
- [x] The text distinguishes tactical disengagement (max-range OOB) from
  physical altitude loss.

**Acceptance gate G2**

- [x] All numerical claims can be reproduced from a named CSV/JSON artifact.
- [x] No table or figure uses the obsolete mixed `Crash/OOB safety rate`.
- [x] The Discussion does not imply that transient actual `n_z` exceedance
  caused every terminal event without a trajectory-level causal audit.

**Fail action:** fix analysis or wording only. A new simulation is unnecessary
unless an input artifact is missing or corrupt.

**Implementation status (2026-07-12):** **G2 MET.** The reconciled main
manuscript and Supplementary Tables S1--S7 use the same canonical dual-opponent
headline evidence. Tables S3--S5 provide the expert-only three-method outcome
counts, paired PPO-versus-fixed deltas, and corrected ego-terminal-event
decomposition; Table S7 and Figure S7 add the separately scoped RQ1 crossing
synthesis. Their source JSON/CSV artifacts are named in the supplement. No new
training was used for this reconciliation.

## 4. Priority 3: Flight-Control Limitation, Not Safety Marketing

### Required artefacts

- [ ] Flight-envelope figure in SVG/PDF/TIFF/PNG plus source-data CSV.
- [ ] `flight_control_protection_diagnostic.md` and JSON.
- [ ] The report verifies recorded `n_z_cmd` stays in `[-2, 7] g` while actual
  JSBSim `n_z` can leave that interval.
- [ ] The report records the default 0.5-rad AoA protection threshold as a
  controller setting, not an airworthiness certification limit.

### Acceptance gate G3

- [ ] The figure is cited as evidence of observed operating occupancy and
  command-versus-response behavior.
- [ ] It is placed in Supplementary Material or a clearly limited Discussion
  subsection unless the paper adds a separately validated response-level
  protection study.
- [ ] The manuscript states that low-altitude termination and actual-load
  overshoot remain low-level limitations.

**Stop rule:** do not start a new safety-control or low-level-specialist family
for this AST paper. Such work becomes future work unless the canonical paper
claim is explicitly reopened.

## 5. Priority 4: One Optional, Pre-Registered Extension

Run this only after G1-G3 pass and only if the paper needs causal evidence on
both opponent splits.

### Permitted run

- Scope: existing manifest60, JSBSim, AoA60, `end_to_end` opponent.
- Methods: the same three methods from G1.
- Training: none.
- Required provenance: clean worktree, clean commit, `run-status=formal`,
  valid artifact contract, `paper_safe=true`.

### Acceptance gate G4

- [ ] The outcome is analyzed with the same tables and terminology as G1-G3.
- [ ] It is included only if it confirms or honestly bounds the expert-split
  causal conclusion.

**Stop rule:** regardless of outcome, no new policy family, reward family,
specialist, or training extension is started for the AST manuscript.

## 6. Priority 5: Reproducibility and Paper-Code Sync

### Reproducibility checklist

- [ ] Freeze the final repository commit and record the exact external
  checkpoint/prediction asset manifest with hashes.
- [ ] Record each formal run's config hash, clean SHA, command line,
  `paper_safe` flag, artifact-contract result, and source-data location.
- [ ] Keep the two telemetry commits separate from the canonical headline
  commit; document their auxiliary role.
- [ ] Verify a clean worktree imports its own `src` first, not an exploratory
  repository via `PYTHONPATH`.

### Paper-code sync checklist

- [ ] Search manuscript, supplement, README, cover letter, and figure captions
  for historical DQN/early PPO contamination.
- [ ] Search for stale paths, stale dates, obsolete `Crash/OOB` terminology,
  and raw local paths in anonymized files.
- [ ] Verify every headline numerical value against its cited aggregate output.
- [ ] Verify every figure caption names task, opponent, sample size, comparator,
  and whether values are episode-level or telemetry-sample-level.

**Hard gate H3 - Reproducibility:** all boxes checked and an independent
paper-code sync audit returns no blocking finding.

## 7. Priority 6: Manuscript and AST Package

### Scientific presentation

- [ ] Abstract leads with the VPP maneuver-interface problem, not generic RL.
- [ ] Introduction names the unresolved interface problem: VPP can encode a
  continuous geometric intent but cannot alone select task-appropriate maneuver
  primitives across engagement geometries.
- [ ] Method distinguishes discrete PPO routing from continuous frozen
  specialist VPP control and the guidance/PID execution layers.
- [ ] Results lead with task-level outcome evidence, then mechanism/trajectory
  evidence, then the bounded control diagnostic.
- [ ] Discussion states both the engineering contribution and the remaining
  low-level response limitation.

### Submission package

- [ ] Anonymized AST manuscript compiles without warnings that affect layout.
- [ ] Separate title page, highlights, cover letter, declarations, and
  supplement index agree with the final manuscript title and claims.
- [ ] All raster figures meet the journal's effective resolution requirement;
  SVG/PDF sources remain archived.
- [ ] A non-author review reads the PDF against this checklist and finds no
  ambiguous oracle privilege, safety, or sample-size claim.

**Hard gate H4 - Package consistency:** all boxes checked.

## 8. Internal Readiness Score

Score only after H1-H4 pass.

| Dimension | Weight | Full-score condition |
|---|---:|---|
| Contribution and AST fit | 20 | VPP-as-maneuver-interface contribution is explicit and differentiated from generic HRL. |
| Causal evaluation | 20 | G1 passes, or the ablation limitation is transparent and the headline claim is narrowed. |
| Statistics and claim discipline | 15 | G2 passes; all denominators, CIs, and residuals are explicit. |
| Control-diagnostic honesty | 15 | G3 passes; no safety overclaim remains. |
| Reproducibility | 15 | H3 passes with clean formal artefacts and reproducible source data. |
| Manuscript and package quality | 15 | H4 passes and figures/captions are publication-ready. |

**Submission rule:** total score >= 80/100 and all H1-H4 pass.  
**Interpretation:** this is an internal quality threshold, not a numerical
promise of AST acceptance.

## 9. Execution Order and Stop Rules

1. Derive the three-method expert ablation from the completed formal telemetry
   run; do not retrain.
2. Correct all termination semantics and insert the diagnostic caveat.
3. Complete statistics, paired deltas, and paper-code sync.
4. Run the optional end-to-end three-method evaluation only if it adds causal
   value after the expert ablation.
5. Freeze final assets, score readiness, conduct one external-style PDF review,
   then submit.

**Global stop rule:** no new RL algorithm, reward family, curriculum, self-play
lane, commander clamp, or specialist training may enter the AST manuscript
after this checklist begins. Negative or mixed evidence changes the claim
boundary; it does not reopen the engineering search.
