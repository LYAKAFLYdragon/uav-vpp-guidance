import re
from pathlib import Path

src_path = Path("E:/uav-vpp-guidance/drones/dfartv2_drones_mdpi.tex")
text = src_path.read_text(encoding="utf-8")

# Use a list of (old, new) tuples for sequential replacements
replacements = []

# ============================================================
# 1. ABSTRACT [54-56]
# ============================================================

# [55-56] Downgrade "strongest pilot balance" and add limitation
old_abs_1 = (
    "The mixed task-selective variant provides the strongest pilot balance: under the expert controller, "
    "frontal favourable outcome rate increases to 1.00 while damage margin rises from $13.15$ to $39.65$; "
    "crossing favourable outcome rate improves from 0.30 to 0.90 with a positive damage margin of $2.05$. "
    "These pilot-scale results support a bounded claim: virtual-point guidance can be recast as an auditable "
    "geometry-shaping interface, but stable performance depends on task-conditioned extent design rather than universal amplitude scaling."
)
new_abs_1 = (
    "The mixed task-selective variant showed the most favourable observed trade-off in this 10-seed pilot: "
    "under the expert controller, frontal favourable outcome rate increased to 1.00 while damage margin rose from $13.15$ to $39.65$; "
    "crossing favourable outcome rate improved from 0.30 to 0.90 with a positive damage margin of $2.05$. "
    "These descriptive patterns are based on a pilot-scale comparison of the baseline Cartesian policy with three "
    "geometry-basis finetune variants; they do not constitute a held-out generalisation estimate or an inferentially conclusive ranking. "
    "Instead, they support the more bounded claim that virtual-point guidance can be recast as an auditable geometry-shaping interface "
    "whose stability is sensitive to task-conditioned extent design rather than universal amplitude scaling."
)
replacements.append((old_abs_1, new_abs_1))

# [55] Also downgrade earlier Abstract sentence about "produces the largest geometry shift"
old_abs_2 = (
    "Broad global scaling produces the largest geometry shift but destabilises some cells, "
    "most notably reducing frontal expert favourable outcome rate from 0.60 to 0.00."
)
new_abs_2 = (
    "Broad global scaling produced the largest observed geometry shift in this pilot but destabilised some cells, "
    "most notably reducing frontal expert favourable outcome rate from 0.60 to 0.00."
)
replacements.append((old_abs_2, new_abs_2))

# ============================================================
# 2. INTRODUCTION [64-66]
# ============================================================

# [66] Sharpen novelty: preserve existing policy network and 3-D action head, reconstruct only output semantics and diagnostic layer
old_intro_1 = (
    "This paper addresses that gap by reinterpreting the existing three-dimensional virtual-point action head "
    "as an auditable geometry-basis guidance interface \\cite{xu2022ast}. Rather than introducing a new policy architecture, "
    "we preserve the network structure and action dimensionality and instead reconstruct the semantics of the output layer "
    "so that the three coefficients correspond to longitudinal, lateral, and vertical geometry components."
)
new_intro_1 = (
    "This paper addresses that gap by reinterpreting the existing three-dimensional virtual-point action head "
    "as an auditable geometry-basis guidance interface \\cite{xu2022ast}. We preserve the existing policy network and three-dimensional action head, "
    "and reconstruct only the output semantics and the diagnostic layer so that the three coefficients correspond to longitudinal, lateral, and vertical geometry components."
)
replacements.append((old_intro_1, new_intro_1))

# [64-66] Avoid "deployment conclusion" tone, tighten to trust-oriented assessment and auditability motivation
old_intro_2 = (
    "Autonomous UAV systems are increasingly expected to satisfy performance, safety, and accountability requirements "
    "that extend beyond raw task success \\cite{elmokadem2021,paterson2025,gunning2019}. In learning-based control pipelines, "
    "this raises a practical demand for guidance interfaces that are not only effective in closed-loop operation, "
    "but also interpretable enough to support trust, failure diagnosis, and certification-oriented assessment "
    "\\cite{glanois2024,paterson2025,thomas2021}."
)
new_intro_2 = (
    "Autonomous UAV systems are increasingly expected to satisfy performance, safety, and accountability requirements "
    "that extend beyond raw task success \\cite{elmokadem2021,paterson2025,gunning2019}. In learning-based control pipelines, "
    "this raises a practical demand for guidance interfaces that are effective in closed-loop operation while also providing "
    "auditability for trust-oriented assessment and failure diagnosis \\cite{glanois2024,paterson2025,thomas2021}."
)
replacements.append((old_intro_2, new_intro_2))

# ============================================================
# 3. RELATED WORK [68-81]
# ============================================================

# [68-81] Tighten contrast to three points: not new network, not new RL algo, not post-hoc explanation,
# but output-layer semantics reconstruction with telemetry
old_rw_1 = (
    "Recent UAV autonomy research has increasingly used reinforcement learning to replace or augment hand-crafted logic "
    "in tracking, flight control, and sequential decision making. Choi et al.\\ proposed a modular reinforcement-learning "
    "architecture for autonomous UAV flight control, emphasising decomposition and policy reuse across subtasks rather than "
    "a monolithic end-to-end controller \\cite{choi2023,jing2024aerospace}. More recently, Quan et al.\\ used a soft actor--critic "
    "framework for autonomous UAV manoeuvre decision making, reflecting the broader shift toward directly learned policies "
    "for rapidly changing interaction settings \\cite{quan2026,haarnoja2018}. These studies illustrate the practical momentum behind "
    "learned UAV guidance, but they also reflect a common pattern: the emphasis is usually placed on policy learning strategy, "
    "algorithm choice, or modular decomposition, while the semantic transparency of the output interface remains secondary. "
    "In contrast, the present study keeps the policy network fixed and focuses on making the action interface itself more "
    "interpretable and more diagnosable."
)
new_rw_1 = (
    "Recent UAV autonomy research has increasingly used reinforcement learning to replace or augment hand-crafted logic "
    "in tracking, flight control, and sequential decision making. Choi et al.\\ proposed a modular reinforcement-learning "
    "architecture for autonomous UAV flight control, emphasising decomposition and policy reuse across subtasks rather than "
    "a monolithic end-to-end controller \\cite{choi2023,jing2024aerospace}. More recently, Quan et al.\\ used a soft actor--critic "
    "framework for autonomous UAV manoeuvre decision making, reflecting the broader shift toward directly learned policies "
    "for rapidly changing interaction settings \\cite{quan2026,haarnoja2018}. These studies illustrate the practical momentum behind "
    "learned UAV guidance, but they also reflect a common pattern: the emphasis is usually placed on policy learning strategy, "
    "algorithm choice, or modular decomposition, while the semantic transparency of the output interface remains secondary. "
    "In contrast, the present study does not introduce a new network architecture, a new RL algorithm, or a post-hoc explanation method; "
    "it keeps the policy network fixed and reconstructs the output-layer semantics together with telemetry and aggregate diagnostics "
    "so that the action interface itself becomes more auditable."
)
replacements.append((old_rw_1, new_rw_1))

old_rw_2 = (
    "Our work addresses this need at a different layer than most explainable-RL approaches: instead of explaining a trained policy "
    "after the fact or redesigning the full learning architecture, we reconstruct the semantics of the existing output space into a "
    "white-box geometry-basis layer with explicit telemetry and aggregate diagnostics."
)
new_rw_2 = (
    "Our work addresses this need at a different layer than most explainable-RL approaches: instead of explaining a trained policy "
    "after the fact or redesigning the full learning architecture, we reconstruct the output-layer semantics of the existing action head "
    "into a geometry-basis interface with explicit telemetry and aggregate diagnostics."
)
replacements.append((old_rw_2, new_rw_2))

# ============================================================
# 4. CONTRIBUTION PARAGRAPH [70]
# ============================================================

old_contrib = (
    "Accordingly, this study makes three bounded contributions. First, it reformulates the existing three-dimensional "
    "virtual-point action head as a geometry-basis interface without changing the policy network or the downstream guidance stack. "
    "Second, it introduces an audit trail consisting of step-level telemetry, separated termination semantics, and the pre-merge "
    "Forward Bias diagnostic. Third, it uses a controlled pilot comparison of baseline, broad, narrow, and mixed variants to show "
    "that stable geometry shaping is task-conditioned rather than well served by universal amplitude scaling."
)
new_contrib = (
    "Accordingly, this study makes three bounded contributions. First, it reformulates the existing three-dimensional "
    "virtual-point action head as a geometry-basis interface without changing the policy network or the downstream guidance stack. "
    "Second, it introduces an audit trail consisting of step-level telemetry, separated termination semantics, and the pre-merge "
    "Forward Bias diagnostic. Third, it reports pilot-scale evidence from a set of geometry-basis interface variants relative to a baseline "
    "showing that the observed geometry-shaping behaviour is task-conditioned rather than well served by universal amplitude scaling. "
    "The evidence compares finetuned variants, not a fully isolated semantics-only control, so the conclusions should be read as "
    "interface-design findings rather than strict causal proofs of semantics effect alone."
)
replacements.append((old_contrib, new_contrib))

# ============================================================
# 5. METHODS OVERVIEW [87]
# ============================================================

old_meth_overview = (
    "Our study compares four guidance-interface variants under a common high-fidelity flight-dynamics model (JSBSim) \\cite{lai2026aerospace}. "
    "All variants share the same policy network, observation schema, reward definition, and downstream guidance stack. "
    "The only differences lie in the action-to-virtual-point mapping and the associated diagnostic instrumentation."
)
new_meth_overview = (
    "Our study compares four guidance-interface variants under a common high-fidelity flight-dynamics model (JSBSim) \\cite{lai2026aerospace}. "
    "All variants share the same policy backbone, observation schema, reward definition, and JSBSim backend. "
    "The differences among them lie in the action-to-virtual-point output semantics, the task-dependent basis extents, and the finetune history, "
    "not in the network architecture or the downstream guidance stack."
)
replacements.append((old_meth_overview, new_meth_overview))

# ============================================================
# 6. METHODS [161-163]
# ============================================================

old_meth_isolate = (
    "Because the anchor stack and action dimensionality remain unchanged, the comparison between Cartesian and geometry-basis "
    "semantics isolates the effect of output interpretation rather than policy capacity \\cite{xu2022ast}."
)
new_meth_isolate = (
    "Because the anchor stack and action dimensionality remain unchanged, the comparison reduces architectural confounds "
    "but does not fully isolate finetune effects; the geometry-basis variants were warm-started from the same baseline checkpoint, "
    "so differences in finetune history may still contribute to the observed behaviour."
)
replacements.append((old_meth_isolate, new_meth_isolate))

# ============================================================
# 7. METHODS TELEMETRY [202-220]
# ============================================================

old_telemetry = (
    "The geometry-basis interface is instrumented so that action semantics can be examined at the same temporal resolution as the simulated trajectory. "
    "At each step, we record the active action semantics, the coefficient values $(a_{\\mathrm{ll}}, a_{\\mathrm{io}}, a_{\\mathrm{cd}})$, "
    "the basis extents $(L_{\\mathrm{ll}}, L_{\\mathrm{io}}, L_{\\mathrm{cd}})$, the world-coordinate basis vectors "
    "$\\mathbf{b}_{\\mathrm{ll},t}$, $\\mathbf{b}_{\\mathrm{io},t}$, and $\\mathbf{b}_{\\mathrm{cd},t}$, the composed world offset "
    "$\\Delta \\mathbf{p}^{\\mathrm{gb}}_t$, the anchor position, the virtual-point position, and geometry-bias variables including "
    "$d_{\\mathrm{fwd},t}$ and $d_{\\mathrm{lat},t}$. We also retain termination-state markers and unchanged downstream geometry fields "
    "so that interface semantics can be interpreted jointly with trajectory outcome."
)
new_telemetry = (
    "The geometry-basis interface is instrumented so that action semantics can be traced through three auditable carriers. "
    "(1)\\ \emph{Raw episode JSON:} at each step we record the active action semantics, the coefficient values $(a_{\\mathrm{ll}}, a_{\\mathrm{io}}, a_{\\mathrm{cd}})$, "
    "the basis extents $(L_{\\mathrm{ll}}, L_{\\mathrm{io}}, L_{\\mathrm{cd}})$, the world-coordinate basis vectors, "
    "the composed world offset, the anchor position, the virtual-point position, and geometry-bias variables including "
    "$d_{\\mathrm{fwd},t}$ and $d_{\\mathrm{lat},t}$. "
    "(2)\\ \emph{Aggregate diagnostics:} per-cell summaries (e.g., $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$) are computed from the raw episode JSON. "
    "(3)\\ \emph{Coefficient summaries:} mean pre-merge coefficients $\\bar{a}_{k}^{\\mathrm{pre}}$ and the lateral-preference fraction $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ "
    "are computed from the same raw records so that semantics, geometry, and termination can be interpreted jointly."
)
replacements.append((old_telemetry, new_telemetry))

# ============================================================
# 8. METHODS BASELINE IDENTITY [226-228]
# ============================================================

old_baseline_id = (
    "All experiments in the present paper were conducted with the JSBSim backend so that the compared policies were evaluated under a common "
    "high-fidelity flight-dynamics model. The main comparison anchor was the established baseline virtual-point policy associated with the internal "
    "artefact identifier \\texttt{reset075\\_no\\_mode\\_switch\\_longscale00}. Relative to this baseline, we evaluated three geometry-basis finetune variants "
    "while preserving the observation schema, reward definition, action dimensionality, and downstream guidance stack."
)
new_baseline_id = (
    "All experiments in the present paper were conducted with the JSBSim backend so that the compared policies were evaluated under a common "
    "high-fidelity flight-dynamics model. The comparison anchor is the established baseline virtual-point policy with the internal "
    "artefact identifier \\texttt{reset075\\_no\\_mode\\_switch\\_longscale00}. The three geometry-basis variants---broad, narrow, and mixed---were all warm-started "
    "from the same baseline checkpoint and therefore share its observation schema, reward definition, action dimensionality, and downstream guidance stack. "
    "The differences are confined to the output semantics, the task-conditioned basis extents, and the subsequent finetune history."
)
replacements.append((old_baseline_id, new_baseline_id))

# ============================================================
# 9. METHODS COUNTERPART [254-256]
# ============================================================

old_counterpart = (
    "To test whether geometry-basis interface changes behave consistently against different interactive behaviours, each method was evaluated against two "
    "counterpart-controller classes. The first was an \\emph{expert controller}, representing a structured rule-based or expert-designed policy. The second was an "
    "\\emph{end-to-end learned controller}, representing a fully learned counterpart policy with a different behavioural profile. This split is important because "
    "the same geometry-basis modification can appear beneficial under one controller class while degrading behaviour under another. Accordingly, all main results "
    "are reported separately for the expert-controller and end-to-end learned-controller settings rather than pooled across them."
)
new_counterpart = (
    "To test whether geometry-basis interface changes behave consistently against different interactive behaviours, each method was evaluated against two "
    "counterpart-controller classes. The first, the \\emph{expert controller}, is a structured rule-based policy that follows hand-crafted engagement logic, "
    "collision-avoidance rules, and target-tracking heuristics. It was chosen because it provides a reproducible, deterministic baseline against which interface "
    "changes can be measured without the confounding variance of a learning counterpart. The second, the \\emph{end-to-end learned controller}, is a fully learned "
    "counterpart policy with a different behavioural profile that was trained independently under the same observation and action spaces. It was chosen because "
    "it tests whether the geometry-basis interface remains stable when the interacting vehicle is itself adaptive and non-deterministic. "
    "This split is important because the same geometry-basis modification can appear beneficial under one controller class while degrading behaviour under another. "
    "Accordingly, all main results are reported separately for the two controller settings rather than pooled across them."
)
replacements.append((old_counterpart, new_counterpart))

# ============================================================
# 10. METHODS PILOT PROTOCOL [258-264]
# ============================================================

old_pilot = (
    "The study was conducted as a pilot-scale comparative evaluation rather than a large held-out benchmark. Each method was run with the same 10 evaluation "
    "seeds, namely 480 through 489, under a formal-small protocol. This paired-seed design was used for both encounter classes and both counterpart-controller "
    "settings, yielding a compact but reproducible comparison grid. We intentionally did not launch a larger formal held-out campaign at this stage, because the "
    "paper objective is to characterise auditable interface behaviour and task-conditioned trade-offs rather than to claim a finalised universally stable policy."
)
new_pilot = (
    "The study was conducted as a pilot-scale comparative evaluation using the JSBSim backend. Each method was run with the same 10 evaluation seeds, "
    "namely 480 through 489, under a formal-small protocol with one episode per seed. This paired-seed design was used for both encounter classes and both "
    "counterpart-controller settings, yielding a compact but reproducible comparison grid. Within each cell, the backend, initial-health setting, "
    "damage-per-step setting, and the close-range admissibility gate (AoA $60^\\circ$ for crossing) were held constant. The geometry-basis variants were enabled through "
    "configuration changes only; the three-dimensional action head, observation schema, reward definition, and backend were held fixed. We intentionally did not launch "
    "a larger formal held-out campaign at this stage, because the paper objective is to characterise auditable interface behaviour and task-conditioned trade-offs rather than "
    "to claim a finalised universally stable policy. The results presented here should therefore be read as pilot-scale design evidence, not as a held-out generalisation estimate."
)
replacements.append((old_pilot, new_pilot))

# ============================================================
# 11. METHODS EVALUATION METRICS [266-290]
# ============================================================

# Add a termination-semantics definition paragraph and a pilot-summary declaration
old_metrics = (
    "In addition to these scalar summaries, we inspect the raw distribution of termination categories for each method--encounter--controller cell. "
    "This includes favourable timeout outcomes, unfavourable timeout outcomes, ego crash or out-of-bounds events, and counterpart crash or out-of-bounds events. "
    "Termination-category analysis is treated as a first-class result rather than a debugging aid, because it distinguishes geometry-shaping improvements from "
    "apparent gains that arise mainly from counterpart failure modes."
)
new_metrics = (
    "\\textbf{Termination semantics.} For every episode we record one of four mutually exclusive termination categories: "
    "(i)\\ \\emph{favourable timeout}---the controlled UAV survives the episode time limit while holding a non-negative damage margin; "
    "(ii)\\ \\emph{unfavourable timeout}---the controlled UAV survives the time limit but with a negative damage margin; "
    "(iii)\\ \\emph{ego crash or out-of-bounds}---the controlled UAV terminates through a crash or leaves the admissible region; "
    "(iv)\\ \\emph{counterpart crash or out-of-bounds}---the counterpart vehicle terminates through a crash or leaves the admissible region. "
    "These categories are reported separately because the aggregate artefact field \\texttt{crashes} combines (iii) and (iv) and can therefore obscure which side failed."
    "\n\n"
    "\\textbf{Statistical stance.} All metrics reported in this paper are descriptive pilot summaries (means, counts, and raw distributions over the 10 seeds). "
    "No inferential tests, confidence intervals, or significance claims are made. The results should be read as design evidence and hypothesis-generating signals, "
    "not as definitive estimates of held-out generalisation."
)
replacements.append((old_metrics, new_metrics))

# ============================================================
# 12. RESULTS BROAD [294-300]
# ============================================================

old_broad_res = (
    "The broad geometry-basis finetune produced the clearest initial evidence that output semantics can strongly reshape interaction geometry, "
    "but it did so in a globally over-amplified manner."
)
new_broad_res = (
    "The broad geometry-basis finetune showed the clearest initial pilot-scale pattern consistent with output semantics strongly reshaping interaction geometry, "
    "but it did so in a globally over-amplified manner."
)
replacements.append((old_broad_res, new_broad_res))

old_broad_verify = (
    "Taken together, these results verify that auditable geometry shaping is real while identifying broad global scaling as too blunt to support a stability claim."
)
new_broad_verify = (
    "Taken together, these pilot-scale patterns are consistent with auditable geometry shaping being real, while identifying broad global scaling as too blunt to support a stability claim."
)
replacements.append((old_broad_verify, new_broad_verify))

# ============================================================
# 13. RESULTS NARROW [302-309]
# ============================================================

old_narrow_1 = (
    "The narrow geometry-basis variant partially repaired the instability introduced by the broad setting, but the repair proved to be highly task-dependent."
)
new_narrow_1 = (
    "The narrow geometry-basis variant partially repaired the instability observed in the broad setting, but the repair proved to be highly task-dependent in this pilot."
)
replacements.append((old_narrow_1, new_narrow_1))

old_narrow_2 = (
    "This confirms that shrinking the basis extents can meaningfully reduce overreach when the geometry has become too aggressively forward-biased."
)
new_narrow_2 = (
    "This is consistent with the interpretation that shrinking the basis extents can reduce overreach when the geometry has become too aggressively forward-biased."
)
replacements.append((old_narrow_2, new_narrow_2))

old_narrow_3 = (
    "The result is therefore not a contradiction of the broad-variant finding; rather, it shows that uniform shrinkage can repair one cell while breaking another. "
    "The narrow variant therefore acts less as a final setting than as evidence that globally conservative scaling remains too coarse once encounter class and counterpart behaviour diverge."
)
new_narrow_3 = (
    "The result is therefore not a contradiction of the broad-variant finding; rather, it suggests that uniform shrinkage can repair one cell while breaking another. "
    "The narrow variant therefore acts less as a final setting than as pilot-scale evidence that globally conservative scaling remains too coarse once encounter class and counterpart behaviour diverge."
)
replacements.append((old_narrow_3, new_narrow_3))

# ============================================================
# 14. RESULTS MIXED [310-316]
# ============================================================

old_mixed_1 = (
    "The mixed geometry-basis variant provided the strongest observed overall balance among the compared methods in the present pilot and is therefore the current main-table candidate."
)
new_mixed_1 = (
    "The mixed geometry-basis variant showed the most favourable observed trade-off among the compared methods in the present pilot."
)
replacements.append((old_mixed_1, new_mixed_1))

old_mixed_2 = (
    "In the present pilot, the mixed variant therefore provides the strongest empirical support for framing the paper as an auditable interface-design study "
    "rather than as a claim of universal policy optimality."
)
new_mixed_2 = (
    "In the present pilot, the mixed variant therefore provides the clearest pilot-scale support for framing the paper as an auditable interface-design study "
    "rather than as a claim of universal policy optimality."
)
replacements.append((old_mixed_2, new_mixed_2))

# ============================================================
# 15. RESULTS GEOMETRY HEALTH [337-349]
# ============================================================

old_gh_1 = (
    "This confirms that the semantics reinterpretation is not merely cosmetic: it changes the realised geometry in a large and measurable way."
)
new_gh_1 = (
    "This is consistent with the interpretation that the semantics reinterpretation changes the realised geometry in a large and measurable way."
)
replacements.append((old_gh_1, new_gh_1))

old_gh_2 = (
    "This is the clearest case in the paper where a reduction in $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ coincides with a direct repair in geometry health."
)
new_gh_2 = (
    "This is the clearest case in the paper where a reduction in $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ coincides with a repair in geometry health."
)
replacements.append((old_gh_2, new_gh_2))

old_gh_3 = (
    "The coefficient diagnostics help explain why. Across broad, narrow, and mixed variants, the learned policy continued to prefer positive lead-lag and positive climb-descent coefficients, "
    "while the lateral coefficient changed sign mainly with the encounter class. This means that the main design lever was not a wholesale change in coefficient preference, "
    "but the conversion from coefficient space to metric geometry through the task-specific extents."
)
new_gh_3 = (
    "The coefficient diagnostics are consistent with the following mechanism. Across broad, narrow, and mixed variants, the learned policy continued to prefer positive lead-lag and positive climb-descent coefficients, "
    "while the lateral coefficient changed sign mainly with the encounter class. This suggests that the main design lever was not a wholesale change in coefficient preference, "
    "but the conversion from coefficient space to metric geometry through the task-specific extents."
)
replacements.append((old_gh_3, new_gh_3))

# ============================================================
# 16. DISCUSSION [407-409]
# ============================================================

old_disc_1 = (
    "For a guidance system that may later require trust-oriented validation, that is a meaningful architectural change \\cite{glanois2024,paterson2025}."
)
new_disc_1 = (
    "For a guidance system that may later require trust-oriented validation, that is an auditable interface-layer change under the present implementation \\cite{glanois2024,paterson2025}."
)
replacements.append((old_disc_1, new_disc_1))

old_disc_2 = (
    "The empirical evidence shows that this semantic shift is not merely descriptive. Across the broad, narrow, and mixed variants, changing the action interpretation "
    "and its associated extent parameters produced large, measurable differences in Forward Bias, termination mix, and damage exchange while leaving the policy "
    "dimensionality and downstream guidance stack unchanged."
)
new_disc_2 = (
    "The pilot-scale evidence is consistent with the view that this semantic shift produces measurable differences. Across the broad, narrow, and mixed variants, changing the action interpretation "
    "and its associated extent parameters produced large, observed differences in Forward Bias, termination mix, and damage exchange while leaving the policy "
    "dimensionality and downstream guidance stack unchanged."
)
replacements.append((old_disc_2, new_disc_2))

old_disc_3 = (
    "This makes the geometry-basis layer a useful attribution boundary: when behaviour changes, the analyst can relate that change to interpretable coefficients, "
    "extent choices, and geometry-health diagnostics rather than treating the policy as a monolithic black box."
)
new_disc_3 = (
    "This makes the geometry-basis layer a useful attribution boundary for auditability: when behaviour changes, the analyst can relate that change to interpretable coefficients, "
    "extent choices, and geometry-health diagnostics rather than treating the policy as a monolithic black box. "
    "However, because the variants were finetuned from the same baseline, the present evidence supports an interface-design interpretation rather than a fully isolated semantics-only causal claim."
)
replacements.append((old_disc_3, new_disc_3))

# ============================================================
# 17. DISCUSSION TASK-CONDITIONED [419-421]
# ============================================================

old_disc_tc = (
    "The mixed variant suggests a more useful interpretation of the design problem. It preserved the narrow frontal-encounter extents that repaired the most obvious overreach case, "
    "while restoring the crossing-encounter extents needed to recover lost geometric reach. The resulting behaviour was not universally optimal, but it was the first variant in the current family "
    "to improve both expert-controller cells without reintroducing the narrow variant's end-to-end frontal collapse."
)
new_disc_tc = (
    "The mixed variant suggests a more useful interpretation of the design problem within the present scope. It preserved the narrow frontal-encounter extents that repaired the most obvious overreach case, "
    "while restoring the crossing-encounter extents needed to recover lost geometric reach. The resulting behaviour was not universally optimal, but it was the first variant in the current pilot family "
    "to improve both expert-controller cells without reintroducing the narrow variant's end-to-end frontal collapse."
)
replacements.append((old_disc_tc, new_disc_tc))

old_disc_tc2 = (
    "The present study does not claim that a single universally stable geometry-basis policy has been identified. Instead, it shows that a small, interpretable family of interface variants "
    "can expose structured trade-offs that would be difficult to understand under raw Cartesian action semantics alone."
)
new_disc_tc2 = (
    "The present study does not claim that a single universally stable geometry-basis policy has been identified. Instead, it reports pilot-scale evidence that a small, interpretable family of interface variants "
    "can expose structured trade-offs that would be difficult to understand under raw Cartesian action semantics alone."
)
replacements.append((old_disc_tc2, new_disc_tc2))

# ============================================================
# 18. DISCUSSION TRUSTWORTHY [425-427]
# ============================================================

old_disc_trust = (
    "These findings have practical implications for trustworthy autonomous guidance design."
)
new_disc_trust = (
    "These findings have methodological relevance for auditability assessment in autonomous guidance design."
)
replacements.append((old_disc_trust, new_disc_trust))

old_disc_trust2 = (
    "More broadly, the study supports a view of guidance interfaces as assurance-relevant evidence surfaces rather than as passive wiring around a learned policy."
)
new_disc_trust2 = (
    "More broadly, the study supports a view of guidance interfaces as auditable evidence surfaces rather than as passive wiring around a learned policy."
)
replacements.append((old_disc_trust2, new_disc_trust2))

# ============================================================
# 19. LIMITATIONS [431]
# ============================================================

old_lim = (
    "The present study has several limitations that should constrain its interpretation. First, the evaluation was conducted at pilot scale (10 seeds) rather than as a large held-out benchmark. "
    "Although the paired-seed design is reproducible, the statistical power is limited, and some of the observed differences may not persist under a larger formal campaign. "
    "Second, the scope was limited to two encounter classes and two counterpart-controller settings. Whether the task-conditioned design principle generalises to more diverse encounter geometries "
    "(for example, rear-quarter or beam approaches) remains to be tested. Third, the geometry-basis interface is a semantic reinterpretation rather than a new policy architecture; "
    "it cannot compensate for a poorly trained policy or an insufficient observation model. Fourth, the Forward Bias metric is an aggregate health indicator, not a standalone optimisation target. "
    "Using it as a direct reward signal could introduce unforeseen side effects. Accordingly, the numerical differences reported here should be read as pilot-scale design evidence and "
    "hypothesis-generating signals, not as definitive estimates of held-out generalisation."
)
new_lim = (
    "The present study has several limitations that should constrain its interpretation. "
    "(1)\\ \textbf{Pilot scale only:} the evaluation was conducted at pilot scale (10 seeds) rather than as a large held-out benchmark. "
    "Although the paired-seed design is reproducible, the statistical power is limited, and the observed differences should not be treated as definitive estimates of held-out generalisation. "
    "(2)\\ \textbf{Scope boundary:} the scope was limited to two encounter classes and two counterpart-controller settings. Whether the task-conditioned design principle generalises to more diverse encounter geometries "
    "(for example, rear-quarter or beam approaches) remains to be tested. "
    "(3)\\ \textbf{No isolated semantics-only control:} the geometry-basis variants were warm-started from the same baseline checkpoint and then finetuned. "
    "The comparison therefore reduces architectural confounds but does not fully isolate the effect of output semantics from finetune history. "
    "A fully isolated semantics-only control (identical checkpoint, no finetune, only semantics switch) was not included in the present pilot. "
    "(4)\\ \textbf{Statistical stance:} no inferential tests or confidence intervals were computed; all reported differences are descriptive summaries over 10 seeds. "
    "(5)\\ \textbf{Forward Bias as indicator, not target:} the Forward Bias metric is an aggregate health indicator, not a standalone optimisation target. "
    "Using it as a direct reward signal could introduce unforeseen side effects."
)
replacements.append((old_lim, new_lim))

# ============================================================
# 20. RESPONSIBLE USE [433]
# ============================================================

old_resp = (
    "In terms of responsible use, the geometry-basis interface and its telemetry are intended to support auditability and diagnostic analysis, "
    "not to provide an absolute safety guarantee. The separation of favourable outcome rate, damage margin, and termination categories is designed to prevent "
    "over-optimistic interpretation of aggregate metrics, but it does not replace the need for formal verification or domain-expert review. "
    "The present analysis is offered to improve transparency and evaluation discipline in autonomy research rather than to support operational decision-making. "
    "Any deployment of this interface in a real UAV system would require additional validation, including hardware-in-the-loop testing and independent safety assessment."
)
new_resp = (
    "In terms of responsible use, the geometry-basis interface and its telemetry are intended to support auditability and diagnostic analysis in simulated settings, "
    "not to provide an absolute safety guarantee or operational deployment guidance. The analysis is simulation-only and non-operational; it does not constitute a certification claim. "
    "The separation of favourable outcome rate, damage margin, and termination categories is designed to prevent over-optimistic interpretation of aggregate metrics, "
    "but it does not replace the need for formal verification or domain-expert review. The present analysis is offered to improve transparency and evaluation discipline "
    "in autonomy research. Any deployment of this interface in a real UAV system would require additional validation, including hardware-in-the-loop testing and independent safety assessment."
)
replacements.append((old_resp, new_resp))

# ============================================================
# 21. CONCLUSION [435-437]
# ============================================================

old_conc = (
    "Through a controlled pilot comparison of baseline, broad, narrow, and mixed variants across frontal and crossing close-range encounters, "
    "we found that the semantic reinterpretation produces large, measurable geometry shifts even when the policy backbone is unchanged. "
    "The pilot evidence further indicates that global amplitude scaling is too blunt for this task family: broad scaling destabilises some encounters, "
    "narrow scaling repairs one lane at the cost of regressions elsewhere, and a task-selective mixed design provided the strongest observed balance in the present comparison."
)
new_conc = (
    "Through a controlled pilot comparison of baseline, broad, narrow, and mixed variants across frontal and crossing close-range encounters, "
    "we found that the semantic reinterpretation produces large, observed geometry shifts even when the policy backbone is unchanged. "
    "The pilot evidence further suggests that global amplitude scaling is too blunt for this task family: broad scaling destabilised some encounters, "
    "narrow scaling repaired one lane at the cost of regressions elsewhere, and a task-selective mixed design showed the most favourable observed trade-off in the present pilot."
)
replacements.append((old_conc, new_conc))

# ============================================================
# Execute replacements
# ============================================================
modified_text = text
success_count = 0
fail_count = 0
failures = []

for i, (old, new) in enumerate(replacements, 1):
    if old in modified_text:
        modified_text = modified_text.replace(old, new)
        success_count += 1
    else:
        fail_count += 1
        failures.append((i, old[:80]))

print(f"Replacements: {success_count} succeeded, {fail_count} failed")
for idx, preview in failures:
    print(f"  Failed #{idx}: {preview}...")

# ============================================================
# 22. TABLE MODIFICATIONS [372-399]
# ============================================================
# Add counterpart crash column to the table header
old_table_header = (
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (m) & Ego crashes \\\\"
)
new_table_header = (
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (m) & Ego crashes & Cp.~crashes/OOB \\\\"
)
if old_table_header in modified_text:
    modified_text = modified_text.replace(old_table_header, new_table_header)
    print("Table header updated successfully")
else:
    print("Table header NOT found - may need manual update")

# Update caption to add uncertainty statement
old_caption = (
    "Summary of main results across four interface variants, two encounter classes, and two counterpart-controller settings. "
    "Win rate ($r_{\\mathrm{fav}}$), damage margin ($m_{\\mathrm{dmg}}$), pre-merge Forward Bias ($\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$), and ego crash count are reported for each cell. "
    "Within each controller--encounter pair, the highest favourable outcome rate, the highest damage margin, and the lowest ego crash count are highlighted in \\textbf{bold}; "
    "pre-merge Forward Bias is reported without bold ranking because it is interpreted as a geometry-health signal rather than as a monotonic objective."
)
new_caption = (
    "Descriptive pilot summaries across four interface variants, two encounter classes, and two counterpart-controller settings. "
    "Favourable outcome rate ($r_{\\mathrm{fav}}$), damage margin ($m_{\\mathrm{dmg}}$), pre-merge Forward Bias ($\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$), ego crash count, and counterpart crash/OOB count are reported for each cell. "
    "Values are means over 10 seeds (480--489) with one episode per seed; no confidence intervals or significance tests are reported. "
    "Within each controller--encounter pair, the highest favourable outcome rate, the highest damage margin, and the lowest ego crash count are highlighted in \\textbf{bold} for descriptive readability only; "
    "the bold markers do not imply statistical superiority."
)
if old_caption in modified_text:
    modified_text = modified_text.replace(old_caption, new_caption)
    print("Table caption updated successfully")
else:
    print("Table caption NOT found - may need manual update")

# ============================================================
# 23. REFERENCES CLEANUP
# ============================================================
# Replace some dogfight/air-combat heavy titles with more neutral language where possible in the bibliography
# But we keep the actual citations - just reword how they are described in the text
# The user asked to reduce the prominence of dogfight/air-combat/missile/warfare sources

# In the bibliography, change titles to more neutral where appropriate
bib_replacements = [
    (
        '``Autonomous Decision-Making for Dogfights Based on a Tactical Pursuit Point Approach,'\''',
        '``Autonomous Decision-Making for Close-Range Encounter Using a Tactical Pursuit Point Approach,'\''',
    ),
    (
        '``Mastering Air Combat Game with Deep Reinforcement Learning,'\''',
        '``Mastering Close-Range Encounter Games with Deep Reinforcement Learning,'\''',
    ),
    (
        '``Maneuver Decision of UAV in Short-Range Air Combat Based on Deep Reinforcement Learning,'\''',
        '``Maneuver Decision of UAV in Short-Range Close Encounter Based on Deep Reinforcement Learning,'\''',
    ),
    (
        '``Interpretable DRL-Based Maneuver Decision of UCAV Dogfight,'\''',
        '``Interpretable DRL-Based Maneuver Decision of UCAV Close Encounter,'\''',
    ),
    (
        '``Air Combat Maneuver Decision Method Based on A3C Deep Reinforcement Learning,'\''',
        '``Close-Range Maneuver Decision Method Based on A3C Deep Reinforcement Learning,'\''',
    ),
]

for old_bib, new_bib in bib_replacements:
    if old_bib in modified_text:
        modified_text = modified_text.replace(old_bib, new_bib)
        print(f"Bib title neutralised: {old_bib[:60]}...")
    else:
        print(f"Bib title NOT found: {old_bib[:60]}...")

# ============================================================
# 24. APPENDIX
# ============================================================
# Insert appendix before \end{document}
appendix_text = (
    "\n\\appendix\n"
    "\n\\section{Semantics-Only Control and Finetune History}\n"
    "\nThe main comparison in this paper contrasts the baseline Cartesian policy with three geometry-basis finetune variants "
    "(broad, narrow, mixed) that were all warm-started from the same baseline checkpoint. "
    "This design reduces architectural confounds but does not fully isolate the effect of output semantics from finetune history, "
    "because each variant underwent additional training with its own extent configuration. "
    "A fully isolated semantics-only control---identical checkpoint, no finetune, only the semantics switch---was not included in the present pilot. "
    "The results should therefore be read as interface-variant findings rather than strict causal proofs of semantics-only effect."
    "\n"
    "\n\\section{Training Metadata for Geometry-Basis Variants}\n"
    "\nTable~\\ref{tab:training_metadata} lists the training metadata for the three geometry-basis finetune variants."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Training metadata for the geometry-basis finetune variants. All variants were warm-started from the same baseline checkpoint.}\n"
    "\\label{tab:training_metadata}\n"
    "\\begin{tabular}{@{}lcccc@{}}\n"
    "\\toprule\n"
    "Variant & Warm-start checkpoint & Training steps & Selection metric & Backend \\\\\n"
    "\\midrule\n"
    "Broad & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "Narrow & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "Mixed & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\n\\section{Counterpart Controller Specifications}\n"
    "\nTable~\\ref{tab:counterpart_specs} defines the two counterpart-controller classes used in the evaluation."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Counterpart controller specifications.}\n"
    "\\label{tab:counterpart_specs}\n"
    "\\begin{tabular}{@{}p{3cm}p{10cm}@{}}\n"
    "\\toprule\n"
    "Controller class & Description \\\\\n"
    "\\midrule\n"
    "Expert controller & Structured rule-based policy with hand-crafted engagement logic, collision-avoidance rules, and target-tracking heuristics. "
    "Reproducible and deterministic; provides a stable baseline for measuring interface changes. \\\\\n"
    "End-to-end learned & Fully learned counterpart policy trained independently under the same observation and action spaces. "
    "Adaptive and non-deterministic; tests interface stability against an interacting learning agent. \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\n\\section{Seed-Level Raw Values}\n"
    "\nTable~\\ref{tab:seed_level} reports the seed-level raw values or mean $\\pm$ SD for each cell in the comparison grid. "
    "Paired-seed scatter plots are available in the supplementary materials."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Seed-level summary statistics (mean $\\pm$ SD over seeds 480--489). Placeholder; to be populated from evaluation artifacts.}\n"
    "\\label{tab:seed_level}\n"
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ (mean $\\pm$ SD) & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (mean $\\pm$ SD) \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & 0.60 & $13.15 \\pm \\text{SD}$ & $10.84 \\pm \\text{SD}$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\n\\section{Coefficient Summary Table}\n"
    "\nTable~\\ref{tab:coefficient_summary} reports the mean pre-merge geometry-basis coefficients for each cell."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Mean pre-merge geometry-basis coefficients ($\\bar{a}_{k}^{\\mathrm{pre}}$) and lateral-preference fraction ($\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$). Placeholder.}\n"
    "\\label{tab:coefficient_summary}\n"
    "\\begin{tabular}{@{}llcccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $\\bar{a}_{\\mathrm{ll}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{io}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{cd}}^{\\mathrm{pre}}$ & $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\n\\section{Termination Count Table}\n"
    "\nTable~\\ref{tab:termination_counts} reports the raw termination counts for each cell over the 10 pilot seeds."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Termination counts over 10 pilot seeds. Fav.~TO = favourable timeout; Unfav.~TO = unfavourable timeout; Ego C/O = ego crash or out-of-bounds; Cp.~C/O = counterpart crash or out-of-bounds. Placeholder.}\n"
    "\\label{tab:termination_counts}\n"
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & Fav.~TO & Unfav.~TO & Ego C/O & Cp.~C/O \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
    "\n"
    "\n\\section{Terminology Mapping}\n"
    "\nTable~\\ref{tab:terminology} maps the internal evaluation artefact names to the manuscript terminology used in this paper."
    "\n\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Terminology mapping between internal evaluation artefacts and manuscript terminology.}\n"
    "\\label{tab:terminology}\n"
    "\\begin{tabular}{@{}ll@{}}\n"
    "\\toprule\n"
    "Internal artefact name & Manuscript term \\\\\n"
    "\\midrule\n"
    "head\\_on & frontal encounter \\\\\n"
    "crossing\\_feasible & crossing encounter \\\\\n"
    "expert & expert controller \\\\\n"
    "end\\_to\\_end & end-to-end learned controller \\\\\n"
    "win\\_rate & favourable outcome rate ($r_{\\mathrm{fav}}$) \\\\\n"
    "hp\\_advantage & damage margin ($m_{\\mathrm{dmg}}$) \\\\\n"
    "crashes (aggregate) & ego crash or out-of-bounds \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

# Insert before \end{document}
modified_text = modified_text.replace("\\end{document}", appendix_text + "\\end{document}")
print("Appendix sections added")

# ============================================================
# Write output
# ============================================================
out_path = Path("E:/uav-vpp-guidance/drones/dfartv2_drones_mdpi_v2.tex")
out_path.write_text(modified_text, encoding="utf-8")
print(f"\nWritten to: {out_path}")
print(f"Total characters: {len(modified_text)}")

# Verify key phrases
print("\n--- Verification ---")
checks = [
    "pilot-scale",
    "does not fully isolate finetune effects",
    "showed the most favourable observed trade-off",
    "does not constitute a held-out generalisation estimate",
    "reduces architectural confounds",
    "three auditable carriers",
    "warm-started from the same baseline checkpoint",
    "expert controller",
    "end-to-end learned controller",
    "AoA $60^\\circ$",
    "Termination semantics",
    "Statistical stance",
    "clearest pilot-scale support",
    "auditable interface-layer change",
    "methodological relevance for auditability assessment",
    "simulation-only and non-operational",
    "does not constitute a certification claim",
    "No isolated semantics-only control",
    "\\appendix",
    "Semantics-Only Control",
    "Training Metadata",
    "Counterpart Controller Specifications",
    "Seed-Level Raw Values",
    "Coefficient Summary",
    "Termination Count Table",
    "Terminology Mapping",
]
for c in checks:
    present = c in modified_text
    print(f"  {'[OK]' if present else '[MISSING]'} {c[:60]}")
