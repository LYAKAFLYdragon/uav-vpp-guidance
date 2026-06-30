# Reviewer Report for MDPI Drones

**Manuscript ID:** (not yet assigned)  
**Title:** From Virtual-Point Tracking to an Auditable Geometry-Basis Guidance Interface for Close-Range UAV Interaction  
**Authors:** Yu Lai, Yong Chen, Yang Yang, Jialong Jian, Yuanfei Liu  
**Journal:** Drones (MDPI)  
**Recommendation:** Major Revision (substantial changes required before acceptance)

---

## 1. Summary and Overall Assessment

This manuscript presents an interface-design study that reinterprets a conventional 3D virtual-point guidance action head as a "geometry-basis" interface, where action coefficients modulate longitudinal, lateral, and vertical geometry components. The work is paired with telemetry diagnostics (notably a "Forward Bias" metric) and evaluated under JSBSim flight dynamics across frontal and crossing close-range encounter scenarios. The authors compare four interface variants: a baseline Cartesian policy, a broad geometry-basis variant, a narrow variant, and a mixed task-selective variant. A notable strength is the explicit disclosure of negative findings (single-policy PPO cannot simultaneously excel in both encounter geometries), which lends scientific credibility.

However, the manuscript currently contains **data inconsistencies**, **incomplete appendix tables**, **insufficient statistical treatment** of pilot-scale results, and **unclear positioning relative to SOTA methods**. These issues prevent acceptance in its current form but are addressable through a major revision. The paper has potential to become a valuable methods contribution to interpretable guidance interface design if these concerns are addressed.

---

## 2. Critical Issues Requiring Resolution (Major Revision)

### 2.1 Data Inconsistency Between Main Text and Appendix (CRITICAL)

**Table 1 (main results, line 390)** reports for Mixed/Expert/Crossing:
- $r_{\mathrm{fav}} = 0.90$, $m_{\mathrm{dmg}} = 2.05$, $\bar{d}_{\mathrm{fwd}}^{\mathrm{pre}} = 234.84$

**Table A2 (Appendix, line 719)** reports for the same cell:
- $r_{\mathrm{fav}} = 0.40 \pm 0.49$, $m_{\mathrm{dmg}} = -3.35$, $\bar{d}_{\mathrm{fwd}}^{\mathrm{pre}} = 200.38$

These are **directly contradictory**. Either the main table or the appendix table contains incorrect data. If the 10-seed pilot (seeds 480–489) produced the appendix values, then the main table must be corrected. If the main table values come from a different evaluation run, the provenance must be explicitly declared. **This inconsistency alone is grounds for rejection** if not resolved.

**Action required:** Reconcile all numerical values across the manuscript. Every number must be traceable to a single `run_id` and `config_hash` from the artifact pipeline. The `run_manifest.json` for the exact evaluation run used in the paper must be cited in the data availability statement.

### 2.2 Incomplete Appendix Tables (CRITICAL)

Multiple appendix tables are still labeled "Placeholder" or contain "TBD" (to be determined):
- Table A2 (line 708): "Placeholder; to be populated from evaluation artifacts"
- Table A3 (line 730): "Placeholder"
- Table A4 (line 750): "Placeholder"

**Action required:** Populate all tables with verified data or remove them if not essential. Partially populated tables damage the manuscript's credibility. If space is a concern, the seed-level table (A2) could be moved to supplementary materials, but the coefficient summary (A3) and termination counts (A4) are essential for reproducibility and should remain in the appendix.

### 2.3 Statistical Treatment of Pilot-Scale Results (MAJOR)

The manuscript repeatedly acknowledges its "pilot-scale" nature (10 seeds, 1 episode per seed) and explicitly states that no inferential tests or confidence intervals were computed. While this honesty is commendable, the **absence of any statistical framing** weakens the paper's contribution. With only 10 binary samples per cell, the standard error for a proportion estimate is substantial ($\sqrt{p(1-p)/10} \approx 0.16$ for $p=0.5$). The difference between 0.55 and 0.35 in the structural trade-off table is not statistically distinguishable from noise at conventional levels.

**Action required:** Either (a) expand the evaluation to a minimum of 30–50 seeds per cell to support descriptive claims, or (b) add binomial confidence intervals (e.g., Clopper–Pearson) to all proportion estimates and explicitly caution that the reported differences are exploratory. If the latter, the abstract and conclusion must be further hedged to avoid implying definitive comparisons.

### 2.4 Missing "Semantics-Only Control" (MAJOR)

The manuscript acknowledges (Limitation #3, line 462) that a "fully isolated semantics-only control (identical checkpoint, no finetune, only the semantics switch) was not included." This is a **serious methodological gap**. The paper's core claim is that the geometry-basis interface produces different behaviour than the Cartesian baseline, but because all variants were warm-started and finetuned, the observed differences may partly reflect finetune history rather than pure semantics.

**Action required:** The authors must either (a) run a no-finetune semantics-only control and report the results, or (b) substantially reframe the contribution as "interface variant findings" rather than "semantics effects." The current hedging in the text is insufficient because the title and abstract still imply a semantic reinterpretation claim.

### 2.5 SOTA Comparison and Baseline Adequacy (MAJOR)

The manuscript compares geometry-basis variants against a single Cartesian baseline but does **not compare against existing SOTA methods** for interpretable UAV guidance or close-range encounter control (e.g., the hierarchical decision-making approach cited as [Xu et al., 2022], or the variable-scale action approach by Wang et al., 2023). For a methods paper in MDPI Drones, reviewers will expect at least a quantitative comparison with one or two recent relevant methods from the literature.

**Action required:** Add a comparison row in the main results table for at least one recent SOTA method from the literature (e.g., the tactical pursuit point approach from Xu et al., 2022, or the variable-scale RL approach from Wang et al., 2023). If direct comparison is impossible due to implementation differences, add a dedicated "Comparison with Related Work" subsection discussing the qualitative and quantitative differences.

---

## 3. Significant Issues (Should be Addressed)

### 3.1 Sample Size and Generalisability

Ten seeds with one episode per seed yields a very small sample. The structural trade-off section (Section 4.3) reports results over 20 seeds for the ablation, but the main table still uses 10 seeds. The manuscript should standardise on a single seed count across all experiments or clearly label which experiments used which protocol.

### 3.2 "Cp. crashes/OOB" Column in Main Table

The newly added 8th column (counterpart crashes/OOB) contains "N/A$^*$" for all broad-variant rows. The footnote states that broad-variant artifacts were not present in the 10-seed directories. This is problematic because the broad variant is discussed extensively in the text. If the data truly does not exist, the broad variant should be removed from the main table and discussed only in the text, or the missing data should be generated.

### 3.3 Appendix H: Evidence Package

Appendix H is a welcome addition that demonstrates the depth of the architectural exploration. However, it presents the ablation results as a standalone table without clear linkage to the main text. The text in Appendix H states the results are over 20 seeds, but the main table uses 10 seeds—this inconsistency should be explicitly noted and justified.

### 3.4 Writing and Terminology

The manuscript is generally well-written, but some phrases are overly verbose or could be misinterpreted:
- "pilot-scale" is used 15+ times; consider varying the phrasing or consolidating the limitation discussion.
- The abstract is truncated at line 55 (the original source had a truncation issue).
- "Cp. crashes/OOB" is non-standard abbreviation; spell out "Counterpart" or use "Adv." (adversary) consistently.

---

## 4. Minor Issues and Suggestions

### 4.1 Bibliography Completeness

The bibliography contains 30 entries, which is adequate. However, several recent relevant works are missing:
- **Meng et al. (2024)** on multi-task policy architectures for UAV air combat (if applicable to the multi-head ablation).
- **Recent survey papers on interpretable reinforcement learning for aerospace** (e.g., 2024–2025).
- The arXiv preprint for Schulman et al. (2017) should be updated to the published ICML version if available, or the citation should remain as preprint but with an explicit note.

### 4.2 Figure Quality

Figures 2–5 are referenced and described in detail, but the manuscript does not discuss their colour accessibility. For a journal with online readership, consider whether the colour palette is distinguishable for colour-blind readers.

### 4.3 Code and Data Availability

The data availability statement is reasonable but vague. To satisfy MDPI Drones' reproducibility standards, the authors should:
- Provide a public repository URL (GitHub/GitLab) or a Zenodo DOI.
- Include a `README.md` with exact commands to reproduce the main table results.

### 4.4 AI Disclosure

The acknowledgment states that generative AI was used for "language revision, structural editing, and figure-code generation." This is acceptable, but the authors should confirm that all numerical values, technical claims, and experimental conclusions were independently verified. I suggest adding a statement that the experimental data was not generated by AI.

---

## 5. Strengths of the Manuscript

Despite the issues above, the manuscript has several commendable strengths:

1. **Honest scientific reporting:** The explicit disclosure of the single-policy trade-off (Section 4.3) and the failure of multi-head architectures is rare and valuable. This "negative finding" framing actually strengthens the paper's credibility.
2. **Multi-dimensional evaluation:** The separation of outcome rate, damage margin, ego crashes, and counterpart crashes is methodologically sound and addresses a real problem in the field (over-optimistic aggregate metrics).
3. **Interface-level contribution:** The focus on auditability and geometry-shaping semantics is a meaningful contribution to trustworthy autonomy, distinct from the more common policy-learning or algorithmic contributions.
4. **Reproducible pipeline:** The mention of automated tests (118 tests), branch names, and artifact manifests suggests a mature engineering practice that supports scientific reproducibility.

---

## 6. Specific Questions for the Authors

1. **Data reconciliation:** Which exact evaluation run produced the numbers in Table 1? Can you provide the `run_id` and confirm that the appendix tables were populated from the same run? If not, explain the discrepancy.
2. **Semantics-only control:** Is it feasible to run a no-finetune semantics-only experiment within the revision timeline? If not, how will you reframe the main contribution to avoid overclaiming?
3. **SOTA comparison:** Can you add a quantitative comparison with the Xu et al. (2022) tactical pursuit point approach or the Wang et al. (2023) variable-scale action approach? If implementation barriers exist, please explain.
4. **Sample size:** Will you expand the evaluation to 20 or 30 seeds to match the ablation study, or will you add confidence intervals to the 10-seed results?
5. **Broad variant data:** Can you generate the missing broad-variant counterpart crash data for the 10-seed evaluation, or should the broad variant be removed from the main table?

---

## 7. Detailed Section-by-Section Comments

### Abstract (lines 54–56)
- **Issue:** Truncated. The final sentence is cut off ("whose stab...").
- **Issue:** The phrase "the strongest pilot balance" has been partially downgraded but still implies a ranking that is not statistically supported.
- **Suggestion:** Add a final sentence explicitly stating the pilot-scale and descriptive nature of the results.

### Introduction (lines 62–70)
- **Strength:** The three-bounded-contribution framing is clear and appropriate.
- **Suggestion:** The phrase "trust-oriented assessment and failure diagnosis" (line 64) could be tightened to "auditability for trust-oriented assessment and failure diagnosis."
- **Comment:** The distinction between the present work and post-hoc explainability methods is well-drawn.

### Methods (lines 83–293)
- **Strength:** The geometric formulation is clear, and Equation (4) is the key technical contribution.
- **Issue (line 161):** The text acknowledges that warm-starting confounds the comparison, but the main contribution claim still implies semantics isolation. This needs clearer alignment.
- **Suggestion:** The telemetry section (lines 200–220) is detailed but could be condensed by moving the full JSON schema description to the appendix.
- **Suggestion:** The pilot protocol (lines 258–264) should explicitly state whether the 10 seeds are the same across all encounter/controller combinations (paired design), which is important for reproducibility.

### Results (lines 294–403)
- **Strength:** The multi-metric interpretation (outcome + damage margin + Forward Bias + termination semantics) is the strongest methodological contribution.
- **Issue:** The broad variant's "Cp. crashes/OOB" column is entirely N/A, which weakens the comparison. This must be resolved.
- **Strength:** Section 4.3 (Structural Limitation) is an excellent addition that pre-empts reviewer skepticism about the residual crossing gap.
- **Issue:** Section 4.4 (Provenance) claims 118 passing tests but provides no citation or link. A footnote with the repository URL would suffice.

### Discussion (lines 405–429)
- **Strength:** The discussion correctly frames the contribution as interface-design evidence rather than policy optimality.
- **Suggestion:** The "Implications for Trustworthy Autonomous Guidance" subsection could be strengthened by referencing one or two specific assurance frameworks (e.g., ISO 21448 SOTIF, or the FAA's interpretable autonomy guidelines) to ground the discussion in regulatory context.
- **Suggestion:** The phrase "methodological relevance for auditability assessment" (line 425) is slightly awkward; "methodological relevance for auditability-oriented assessment" or "implications for auditability assessment" would read better.

### Limitations (lines 460–464)
- **Strength:** The six numbered limitations are comprehensive and honest. This is a model of responsible scientific reporting.
- **Suggestion:** Limitation #6 (single-policy trade-off) could reference the specific Appendix table (Table H.1) for readers who want the full evidence.

### Conclusion (lines 466–468)
- **Strength:** The conclusion appropriately hedges the contribution and explicitly notes the "known architectural limitations."
- **Suggestion:** The final sentence is very long and could be split for readability.
- **Suggestion:** Add a sentence about future work (multi-policy architectures, curriculum learning, etc.) to guide the reader.

### Appendix (lines 660–849)
- **Issue:** Tables A2, A3, and A4 contain "Placeholder" or "TBD" text. These must be populated or removed.
- **Issue:** The data in Table A2 does not match the main table. This is the most critical issue in the appendix.
- **Strength:** Appendix H is well-structured and provides valuable evidence for the structural trade-off claim.
- **Suggestion:** The warmstart recovery results (0.45/0.25 and 0.50/0.30) are described as "worse than baseline" but the baseline values in the main table are 0.55/0.35, while in Appendix H they are 0.55/0.35 (expert) and 0.95/1.00 (e2e). This is consistent, but the text should be explicit about which baseline is being used for comparison.

---

## 8. Final Recommendation and Decision Rationale

**Recommendation: Major Revision**

The manuscript presents a valuable and honest contribution to interpretable guidance interface design. The explicit disclosure of negative findings (single-policy structural trade-off) and the multi-dimensional evaluation framework are significant strengths that distinguish this work from typical RL policy papers. However, **the manuscript cannot be accepted in its current form due to data inconsistencies, incomplete appendix tables, and the lack of a semantics-only control**.

The authors must:
1. Reconcile all numerical values between the main text and appendix (Priority 1).
2. Populate or remove all placeholder tables (Priority 1).
3. Either run a semantics-only control or reframe the contribution claim (Priority 1).
4. Add statistical intervals (e.g., Clopper–Pearson) to all proportion estimates (Priority 2).
5. Add a SOTA comparison or a clear justification for its absence (Priority 2).
6. Resolve the missing broad-variant data in the main table (Priority 2).

If these issues are addressed satisfactorily, the revised manuscript would be a strong candidate for **Accept** in MDPI Drones. The topic is appropriate for the journal, the JSBSim flight-dynamics evaluation is rigorous, and the interface-design framing is a genuine methodological contribution. I look forward to reviewing the revised version.

---

**Reviewer Signature:** (Anonymous Reviewer for MDPI Drones)  
**Date:** 2026-06-30
