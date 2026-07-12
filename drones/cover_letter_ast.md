To the Editors of Aerospace Science and Technology

**Subject:** Submission of Manuscript — A Hierarchical Virtual-Pursuit-Point Maneuver Interface for Close-Range Air Combat: Geometry Shaping via Task-Label Routing

Dear Editors,

We are pleased to submit our manuscript, *"A Hierarchical Virtual-Pursuit-Point Maneuver Interface for Close-Range Air Combat: Geometry Shaping via Task-Label Routing,"* for consideration for publication in *Aerospace Science and Technology*.

**Positioning and Scope Fit.** This paper is positioned at the intersection of guidance-and-control and interpretable maneuver-decision interfaces. Its central question is not whether reinforcement learning can improve air-combat win rates, but whether a classical virtual-pursuit-point (VPP) guidance law can be advanced from pure virtual-point chasing toward a combat-geometry shaping maneuver interface *without* replacing its underlying three-dimensional low-level action head. We believe this scope aligns closely with AST's stated coverage areas of decision aid, flight mechanics, navigation/guidance/control, and robotics and intelligent systems. The contribution is therefore directed primarily at the guidance community, not merely at the reinforcement-learning community.

**Novel Contribution.** The paper introduces a hierarchical architecture in which a proximal-policy-optimization (PPO) high-level policy routes among frozen low-level VPP specialists, each producing task-dependent geometry behavior. The critical design choice is that the low-level VPP action head — the three-dimensional virtual-point command that makes VPP interpretable — is preserved unchanged. The decision intelligence is moved upward into a routing layer, turning tactical mode selection into an auditable interface decision while retaining geometric transparency at the low level. This advances VPP from a trajectory-tracking mechanism toward a maneuver-decision interface that shapes favorable engagement geometry under opponent-dependent conditions.

**Evaluation and Claim Discipline.** The evaluation is conducted in a high-fidelity JSBSim six-degree-of-freedom simulator, with two task families (head-on and crossing-feasible) and two opponent splits (expert and end-to-end). The paper reports bounded claims with explicit residual boundaries: the learned commander is no longer weaker than an oracle static gate on the previously blocking expert/head-on split, preserves crossing-feasible performance on both opponent splits, and admits a localized residual of three specific scenarios on end-to-end/head-on. All claims are supported by Wilson 95% confidence intervals, and the paper explicitly states what the evidence does and does not support. This claim discipline is intended to meet the rigor expected in guidance-and-control archival literature.

**Why AST Readership.** We believe the work is of interest to AST readers because it addresses a classical guidance formalism (VPP) and asks how far it can be extended toward autonomous maneuver decision before its low-level geometric interpretability is sacrificed. The answer — that a frozen hierarchical routing layer can advance VPP toward geometry shaping while preserving the original action head — offers a concrete path for guidance researchers who are interested in learning-based augmentation but cautious about losing the transparency and auditability of traditional geometric guidance laws.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and agree to the publication policies of *Aerospace Science and Technology*.

We look forward to your consideration.

Sincerely,

[Corresponding Author on behalf of all authors]

**Corresponding Author:** Jialong Jian  
**Affiliation:** Aviation Engineering School, Air Force Engineering University, Xi'an 710038, China  
**Email:** j330809@163.com

**Co-Corresponding Author:** Yuanfei Liu  
**Email:** 13572287887@163.com
