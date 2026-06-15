#!/usr/bin/env python3
"""
Draw a decision flowchart for an Air Combat Maneuver Library Expert System.
Outputs: paper_materials/figures/aircombat_expert_flowchart.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


class Box:
    def __init__(self, ax, x, y, w, h, text, color="#E8F4FD", fontsize=10, bold=False):
        self.ax = ax
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor="#333333", linewidth=1.2
        )
        ax.add_patch(self.box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                weight=weight, linespacing=1.1)

    # anchor points
    def top(self):
        return (self.x, self.y + self.h/2)

    def bottom(self):
        return (self.x, self.y - self.h/2)

    def left(self):
        return (self.x - self.w/2, self.y)

    def right(self):
        return (self.x + self.w/2, self.y)


class Diamond:
    def __init__(self, ax, x, y, w, h, text, color="#FFF3CD", fontsize=10):
        self.ax = ax
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dia = Polygon(
            [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)],
            facecolor=color, edgecolor="#333333", linewidth=1.2
        )
        ax.add_patch(self.dia)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                weight="bold", linespacing=1.0)

    def top(self):
        return (self.x, self.y + self.h/2)

    def bottom(self):
        return (self.x, self.y - self.h/2)

    def left(self):
        return (self.x - self.w/2, self.y)

    def right(self):
        return (self.x + self.w/2, self.y)


def arrow(ax, p1, p2, label="", label_offset=(0, 0.12), color="#333333"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                fontsize=8, ha="center", va="bottom", color="#555555")


def elbow_arrow(ax, p1, p2, label="", label_offset=(0, 0.08), rad=0.2, color="#333333"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                fontsize=8, ha="center", va="bottom", color="#555555")


fig, ax = plt.subplots(figsize=(12, 15))
ax.set_xlim(0, 12)
ax.set_ylim(0, 15)
ax.axis("off")

# Title
ax.text(6, 14.5, "Air Combat Maneuver Library Expert System",
        fontsize=18, weight="bold", ha="center")
ax.text(6, 14.1, "Decision Flowchart", fontsize=13, ha="center", color="#555555")

# Main vertical chain (center x=6)
start       = Box(ax, 6, 13.3, 3.0, 0.6, "START\nMission / Engagement Trigger", color="#D1ECF1", fontsize=11, bold=True)
sensor      = Box(ax, 6, 12.2, 5.0, 0.7, "Sensor Fusion & Tactical Picture\nown state · bandit state · missiles · weather/terrain", color="#E2E3E5", fontsize=10)
geometry    = Box(ax, 6, 11.1, 4.8, 0.7, "Relative Geometry Computation\nrange · ATA · aspect · speed/altitude · energy state", color="#E8F4FD", fontsize=10)
threat      = Diamond(ax, 6, 9.9, 1.6, 1.0, "Threat\nAssessment", color="#FFF3CD", fontsize=10)

# Branch libraries
defensive   = Box(ax, 1.8, 9.9, 2.6, 0.9, "Defensive Library\nSplit-S\nScissors\nMissile-Evade Jink", color="#F8D7DA", fontsize=9)
offensive   = Box(ax, 10.2, 9.9, 2.6, 0.9, "Offensive Library\nHigh Yo-Yo\nLag Roll\nCobra / Pugachev", color="#D4EDDA", fontsize=9)
neutral     = Box(ax, 6, 8.5, 2.8, 0.8, "Neutral / Energy Library\nTurn-Reversal · Vertical Slice · Energy Extension", color="#FFF3CD", fontsize=9)

candidate   = Box(ax, 6, 7.2, 4.0, 0.7, "Generate Candidate Maneuvers\nfilter by geometry, energy, weapons envelope", color="#E8F4FD", fontsize=10)
cost        = Box(ax, 6, 6.1, 4.4, 0.7, "Cost-Reward Evaluation\nkill probability · escape probability · fuel · time-to-kill", color="#E2E3E5", fontsize=10)
feasible    = Diamond(ax, 6, 5.0, 1.6, 1.0, "Feasible?", color="#FFF3CD", fontsize=10)
relax       = Box(ax, 1.8, 5.0, 2.2, 0.7, "Relax Constraints\n& Replan", color="#FFF3CD", fontsize=9)
select      = Box(ax, 6, 3.8, 3.2, 0.7, "Select Best Maneuver\nargmax over utility score", color="#D1ECF1", fontsize=10, bold=True)
command     = Box(ax, 6, 2.9, 3.6, 0.7, "Flight-Command Generation\nnz_cmd · roll_rate_cmd · throttle_cmd", color="#E8F4FD", fontsize=10)
actuator    = Box(ax, 6, 2.0, 3.4, 0.7, "Actuator Dynamics & Execution\nlag · rate limits · saturation", color="#E2E3E5", fontsize=10)
state       = Box(ax, 6, 1.1, 4.0, 0.7, "State Update & Outcome Feedback\nrange/ATA/energy change · hit/miss", color="#E8F4FD", fontsize=10)
end         = Box(ax, 6, 0.2, 2.2, 0.5, "END / Disengage", color="#D1ECF1", fontsize=11, bold=True)

# Arrows: main chain
arrow(ax, start.bottom(), sensor.top())
arrow(ax, sensor.bottom(), geometry.top())
arrow(ax, geometry.bottom(), threat.top())

# Branches from threat diamond
arrow(ax, threat.left(), defensive.right(), label="defensive", label_offset=(-0.05, 0.08))
arrow(ax, threat.right(), offensive.left(), label="offensive", label_offset=(0.05, 0.08))
arrow(ax, threat.bottom(), neutral.top(), label="neutral / beam", label_offset=(0.45, 0.05))

# Converge to candidate
elbow_arrow(ax, defensive.bottom(), candidate.left(), rad=-0.15)
elbow_arrow(ax, offensive.bottom(), candidate.right(), rad=0.15)
arrow(ax, neutral.bottom(), candidate.top())

# Main chain below candidate
arrow(ax, candidate.bottom(), cost.top())
arrow(ax, cost.bottom(), feasible.top())

# Feasible no -> relax -> replan loop
arrow(ax, feasible.left(), relax.right(), label="No", label_offset=(-0.05, 0.08), color="#856404")
elbow_arrow(ax, relax.top(), candidate.left(), rad=-0.25, color="#856404")

# Feasible yes -> select
arrow(ax, feasible.bottom(), select.top(), label="Yes", label_offset=(0.08, 0.05))

# Continue main chain
arrow(ax, select.bottom(), command.top())
arrow(ax, command.bottom(), actuator.top())
arrow(ax, actuator.bottom(), state.top())

# Feedback loop
elbow_arrow(ax, state.right(), offensive.right(), label="re-evaluate", label_offset=(0.15, 0.0), rad=0.25)
# Termination
arrow(ax, state.bottom(), end.top(), label="termination", label_offset=(0.45, 0.05))

# Legend
ax.text(0.4, 0.05, "Legend:", fontsize=9, weight="bold")
ax.add_patch(FancyBboxPatch((1.1, -0.05), 0.4, 0.25, boxstyle="round,pad=0.02",
                             facecolor="#E8F4FD", edgecolor="#333"))
ax.text(1.65, 0.08, "Process", fontsize=9, va="center")
ax.add_patch(Polygon([(3.0, 0.2), (3.35, 0.08), (3.0, -0.05), (2.65, 0.08)],
                     facecolor="#FFF3CD", edgecolor="#333"))
ax.text(3.5, 0.08, "Decision", fontsize=9, va="center")

plt.tight_layout()
output_path_png = "paper_materials/figures/aircombat_expert_flowchart.png"
output_path_svg = "paper_materials/figures/aircombat_expert_flowchart.svg"
fig.savefig(output_path_png, dpi=200, bbox_inches="tight")
fig.savefig(output_path_svg, format="svg", bbox_inches="tight")
print(f"Saved flowchart to {output_path_png}")
print(f"Saved flowchart to {output_path_svg}")
