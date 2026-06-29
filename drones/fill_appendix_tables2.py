
import re
from pathlib import Path
import math

src = Path("E:/uav-vpp-guidance/drones/dfartv2_drones_mdpi_v2.tex")
text = src.read_text(encoding="utf-8")

# ============================================================
# 1. Fill main table Cp. crashes/OOB column (already done, skip)
# ============================================================

# ============================================================
# 2. Fill tab:training_metadata
# ============================================================
old_training = (
    "\\begin{tabular}{@{}lcccc@{}}\n"
    "\\toprule\n"
    "Variant & Warm-start checkpoint & Training steps & Selection metric & Backend \\\\\n"
    "\\midrule\n"
    "Broad & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "Narrow & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "Mixed & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_training = (
    "\\begin{tabular}{@{}lcccc@{}}\n"
    "\\toprule\n"
    "Variant & Warm-start checkpoint & Training steps & Selection metric & Backend \\\\\n"
    "\\midrule\n"
    "Broad & reset075\\_no\\_mode\\_switch\\_longscale00 & N/A & N/A & JSBSim \\\\\n"
    "Narrow & reset075\\_no\\_mode\\_switch\\_longscale00 & 4\\,096--16\\,384 & best by win rate (step 4096) & JSBSim \\\\\n"
    "Mixed & reset075\\_no\\_mode\\_switch\\_longscale00 & 4\\,096--16\\,384 & best by win rate (step 4096) & JSBSim \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

if old_training in text:
    text = text.replace(old_training, new_training)
    print("[OK] Replaced training_metadata")
else:
    print("[MISSING] Could not find training_metadata")


# ============================================================
# 3. Fill tab:seed_level
# ============================================================
old_seed = (
    "\\begin{tabular}{@{}llcccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ (mean $\\pm$ SD) & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (mean $\\pm$ SD) \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & 0.60 & $13.15 \\pm \\text{SD}$ & $10.84 \\pm \\text{SD}$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_seed = (
    "\\begin{tabular}{@{}llcccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ (mean $\\pm$ SD) & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (mean $\\pm$ SD) \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & $0.60 \\pm 0.49$ & $13.15$ & $10.84$ \\\\\n"
    "Expert & Crossing & Baseline & $0.30 \\pm 0.46$ & $-0.80$ & $-598.92$ \\\\\n"
    "Expert & Frontal & Narrow & $0.90 \\pm 0.30$ & $23.70$ & $185.92$ \\\\\n"
    "Expert & Crossing & Narrow & $0.20 \\pm 0.40$ & $-6.20$ & $148.90$ \\\\\n"
    "Expert & Frontal & Mixed & $1.00 \\pm 0.00$ & $37.45$ & $196.58$ \\\\\n"
    "Expert & Crossing & Mixed & $0.40 \\pm 0.49$ & $-3.35$ & $200.38$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

if old_seed in text:
    text = text.replace(old_seed, new_seed)
    print("[OK] Replaced seed_level")
else:
    print("[MISSING] Could not find seed_level")


# ============================================================
# 4. Fill tab:coefficient_summary
# ============================================================
old_coeff = (
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $\\bar{a}_{\\mathrm{ll}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{io}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{cd}}^{\\mathrm{pre}}$ & $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_coeff = (
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $\\bar{a}_{\\mathrm{ll}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{io}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{cd}}^{\\mathrm{pre}}$ & $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & N/A & N/A & N/A & N/A \\\\\n"
    "Expert & Crossing & Baseline & N/A & N/A & N/A & N/A \\\\\n"
    "Expert & Frontal & Narrow & $0.719$ & $-0.120$ & $0.667$ & $0.000$ \\\\\n"
    "Expert & Crossing & Narrow & $0.473$ & $0.190$ & $0.652$ & $1.000$ \\\\\n"
    "Expert & Frontal & Mixed & $0.610$ & $-0.017$ & $0.561$ & $0.037$ \\\\\n"
    "Expert & Crossing & Mixed & $0.279$ & $0.307$ & $0.562$ & $1.000$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

if old_coeff in text:
    text = text.replace(old_coeff, new_coeff)
    print("[OK] Replaced coefficient_summary")
else:
    print("[MISSING] Could not find coefficient_summary")


# ============================================================
# 5. Fill tab:termination_counts
# ============================================================
old_term = (
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & Fav.~TO & Unfav.~TO & Ego C/O & Cp.~C/O \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_term = (
    "\\begin{tabular}{@{}llccccc@{}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & Fav.~TO & Unfav.~TO & Ego C/O & Cp.~C/O \\\\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & 6 & 2 & 2 & 0 \\\\\n"
    "Expert & Crossing & Baseline & 3 & 6 & 1 & 0 \\\\\n"
    "Expert & Frontal & Narrow & 9 & 1 & 0 & 1 \\\\\n"
    "Expert & Crossing & Narrow & 2 & 8 & 0 & 1 \\\\\n"
    "Expert & Frontal & Mixed & 10 & 0 & 0 & 3 \\\\\n"
    "Expert & Crossing & Mixed & 4 & 5 & 1 & 1 \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

if old_term in text:
    text = text.replace(old_term, new_term)
    print("[OK] Replaced termination_counts")
else:
    print("[MISSING] Could not find termination_counts")


# ============================================================
# 6. Write back
# ============================================================
src.write_text(text, encoding="utf-8")
print(f"\nWritten to: {src}")
print(f"Total characters: {len(text)}")
