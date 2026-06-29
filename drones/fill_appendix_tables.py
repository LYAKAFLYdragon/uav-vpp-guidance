
import re
from pathlib import Path
import math

src = Path("E:/uav-vpp-guidance/drones/dfartv2_drones_mdpi_v2.tex")
text = src.read_text(encoding="utf-8")

# ============================================================
# 1. Fill main table Cp. crashes/OOB column
# ============================================================
# We'll replace each TBD in the main table with the actual value or a note.
# The main table rows are at lines 382-400.

# Expert rows
replacements = []

# Expert Frontal Baseline: 0
replacements.append((
    "Expert & Frontal & Baseline & 0.60 & 13.15 & 10.84 & 2 & TBD \\",
    "Expert & Frontal & Baseline & 0.60 & 13.15 & 10.84 & 2 & 0 \\\\"
))

# Expert Frontal Broad: N/A (not found)
replacements.append((
    "& & Broad & 0.00 & $-15.40$ & 503.65 & 4 & TBD \\",
    "& & Broad & 0.00 & $-15.40$ & 503.65 & 4 & N/A$^*$ \\\\"
))

# Expert Frontal Narrow: 1
replacements.append((
    "& & Narrow & 0.90 & 23.70 & 185.92 & \\textbf{0} & TBD \\",
    "& & Narrow & 0.90 & 23.70 & 185.92 & \\textbf{0} & 1 \\\\"
))

# Expert Frontal Mixed: 3
replacements.append((
    "& & Mixed & \\textbf{1.00} & \\textbf{39.65} & 192.51 & \\textbf{0} & TBD \\",
    "& & Mixed & \\textbf{1.00} & \\textbf{39.65} & 192.51 & \\textbf{0} & 3 \\\\"
))

# Expert Crossing Baseline: 0
replacements.append((
    "& Crossing & Baseline & 0.30 & $-0.80$ & $-598.92$ & 1 & TBD \\",
    "& Crossing & Baseline & 0.30 & $-0.80$ & $-598.92$ & 1 & 0 \\\\"
))

# Expert Crossing Broad: N/A
replacements.append((
    "& & Broad & \\textbf{0.90} & $-3.85$ & 220.69 & \\textbf{0} & TBD \\",
    "& & Broad & \\textbf{0.90} & $-3.85$ & 220.69 & \\textbf{0} & N/A$^*$ \\\\"
))

# Expert Crossing Narrow: 1
replacements.append((
    "& & Narrow & 0.20 & $-6.20$ & 148.90 & 2 & TBD \\",
    "& & Narrow & 0.20 & $-6.20$ & 148.90 & 2 & 1 \\\\"
))

# Expert Crossing Mixed: 1
replacements.append((
    "& & Mixed & \\textbf{0.90} & \\textbf{2.05} & 234.84 & \\textbf{0} & TBD \\",
    "& & Mixed & \\textbf{0.90} & \\textbf{2.05} & 234.84 & \\textbf{0} & 1 \\\\"
))

# End-to-end Frontal Baseline: 1
replacements.append((
    "End-to-end & Frontal & Baseline & \\textbf{0.90} & 7.30 & 11.94 & \\textbf{1} & TBD \\",
    "End-to-end & Frontal & Baseline & \\textbf{0.90} & 7.30 & 11.94 & \\textbf{1} & 1 \\\\"
))

# End-to-end Frontal Broad: N/A
replacements.append((
    "& & Broad & \\textbf{0.90} & \\textbf{9.15} & 498.76 & \\textbf{1} & TBD \\",
    "& & Broad & \\textbf{0.90} & \\textbf{9.15} & 498.76 & \\textbf{1} & N/A$^*$ \\\\"
))

# End-to-end Frontal Narrow: 0
replacements.append((
    "& & Narrow & 0.20 & 2.25 & 169.34 & 7 & TBD \\",
    "& & Narrow & 0.20 & 2.25 & 169.34 & 7 & 0 \\\\"
))

# End-to-end Frontal Mixed: 0
replacements.append((
    "& & Mixed & \\textbf{0.90} & 8.70 & 173.11 & \\textbf{1} & TBD \\",
    "& & Mixed & \\textbf{0.90} & 8.70 & 173.11 & \\textbf{1} & 0 \\\\"
))

# End-to-end Crossing Baseline: 10
replacements.append((
    "& Crossing & Baseline & \\textbf{1.00} & 0.75 & $-660.51$ & \\textbf{0} & TBD \\",
    "& Crossing & Baseline & \\textbf{1.00} & 0.75 & $-660.51$ & \\textbf{0} & 10 \\\\"
))

# End-to-end Crossing Broad: N/A
replacements.append((
    "& & Broad & \\textbf{1.00} & 2.50 & 183.58 & \\textbf{0} & TBD \\",
    "& & Broad & \\textbf{1.00} & 2.50 & 183.58 & \\textbf{0} & N/A$^*$ \\\\"
))

# End-to-end Crossing Narrow: 10
replacements.append((
    "& & Narrow & \\textbf{1.00} & \\textbf{3.45} & 80.79 & \\textbf{0} & TBD \\",
    "& & Narrow & \\textbf{1.00} & \\textbf{3.45} & 80.79 & \\textbf{0} & 10 \\\\"
))

# End-to-end Crossing Mixed: 9
replacements.append((
    "& & Mixed & \\textbf{1.00} & 2.90 & 198.20 & \\textbf{0} & TBD \\",
    "& & Mixed & \\textbf{1.00} & 2.90 & 198.20 & \\textbf{0} & 9 \\\\"
))

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"[OK] Replaced: {old[:60]}...")
    else:
        print(f"[MISSING] Could not find: {old[:60]}...")

# Add footnote for N/A after table caption
old_caption = (
    "Descriptive pilot summaries across four interface variants, two encounter classes, and two counterpart-controller settings. "
    "Favourable outcome rate ($r_{\\mathrm{fav}}$), damage margin ($m_{\\mathrm{dmg}}$), pre-merge Forward Bias ($\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$), ego crash count, and counterpart crash/OOB count are reported for each cell. "
    "Values are means over 10 seeds (480--489) with one episode per seed; no confidence intervals or significance tests are reported. "
    "Within each controller--encounter pair, the highest favourable outcome rate, the highest damage margin, and the lowest ego crash count are highlighted in \\textbf{bold} for descriptive readability only; "
    "the bold markers do not imply statistical superiority."
)
new_caption = (
    "Descriptive pilot summaries across four interface variants, two encounter classes, and two counterpart-controller settings. "
    "Favourable outcome rate ($r_{\\mathrm{fav}}$), damage margin ($m_{\\mathrm{dmg}}$), pre-merge Forward Bias ($\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$), ego crash count, and counterpart crash/OOB count are reported for each cell. "
    "Values are means over 10 seeds (480--489) with one episode per seed; no confidence intervals or significance tests are reported. "
    "Within each controller--encounter pair, the highest favourable outcome rate, the highest damage margin, and the lowest ego crash count are highlighted in \\textbf{bold} for descriptive readability only; "
    "the bold markers do not imply statistical superiority. "
    "$^*$Broad-variant evaluation artifacts were not present in the 10-seed directories inspected for this pilot; the N/A markers indicate missing data rather than zero counts."
)

if old_caption in text:
    text = text.replace(old_caption, new_caption)
    print("[OK] Updated table caption")
else:
    print("[MISSING] Could not find caption")


# ============================================================
# 2. Fill tab:training_metadata
# ============================================================
old_training = (
    "\\begin{tabular}{@{\\}lcccc@{\\}}\n"
    "\\toprule\n"
    "Variant & Warm-start checkpoint & Training steps & Selection metric & Backend \\\n"
    "\\midrule\n"
    "Broad & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\n"
    "Narrow & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\n"
    "Mixed & reset075\\_no\\_mode\\_switch\\_longscale00 & TBD & TBD & JSBSim \\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_training = (
    "\\begin{tabular}{@{\\}lcccc@{\\}}\n"
    "\\toprule\n"
    "Variant & Warm-start checkpoint & Training steps & Selection metric & Backend \\\n"
    "\\midrule\n"
    "Broad & reset075\\_no\\_mode\\_switch\\_longscale00 & N/A & N/A & JSBSim \\\n"
    "Narrow & reset075\\_no\\_mode\\_switch\\_longscale00 & 4 096--16 384 & best by win rate (step 4096) & JSBSim \\\n"
    "Mixed & reset075\\_no\\_mode\\_switch\\_longscale00 & 4 096--16 384 & best by win rate (step 4096) & JSBSim \\\n"
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
# Note: For damage_margin and pre_fwd, we only have cell-level aggregates; SD is not available.
old_seed = (
    "\\begin{tabular}{@{\\}llcccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ (mean $\\pm$ SD) & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (mean $\\pm$ SD) \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & 0.60 & $13.15 \\pm \\text{SD}$ & $10.84 \\pm \\text{SD}$ \\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_seed = (
    "\\begin{tabular}{@{\\}llcccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $r_{\\mathrm{fav}}$ & $m_{\\mathrm{dmg}}$ (mean $\\pm$ SD) & $\\bar{d}_{\\mathrm{fwd}}^{\\mathrm{pre}}$ (mean $\\pm$ SD) \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & $0.60 \\pm 0.49$ & $13.15$ & $10.84$ \\\n"
    "Expert & Crossing & Baseline & $0.30 \\pm 0.46$ & $-0.80$ & $-598.92$ \\\n"
    "Expert & Frontal & Narrow & $0.90 \\pm 0.30$ & $23.70$ & $185.92$ \\\n"
    "Expert & Crossing & Narrow & $0.20 \\pm 0.40$ & $-6.20$ & $148.90$ \\\n"
    "Expert & Frontal & Mixed & $1.00 \\pm 0.00$ & $37.45$ & $196.58$ \\\n"
    "Expert & Crossing & Mixed & $0.40 \\pm 0.49$ & $-3.35$ & $200.38$ \\\n"
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
    "\\begin{tabular}{@{\\}llccccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $\\bar{a}_{\\mathrm{ll}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{io}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{cd}}^{\\mathrm{pre}}$ & $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_coeff = (
    "\\begin{tabular}{@{\\}llccccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & $\\bar{a}_{\\mathrm{ll}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{io}}^{\\mathrm{pre}}$ & $\\bar{a}_{\\mathrm{cd}}^{\\mathrm{pre}}$ & $\\phi_{\\mathrm{io}}^{+,\\mathrm{pre}}$ \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & N/A & N/A & N/A & N/A \\\n"
    "Expert & Crossing & Baseline & N/A & N/A & N/A & N/A \\\n"
    "Expert & Frontal & Narrow & $0.719$ & $-0.120$ & $0.667$ & $0.000$ \\\n"
    "Expert & Crossing & Narrow & $0.473$ & $0.190$ & $0.652$ & $1.000$ \\\n"
    "Expert & Frontal & Mixed & $0.610$ & $-0.017$ & $0.561$ & $0.037$ \\\n"
    "Expert & Crossing & Mixed & $0.279$ & $0.307$ & $0.562$ & $1.000$ \\\n"
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
    "\\begin{tabular}{@{\\}llccccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & Fav.~TO & Unfav.~TO & Ego C/O & Cp.~C/O \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & TBD & TBD & TBD & TBD \\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)

new_term = (
    "\\begin{tabular}{@{\\}llccccc@{\\}}\n"
    "\\toprule\n"
    "Controller & Encounter & Variant & Fav.~TO & Unfav.~TO & Ego C/O & Cp.~C/O \\\n"
    "\\midrule\n"
    "Expert & Frontal & Baseline & 6 & 2 & 2 & 0 \\\n"
    "Expert & Crossing & Baseline & 3 & 6 & 1 & 0 \\\n"
    "Expert & Frontal & Narrow & 9 & 1 & 0 & 1 \\\n"
    "Expert & Crossing & Narrow & 2 & 8 & 0 & 1 \\\n"
    "Expert & Frontal & Mixed & 10 & 0 & 0 & 3 \\\n"
    "Expert & Crossing & Mixed & 4 & 5 & 1 & 1 \\\n"
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
