"""Plot No-VPP vs VPP trajectories on mild-disadvantage for root-cause analysis."""
import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

OUT_DIR = "outputs/jsbsim_diagnose_figures"
os.makedirs(OUT_DIR, exist_ok=True)

TRAJ_FILES = {
    "No-VPP (timeout)": "outputs/jsbsim_diagnose/No-VPP_seed0_ep0_timeout_traj.json",
    "VPP (success)": "outputs/jsbsim_diagnose/VPP_seed0_ep0_success_traj.json",
}
SUCCESS_RANGE_M = 900.0
SUCCESS_ATA_DEG = 25.0


def load_traj(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_2d_trajectories(axes=None):
    fig, ax = plt.subplots(figsize=(7, 6)) if axes is None else (None, axes)
    for label, path in TRAJ_FILES.items():
        traj = load_traj(path)
        own_n = np.array([t["own_n"] for t in traj])
        own_e = np.array([t["own_e"] for t in traj])
        tgt_n = np.array([t["tgt_n"] for t in traj])
        tgt_e = np.array([t["tgt_e"] for t in traj])
        ax.plot(own_e, own_n, label=f"{label} own", linewidth=1.5)
        ax.plot(tgt_e, tgt_n, label=f"{label} target", linewidth=1.5, linestyle="--")
        ax.scatter(own_e[0], own_n[0], marker="o", s=30)
        ax.scatter(tgt_e[0], tgt_n[0], marker="x", s=30)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Mild-disadvantage trajectories: No-VPP vs VPP")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    if axes is None:
        fig.tight_layout()
        out = os.path.join(OUT_DIR, "disadvantage_trajectories.png")
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


def plot_time_series(key, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path in TRAJ_FILES.items():
        traj = load_traj(path)
        t = np.array([t["time_s"] for t in traj])
        y = np.array([t[key] for t in traj])
        ax.plot(t, y, label=label, linewidth=1.5)
    if key == "range_m":
        ax.axhline(SUCCESS_RANGE_M, color="green", linestyle="--", label="success range")
    if key == "ata_deg":
        ax.axhline(SUCCESS_ATA_DEG, color="green", linestyle="--", label="success ATA")
        ax.axhline(-SUCCESS_ATA_DEG, color="green", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    plot_2d_trajectories()
    plot_time_series("range_m", "Range (m)", "Range vs time", "disadvantage_range.png")
    plot_time_series("ata_deg", "ATA (deg)", "Antenna train angle vs time", "disadvantage_ata.png")
    plot_time_series("own_roll_deg", "Roll angle (deg)", "Ownship roll angle vs time", "disadvantage_roll.png")
    plot_time_series("own_alt", "Altitude (m)", "Ownship altitude vs time", "disadvantage_altitude.png")


if __name__ == "__main__":
    main()
