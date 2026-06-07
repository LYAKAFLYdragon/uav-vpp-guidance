"""
Plotting utilities for training curves and evaluation figures.

TODO: Migrate plotting helpers from legacy project if reusable.
  <JSBSIM_ROOT>/get_reward_figure/ contains many ad-hoc scripts;
  extract only general-purpose functions.
"""

import os
import matplotlib.pyplot as plt


def plot_training_curve(steps, values, label, title, ylabel, output_path):
    """
    Plot and save a training curve.

    Args:
        steps (list): X-axis steps.
        values (list): Y-axis values.
        label (str): Line label.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        output_path (str): Output file path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=label)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
