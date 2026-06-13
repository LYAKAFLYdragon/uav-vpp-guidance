"""Optional SwanLab logging integration.

This module provides a thin wrapper around SwanLab so that the training script
can log metrics remotely without hard-depending on the `swanlab` package. If
SwanLab is not installed, the logger falls back to a no-op with a warning.
"""
import os
import warnings


class SwanLabLogger:
    """Lightweight SwanLab logger with graceful fallback."""

    def __init__(self, project=None, experiment=None, config=None, enabled=True):
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import swanlab
        except ImportError as exc:  # pragma: no cover
            warnings.warn(
                "SwanLab is not installed (`pip install swanlab`). "
                "Remote logging will be disabled."
            )
            self.enabled = False
            return

        project = project or os.environ.get("SWANLAB_PROJECT", "uav-vpp-guidance")
        experiment = experiment or os.environ.get("SWANLAB_EXP_NAME", None)
        self._run = swanlab.init(
            project=project,
            experiment_name=experiment,
            config=config or {},
        )

    def log(self, metrics, step=None):
        """Log a dictionary of metrics at the given step."""
        if not self.enabled or self._run is None:
            return
        try:
            import swanlab
            swanlab.log(metrics, step=step)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"SwanLab log failed: {exc}")

    def finish(self):
        """Finalize the SwanLab run."""
        if not self.enabled or self._run is None:
            return
        try:
            import swanlab
            swanlab.finish()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"SwanLab finish failed: {exc}")
