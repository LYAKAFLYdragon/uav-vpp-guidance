"""
Flight data recording utilities (ACMI, CSV).

TODO: Migrate ACMI recording from legacy project:
  <JSBSIM_ROOT>/scripts/render/ or similar.

Design principle: recording should not block or interfere with training logic.
"""

import os
import csv


class ACMIRecorder:
    """
    Recorder for ACMI (TacView) format flight data.

    Writes aircraft state per simulation step for later visualization.
    """

    def __init__(self, output_path):
        """
        Args:
            output_path (str): Path to .acmi output file.
        """
        self.output_path = output_path
        self._file = None
        self._step = 0

    def open(self):
        """Open the ACMI file for writing."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        # TODO: Write ACMI header based on TacView format.

    def record(self, timestamp, own_state, target_state):
        """
        Record one timestep.

        Args:
            timestamp (float): Simulation time in seconds.
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
        """
        if self._file is None:
            return
        # TODO: Format and write ACMI line.

    def close(self):
        """Close the ACMI file."""
        if self._file is not None:
            self._file.close()
            self._file = None


class CSVRecorder:
    """
    Simple CSV recorder for episode metrics.
    """

    def __init__(self, output_path, fieldnames):
        """
        Args:
            output_path (str): Path to CSV file.
            fieldnames (list): Column names.
        """
        self.output_path = output_path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def open(self):
        """Open CSV file and write header."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def record(self, row):
        """Record a row."""
        if self._writer is not None:
            self._writer.writerow(row)

    def close(self):
        """Close CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
