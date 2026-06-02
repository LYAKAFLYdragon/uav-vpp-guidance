"""
Curriculum learning utilities.

TODO: Design curriculum schedule if needed for difficult scenarios.
"""


class CurriculumScheduler:
    """
    Gradually increase scenario difficulty during training.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Curriculum configuration.
        """
        self.config = config
        self.current_level = 0

    def update(self, performance_metrics):
        """
        Update curriculum level based on recent performance.

        Args:
            performance_metrics (dict): Recent evaluation metrics.
        """
        # TODO: Implement curriculum update logic.
        raise NotImplementedError

    def get_current_scenario_weights(self):
        """
        Get sampling weights for scenario types at current curriculum level.

        Returns:
            dict: Scenario type -> sampling weight.
        """
        # TODO: Implement scenario weight schedule.
        raise NotImplementedError
