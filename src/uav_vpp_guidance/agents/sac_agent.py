"""
SAC agent implementation (optional).

TODO: Implement SAC if needed for comparison.
"""


class SACAgent:
    """
    Soft Actor-Critic agent.

    Optional alternative to PPO for continuous control.
    """

    def __init__(self, env, config):
        """
        Args:
            env (gym.Env or CloseRangeTrackingEnv): Training environment.
            config (dict): SAC hyperparameters.
        """
        self.env = env
        self.config = config

    def learn(self, total_timesteps):
        """
        Train the agent.

        Args:
            total_timesteps (int): Total number of environment steps.

        Returns:
            SACAgent: Self.
        """
        # TODO: Implement SAC training loop.
        raise NotImplementedError

    def predict(self, observation, deterministic=False):
        """
        Predict action from observation.

        Args:
            observation (np.ndarray): Observation vector.
            deterministic (bool): If True, return mean action.

        Returns:
            tuple: (action, state)
        """
        # TODO: Implement action prediction.
        raise NotImplementedError

    def save(self, path):
        """Save model checkpoint."""
        raise NotImplementedError

    def load(self, path):
        """Load model checkpoint."""
        raise NotImplementedError
