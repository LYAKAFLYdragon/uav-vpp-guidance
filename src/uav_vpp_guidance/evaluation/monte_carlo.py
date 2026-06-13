"""
Monte Carlo evaluation utilities.

Self-contained JSBSim integration.
"""



def run_monte_carlo(policy, env, config):
    """
    Evaluate policy under randomized initial configurations.

    Args:
        policy: Trained policy with predict() method.
        env (CloseRangeTrackingEnv): Evaluation environment.
        config (dict): Evaluation configuration.

    Returns:
        dict: Evaluation metrics including:
            - success_rate
            - non_crash_success_rate
            - crash_rate
            - timeout_rate
            - mean_return
            - mean_episode_length
    """
    # TODO: Implement Monte Carlo rollouts.
    raise NotImplementedError
