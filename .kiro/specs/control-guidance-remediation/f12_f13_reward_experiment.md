# F12/F13 Reward Decomposition Evidence

This analysis is offline-only and does not modify `reward.py` or reward defaults. It records the current step terms, injected terminal term, low-altitude safety occupancy, and command-saturation occupancy on a deterministic synthetic trace. A counterfactual changes only `w_alive` and `w_closing` to zero, so their contribution can be isolated without altering the remaining formula.

The trace is not policy, training, task, exploit-resistance, or JSBSim evidence. The frozen scenario matrix and required checkpoints are unavailable. Therefore F12/F13 remain `needs_more_evidence`, with default reward weights retained. A later gate requires one-at-a-time fixed-protocol ablations, per-seed task/safety/return decomposition, anti-exploitation evidence, and strict-JSBSim results. Any loaded YAML mutation must be recorded with `record_config_override`.
