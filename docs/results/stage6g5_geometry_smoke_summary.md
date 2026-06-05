# Stage 6G.5A Geometry Smoke Summary

- **Experiment**: stage6g5_wide_geometry_smoke
- **Timestamp**: 20260605_224921
- **Total grid size**: 324
- **Sampled points**: 40
- **Sampling method**: random
- **Evaluated episodes**: 120
- **Methods evaluated**: ['no_prediction']
- **Scope note**: Stage 6G.5A smoke tests baseline geometric feasibility only; predictor-policy feasibility requires Stage 6G.5B.
- **Any success >20%**: False
- **Best success rate**: 0.0%
- **Best geometry**: {'initial_range_m': 3200, 'ego_speed_mps': 280, 'target_speed_mps': 160, 'aspect_angle_deg': 0, 'altitude_diff_m': 500, 'closure_rate_mps': 120.0, 'range_rate_mps': -120.0, 'estimated_time_to_capture_s': 26.67, 'expected_feasible_flag': True}
- **Bilevel unblocked candidate**: False

> **Note**: `bilevel_unblocked_candidate` is `true` only when a geometry combo
> shows >20% success *and* the closure rate suggests the failure mode is
> gain-sensitive rather than geometrically infeasible.
