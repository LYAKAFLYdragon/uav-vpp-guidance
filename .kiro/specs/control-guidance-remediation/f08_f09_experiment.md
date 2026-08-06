# F08 CEM Credibility and F09 Metric Terminology Evidence

## F08 fixed-budget offline comparison
The frozen optimizer uses a 5-D, clipped diagonal Gaussian with 12 candidates and `elite_ratio=0.25`, which produces 3 elites. This evidence compares it with population, elite-fraction/covariance, and early-stop variants on a deterministic bounded 5-D surrogate. Every plan uses at most 120 objective evaluations and seeds `[0, 1, 2]`; it reports coverage bins, convergence spread, and best-score distributions.

The surrogate is not a rollout, policy, gain, safety, compute-time, or JSBSim result. The workspace only has zero-iteration dry-run CEM outputs and unavailable frozen checkpoints, so the F08 gate remains `needs_more_evidence`. Runtime optimizer defaults remain unchanged.

## F09 truthful label
`compute_empirical_regret(candidate_scores, current_index)` computes the non-negative score gap between the current candidate and the best candidate in the same sampled batch. Its truthful label is **within-batch empirical score gap to the best sampled candidate**. It is neither paired regret nor a rollback signal. This documentation changes no bilevel update, snapshot, or rollback behavior.

## Gate
Any CEM change needs identical-budget paired rollouts, seed distributions, wall-clock and coverage measurements, safety metrics, and strict-JSBSim validation. Any training/rollback behavior change requires a separate approved specification.
