import pytest
import numpy as np
from uav_vpp_guidance.evaluation.statistical_comparison import (
    bootstrap_confidence_interval,
    bootstrap_ci,
    paired_t_test,
    cohens_d,
    mann_whitney_u,
    mcnemar_exact_pvalue,
    mean_std,
    paired_delta,
)


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap confidence interval (paper-level API)."""

    def test_empty_data_returns_nan(self):
        mean, lower, upper = bootstrap_confidence_interval([])
        assert np.isnan(mean) and np.isnan(lower) and np.isnan(upper)

    def test_single_value_ci_is_that_value(self):
        mean, lower, upper = bootstrap_confidence_interval([5.0], n_bootstrap=100)
        assert mean == pytest.approx(5.0)
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)

    def test_ci_bounds_contain_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lower, upper = bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95)
        assert lower <= mean <= upper

    def test_higher_ci_is_wider(self):
        data = np.random.default_rng(42).normal(0, 1, 50).tolist()
        _, lower_90, upper_90 = bootstrap_confidence_interval(data, n_bootstrap=2000, ci=0.90)
        _, lower_95, upper_95 = bootstrap_confidence_interval(data, n_bootstrap=2000, ci=0.95)
        assert (upper_95 - lower_95) >= (upper_90 - lower_90)

    def test_reproducible_with_same_seed(self):
        data = np.random.default_rng(123).normal(10, 2, 30).tolist()
        r1 = bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95, random_seed=42)
        r2 = bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95, random_seed=42)
        assert r1 == r2

    def test_coverage_on_known_distribution(self):
        """Bootstrap CI should cover true mean ~95% of the time for normal data."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        covered = 0
        n_trials = 100
        for _ in range(n_trials):
            sample = rng.normal(true_mean, 1.0, 100).tolist()
            _, lower, upper = bootstrap_confidence_interval(sample, n_bootstrap=1000, ci=0.95)
            if lower <= true_mean <= upper:
                covered += 1
        # Should be roughly 95% — allow generous margin for small n_trials
        assert covered >= 80, f"Coverage was only {covered}%"

    def test_alias_bootstrap_ci_matches(self):
        """bootstrap_confidence_interval and bootstrap_ci should return identical results."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        r1 = bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95, random_seed=123)
        r2 = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, random_seed=123)
        assert r1 == r2


class TestPairedTTest:
    """Tests for paired_t_test."""

    def test_identical_samples_returns_high_pvalue(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(data, data)
        assert result['p_value'] > 0.05
        assert result['significant_at_05'] is False
        assert result['mean_diff'] == pytest.approx(0.0, abs=1e-6)

    def test_clear_difference_is_significant(self):
        """Two clearly different distributions should show significant p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50).tolist()
        b = rng.normal(2, 1, 50).tolist()
        result = paired_t_test(a, b)
        assert result['significant_at_05'] is True
        assert result['significant_at_01'] is True
        assert result['mean_diff'] > 1.0

    def test_small_n_insufficient(self):
        """With only 1 pair, t-test is not computable (needs df > 0)."""
        result = paired_t_test([1.0], [2.0])
        assert result['n_pairs'] == 1
        assert np.isnan(result['t_statistic'])
        assert np.isnan(result['p_value'])

    def test_empty_returns_nan(self):
        result = paired_t_test([], [])
        assert np.isnan(result['p_value'])
        assert result['n_pairs'] == 0

    def test_nan_pairs_excluded(self):
        a = [1.0, 2.0, np.nan, 4.0]
        b = [1.1, 2.1, 3.1, np.nan]
        result = paired_t_test(a, b)
        assert result['n_pairs'] == 2  # Only first two pairs are valid

    def test_mismatched_lengths_handled(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1]
        result = paired_t_test(a, b)
        assert result['n_pairs'] == 2

    def test_negative_treatment_better(self):
        """If method B is worse than A, mean_diff should be negative."""
        a = [5.0, 5.0, 5.0, 5.0, 5.0]
        b = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = paired_t_test(a, b)
        assert result['mean_diff'] == pytest.approx(-4.0)
        assert result['significant_at_05'] is True


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_identical_samples_d_is_zero(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = cohens_d(data, data)
        assert result['d'] == pytest.approx(0.0, abs=1e-6)
        assert result['magnitude'] == 'negligible'

    def test_small_effect_size(self):
        """d ≈ 0.3 should classify as 'small'."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100).tolist()
        noise = rng.normal(0, 1, 100).tolist()
        b = [a[i] + 0.3 + noise[i] for i in range(100)]
        result = cohens_d(a, b)
        assert 0.15 <= abs(result['d']) < 0.5
        assert result['magnitude'] == 'small'

    def test_medium_effect_size(self):
        """d ≈ 0.6 should classify as 'medium'."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100).tolist()
        noise = rng.normal(0, 1, 100).tolist()
        b = [a[i] + 0.6 + noise[i] for i in range(100)]
        result = cohens_d(a, b)
        assert 0.5 <= abs(result['d']) < 0.8
        assert result['magnitude'] == 'medium'

    def test_large_effect_size(self):
        """d ≈ 1.0 should classify as 'large'."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100).tolist()
        noise = rng.normal(0, 1, 100).tolist()
        b = [a[i] + 1.0 + noise[i] for i in range(100)]
        result = cohens_d(a, b)
        assert abs(result['d']) >= 0.8
        assert result['magnitude'] == 'large'

    def test_negligible_effect_size(self):
        """d ≈ 0.1 should classify as 'negligible'."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100).tolist()
        noise = rng.normal(0, 1, 100).tolist()
        b = [a[i] + 0.1 + noise[i] for i in range(100)]
        result = cohens_d(a, b)
        assert abs(result['d']) < 0.2
        assert result['magnitude'] == 'negligible'

    def test_empty_returns_nan(self):
        result = cohens_d([], [])
        assert np.isnan(result['d'])
        assert result['magnitude'] == 'unknown'

    def test_zero_std_returns_inf(self):
        """If all differences are identical and non-zero, d = +/-inf, magnitude = large."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]  # Constant +1.0 difference
        result = cohens_d(a, b)
        assert result['std_diff'] == 0.0
        assert np.isinf(result['d'])
        assert result['magnitude'] == 'large'

    def test_negative_diff_magnitude_positive(self):
        """Magnitude should reflect |d|, not signed d."""
        a = [5.0, 5.0, 5.0]
        b = [1.0, 1.0, 1.0]
        result = cohens_d(a, b)
        assert result['d'] < 0
        assert result['magnitude'] == 'large'

    def test_n_pairs_matches(self):
        a = [1.0, 2.0, np.nan, 4.0]
        b = [1.1, 2.1, 3.1, np.nan]
        result = cohens_d(a, b)
        assert result['n_pairs'] == 2


class TestMannWhitneyU:
    """Tests for Mann-Whitney U test."""

    def test_identical_samples_pvalue_one(self):
        """Identical samples should yield p=1 (or close to it)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mann_whitney_u(data, data)
        assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        assert result['significant_at_05'] is False

    def test_clear_difference_is_significant(self):
        """Two clearly different distributions should show significant p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50).tolist()
        b = rng.normal(3, 1, 50).tolist()
        result = mann_whitney_u(a, b)
        assert result['significant_at_05'] is True
        assert result['significant_at_01'] is True

    def test_empty_returns_nan(self):
        result = mann_whitney_u([], [1.0, 2.0])
        assert np.isnan(result['p_value'])
        assert result['n_a'] == 0

    def test_both_empty_returns_nan(self):
        result = mann_whitney_u([], [])
        assert np.isnan(result['p_value'])

    def test_nan_values_excluded(self):
        a = [1.0, 2.0, np.nan, 4.0]
        b = [1.1, 2.1, 3.1, 4.1]
        result = mann_whitney_u(a, b)
        assert result['n_a'] == 3
        assert result['n_b'] == 4

    def test_small_samples_valid(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        result = mann_whitney_u(a, b)
        assert 0 <= result['p_value'] <= 1
        assert result['n_a'] == 2
        assert result['n_b'] == 2

    def test_non_parametric_vs_ttest(self):
        """Mann-Whitney should be more robust to outliers than t-test."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 100.0]  # Outlier
        mw = mann_whitney_u(a, b)
        # With outlier, t-test might be significant, but MW should not be
        assert mw['significant_at_05'] is False

    def test_all_identical_values_returns_one(self):
        """All identical values should gracefully return p=1."""
        a = [5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0]
        result = mann_whitney_u(a, b)
        assert result['p_value'] == pytest.approx(1.0, abs=0.01)


class TestMcNemarExactPvalue:
    """Unit tests for exact McNemar p-value using scipy.stats.binomtest."""

    def test_b_zero_c_zero_returns_one(self):
        assert mcnemar_exact_pvalue(0, 0) == 1.0

    def test_b_one_c_zero_returns_one(self):
        assert mcnemar_exact_pvalue(1, 0) == 1.0

    def test_b_zero_c_one_returns_one(self):
        assert mcnemar_exact_pvalue(0, 1) == 1.0

    def test_b_five_c_zero_is_within_range(self):
        p = mcnemar_exact_pvalue(5, 0)
        assert 0 <= p <= 1
        assert p == pytest.approx(0.0625, abs=0.001)

    def test_b_ten_c_two_is_within_range(self):
        p = mcnemar_exact_pvalue(10, 2)
        assert 0 <= p <= 1

    def test_negative_b_raises(self):
        with pytest.raises(ValueError, match="b and c must be non-negative"):
            mcnemar_exact_pvalue(-1, 0)

    def test_negative_c_raises(self):
        with pytest.raises(ValueError, match="b and c must be non-negative"):
            mcnemar_exact_pvalue(0, -1)

    def test_symmetry_b_and_c(self):
        p1 = mcnemar_exact_pvalue(5, 2)
        p2 = mcnemar_exact_pvalue(2, 5)
        assert p1 == pytest.approx(p2)

    def test_large_n_within_range(self):
        p = mcnemar_exact_pvalue(50, 30)
        assert 0 <= p <= 1


class TestMeanStd:
    """Tests for mean_std helper."""

    def test_basic(self):
        m, s = mean_std([1.0, 2.0, 3.0])
        assert m == pytest.approx(2.0)
        assert s == pytest.approx(1.0)

    def test_empty(self):
        m, s = mean_std([])
        assert np.isnan(m) and np.isnan(s)

    def test_with_nan(self):
        m, s = mean_std([1.0, np.nan, 3.0])
        assert m == pytest.approx(2.0)


class TestPairedDelta:
    """Tests for paired_delta helper."""

    def test_basic(self):
        mean_d, std_d, n = paired_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert mean_d == pytest.approx(1.0)
        assert n == 3

    def test_empty(self):
        mean_d, std_d, n = paired_delta([], [])
        assert np.isnan(mean_d)
        assert n == 0
