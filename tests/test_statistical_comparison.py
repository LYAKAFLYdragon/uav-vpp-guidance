import pytest
from uav_vpp_guidance.evaluation.statistical_comparison import mcnemar_exact_pvalue


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
