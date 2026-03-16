"""Tests for the metacognition module.

Tests SDT computations and meta-d' estimation, verifying:
1. d' computation is correct for known hit/FA rates.
2. Meta-d' MLE returns sensible values for simulated data.
3. m-ratio to gamma mapping is monotonic and bounded.
4. Online precision update adjusts gamma correctly.
"""

import numpy as np

from alf.metacognition import (
    compute_sdt_type1,
    compute_type1_from_counts,
    fit_meta_d_mle,
    m_ratio_to_gamma,
    update_gamma_from_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sdt_data(
    d_prime: float = 1.5,
    c: float = 0.0,
    n_s1: int = 200,
    n_s2: int = 200,
    n_ratings: int = 4,
    meta_d: float = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SDT confidence rating data.

    Args:
        d_prime: True d'.
        c: Response criterion.
        n_s1: Number of stimulus-absent (S1) trials.
        n_s2: Number of stimulus-present (S2) trials.
        n_ratings: Number of confidence levels per response.
        meta_d: Meta-d' for confidence. Default: same as d_prime.
        seed: Random seed.

    Returns:
        Tuple of (nR_S1, nR_S2) count arrays.
    """
    if meta_d is None:
        meta_d = d_prime

    rng = np.random.RandomState(seed)

    # Evenly spaced confidence criteria
    criteria = np.linspace(-2.0, 2.0, 2 * n_ratings - 1)

    # Simulate S1 trials (noise distribution: N(-d/2, 1))
    internal_S1 = rng.normal(-d_prime / 2.0, 1.0, n_s1)
    # Simulate S2 trials (signal distribution: N(d/2, 1))
    internal_S2 = rng.normal(d_prime / 2.0, 1.0, n_s2)

    # Bin responses using criteria
    nR_S1 = np.histogram(internal_S1, bins=np.concatenate([[-np.inf], criteria, [np.inf]]))[0]
    nR_S2 = np.histogram(internal_S2, bins=np.concatenate([[-np.inf], criteria, [np.inf]]))[0]

    return nR_S1.astype(float), nR_S2.astype(float)


# ---------------------------------------------------------------------------
# Type 1 SDT tests
# ---------------------------------------------------------------------------

def test_sdt_type1_perfect_observer():
    """Test d' for a near-perfect observer."""
    # 99 hits, 1 miss, 1 FA, 99 CR
    d, c = compute_sdt_type1(hits=99, misses=1, false_alarms=1, correct_rejections=99)

    assert d > 3.0, f"Near-perfect observer should have high d': {d:.3f}"
    assert abs(c) < 1.0, f"Criterion should be near zero: {c:.3f}"


def test_sdt_type1_chance():
    """Test d' for a chance observer."""
    d, c = compute_sdt_type1(hits=50, misses=50, false_alarms=50, correct_rejections=50)

    assert abs(d) < 0.5, f"Chance observer should have d' near 0: {d:.3f}"


def test_sdt_type1_liberal_criterion():
    """Test that a liberal (low) criterion produces negative c."""
    # High hit rate, high FA rate = liberal
    d, c = compute_sdt_type1(hits=90, misses=10, false_alarms=60, correct_rejections=40)

    assert c < 0, f"Liberal criterion should be negative: {c:.3f}"


def test_sdt_type1_conservative_criterion():
    """Test that a conservative (high) criterion produces positive c."""
    # Low hit rate, low FA rate = conservative
    d, c = compute_sdt_type1(hits=40, misses=60, false_alarms=10, correct_rejections=90)

    assert c > 0, f"Conservative criterion should be positive: {c:.3f}"


def test_compute_type1_from_counts():
    """Test d' computation from confidence rating counts."""
    nR_S1, nR_S2 = generate_sdt_data(d_prime=1.5, n_s1=500, n_s2=500)
    d, c = compute_type1_from_counts(nR_S1, nR_S2)

    assert abs(d - 1.5) < 0.5, f"Recovered d' should be near 1.5: {d:.3f}"


# ---------------------------------------------------------------------------
# Meta-d' estimation tests
# ---------------------------------------------------------------------------

def test_meta_d_mle_perfect_metacognition():
    """Test meta-d' for data with perfect metacognition (meta-d' ~ d')."""
    nR_S1, nR_S2 = generate_sdt_data(
        d_prime=1.5, meta_d=1.5, n_s1=1000, n_s2=1000, seed=42
    )

    result = fit_meta_d_mle(nR_S1, nR_S2)

    assert abs(result.d_prime - 1.5) < 0.5, (
        f"d' should be near 1.5: {result.d_prime:.3f}"
    )
    # m-ratio should be near 1.0 for optimal metacognition
    assert result.m_ratio > 0.3, (
        f"m-ratio should be positive for good metacognition: {result.m_ratio:.3f}"
    )


def test_meta_d_mle_returns_valid():
    """Test that meta-d' MLE returns valid results."""
    nR_S1, nR_S2 = generate_sdt_data(d_prime=1.0, n_s1=200, n_s2=200)

    result = fit_meta_d_mle(nR_S1, nR_S2)

    assert np.isfinite(result.d_prime), f"d' not finite: {result.d_prime}"
    assert np.isfinite(result.meta_d), f"meta-d' not finite: {result.meta_d}"
    assert np.isfinite(result.m_ratio), f"m-ratio not finite: {result.m_ratio}"
    assert np.isfinite(result.c), f"c not finite: {result.c}"


# ---------------------------------------------------------------------------
# Precision bridge tests
# ---------------------------------------------------------------------------

def test_m_ratio_to_gamma_perfect():
    """Test that m_ratio=1 maps to base_gamma."""
    gamma = m_ratio_to_gamma(m_ratio=1.0, base_gamma=4.0)
    assert abs(gamma - 4.0) < 1e-6, f"m_ratio=1 should give base_gamma: {gamma}"


def test_m_ratio_to_gamma_monotonic():
    """Test that higher m-ratio gives higher gamma."""
    gamma_low = m_ratio_to_gamma(m_ratio=0.5)
    gamma_mid = m_ratio_to_gamma(m_ratio=1.0)
    gamma_high = m_ratio_to_gamma(m_ratio=1.5)

    assert gamma_low < gamma_mid < gamma_high, (
        f"Gamma should be monotonic: {gamma_low:.3f} < {gamma_mid:.3f} < {gamma_high:.3f}"
    )


def test_m_ratio_to_gamma_bounded():
    """Test that gamma is clamped within bounds."""
    gamma_extreme_low = m_ratio_to_gamma(m_ratio=0.001)
    gamma_extreme_high = m_ratio_to_gamma(m_ratio=100.0)

    assert gamma_extreme_low == 0.1, f"Should be clamped to min: {gamma_extreme_low}"
    assert gamma_extreme_high == 16.0, f"Should be clamped to max: {gamma_extreme_high}"


def test_update_gamma_from_confidence_overconfident():
    """Test that overconfidence reduces gamma."""
    gamma = 4.0
    # Agent is 90% confident but only 50% accurate -> overconfident
    new_gamma = update_gamma_from_confidence(
        gamma, predicted_confidence=0.9, actual_accuracy=0.5
    )
    assert new_gamma < gamma, (
        f"Overconfidence should reduce gamma: {new_gamma:.3f} vs {gamma:.3f}"
    )


def test_update_gamma_from_confidence_underconfident():
    """Test that underconfidence increases gamma."""
    gamma = 4.0
    # Agent is 50% confident but 90% accurate -> underconfident
    new_gamma = update_gamma_from_confidence(
        gamma, predicted_confidence=0.5, actual_accuracy=0.9
    )
    assert new_gamma > gamma, (
        f"Underconfidence should increase gamma: {new_gamma:.3f} vs {gamma:.3f}"
    )


def test_update_gamma_from_confidence_calibrated():
    """Test that calibrated confidence leaves gamma approximately unchanged."""
    gamma = 4.0
    new_gamma = update_gamma_from_confidence(
        gamma, predicted_confidence=0.7, actual_accuracy=0.7
    )
    assert abs(new_gamma - gamma) < 0.01, (
        f"Calibrated agent should keep gamma: {new_gamma:.3f} vs {gamma:.3f}"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
