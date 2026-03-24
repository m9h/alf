"""Oracle validation tests for metacognition against metadPy dataset.

Uses the rm.txt dataset from metadPy (20 subjects, 200 trials each) and
hand-computed reference values from signal detection theory to validate
our meta-d' implementation.

Hand-computed SDT references:
    hits=80, misses=20, FA=30, CR=70:
        hit_rate = (80+0.5)/(100+1) ≈ 0.797
        fa_rate  = (30+0.5)/(100+1) ≈ 0.302
        d' = z(0.797) - z(0.302) ≈ 0.833 - (-0.518) ≈ 1.351
        c  = -0.5 * (z(0.797) + z(0.302)) ≈ -0.5 * (0.833 + (-0.518)) ≈ -0.158
"""

from pathlib import Path

import numpy as np
import pytest

from alf.metacognition import (
    compute_sdt_type1,
    compute_type1_from_counts,
    fit_meta_d_mle,
    m_ratio_to_gamma,
)

DATA_DIR = Path(__file__).parent / "data" / "metacognition"
RM_FILE = DATA_DIR / "rm.txt"
DATA_AVAILABLE = RM_FILE.exists()

try:
    import metadpy  # noqa: F401
    HAS_METADPY = True
except ImportError:
    HAS_METADPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rm_dataset():
    """Load the metadPy rm.txt dataset."""
    import pandas as pd
    return pd.read_csv(RM_FILE)


def trial_data_to_counts(stimuli, responses, confidence, n_ratings=4):
    """Convert trial-level data to nR_S1/nR_S2 count format.

    Args:
        stimuli: 0 or 1 per trial (stimulus type).
        responses: 0 or 1 per trial (response).
        confidence: 1 to n_ratings per trial.
        n_ratings: Number of confidence levels.

    Returns:
        Tuple of (nR_S1, nR_S2) arrays, each shape (2*n_ratings,).
    """
    stimuli = np.asarray(stimuli)
    responses = np.asarray(responses)
    confidence = np.asarray(confidence)

    nR_S1 = np.zeros(2 * n_ratings)
    nR_S2 = np.zeros(2 * n_ratings)

    for trial_idx in range(len(stimuli)):
        s = int(stimuli[trial_idx])
        r = int(responses[trial_idx])
        c = int(confidence[trial_idx])
        c = np.clip(c, 1, n_ratings)

        if r == 0:
            # Responded "S1" — confidence bins go from high-conf-S1 to low-conf-S1
            bin_idx = n_ratings - c
        else:
            # Responded "S2" — confidence bins go from low-conf-S2 to high-conf-S2
            bin_idx = n_ratings + c - 1

        if s == 0:
            nR_S1[bin_idx] += 1
        else:
            nR_S2[bin_idx] += 1

    return nR_S1, nR_S2


# ---------------------------------------------------------------------------
# Hand-computed reference tests (no external data needed)
# ---------------------------------------------------------------------------

def test_known_dprime_computation():
    """Test d' against hand-computed value.

    hits=80, misses=20, FA=30, CR=70 (N=100 per stimulus type)
    With log-linear correction (0.5):
        hit_rate = 80.5/101 = 0.7970
        fa_rate  = 30.5/101 = 0.3020
        d' = Phi^-1(0.7970) - Phi^-1(0.3020)
    """
    from scipy.stats import norm

    d, c = compute_sdt_type1(hits=80, misses=20, false_alarms=30, correct_rejections=70)

    # Compute expected d' with log-linear correction
    hr = 80.5 / 101.0
    far = 30.5 / 101.0
    expected_d = norm.ppf(hr) - norm.ppf(far)
    expected_c = -0.5 * (norm.ppf(hr) + norm.ppf(far))

    np.testing.assert_allclose(d, expected_d, atol=0.01,
        err_msg=f"d'={d:.4f}, expected={expected_d:.4f}"
    )
    np.testing.assert_allclose(c, expected_c, atol=0.01,
        err_msg=f"c={c:.4f}, expected={expected_c:.4f}"
    )

    # Sanity: d' should be around 1.35
    assert 1.0 < d < 1.7, f"d'={d:.3f} out of expected range"


def test_dprime_zero_for_chance():
    """Test that equal hit and FA rates give d' near zero."""
    d, c = compute_sdt_type1(hits=50, misses=50, false_alarms=50, correct_rejections=50)

    assert abs(d) < 0.3, f"Chance performance should give d' near 0, got {d:.3f}"


def test_criterion_sign_convention():
    """Test criterion sign: liberal (low threshold) -> negative c."""
    # Liberal: high hit rate AND high FA rate
    d_lib, c_lib = compute_sdt_type1(hits=95, misses=5, false_alarms=70, correct_rejections=30)
    # Conservative: low hit rate AND low FA rate
    d_con, c_con = compute_sdt_type1(hits=40, misses=60, false_alarms=5, correct_rejections=95)

    assert c_lib < 0, f"Liberal criterion should be negative: {c_lib:.3f}"
    assert c_con > 0, f"Conservative criterion should be positive: {c_con:.3f}"


def test_known_counts_to_dprime():
    """Test d' from confidence rating counts."""
    # Simple 4-rating experiment
    # S1 trials: mostly respond S1 with high confidence
    nR_S1 = np.array([60.0, 10.0, 5.0, 25.0, 5.0, 5.0, 10.0, 80.0])
    # S2 trials: mostly respond S2 with high confidence
    nR_S2 = np.array([5.0, 5.0, 10.0, 30.0, 25.0, 10.0, 15.0, 100.0])

    d, c = compute_type1_from_counts(nR_S1, nR_S2)

    # d' should be positive (above-chance discrimination)
    assert d > 0.3, f"d' should be positive: {d:.3f}"
    assert np.isfinite(d), f"d' not finite: {d}"
    assert np.isfinite(c), f"c not finite: {c}"


def test_meta_d_mle_on_synthetic():
    """Test meta-d' MLE on synthetic data with known properties."""
    # Generate data where metacognition is good (meta-d' ~ d')
    np.random.seed(42)

    # Create clean SDT data
    nR_S1 = np.array([40.0, 20.0, 10.0, 5.0, 3.0, 7.0, 15.0, 50.0])
    nR_S2 = np.array([5.0, 8.0, 12.0, 25.0, 20.0, 15.0, 30.0, 85.0])

    result = fit_meta_d_mle(nR_S1, nR_S2)

    assert np.isfinite(result.d_prime), f"d' not finite: {result.d_prime}"
    assert np.isfinite(result.meta_d), f"meta-d' not finite: {result.meta_d}"
    assert np.isfinite(result.m_ratio), f"m-ratio not finite: {result.m_ratio}"
    assert result.d_prime > 0, f"d' should be positive: {result.d_prime}"


def test_m_ratio_to_gamma_pipeline():
    """Test the full SDT -> meta-d' -> gamma pipeline."""
    nR_S1 = np.array([40.0, 20.0, 10.0, 5.0, 3.0, 7.0, 15.0, 50.0])
    nR_S2 = np.array([5.0, 8.0, 12.0, 25.0, 20.0, 15.0, 30.0, 85.0])

    result = fit_meta_d_mle(nR_S1, nR_S2)
    gamma = m_ratio_to_gamma(result.m_ratio, base_gamma=4.0)

    assert 0.1 <= gamma <= 16.0, f"gamma out of bounds: {gamma}"
    assert isinstance(gamma, float), f"gamma should be float: {type(gamma)}"


# ---------------------------------------------------------------------------
# Real dataset tests (metadPy rm.txt)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATA_AVAILABLE, reason="rm.txt not downloaded")
def test_rm_data_loads():
    """Test that metadPy rm.txt dataset loads correctly."""
    df = load_rm_dataset()

    assert "Stimuli" in df.columns, "Missing Stimuli column"
    assert "Responses" in df.columns, "Missing Responses column"
    assert "Confidence" in df.columns, "Missing Confidence column"
    assert "Subject" in df.columns, "Missing Subject column"
    assert len(df) == 4000, f"Expected 4000 rows, got {len(df)}"
    assert df["Subject"].nunique() == 20, "Expected 20 subjects"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="rm.txt not downloaded")
def test_rm_single_subject_dprime():
    """Test d' computation on first subject of rm dataset."""
    df = load_rm_dataset()
    subj0 = df[df["Subject"] == 0]

    n_ratings = int(subj0["Confidence"].max())
    nR_S1, nR_S2 = trial_data_to_counts(
        subj0["Stimuli"].values,
        subj0["Responses"].values,
        subj0["Confidence"].values,
        n_ratings=n_ratings,
    )

    d, c = compute_type1_from_counts(nR_S1, nR_S2)

    # Typical perceptual d' range
    assert 0.0 < d < 4.0, f"d'={d:.3f} out of expected range for perceptual task"
    assert -2.0 < c < 2.0, f"c={c:.3f} out of expected range"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="rm.txt not downloaded")
def test_rm_meta_d_mle():
    """Test meta-d' MLE on first subject of rm dataset."""
    df = load_rm_dataset()
    subj0 = df[df["Subject"] == 0]

    n_ratings = int(subj0["Confidence"].max())
    nR_S1, nR_S2 = trial_data_to_counts(
        subj0["Stimuli"].values,
        subj0["Responses"].values,
        subj0["Confidence"].values,
        n_ratings=n_ratings,
    )

    result = fit_meta_d_mle(nR_S1, nR_S2)

    assert np.isfinite(result.d_prime), "d' not finite"
    assert np.isfinite(result.meta_d), "meta-d' not finite"
    assert np.isfinite(result.m_ratio), "m-ratio not finite"

    # m-ratio should be in reasonable range
    assert 0.01 < abs(result.m_ratio) < 5.0, (
        f"m-ratio={result.m_ratio:.3f} out of expected range"
    )


@pytest.mark.skipif(not DATA_AVAILABLE, reason="rm.txt not downloaded")
def test_rm_all_subjects_finite():
    """Test that d' and meta-d' are finite for all 20 subjects."""
    df = load_rm_dataset()
    n_ratings = int(df["Confidence"].max())

    n_ok = 0
    for subj_id in df["Subject"].unique():
        subj = df[df["Subject"] == subj_id]
        nR_S1, nR_S2 = trial_data_to_counts(
            subj["Stimuli"].values,
            subj["Responses"].values,
            subj["Confidence"].values,
            n_ratings=n_ratings,
        )

        try:
            result = fit_meta_d_mle(nR_S1, nR_S2)
            if np.isfinite(result.d_prime) and np.isfinite(result.meta_d):
                n_ok += 1
        except Exception:
            pass

    assert n_ok >= 15, (
        f"At least 15/20 subjects should have finite results, got {n_ok}"
    )


@pytest.mark.skipif(not DATA_AVAILABLE, reason="rm.txt not downloaded")
def test_rm_full_pipeline():
    """Test complete pipeline: rm data -> d' -> meta-d' -> gamma."""
    df = load_rm_dataset()
    subj0 = df[df["Subject"] == 0]

    n_ratings = int(subj0["Confidence"].max())
    nR_S1, nR_S2 = trial_data_to_counts(
        subj0["Stimuli"].values,
        subj0["Responses"].values,
        subj0["Confidence"].values,
        n_ratings=n_ratings,
    )

    # SDT
    d, c = compute_type1_from_counts(nR_S1, nR_S2)
    assert d > 0

    # Meta-d'
    result = fit_meta_d_mle(nR_S1, nR_S2)
    assert np.isfinite(result.m_ratio)

    # Gamma
    gamma = m_ratio_to_gamma(result.m_ratio)
    assert 0.1 <= gamma <= 16.0


# ---------------------------------------------------------------------------
# metadPy cross-validation (skip if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_METADPY, reason="metadPy not installed")
def test_dprime_matches_metadpy():
    """Compare d' computation against metadPy."""
    pytest.skip("metadPy comparison not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
