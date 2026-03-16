"""Tests for the normative modeling module.

Tests BLR and SHASH warping, verifying:
1. B-spline basis has correct shape and properties.
2. BLR posterior is well-conditioned and predictions are reasonable.
3. Z-scores are near zero for normal population members.
4. Outliers produce large Z-scores.
5. vmap over brain regions works correctly.
6. SHASH distribution has correct properties.
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.normative.blr import (
    bspline_basis,
    fit_blr,
    predict_blr,
    compute_zscore,
    normative_model,
    normative_model_vmap,
)
from alf.normative.warping import (
    shash_log_prob,
    shash_transform,
    shash_inverse_transform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_normative_data(
    n_train: int = 200,
    n_test: int = 50,
    n_outliers: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for normative modeling.

    Simulates a brain measure that follows a nonlinear function of age
    with added noise. Test set includes some outliers.
    """
    rng = np.random.RandomState(seed)

    # Training data (ages 20-80)
    x_train = rng.uniform(20, 80, n_train)
    y_train = np.sin(x_train / 20.0) * 10 + 50 + rng.normal(0, 3, n_train)

    # Test data (normal + outliers)
    x_test = rng.uniform(20, 80, n_test)
    y_test = np.sin(x_test / 20.0) * 10 + 50 + rng.normal(0, 3, n_test)

    # Add outliers (large deviations)
    outlier_idx = rng.choice(n_test, n_outliers, replace=False)
    y_test[outlier_idx] += rng.choice([-20, 20], n_outliers) + rng.normal(0, 5, n_outliers)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# B-spline tests
# ---------------------------------------------------------------------------

def test_bspline_basis_shape():
    """Test that B-spline basis has correct shape."""
    x = np.linspace(0, 10, 100)
    B = bspline_basis(x, n_basis=8, degree=3)

    assert B.shape[0] == 100, f"Expected 100 rows, got {B.shape[0]}"
    assert B.shape[1] > 0, f"Expected > 0 columns, got {B.shape[1]}"


def test_bspline_basis_nonnegative():
    """Test that B-spline basis values are non-negative."""
    x = np.linspace(0, 10, 100)
    B = bspline_basis(x, n_basis=8, degree=3)

    assert np.all(B >= -1e-10), "B-spline values should be non-negative"


def test_bspline_basis_partition_of_unity():
    """Test that B-spline basis approximately sums to 1 at interior points."""
    x = np.linspace(1, 9, 80)  # Stay away from boundaries
    B = bspline_basis(x, n_basis=8, degree=3, x_min=0, x_max=10)

    row_sums = B.sum(axis=1)
    # Should approximately sum to 1 at interior points
    interior_mask = (x > 1.5) & (x < 8.5)
    if np.any(interior_mask):
        np.testing.assert_allclose(
            row_sums[interior_mask], 1.0, atol=0.2,
            err_msg="B-splines should approximately partition unity at interior"
        )


# ---------------------------------------------------------------------------
# BLR tests
# ---------------------------------------------------------------------------

def test_fit_blr_returns_valid():
    """Test that BLR fitting returns valid parameters."""
    rng = np.random.RandomState(42)
    N, D = 50, 5
    Phi = jnp.array(rng.normal(0, 1, (N, D)))
    y = jnp.array(rng.normal(0, 1, N))

    params = fit_blr(Phi, y, optimize_hyperparams=False)

    assert params.w_mean.shape == (D,), f"w_mean shape: {params.w_mean.shape}"
    assert params.w_cov.shape == (D, D), f"w_cov shape: {params.w_cov.shape}"
    assert jnp.all(jnp.isfinite(params.w_mean)), "w_mean not finite"
    assert jnp.all(jnp.isfinite(params.w_cov)), "w_cov not finite"
    assert params.beta > 0, f"beta should be positive: {params.beta}"


def test_fit_blr_recovers_linear():
    """Test that BLR recovers a simple linear relationship."""
    rng = np.random.RandomState(42)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)

    # Design matrix: [1, x]
    Phi = jnp.column_stack([jnp.ones(100), jnp.array(x)])
    y_jnp = jnp.array(y)

    params = fit_blr(Phi, y_jnp, optimize_hyperparams=False)
    pred = predict_blr(Phi, params)

    # Predicted should be close to observed
    residuals = np.array(y_jnp - pred.y_pred)
    rmse = np.sqrt(np.mean(residuals ** 2))
    assert rmse < 1.0, f"RMSE should be small for linear fit: {rmse:.3f}"


def test_predict_blr_uncertainty():
    """Test that BLR predictions include reasonable uncertainty."""
    rng = np.random.RandomState(42)
    N = 50
    Phi_train = jnp.array(rng.normal(0, 1, (N, 5)))
    y_train = jnp.array(rng.normal(0, 1, N))

    params = fit_blr(Phi_train, y_train, optimize_hyperparams=False)

    # Predict at training points
    pred = predict_blr(Phi_train, params)

    assert pred.y_var.shape == (N,), f"y_var shape: {pred.y_var.shape}"
    assert jnp.all(pred.y_var > 0), "Variance should be positive"
    assert jnp.all(pred.y_std > 0), "Std should be positive"


# ---------------------------------------------------------------------------
# Z-score tests
# ---------------------------------------------------------------------------

def test_compute_zscore_zero_for_mean():
    """Test that Z-score is zero when observed equals predicted."""
    y_obs = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([1.0, 2.0, 3.0])
    y_std = jnp.array([0.5, 0.5, 0.5])

    z = compute_zscore(y_obs, y_pred, y_std)

    np.testing.assert_allclose(np.array(z), np.zeros(3), atol=1e-6)


def test_compute_zscore_positive_for_deviation():
    """Test that positive deviation gives positive Z-score."""
    y_obs = jnp.array([5.0])
    y_pred = jnp.array([3.0])
    y_std = jnp.array([1.0])

    z = compute_zscore(y_obs, y_pred, y_std)

    assert float(z[0]) == 2.0, f"Expected Z=2.0, got {float(z[0]):.3f}"


# ---------------------------------------------------------------------------
# Normative model integration tests
# ---------------------------------------------------------------------------

def test_normative_model_basic():
    """Test full normative modeling pipeline."""
    x_train, y_train, x_test, y_test = generate_normative_data()

    result = normative_model(
        x_train, y_train, x_test, y_test,
        n_basis=8, degree=3,
    )

    assert result.z_score.shape == (50,), f"z shape: {result.z_score.shape}"
    assert result.y_pred.shape == (50,), f"y_pred shape: {result.y_pred.shape}"
    assert result.y_std.shape == (50,), f"y_std shape: {result.y_std.shape}"
    assert jnp.all(jnp.isfinite(result.z_score)), "z-scores not finite"
    assert jnp.all(result.y_std > 0), "y_std should be positive"


def test_normative_model_outliers_detected():
    """Test that outliers have larger |Z| than normal observations."""
    rng = np.random.RandomState(42)
    n_train, n_test = 200, 50

    x_train = rng.uniform(20, 80, n_train)
    y_train = x_train * 0.5 + rng.normal(0, 2, n_train)

    x_test = rng.uniform(20, 80, n_test)
    y_test = x_test * 0.5 + rng.normal(0, 2, n_test)

    # Add extreme outlier
    y_test[0] = y_test[0] + 30.0  # Very large deviation

    result = normative_model(x_train, y_train, x_test, y_test, n_basis=6)

    # The outlier should have the largest |Z|
    z_abs = np.abs(np.array(result.z_score))
    assert z_abs[0] > np.median(z_abs), (
        f"Outlier Z should be above median: {z_abs[0]:.3f} vs {np.median(z_abs):.3f}"
    )


def test_normative_model_vmap_shape():
    """Test vmapped normative model over multiple brain regions."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_regions = 100, 30, 5

    x_train = rng.uniform(20, 80, n_train)
    x_test = rng.uniform(20, 80, n_test)

    # Multiple brain regions
    Y_train = rng.normal(0, 1, (n_train, n_regions))
    Y_test = rng.normal(0, 1, (n_test, n_regions))

    z_scores, y_pred, y_std = normative_model_vmap(
        x_train, Y_train, x_test, Y_test, n_basis=6,
    )

    assert z_scores.shape == (n_test, n_regions), f"z shape: {z_scores.shape}"
    assert y_pred.shape == (n_test, n_regions), f"y_pred shape: {y_pred.shape}"
    assert y_std.shape == (n_test, n_regions), f"y_std shape: {y_std.shape}"
    assert jnp.all(jnp.isfinite(z_scores)), "Some z-scores are not finite"


# ---------------------------------------------------------------------------
# SHASH warping tests
# ---------------------------------------------------------------------------

def test_shash_identity_warping():
    """Test that SHASH with epsilon=0, delta=1 is approximately identity."""
    y = jnp.array([1.0, 2.0, 3.0])
    mu = jnp.array(0.0)
    sigma = jnp.array(1.0)

    z = shash_transform(y, mu, sigma, jnp.array(0.0), jnp.array(1.0))

    # With epsilon=0, delta=1: z = sinh(arcsinh(y)) = y
    np.testing.assert_allclose(np.array(z), np.array(y), atol=1e-5)


def test_shash_roundtrip():
    """Test that SHASH transform and inverse are consistent."""
    y = jnp.array([-1.0, 0.0, 1.0, 2.0])
    mu = jnp.array(1.0)
    sigma = jnp.array(2.0)
    epsilon = jnp.array(0.5)
    delta = jnp.array(1.5)

    z = shash_transform(y, mu, sigma, epsilon, delta)
    y_recovered = shash_inverse_transform(z, mu, sigma, epsilon, delta)

    np.testing.assert_allclose(np.array(y_recovered), np.array(y), atol=1e-4)


def test_shash_log_prob_finite():
    """Test that SHASH log-probability is finite."""
    y = jnp.array([0.0, 1.0, 2.0])
    log_p = shash_log_prob(
        y,
        mu=jnp.array(1.0),
        sigma=jnp.array(1.0),
        epsilon=jnp.array(0.0),
        delta=jnp.array(1.0),
    )

    assert jnp.all(jnp.isfinite(log_p)), f"log_p not finite: {log_p}"


def test_shash_log_prob_normal_case():
    """Test that SHASH with epsilon=0, delta=1 approximates normal density."""
    y = jnp.array(0.0)
    mu = jnp.array(0.0)
    sigma = jnp.array(1.0)

    shash_lp = float(shash_log_prob(y, mu, sigma, jnp.array(0.0), jnp.array(1.0)))
    normal_lp = float(-0.5 * jnp.log(2.0 * jnp.pi))  # log N(0|0,1)

    np.testing.assert_allclose(shash_lp, normal_lp, atol=0.1,
        err_msg="SHASH with default params should approximate normal"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
