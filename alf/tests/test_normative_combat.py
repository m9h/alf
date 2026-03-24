"""Tests for ComBat harmonization and Z-score -> VFE bridge.

Verifies that:
1. ComBat removes site effects while preserving biological signal.
2. The vmapped ComBat implementation produces correct results.
3. Z-score to VFE conversion matches analytical formulas.
4. Individual free energy computation is correct.
5. Deviation profiling detects outlier regions.
6. The full normative-AIF pipeline runs end-to-end.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from alf.normative.combat import (
    fit_combat,
    apply_combat,
    combat_harmonize,
    fit_combat_vmap,
    apply_combat_vmap,
)
from alf.normative.bridge import (
    NormativeAIFResult,
    zscore_to_vfe,
    individual_free_energy,
    deviation_profile_to_beliefs,
    deviation_mask,
    normative_aif_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_multisite_data(
    n_subjects_per_site: int = 100,
    n_sites: int = 3,
    n_features: int = 10,
    site_effect_strength: float = 5.0,
    biological_signal_strength: float = 2.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multi-site neuroimaging data.

    Creates data with:
    - A biological signal (shared across sites).
    - Site-specific offsets (additive batch effects).
    - Random noise.

    Args:
        n_subjects_per_site: Subjects per site.
        n_sites: Number of sites.
        n_features: Number of brain features.
        site_effect_strength: Magnitude of site effects.
        biological_signal_strength: Magnitude of biological signal.
        noise_std: Standard deviation of noise.
        seed: Random seed.

    Returns:
        Tuple of (Y, site_labels, biological_signal):
            Y: Data matrix, shape (n_total, n_features).
            site_labels: Site IDs, shape (n_total,).
            biological_signal: True signal, shape (n_total, n_features).
    """
    rng = np.random.RandomState(seed)
    n_total = n_subjects_per_site * n_sites

    # Biological signal (shared across sites)
    biological_signal = biological_signal_strength * rng.normal(
        0, 1, (n_total, n_features)
    )

    # Site labels
    site_labels = np.repeat(np.arange(n_sites), n_subjects_per_site)

    # Site effects (additive offset per site per feature)
    site_offsets = site_effect_strength * rng.normal(0, 1, (n_sites, n_features))

    # Construct data
    Y = biological_signal.copy()
    for s in range(n_sites):
        mask = site_labels == s
        Y[mask] += site_offsets[s]

    # Add noise
    Y += noise_std * rng.normal(0, 1, (n_total, n_features))

    return Y, site_labels, biological_signal


def generate_pipeline_data(
    n_train: int = 150,
    n_test: int = 50,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Generate data for the full pipeline test.

    Returns:
        Tuple of (x_train, Y_train, x_test, Y_test, site_train, site_test).
    """
    rng = np.random.RandomState(seed)

    # Ages
    x_train = rng.uniform(20, 80, n_train)
    x_test = rng.uniform(20, 80, n_test)

    # Brain measures with age-related trend
    Y_train = np.column_stack([
        np.sin(x_train / 20.0) * 10 + 50 + rng.normal(0, 3, n_train)
        for _ in range(n_features)
    ])
    Y_test = np.column_stack([
        np.sin(x_test / 20.0) * 10 + 50 + rng.normal(0, 3, n_test)
        for _ in range(n_features)
    ])

    # Site labels (2 sites, 3:1 split for train, alternating for test)
    site_train = np.zeros(n_train, dtype=int)
    site_train[n_train // 4 * 3:] = 1

    # Add site effect to site 1
    site_1_mask_train = site_train == 1
    Y_train[site_1_mask_train] += 5.0

    site_test = rng.randint(0, 2, n_test)
    site_1_mask_test = site_test == 1
    Y_test[site_1_mask_test] += 5.0

    return x_train, Y_train, x_test, Y_test, site_train, site_test


# ---------------------------------------------------------------------------
# ComBat tests
# ---------------------------------------------------------------------------

def test_combat_removes_site_effect():
    """Harmonized data should have negligible site mean differences."""
    Y, site_labels, _ = generate_multisite_data(
        n_subjects_per_site=100,
        n_sites=3,
        n_features=10,
        site_effect_strength=5.0,
        seed=42,
    )

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)

    params = fit_combat(Y_jnp, site_jnp)
    Y_harm = apply_combat(Y_jnp, site_jnp, params)

    # Check that site means are much closer after harmonization
    Y_harm_np = np.array(Y_harm)

    site_means_before = []
    site_means_after = []
    for s in range(3):
        mask = site_labels == s
        site_means_before.append(np.mean(Y[mask], axis=0))
        site_means_after.append(np.mean(Y_harm_np[mask], axis=0))

    # Max site mean difference before vs after
    max_diff_before = np.max(np.abs(
        np.array(site_means_before) - np.mean(site_means_before, axis=0)
    ))
    max_diff_after = np.max(np.abs(
        np.array(site_means_after) - np.mean(site_means_after, axis=0)
    ))

    assert max_diff_after < max_diff_before, (
        f"ComBat should reduce site differences: "
        f"before={max_diff_before:.3f}, after={max_diff_after:.3f}"
    )
    # Harmonized site differences should be small
    assert max_diff_after < 1.0, (
        f"Harmonized site mean diff should be < 1.0, got {max_diff_after:.3f}"
    )


def test_combat_preserves_biological():
    """Biological signal should be preserved after harmonization."""
    Y, site_labels, bio_signal = generate_multisite_data(
        n_subjects_per_site=100,
        n_sites=3,
        n_features=10,
        site_effect_strength=5.0,
        biological_signal_strength=3.0,
        seed=42,
    )

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)

    params = fit_combat(Y_jnp, site_jnp)
    Y_harm = apply_combat(Y_jnp, site_jnp, params)
    Y_harm_np = np.array(Y_harm)

    # Correlation between harmonized data and biological signal should be high
    # (Compare centered versions to remove mean effects)
    Y_harm_centered = Y_harm_np - Y_harm_np.mean(axis=0)
    bio_centered = bio_signal - bio_signal.mean(axis=0)

    for f in range(10):
        corr = np.corrcoef(Y_harm_centered[:, f], bio_centered[:, f])[0, 1]
        assert corr > 0.3, (
            f"Feature {f}: biological correlation {corr:.3f} too low after ComBat"
        )


def test_combat_vmap():
    """Vmapped ComBat should produce valid outputs."""
    Y, site_labels, _ = generate_multisite_data(
        n_subjects_per_site=50,
        n_sites=2,
        n_features=8,
        seed=42,
    )

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)

    grand_means, gamma_star, delta_star, pooled_stds = fit_combat_vmap(
        Y_jnp, site_jnp
    )

    n_features = 8
    n_sites = 2

    assert grand_means.shape == (n_features,), (
        f"grand_means shape: {grand_means.shape}"
    )
    assert gamma_star.shape == (n_features, n_sites), (
        f"gamma_star shape: {gamma_star.shape}"
    )
    assert delta_star.shape == (n_features, n_sites), (
        f"delta_star shape: {delta_star.shape}"
    )
    assert pooled_stds.shape == (n_features,), (
        f"pooled_stds shape: {pooled_stds.shape}"
    )
    assert jnp.all(jnp.isfinite(grand_means)), "grand_means not finite"
    assert jnp.all(jnp.isfinite(gamma_star)), "gamma_star not finite"
    assert jnp.all(delta_star > 0), "delta_star should be positive"
    assert jnp.all(pooled_stds > 0), "pooled_stds should be positive"


def test_combat_vmap_apply():
    """Vmapped ComBat fit + apply should reduce site effects."""
    Y, site_labels, _ = generate_multisite_data(
        n_subjects_per_site=80,
        n_sites=2,
        n_features=5,
        site_effect_strength=8.0,
        seed=42,
    )

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)

    grand_means, gamma_star, delta_star, pooled_stds = fit_combat_vmap(
        Y_jnp, site_jnp
    )
    Y_harm = apply_combat_vmap(
        Y_jnp, site_jnp, grand_means, gamma_star, delta_star, pooled_stds
    )

    Y_harm_np = np.array(Y_harm)

    # Site mean difference after harmonization
    mean_0 = np.mean(Y_harm_np[site_labels == 0], axis=0)
    mean_1 = np.mean(Y_harm_np[site_labels == 1], axis=0)
    max_diff = np.max(np.abs(mean_0 - mean_1))

    assert max_diff < 2.0, (
        f"Vmapped ComBat should reduce site diff, got {max_diff:.3f}"
    )


def test_combat_roundtrip():
    """Fit and apply should produce expected shapes."""
    Y, site_labels, _ = generate_multisite_data(
        n_subjects_per_site=60,
        n_sites=3,
        n_features=15,
        seed=42,
    )

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)

    params = fit_combat(Y_jnp, site_jnp)
    Y_harm = apply_combat(Y_jnp, site_jnp, params)

    assert Y_harm.shape == Y_jnp.shape, (
        f"Shape mismatch: {Y_harm.shape} vs {Y_jnp.shape}"
    )
    assert jnp.all(jnp.isfinite(Y_harm)), "Harmonized data has non-finite values"

    # ComBat params should have expected shapes
    assert params.grand_mean.shape == (15,)
    assert params.gamma_star.shape == (3, 15)
    assert params.delta_star.shape == (3, 15)
    assert params.n_sites == 3


def test_combat_with_covariates():
    """ComBat with covariates should preserve covariate effects."""
    rng = np.random.RandomState(42)
    n_per_site = 80
    n_sites = 2
    n_features = 5
    n_total = n_per_site * n_sites

    # Covariates (age, sex)
    age = rng.uniform(20, 80, n_total)
    sex = rng.binomial(1, 0.5, n_total).astype(float)
    covariates = np.column_stack([age, sex])

    # Biological signal depends on covariates
    beta_age = rng.normal(0, 0.1, n_features)
    beta_sex = rng.normal(0, 2, n_features)
    bio = age[:, None] * beta_age[None, :] + sex[:, None] * beta_sex[None, :]

    # Site labels and effects
    site_labels = np.repeat(np.arange(n_sites), n_per_site)
    site_offsets = 5.0 * rng.normal(0, 1, (n_sites, n_features))

    Y = bio.copy()
    for s in range(n_sites):
        mask = site_labels == s
        Y[mask] += site_offsets[s]
    Y += rng.normal(0, 1, (n_total, n_features))

    Y_jnp = jnp.array(Y)
    site_jnp = jnp.array(site_labels)
    cov_jnp = jnp.array(covariates)

    params = fit_combat(Y_jnp, site_jnp, covariates=cov_jnp)
    Y_harm = apply_combat(Y_jnp, site_jnp, params, covariates=cov_jnp)

    assert Y_harm.shape == Y_jnp.shape
    assert params.beta_hat is not None
    assert params.beta_hat.shape == (2, n_features)


def test_combat_harmonize_convenience():
    """Test the convenience function for train/test harmonization."""
    Y, site_labels, _ = generate_multisite_data(
        n_subjects_per_site=80,
        n_sites=2,
        n_features=5,
        seed=42,
    )

    # Split into train/test
    n_total = len(site_labels)
    n_train = int(0.7 * n_total)

    Y_train = jnp.array(Y[:n_train])
    Y_test = jnp.array(Y[n_train:])
    site_train = jnp.array(site_labels[:n_train])
    site_test = jnp.array(site_labels[n_train:])

    Y_train_harm, Y_test_harm = combat_harmonize(
        Y_train, site_train, Y_test, site_test
    )

    assert Y_train_harm.shape == Y_train.shape
    assert Y_test_harm.shape == Y_test.shape
    assert jnp.all(jnp.isfinite(Y_train_harm))
    assert jnp.all(jnp.isfinite(Y_test_harm))


# ---------------------------------------------------------------------------
# Bridge tests: Z-score -> VFE
# ---------------------------------------------------------------------------

def test_zscore_to_vfe_known_values():
    """Known Z-scores should give correct VFE values."""
    # Z = 0 -> VFE = 0.5 * log(2*pi) ~ 0.9189
    vfe_0 = zscore_to_vfe(jnp.array(0.0))
    expected_0 = 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(vfe_0), expected_0, atol=1e-5,
        err_msg="VFE at z=0 should be 0.5*log(2*pi)"
    )

    # Z = 1 -> VFE = 0.5 + 0.5*log(2*pi) ~ 1.4189
    vfe_1 = zscore_to_vfe(jnp.array(1.0))
    expected_1 = 0.5 + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(vfe_1), expected_1, atol=1e-5,
        err_msg="VFE at z=1 should be 0.5 + 0.5*log(2*pi)"
    )

    # Z = -2 -> VFE = 2.0 + 0.5*log(2*pi) ~ 2.9189
    vfe_neg2 = zscore_to_vfe(jnp.array(-2.0))
    expected_neg2 = 2.0 + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(vfe_neg2), expected_neg2, atol=1e-5,
        err_msg="VFE at z=-2 should be 2.0 + 0.5*log(2*pi)"
    )


def test_zscore_to_vfe_symmetry():
    """VFE should be symmetric: VFE(z) == VFE(-z)."""
    z = jnp.array([1.0, 2.0, 3.0])
    vfe_pos = zscore_to_vfe(z)
    vfe_neg = zscore_to_vfe(-z)

    np.testing.assert_allclose(np.array(vfe_pos), np.array(vfe_neg), atol=1e-6,
        err_msg="VFE should be symmetric in z"
    )


def test_zscore_to_vfe_batch():
    """VFE should work on batched input (2D arrays)."""
    z = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
    vfe = zscore_to_vfe(z)

    assert vfe.shape == (2, 2), f"Expected shape (2, 2), got {vfe.shape}"
    assert jnp.all(jnp.isfinite(vfe)), "VFE should be finite"
    assert jnp.all(vfe > 0), "VFE should be positive"


def test_individual_free_energy_matches_manual():
    """Individual free energy should match manual computation."""
    y_obs = jnp.array(5.0)
    y_pred = jnp.array(3.0)
    y_var = jnp.array(4.0)  # sigma^2 = 4

    fe = individual_free_energy(y_obs, y_pred, y_var)

    # Manual: 0.5 * log(2*pi*4) + 0.5 * (5-3)^2 / 4
    #       = 0.5 * log(8*pi) + 0.5 * 4/4
    #       = 0.5 * log(8*pi) + 0.5
    expected = 0.5 * np.log(2 * np.pi * 4.0) + 0.5 * (5.0 - 3.0) ** 2 / 4.0

    np.testing.assert_allclose(float(fe), expected, atol=1e-5,
        err_msg="Individual VFE should match analytical formula"
    )


def test_individual_free_energy_zero_deviation():
    """When y_obs == y_pred, VFE should equal just the complexity term."""
    y = jnp.array(3.0)
    y_var = jnp.array(2.0)

    fe = individual_free_energy(y, y, y_var)

    # Just the complexity: 0.5 * log(2*pi*sigma^2)
    expected = 0.5 * np.log(2 * np.pi * 2.0)

    np.testing.assert_allclose(float(fe), expected, atol=1e-5,
        err_msg="VFE at zero deviation should be the complexity term only"
    )


def test_individual_free_energy_consistency_with_zscore():
    """Individual VFE should equal zscore_to_vfe when sigma = 1."""
    y_obs = jnp.array(2.5)
    y_pred = jnp.array(0.5)
    y_var = jnp.array(1.0)  # unit variance

    fe = individual_free_energy(y_obs, y_pred, y_var)

    z = (y_obs - y_pred) / jnp.sqrt(y_var)
    vfe_from_z = zscore_to_vfe(z)

    np.testing.assert_allclose(float(fe), float(vfe_from_z), atol=1e-5,
        err_msg="Individual VFE with sigma=1 should match zscore_to_vfe"
    )


# ---------------------------------------------------------------------------
# Deviation profile tests
# ---------------------------------------------------------------------------

def test_deviation_profile_basic():
    """Outlier regions should have high P(Deviant)."""
    # 5 regions: first 3 normal, last 2 deviant
    z = jnp.array([0.1, -0.5, 0.3, 3.5, -4.0])

    beliefs = deviation_profile_to_beliefs(z, threshold=2.0)

    assert beliefs.shape == (5, 2), f"Shape: {beliefs.shape}"

    # P(Normal) + P(Deviant) = 1
    np.testing.assert_allclose(
        np.array(jnp.sum(beliefs, axis=-1)),
        np.ones(5),
        atol=1e-6,
    )

    # Deviant regions should have high P(Deviant)
    assert float(beliefs[3, 1]) > 0.8, (
        f"Region 3 (z=3.5) should have high P(Deviant): {float(beliefs[3, 1]):.3f}"
    )
    assert float(beliefs[4, 1]) > 0.8, (
        f"Region 4 (z=-4.0) should have high P(Deviant): {float(beliefs[4, 1]):.3f}"
    )

    # Normal regions should have low P(Deviant)
    assert float(beliefs[0, 1]) < 0.2, (
        f"Region 0 (z=0.1) should have low P(Deviant): {float(beliefs[0, 1]):.3f}"
    )


def test_deviation_profile_batch():
    """Deviation profile should work for batched Z-scores."""
    z = jnp.array([
        [0.0, 3.0],
        [-3.0, 0.5],
    ])

    beliefs = deviation_profile_to_beliefs(z, threshold=2.0)

    assert beliefs.shape == (2, 2, 2), f"Shape: {beliefs.shape}"


def test_deviation_mask_basic():
    """Hard threshold should correctly identify deviants."""
    z = jnp.array([0.5, 1.9, 2.1, -2.5, 0.0])

    mask = deviation_mask(z, threshold=2.0)

    expected = np.array([False, False, True, True, False])
    np.testing.assert_array_equal(np.array(mask), expected)


def test_deviation_mask_symmetry():
    """Mask should be symmetric: positive and negative Z above threshold."""
    z = jnp.array([2.5, -2.5])
    mask = deviation_mask(z, threshold=2.0)

    assert bool(mask[0])
    assert bool(mask[1])


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_full_pipeline():
    """End-to-end pipeline: ComBat -> BLR -> Z-scores -> VFE."""
    x_train, Y_train, x_test, Y_test, site_train, site_test = (
        generate_pipeline_data(n_train=150, n_test=50, n_features=5, seed=42)
    )

    result = normative_aif_pipeline(
        x_train, Y_train, x_test, Y_test,
        site_train=site_train,
        site_test=site_test,
        n_basis=6,
        z_threshold=2.0,
    )

    assert isinstance(result, NormativeAIFResult)

    n_test, n_features = 50, 5

    # Shape checks
    assert result.z_scores.shape == (n_test, n_features), (
        f"z_scores shape: {result.z_scores.shape}"
    )
    assert result.vfe.shape == (n_test, n_features), (
        f"vfe shape: {result.vfe.shape}"
    )
    assert result.vfe_total.shape == (n_test,), (
        f"vfe_total shape: {result.vfe_total.shape}"
    )
    assert result.deviation_mask.shape == (n_test, n_features), (
        f"deviation_mask shape: {result.deviation_mask.shape}"
    )
    assert result.belief_vectors.shape == (n_test, n_features, 2), (
        f"belief_vectors shape: {result.belief_vectors.shape}"
    )
    assert result.y_pred.shape == (n_test, n_features), (
        f"y_pred shape: {result.y_pred.shape}"
    )
    assert result.y_std.shape == (n_test, n_features), (
        f"y_std shape: {result.y_std.shape}"
    )

    # Finite checks
    assert jnp.all(jnp.isfinite(result.z_scores)), "z_scores not finite"
    assert jnp.all(jnp.isfinite(result.vfe)), "vfe not finite"
    assert jnp.all(result.vfe > 0), "VFE should be positive"
    assert jnp.all(result.y_std > 0), "y_std should be positive"


def test_full_pipeline_without_combat():
    """Pipeline should work without site labels (skip ComBat)."""
    x_train, Y_train, x_test, Y_test, _, _ = (
        generate_pipeline_data(n_train=100, n_test=30, n_features=3, seed=123)
    )

    result = normative_aif_pipeline(
        x_train, Y_train, x_test, Y_test,
        site_train=None,
        site_test=None,
        n_basis=6,
    )

    assert result.z_scores.shape == (30, 3)
    assert jnp.all(jnp.isfinite(result.z_scores))
    assert jnp.all(result.vfe > 0)


def test_vfe_increases_with_deviation():
    """VFE should increase monotonically with |z|."""
    z = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    vfe = zscore_to_vfe(z)

    vfe_np = np.array(vfe)
    for i in range(len(vfe_np) - 1):
        assert vfe_np[i + 1] > vfe_np[i], (
            f"VFE should increase: VFE({float(z[i]):.1f})={vfe_np[i]:.4f} >= "
            f"VFE({float(z[i+1]):.1f})={vfe_np[i+1]:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
