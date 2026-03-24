"""ComBat (Combining Batches) harmonization for multi-site neuroimaging data.

Removes site-related batch effects from neuroimaging features while preserving
biological variability. This is essential for multi-site normative modeling,
where scanner differences and acquisition protocols introduce unwanted variance
that can dominate true neurobiological signal.

The parametric ComBat algorithm uses empirical Bayes (EB) to shrink site-effect
estimates toward a common prior, yielding stable corrections even with small
per-site sample sizes. The procedure:

    1. Standardize data: remove covariate effects and grand mean.
    2. Estimate additive (gamma) and multiplicative (delta) site effects.
    3. Apply EB shrinkage to obtain gamma_star and delta_star.
    4. Adjust data: remove estimated site effects, restore covariates.

All computations are JAX-native, with jax.vmap over features (brain regions)
for massive parallelism on GPU/TPU.

References:
    Johnson, Li & Rabinovic (2007). Adjusting batch effects in microarray
        expression data using empirical Bayes methods. Biostatistics.
    Fortin, Parker, Tunc, Watanabe, Elliott, Ruparel, ... & Shinohara (2018).
        Harmonization of multi-site diffusion tensor imaging data. NeuroImage.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Numerical stability constant
# ---------------------------------------------------------------------------

eps = 1e-16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ComBatParams(NamedTuple):
    """Fitted ComBat harmonization parameters.

    Attributes:
        grand_mean: Grand mean across all subjects, shape (n_features,).
        site_means: Raw site-specific additive effects (gamma),
            shape (n_sites, n_features).
        site_vars: Raw site-specific multiplicative effects (delta),
            shape (n_sites, n_features).
        gamma_star: EB-adjusted additive site effects,
            shape (n_sites, n_features).
        delta_star: EB-adjusted multiplicative site effects,
            shape (n_sites, n_features).
        beta_hat: Covariate regression coefficients,
            shape (n_covariates, n_features) or None.
        covariate_design: Design matrix used during fitting (for apply),
            shape (n_subjects, n_covariates) or None.
        n_sites: Number of sites (scalar).
    """
    grand_mean: jnp.ndarray
    site_means: jnp.ndarray
    site_vars: jnp.ndarray
    gamma_star: jnp.ndarray
    delta_star: jnp.ndarray
    beta_hat: Optional[jnp.ndarray]
    covariate_design: Optional[jnp.ndarray]
    n_sites: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_site_design(
    site_labels: jnp.ndarray,
    n_sites: int,
) -> jnp.ndarray:
    """Create a one-hot site indicator matrix.

    Args:
        site_labels: Integer site IDs, shape (n_subjects,).
        n_sites: Total number of sites.

    Returns:
        One-hot matrix, shape (n_subjects, n_sites).
    """
    return jax.nn.one_hot(site_labels, n_sites)


def _estimate_site_effects(
    Y_stand: jnp.ndarray,
    site_labels: jnp.ndarray,
    n_sites: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate per-site additive (gamma) and multiplicative (delta) effects.

    Args:
        Y_stand: Standardized data, shape (n_subjects, n_features).
        site_labels: Integer site IDs, shape (n_subjects,).
        n_sites: Total number of sites.

    Returns:
        Tuple of (gamma, delta):
            gamma: Site means, shape (n_sites, n_features).
            delta: Site variances, shape (n_sites, n_features).
    """
    gamma = jnp.zeros((n_sites, Y_stand.shape[1]))
    delta = jnp.zeros((n_sites, Y_stand.shape[1]))

    for s in range(n_sites):
        mask = site_labels == s
        n_s = jnp.sum(mask)
        # Site mean
        site_data = jnp.where(mask[:, None], Y_stand, 0.0)
        gamma_s = jnp.sum(site_data, axis=0) / jnp.clip(n_s, eps)
        gamma = gamma.at[s].set(gamma_s)

        # Site variance
        resid = jnp.where(mask[:, None], Y_stand - gamma_s[None, :], 0.0)
        delta_s = jnp.sum(resid ** 2, axis=0) / jnp.clip(n_s - 1, eps)
        delta_s = jnp.clip(delta_s, eps)
        delta = delta.at[s].set(delta_s)

    return gamma, delta


def _eb_shrinkage_gamma(
    gamma: jnp.ndarray,
    delta: jnp.ndarray,
    site_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Empirical Bayes shrinkage for additive site effects (gamma).

    Uses the parametric EB approach: assume gamma_g ~ N(gamma_bar, tau^2).
    The posterior mean (gamma_star) is a weighted average of the site-specific
    estimate and the overall mean.

    Args:
        gamma: Site means, shape (n_sites, n_features).
        delta: Site variances, shape (n_sites, n_features).
        site_counts: Number of subjects per site, shape (n_sites,).

    Returns:
        gamma_star: EB-shrunk site means, shape (n_sites, n_features).
    """
    n_sites = gamma.shape[0]

    # Hyperparameters: prior mean and variance of gamma across sites
    gamma_bar = jnp.mean(gamma, axis=0)  # (n_features,)
    tau_sq = jnp.var(gamma, axis=0, ddof=0)  # (n_features,)
    tau_sq = jnp.clip(tau_sq, eps)

    # Posterior mean for each site (shrinkage toward gamma_bar)
    gamma_star = jnp.zeros_like(gamma)
    for s in range(n_sites):
        n_s = site_counts[s]
        # Weight = n_s * tau^2 / (n_s * tau^2 + delta_s)
        numerator = n_s * tau_sq
        denominator = numerator + delta[s]
        weight = numerator / jnp.clip(denominator, eps)

        gamma_star_s = weight * gamma[s] + (1.0 - weight) * gamma_bar
        gamma_star = gamma_star.at[s].set(gamma_star_s)

    return gamma_star


def _eb_shrinkage_delta(
    delta: jnp.ndarray,
    site_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Empirical Bayes shrinkage for multiplicative site effects (delta).

    Uses the parametric EB approach: assume delta_g ~ InvGamma(alpha, beta).
    Method of moments to estimate alpha, beta from the sample of deltas.

    Args:
        delta: Site variances, shape (n_sites, n_features).
        site_counts: Number of subjects per site, shape (n_sites,).

    Returns:
        delta_star: EB-shrunk site variances, shape (n_sites, n_features).
    """
    n_sites = delta.shape[0]

    # Method of moments for Inverse Gamma prior on delta
    delta_bar = jnp.mean(delta, axis=0)  # (n_features,)
    delta_var = jnp.var(delta, axis=0, ddof=0)  # (n_features,)
    delta_var = jnp.clip(delta_var, eps)

    # Inverse Gamma parameters via method of moments:
    #   mean = beta / (alpha - 1), var = beta^2 / ((alpha-1)^2 * (alpha-2))
    #   => alpha = 2 + mean^2 / var
    #   => beta = mean * (alpha - 1)
    alpha = 2.0 + delta_bar ** 2 / delta_var
    beta = delta_bar * (alpha - 1.0)

    # Posterior mode for Inverse Gamma: beta / (alpha + 1)
    # But we use the posterior mean with data:
    #   delta_star = (sum_i (y_i - gamma)^2 / 2 + beta) / (n/2 + alpha + 1)
    delta_star = jnp.zeros_like(delta)
    for s in range(n_sites):
        n_s = site_counts[s]
        numerator = 0.5 * n_s * delta[s] + beta
        denominator = 0.5 * n_s + alpha + 1.0
        delta_star_s = numerator / jnp.clip(denominator, eps)
        delta_star_s = jnp.clip(delta_star_s, eps)
        delta_star = delta_star.at[s].set(delta_star_s)

    return delta_star


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_combat(
    Y: jnp.ndarray,
    site_labels: jnp.ndarray,
    covariates: Optional[jnp.ndarray] = None,
) -> ComBatParams:
    """Fit parametric ComBat harmonization model.

    Estimates site effects (additive and multiplicative) using empirical Bayes,
    following Johnson et al. (2007). The procedure:

        1. Regress out covariate effects (if provided) and grand mean.
        2. Estimate per-site additive (gamma) and multiplicative (delta) effects.
        3. Apply EB shrinkage to yield stable gamma_star and delta_star.

    Args:
        Y: Data matrix, shape (n_subjects, n_features). Each row is a
            subject, each column a brain region / feature.
        site_labels: Integer site IDs, shape (n_subjects,). Must be
            0-indexed contiguous integers (0, 1, ..., n_sites-1).
        covariates: Optional design matrix for biological variables to
            preserve, shape (n_subjects, n_covariates). For example,
            age and sex regressors. If None, no covariate correction.

    Returns:
        ComBatParams with estimated harmonization parameters.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    site_labels = jnp.asarray(site_labels, dtype=jnp.int32)
    n_subjects, n_features = Y.shape
    n_sites = int(jnp.max(site_labels)) + 1

    # Site counts
    site_counts = jnp.array([
        jnp.sum(site_labels == s) for s in range(n_sites)
    ], dtype=jnp.float32)

    # Step 1: Grand mean
    grand_mean = jnp.mean(Y, axis=0)  # (n_features,)

    # Step 2: Regress out covariates (if provided)
    beta_hat = None
    covariate_design = None
    Y_adjusted = Y - grand_mean[None, :]

    if covariates is not None:
        covariates = jnp.asarray(covariates, dtype=jnp.float32)
        covariate_design = covariates
        # OLS: beta_hat = (X^T X)^{-1} X^T Y
        XtX = covariates.T @ covariates
        XtX_inv = jnp.linalg.inv(XtX + eps * jnp.eye(covariates.shape[1]))
        beta_hat = XtX_inv @ covariates.T @ Y_adjusted  # (n_covariates, n_features)
        Y_adjusted = Y_adjusted - covariates @ beta_hat

    # Step 3: Standardize by pooled variance (across sites)
    pooled_var = jnp.var(Y_adjusted, axis=0)
    pooled_std = jnp.sqrt(jnp.clip(pooled_var, eps))
    Y_stand = Y_adjusted / pooled_std[None, :]

    # Step 4: Estimate raw site effects
    gamma, delta = _estimate_site_effects(Y_stand, site_labels, n_sites)

    # Step 5: EB shrinkage
    gamma_star = _eb_shrinkage_gamma(gamma, delta, site_counts)
    delta_star = _eb_shrinkage_delta(delta, site_counts)

    # Store raw site effects scaled back to original units
    site_means = gamma * pooled_std[None, :]
    site_vars = delta * pooled_std[None, :] ** 2

    return ComBatParams(
        grand_mean=grand_mean,
        site_means=site_means,
        site_vars=site_vars,
        gamma_star=gamma_star,
        delta_star=delta_star,
        beta_hat=beta_hat,
        covariate_design=covariate_design,
        n_sites=n_sites,
    )


def apply_combat(
    Y: jnp.ndarray,
    site_labels: jnp.ndarray,
    params: ComBatParams,
    covariates: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Apply fitted ComBat parameters to harmonize data.

    Removes site effects using the EB-estimated gamma_star and delta_star:

        Y_harmonized = pooled_std * (Y_stand - gamma_star[site]) /
                       sqrt(delta_star[site]) + grand_mean + X @ beta

    Args:
        Y: Data matrix, shape (n_subjects, n_features).
        site_labels: Integer site IDs, shape (n_subjects,).
        params: Fitted ComBatParams from fit_combat.
        covariates: Optional covariate matrix, shape (n_subjects, n_covariates).
            If params were fit with covariates, this should be provided.

    Returns:
        Harmonized data matrix, shape (n_subjects, n_features).
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    site_labels = jnp.asarray(site_labels, dtype=jnp.int32)
    n_subjects, n_features = Y.shape

    # Reconstruct standardized data
    Y_adjusted = Y - params.grand_mean[None, :]

    if covariates is not None and params.beta_hat is not None:
        covariates = jnp.asarray(covariates, dtype=jnp.float32)
        Y_adjusted = Y_adjusted - covariates @ params.beta_hat

    # Pooled standard deviation (recompute from grand_mean and site_means)
    # We need it in the standardized space, so derive from stored params
    pooled_var = jnp.var(Y_adjusted, axis=0)
    pooled_std = jnp.sqrt(jnp.clip(pooled_var, eps))
    Y_stand = Y_adjusted / pooled_std[None, :]

    # Remove site effects using EB-estimated parameters
    Y_combat = jnp.zeros_like(Y_stand)
    for s in range(params.n_sites):
        mask = site_labels == s
        site_data = Y_stand  # operate on all, mask at the end

        # Remove additive and multiplicative effects
        corrected = (site_data - params.gamma_star[s][None, :]) / jnp.sqrt(
            jnp.clip(params.delta_star[s][None, :], eps)
        )

        Y_combat = jnp.where(mask[:, None], corrected, Y_combat)

    # Transform back to original scale
    Y_harmonized = Y_combat * pooled_std[None, :] + params.grand_mean[None, :]

    # Add back covariate effects (biological signal to preserve)
    if covariates is not None and params.beta_hat is not None:
        Y_harmonized = Y_harmonized + covariates @ params.beta_hat

    return Y_harmonized


def combat_harmonize(
    Y_train: jnp.ndarray,
    site_train: jnp.ndarray,
    Y_test: jnp.ndarray,
    site_test: jnp.ndarray,
    covariates_train: Optional[jnp.ndarray] = None,
    covariates_test: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convenience function: fit ComBat on training data, apply to both sets.

    This is the typical workflow for normative modeling: fit harmonization
    on a reference/training cohort, then apply the same correction to
    held-out test subjects (potentially from different sites).

    Args:
        Y_train: Training data matrix, shape (n_train, n_features).
        site_train: Training site labels, shape (n_train,).
        Y_test: Test data matrix, shape (n_test, n_features).
        site_test: Test site labels, shape (n_test,).
        covariates_train: Optional training covariates,
            shape (n_train, n_covariates).
        covariates_test: Optional test covariates,
            shape (n_test, n_covariates).

    Returns:
        Tuple of (Y_train_harmonized, Y_test_harmonized), each with the
        same shape as the corresponding input.
    """
    # Fit on training data
    params = fit_combat(Y_train, site_train, covariates=covariates_train)

    # Apply to both
    Y_train_harm = apply_combat(
        Y_train, site_train, params, covariates=covariates_train
    )
    Y_test_harm = apply_combat(
        Y_test, site_test, params, covariates=covariates_test
    )

    return Y_train_harm, Y_test_harm


# ---------------------------------------------------------------------------
# Vectorized ComBat (single-feature, vmappable)
# ---------------------------------------------------------------------------

def _fit_combat_single_feature(
    y: jnp.ndarray,
    site_indicators: jnp.ndarray,
    site_counts: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit ComBat for a single feature (vmappable).

    This is a simplified version designed for jax.vmap over the feature
    dimension. It does not handle covariates (those should be regressed
    out before calling this).

    Args:
        y: Data for one feature, shape (n_subjects,).
        site_indicators: One-hot site matrix, shape (n_subjects, n_sites).
        site_counts: Number of subjects per site, shape (n_sites,).

    Returns:
        Tuple of (grand_mean, gamma_star, delta_star, pooled_std).
    """
    site_indicators.shape[1]

    # Grand mean
    grand_mean = jnp.mean(y)

    # Standardize
    y_centered = y - grand_mean
    pooled_std = jnp.sqrt(jnp.clip(jnp.var(y_centered), eps))
    y_stand = y_centered / pooled_std

    # Site means (gamma) and variances (delta)
    # gamma_s = (site_indicator_s^T @ y_stand) / n_s
    gamma = (site_indicators.T @ y_stand) / jnp.clip(site_counts, eps)

    # delta_s = sum_i (y_stand_i - gamma_s)^2 * I(site_i = s) / (n_s - 1)
    residuals = y_stand[:, None] - gamma[None, :]  # (n_subjects, n_sites)
    sq_resid_weighted = residuals ** 2 * site_indicators  # zero out other sites
    delta = jnp.sum(sq_resid_weighted, axis=0) / jnp.clip(site_counts - 1, eps)
    delta = jnp.clip(delta, eps)

    # EB shrinkage for gamma
    gamma_bar = jnp.mean(gamma)
    tau_sq = jnp.clip(jnp.var(gamma), eps)
    weight = site_counts * tau_sq / jnp.clip(
        site_counts * tau_sq + delta, eps
    )
    gamma_star = weight * gamma + (1.0 - weight) * gamma_bar

    # EB shrinkage for delta (Inverse Gamma)
    delta_bar = jnp.mean(delta)
    delta_var = jnp.clip(jnp.var(delta), eps)
    alpha = 2.0 + delta_bar ** 2 / delta_var
    beta = delta_bar * (alpha - 1.0)
    delta_star = (0.5 * site_counts * delta + beta) / jnp.clip(
        0.5 * site_counts + alpha + 1.0, eps
    )
    delta_star = jnp.clip(delta_star, eps)

    return grand_mean, gamma_star, delta_star, pooled_std


def fit_combat_vmap(
    Y: jnp.ndarray,
    site_labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit ComBat for all features in parallel via jax.vmap.

    A streamlined version that vmaps the single-feature ComBat over all
    columns of Y simultaneously. Does not support covariates (regress
    them out beforehand).

    Args:
        Y: Data matrix, shape (n_subjects, n_features).
        site_labels: Integer site IDs, shape (n_subjects,).

    Returns:
        Tuple of (grand_means, gamma_star, delta_star, pooled_stds):
            grand_means: shape (n_features,).
            gamma_star: shape (n_features, n_sites).
            delta_star: shape (n_features, n_sites).
            pooled_stds: shape (n_features,).
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    site_labels = jnp.asarray(site_labels, dtype=jnp.int32)
    n_sites = int(jnp.max(site_labels)) + 1

    site_indicators = _make_site_design(site_labels, n_sites)
    site_counts = jnp.sum(site_indicators, axis=0)

    # vmap over features (columns of Y)
    grand_means, gamma_star, delta_star, pooled_stds = jax.vmap(
        lambda y: _fit_combat_single_feature(y, site_indicators, site_counts),
        in_axes=0,
    )(Y.T)  # Y.T has shape (n_features, n_subjects)

    return grand_means, gamma_star, delta_star, pooled_stds


def apply_combat_vmap(
    Y: jnp.ndarray,
    site_labels: jnp.ndarray,
    grand_means: jnp.ndarray,
    gamma_star: jnp.ndarray,
    delta_star: jnp.ndarray,
    pooled_stds: jnp.ndarray,
) -> jnp.ndarray:
    """Apply vmapped ComBat parameters to harmonize data.

    Args:
        Y: Data matrix, shape (n_subjects, n_features).
        site_labels: Integer site IDs, shape (n_subjects,).
        grand_means: shape (n_features,).
        gamma_star: shape (n_features, n_sites).
        delta_star: shape (n_features, n_sites).
        pooled_stds: shape (n_features,).

    Returns:
        Harmonized data, shape (n_subjects, n_features).
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    site_labels = jnp.asarray(site_labels, dtype=jnp.int32)
    n_sites = int(jnp.max(site_labels)) + 1

    site_indicators = _make_site_design(site_labels, n_sites)

    def harmonize_one_feature(y, gm, gs, ds, ps):
        """Harmonize a single feature across subjects."""
        y_stand = (y - gm) / jnp.clip(ps, eps)

        # Per-subject gamma_star and delta_star (looked up by site)
        subj_gamma = site_indicators @ gs  # (n_subjects,)
        subj_delta = site_indicators @ ds  # (n_subjects,)

        y_combat = (y_stand - subj_gamma) / jnp.sqrt(jnp.clip(subj_delta, eps))
        return y_combat * ps + gm

    # vmap over features
    Y_harm = jax.vmap(
        harmonize_one_feature,
        in_axes=(0, 0, 0, 0, 0),
    )(Y.T, grand_means, gamma_star, delta_star, pooled_stds)

    return Y_harm.T  # Back to (n_subjects, n_features)
