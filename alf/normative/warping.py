"""SHASH (sinh-arcsinh) likelihood warping for normative models.

The SHASH distribution (Jones & Pewsey, 2009) generalizes the normal
distribution with skewness and kurtosis parameters, enabling normative
models to handle non-Gaussian brain measures. The transformation:

    Y = sinh((arcsinh(Z) + epsilon) / delta)

maps a standard normal Z to a SHASH-distributed Y, where:
    epsilon: skewness parameter
    delta: kurtosis parameter (delta > 0; delta=1 gives normal)

Following Fraza et al. (2021), the SHASH warping is applied BEFORE BLR
fitting, transforming the targets to be approximately normal.

References:
    Jones & Pewsey (2009). Sinh-arcsinh distributions. Biometrika.
    Fraza, Dinga, Beckmann & Marquand (2021). Warped Bayesian linear regression
        for normative modelling of big data. NeuroImage.
"""


import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# SHASH distribution
# ---------------------------------------------------------------------------

def shash_transform(
    y: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    epsilon: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    """Forward SHASH transform: map y to approximately standard normal.

    z = sinh((arcsinh((y - mu) / sigma) - epsilon) * delta)

    Args:
        y: Observed values, shape (*,).
        mu: Location parameter.
        sigma: Scale parameter (> 0).
        epsilon: Skewness parameter.
        delta: Kurtosis parameter (> 0, 1 = normal).

    Returns:
        Transformed values (approximately standard normal), shape (*,).
    """
    eps = 1e-16
    standardized = (y - mu) / jnp.clip(sigma, eps)
    return jnp.sinh((jnp.arcsinh(standardized) - epsilon) * delta)


def shash_inverse_transform(
    z: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    epsilon: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    """Inverse SHASH transform: map standard normal z back to y-space.

    y = mu + sigma * sinh(arcsinh(z) / delta + epsilon)

    Args:
        z: Standard normal values, shape (*,).
        mu: Location parameter.
        sigma: Scale parameter.
        epsilon: Skewness parameter.
        delta: Kurtosis parameter.

    Returns:
        Values in original space, shape (*,).
    """
    eps = 1e-16
    return mu + sigma * jnp.sinh(jnp.arcsinh(z) / jnp.clip(delta, eps) + epsilon)


def shash_log_prob(
    y: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    epsilon: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability of the SHASH distribution.

    log p(y | mu, sigma, epsilon, delta) =
        log(delta / sigma) + log(C(y)) - 0.5*log(2*pi) - 0.5*S(y)^2

    where S(y) = sinh(delta * arcsinh((y-mu)/sigma) - epsilon)
    and C(y) = delta * cosh(delta * arcsinh((y-mu)/sigma) - epsilon)
               / sqrt((y-mu)^2 / sigma^2 + 1)

    Args:
        y: Observed values, shape (*,).
        mu: Location parameter.
        sigma: Scale parameter (> 0).
        epsilon: Skewness parameter.
        delta: Kurtosis parameter (> 0).

    Returns:
        Log-probability, shape (*,).
    """
    eps = 1e-16
    standardized = (y - mu) / jnp.clip(sigma, eps)
    asinh_s = jnp.arcsinh(standardized)

    # S(y) and C(y)
    arg = delta * asinh_s - epsilon
    S = jnp.sinh(arg)
    C = delta * jnp.cosh(arg) / jnp.sqrt(standardized ** 2 + 1.0)

    log_p = (
        jnp.log(jnp.clip(C, eps))
        - jnp.log(jnp.clip(sigma, eps))
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * S ** 2
    )

    return log_p


# ---------------------------------------------------------------------------
# Warped BLR
# ---------------------------------------------------------------------------

def _shash_nll(
    shash_params: tuple[jnp.ndarray, jnp.ndarray],
    y: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Negative log-likelihood for SHASH parameter estimation.

    Args:
        shash_params: Tuple of (epsilon, log_delta).
        y: Observed values, shape (N,).
        mu: BLR predicted means, shape (N,).
        sigma: BLR predicted stds, shape (N,).

    Returns:
        Negative log-likelihood (scalar).
    """
    epsilon, log_delta = shash_params
    delta = jnp.exp(log_delta)
    return -jnp.sum(shash_log_prob(y, mu, sigma, epsilon, delta))


def fit_blr_shash(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
    num_warp_iter: int = 5,
    lr: float = 0.01,
    num_opt_steps: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit warped BLR normative model with SHASH likelihood.

    Alternates between:
    1. Fit BLR on SHASH-transformed targets.
    2. Optimize SHASH parameters (epsilon, delta) given BLR predictions.

    Args:
        x_train: Training covariates, shape (N_train,).
        y_train: Training targets, shape (N_train,).
        x_test: Test covariates, shape (N_test,).
        y_test: Test targets, shape (N_test,).
        n_basis: Number of B-spline basis functions.
        degree: Spline degree.
        num_warp_iter: Number of BLR <-> SHASH alternation steps.
        lr: Learning rate for SHASH parameter optimization.
        num_opt_steps: Steps per SHASH optimization.

    Returns:
        Tuple of (z_scores, y_pred, y_std) in original space.
    """
    from alf.normative.blr import (
        bspline_basis,
        fit_blr,
        predict_blr,
        compute_zscore,
    )

    x_min = float(min(np.min(x_train), np.min(x_test)))
    x_max = float(max(np.max(x_train), np.max(x_test)))

    Phi_train = jnp.array(bspline_basis(x_train, n_basis, degree, x_min, x_max))
    Phi_test = jnp.array(bspline_basis(x_test, n_basis, degree, x_min, x_max))
    y_train_jnp = jnp.array(y_train)
    y_test_jnp = jnp.array(y_test)

    # Initialize SHASH parameters (identity warping)
    epsilon = jnp.array(0.0)
    log_delta = jnp.array(0.0)  # delta = 1 -> standard normal

    grad_fn = jax.grad(_shash_nll, argnums=0)

    for warp_iter in range(num_warp_iter):
        # 1. Transform targets using current SHASH params
        delta = jnp.exp(log_delta)
        y_warped = shash_transform(
            y_train_jnp,
            jnp.mean(y_train_jnp),
            jnp.std(y_train_jnp),
            epsilon,
            delta,
        )

        # 2. Fit BLR on warped targets
        blr_params = fit_blr(Phi_train, y_warped)

        # 3. Predict on training data
        pred_train = predict_blr(Phi_train, blr_params)

        # 4. Optimize SHASH parameters given BLR predictions
        shash_params = (epsilon, log_delta)
        for _ in range(num_opt_steps):
            grads = grad_fn(
                shash_params, y_train_jnp,
                pred_train.y_pred + jnp.mean(y_train_jnp),
                pred_train.y_std + jnp.std(y_train_jnp),
            )
            shash_params = (
                shash_params[0] - lr * grads[0],
                shash_params[1] - lr * grads[1],
            )

        epsilon, log_delta = shash_params

    # Final prediction on test data (in warped space)
    delta = jnp.exp(log_delta)
    y_test_warped = shash_transform(
        y_test_jnp,
        jnp.mean(y_train_jnp),
        jnp.std(y_train_jnp),
        epsilon,
        delta,
    )
    pred_test = predict_blr(Phi_test, blr_params)

    z_scores = compute_zscore(y_test_warped, pred_test.y_pred, pred_test.y_std)

    # Transform predictions back to original space
    y_pred_orig = shash_inverse_transform(
        pred_test.y_pred,
        jnp.mean(y_train_jnp),
        jnp.std(y_train_jnp),
        epsilon,
        delta,
    )
    y_std_orig = pred_test.y_std * jnp.std(y_train_jnp)

    return z_scores, y_pred_orig, y_std_orig
