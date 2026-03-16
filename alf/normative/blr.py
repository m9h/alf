"""Bayesian Linear Regression with B-spline basis for normative modeling.

Fits population distributions of brain measures as a function of covariates
(e.g., age) using Bayesian Linear Regression. Individual deviations from the
population norm are quantified as Z-scores:

    Z = (y_obs - y_pred) / sigma_pred

where y_pred and sigma_pred come from the BLR posterior predictive distribution.

The BLR uses a B-spline basis expansion for flexible nonlinear modeling of
age-related trajectories, following Fraza et al. (2021). All computations are
JAX-native, enabling jax.vmap over brain regions for massive parallelism.

References:
    Fraza, Dinga, Beckmann & Marquand (2021). Warped Bayesian linear regression
        for normative modelling of big data. NeuroImage.
    Marquand, Rezek, Buitelaar & Beckmann (2016). Understanding heterogeneity
        in clinical cohorts using normative models. Biological Psychiatry.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BLRParams(NamedTuple):
    """Posterior parameters of Bayesian Linear Regression.

    Attributes:
        w_mean: Posterior mean of weights, shape (n_basis,).
        w_cov: Posterior covariance of weights, shape (n_basis, n_basis).
        beta: Noise precision (inverse variance), scalar.
        alpha: Prior precision on weights, scalar.
    """
    w_mean: jnp.ndarray
    w_cov: jnp.ndarray
    beta: jnp.ndarray
    alpha: jnp.ndarray


class BLRResult(NamedTuple):
    """Prediction from BLR.

    Attributes:
        y_pred: Predicted mean, shape (N,).
        y_var: Predicted variance, shape (N,).
        y_std: Predicted standard deviation, shape (N,).
    """
    y_pred: jnp.ndarray
    y_var: jnp.ndarray
    y_std: jnp.ndarray


class NormativeResult(NamedTuple):
    """Result of normative model evaluation.

    Attributes:
        z_score: Deviation score (y_obs - y_pred) / y_std, shape (N,).
        y_pred: Predicted mean, shape (N,).
        y_std: Predicted std, shape (N,).
        blr_params: Fitted BLR parameters.
    """
    z_score: jnp.ndarray
    y_pred: jnp.ndarray
    y_std: jnp.ndarray
    blr_params: BLRParams


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------

def _bspline_basis_element(
    x: jnp.ndarray,
    knots: jnp.ndarray,
    i: int,
    degree: int,
) -> jnp.ndarray:
    """Compute a single B-spline basis element via de Boor recursion.

    Uses the Cox-de Boor recursion formula. JAX-compatible via jnp operations.

    Args:
        x: Evaluation points, shape (N,).
        knots: Knot vector, shape (n_knots,).
        i: Basis function index.
        degree: Spline degree (3 = cubic).

    Returns:
        Basis values, shape (N,).
    """
    if degree == 0:
        return jnp.where(
            (x >= knots[i]) & (x < knots[i + 1]),
            1.0, 0.0,
        )

    eps = 1e-16

    denom_left = knots[i + degree] - knots[i]
    denom_right = knots[i + degree + 1] - knots[i + 1]

    left = jnp.where(
        jnp.abs(denom_left) > eps,
        (x - knots[i]) / denom_left * _bspline_basis_element(
            x, knots, i, degree - 1
        ),
        0.0,
    )
    right = jnp.where(
        jnp.abs(denom_right) > eps,
        (knots[i + degree + 1] - x) / denom_right * _bspline_basis_element(
            x, knots, i + 1, degree - 1
        ),
        0.0,
    )

    return left + right


def bspline_basis(
    x: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> np.ndarray:
    """Compute B-spline design matrix.

    Creates a B-spline basis with evenly-spaced interior knots and
    appropriate boundary padding.

    Note: This uses NumPy (not JAX) for basis construction since it's
    typically done once during data preparation, not during optimization.

    Args:
        x: Covariate values, shape (N,).
        n_basis: Number of basis functions.
        degree: Spline degree (3 = cubic). Default 3.
        x_min: Minimum for knot range. Default: min(x).
        x_max: Maximum for knot range. Default: max(x).

    Returns:
        Design matrix, shape (N, n_basis).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)

    if x_min is None:
        x_min = float(x.min())
    if x_max is None:
        x_max = float(x.max())

    # Interior knots
    n_interior = n_basis - degree + 1
    n_interior = max(n_interior, 2)
    interior = np.linspace(x_min, x_max, n_interior)

    # Pad with boundary knots (repeated for clamped spline)
    knots = np.concatenate([
        np.full(degree, x_min),
        interior,
        np.full(degree, x_max),
    ])

    # Evaluate each basis function
    n_funcs = len(knots) - degree - 1
    n_funcs = min(n_funcs, n_basis)

    B = np.zeros((N, n_funcs))
    for i in range(n_funcs):
        B[:, i] = np.array(
            _bspline_basis_element(jnp.array(x), jnp.array(knots), i, degree)
        )

    # Handle right boundary (last basis function should include x_max)
    if N > 0:
        B[x >= x_max - 1e-10, -1] = 1.0

    return B


# ---------------------------------------------------------------------------
# Bayesian Linear Regression
# ---------------------------------------------------------------------------

def fit_blr(
    Phi: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    optimize_hyperparams: bool = True,
    max_iter: int = 100,
) -> BLRParams:
    """Fit Bayesian Linear Regression with evidence optimization.

    Computes the posterior p(w | y, Phi, alpha, beta) = N(w_mean, w_cov)
    where:
        w_cov = (alpha * I + beta * Phi^T @ Phi)^{-1}
        w_mean = beta * w_cov @ Phi^T @ y

    Optionally optimizes alpha and beta via iterative evidence maximization
    (type-II ML / empirical Bayes).

    Args:
        Phi: Design matrix, shape (N, D).
        y: Target values, shape (N,).
        alpha: Prior precision on weights. Default 1.0.
        beta: Noise precision. Default 1.0.
        optimize_hyperparams: If True, optimize alpha/beta. Default True.
        max_iter: Max iterations for hyperparameter optimization.

    Returns:
        BLRParams with posterior parameters.
    """
    N, D = Phi.shape
    eps = 1e-16

    alpha = jnp.array(alpha)
    beta = jnp.array(beta)

    for _ in range(max_iter if optimize_hyperparams else 1):
        # Posterior
        S_inv = alpha * jnp.eye(D) + beta * Phi.T @ Phi
        w_cov = jnp.linalg.inv(S_inv)
        w_mean = beta * w_cov @ Phi.T @ y

        if not optimize_hyperparams:
            break

        # Evidence optimization (Bishop, 2006, Section 3.5.2)
        eigenvalues = jnp.linalg.eigvalsh(beta * Phi.T @ Phi)
        gamma = jnp.sum(eigenvalues / (alpha + eigenvalues))

        # Update alpha
        alpha = jnp.clip(gamma / (w_mean @ w_mean + eps), eps, 1e6)

        # Update beta
        residual = y - Phi @ w_mean
        beta = jnp.clip(
            (N - gamma) / (residual @ residual + eps), eps, 1e6
        )

    return BLRParams(w_mean=w_mean, w_cov=w_cov, beta=beta, alpha=alpha)


def predict_blr(
    Phi_test: jnp.ndarray,
    params: BLRParams,
) -> BLRResult:
    """Predict with BLR posterior.

    Computes the posterior predictive distribution:
        y_pred = Phi_test @ w_mean
        y_var = 1/beta + diag(Phi_test @ w_cov @ Phi_test^T)

    Args:
        Phi_test: Test design matrix, shape (N_test, D).
        params: Fitted BLR parameters.

    Returns:
        BLRResult with predicted mean and variance.
    """
    eps = 1e-16
    y_pred = Phi_test @ params.w_mean

    # Predictive variance = noise variance + epistemic uncertainty
    epistemic = jnp.sum(Phi_test @ params.w_cov * Phi_test, axis=1)
    noise_var = 1.0 / jnp.clip(params.beta, eps)
    y_var = noise_var + epistemic
    y_var = jnp.clip(y_var, eps)
    y_std = jnp.sqrt(y_var)

    return BLRResult(y_pred=y_pred, y_var=y_var, y_std=y_std)


# ---------------------------------------------------------------------------
# Normative model (Z-scoring)
# ---------------------------------------------------------------------------

def compute_zscore(
    y_obs: jnp.ndarray,
    y_pred: jnp.ndarray,
    y_std: jnp.ndarray,
) -> jnp.ndarray:
    """Compute deviation Z-scores.

    Z = (y_obs - y_pred) / y_std

    Args:
        y_obs: Observed values, shape (N,).
        y_pred: Predicted means, shape (N,).
        y_std: Predicted standard deviations, shape (N,).

    Returns:
        Z-scores, shape (N,).
    """
    eps = 1e-16
    return (y_obs - y_pred) / jnp.clip(y_std, eps)


def normative_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
) -> NormativeResult:
    """Fit normative model and compute Z-scores for test subjects.

    Full pipeline: B-spline basis -> BLR fit -> predict -> Z-score.

    Args:
        x_train: Training covariates (e.g., age), shape (N_train,).
        y_train: Training targets (e.g., brain measure), shape (N_train,).
        x_test: Test covariates, shape (N_test,).
        y_test: Test observations, shape (N_test,).
        n_basis: Number of B-spline basis functions.
        degree: Spline degree.

    Returns:
        NormativeResult with Z-scores and model parameters.
    """
    # Build design matrices
    x_min = float(min(np.min(x_train), np.min(x_test)))
    x_max = float(max(np.max(x_train), np.max(x_test)))

    Phi_train = jnp.array(bspline_basis(x_train, n_basis, degree, x_min, x_max))
    Phi_test = jnp.array(bspline_basis(x_test, n_basis, degree, x_min, x_max))

    y_train_jnp = jnp.array(y_train)
    y_test_jnp = jnp.array(y_test)

    # Fit BLR on training data
    blr_params = fit_blr(Phi_train, y_train_jnp)

    # Predict on test data
    pred = predict_blr(Phi_test, blr_params)

    # Compute Z-scores
    z = compute_zscore(y_test_jnp, pred.y_pred, pred.y_std)

    return NormativeResult(
        z_score=z,
        y_pred=pred.y_pred,
        y_std=pred.y_std,
        blr_params=blr_params,
    )


def normative_model_vmap(
    x_train: np.ndarray,
    Y_train: np.ndarray,
    x_test: np.ndarray,
    Y_test: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit normative models across multiple brain regions via vmap.

    This is the key performance advantage: independent BLR fits across
    1000+ brain regions, parallelized by jax.vmap.

    Args:
        x_train: Training covariates, shape (N_train,).
        Y_train: Training targets, shape (N_train, n_regions).
        x_test: Test covariates, shape (N_test,).
        Y_test: Test observations, shape (N_test, n_regions).
        n_basis: Number of B-spline basis functions.
        degree: Spline degree.

    Returns:
        Tuple of (Z_scores, Y_pred, Y_std), each shape (N_test, n_regions).
    """
    # Shared design matrices (same covariates for all regions)
    x_min = float(min(np.min(x_train), np.min(x_test)))
    x_max = float(max(np.max(x_train), np.max(x_test)))

    Phi_train = jnp.array(bspline_basis(x_train, n_basis, degree, x_min, x_max))
    Phi_test = jnp.array(bspline_basis(x_test, n_basis, degree, x_min, x_max))

    Y_train_jnp = jnp.array(Y_train)  # (N_train, n_regions)
    Y_test_jnp = jnp.array(Y_test)    # (N_test, n_regions)

    def fit_and_score_one_region(y_train, y_test):
        """Fit BLR and compute Z-scores for one region."""
        params = fit_blr(Phi_train, y_train, optimize_hyperparams=False)
        pred = predict_blr(Phi_test, params)
        z = compute_zscore(y_test, pred.y_pred, pred.y_std)
        return z, pred.y_pred, pred.y_std

    # vmap over regions (second axis of Y)
    z_scores, y_pred, y_std = jax.vmap(
        fit_and_score_one_region,
        in_axes=(1, 1),
        out_axes=(1, 1, 1),
    )(Y_train_jnp, Y_test_jnp)

    return z_scores, y_pred, y_std
