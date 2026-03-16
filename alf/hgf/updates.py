"""Core HGF update equations via jax.lax.scan.

Implements the Hierarchical Gaussian Filter update equations from Mathys et al.
(2011, 2014). Each level tracks Gaussian sufficient statistics (mu, pi) and
updates them via precision-weighted prediction errors propagating bottom-up.

Two model variants:
    - 2-level binary HGF: Binary observations, sigmoid link to level 2.
    - 3-level continuous HGF: Continuous observations, volatility coupling
      between all levels.

All functions are jax.jit and jax.grad compatible. Temporal sequences use
jax.lax.scan — the same pattern as alf.learning's HMM forward algorithm.

References:
    Mathys, Daunizeau, Friston & Stephan (2011). A Bayesian foundation for
        individual learning under uncertainty. Frontiers in Human Neuroscience.
    Mathys, Lomakina, Daunizeau et al. (2014). Uncertainty in perception and
        the Hierarchical Gaussian Filter. Frontiers in Human Neuroscience.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HGFBelief(NamedTuple):
    """Gaussian belief at one HGF level.

    Attributes:
        mu: Posterior mean.
        pi: Posterior precision (inverse variance).
    """
    mu: jnp.ndarray
    pi: jnp.ndarray


class BinaryHGFParams(NamedTuple):
    """Parameters for a 2-level binary HGF.

    Attributes:
        omega_2: Tonic log-volatility of level 2 (controls step size).
        mu_2_0: Initial mean of level 2.
        sigma_2_0: Initial variance of level 2.
    """
    omega_2: jnp.ndarray
    mu_2_0: jnp.ndarray
    sigma_2_0: jnp.ndarray


class ContinuousHGFParams(NamedTuple):
    """Parameters for a 3-level continuous HGF.

    Attributes:
        omega_1: Tonic log-volatility at level 1.
        omega_2: Tonic log-volatility at level 2.
        kappa_1: Volatility coupling between levels 1 and 2.
        kappa_2: Volatility coupling between levels 2 and 3.
        theta: Meta-volatility (step size of the top level).
        pi_u: Input precision (inverse observation noise variance).
        mu_1_0: Initial mean of level 1.
        sigma_1_0: Initial variance of level 1.
        mu_2_0: Initial mean of level 2.
        sigma_2_0: Initial variance of level 2.
        mu_3_0: Initial mean of level 3.
        sigma_3_0: Initial variance of level 3.
    """
    omega_1: jnp.ndarray
    omega_2: jnp.ndarray
    kappa_1: jnp.ndarray
    kappa_2: jnp.ndarray
    theta: jnp.ndarray
    pi_u: jnp.ndarray
    mu_1_0: jnp.ndarray
    sigma_1_0: jnp.ndarray
    mu_2_0: jnp.ndarray
    sigma_2_0: jnp.ndarray
    mu_3_0: jnp.ndarray
    sigma_3_0: jnp.ndarray


class HGFResult(NamedTuple):
    """Result of running HGF on a sequence.

    Attributes:
        mu: Posterior means at each level, shape (T, num_levels).
        pi: Posterior precisions at each level, shape (T, num_levels).
        surprise: Per-trial surprise (negative log-likelihood), shape (T,).
    """
    mu: jnp.ndarray
    pi: jnp.ndarray
    surprise: jnp.ndarray


# ---------------------------------------------------------------------------
# 2-level binary HGF
# ---------------------------------------------------------------------------

def binary_hgf_update(
    mu_2: jnp.ndarray,
    pi_2: jnp.ndarray,
    u: jnp.ndarray,
    omega_2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-trial update for 2-level binary HGF.

    Level 1 uses a sigmoid link function: P(u=1) = sigmoid(x_2).
    Level 2 evolves as a Gaussian random walk with step size exp(omega_2).

    Args:
        mu_2: Current posterior mean of level 2 (scalar).
        pi_2: Current posterior precision of level 2 (scalar).
        u: Binary observation (0 or 1, scalar).
        omega_2: Tonic log-volatility (scalar).

    Returns:
        Tuple of (new_mu_2, new_pi_2, surprise).
    """
    eps = 1e-16

    # Prediction
    hat_mu_2 = mu_2
    sigma_2 = 1.0 / jnp.clip(pi_2, eps)
    hat_sigma_2 = sigma_2 + jnp.exp(omega_2)
    hat_pi_2 = 1.0 / jnp.clip(hat_sigma_2, eps)

    # Level 1 prediction via sigmoid link
    hat_mu_1 = jax.nn.sigmoid(hat_mu_2)
    hat_mu_1 = jnp.clip(hat_mu_1, eps, 1.0 - eps)

    # Prediction error
    delta_1 = u - hat_mu_1

    # Update level 2
    # Information from observation: d^2/dx_2^2 [-log p(u|x_2)] = s(x)(1-s(x))
    info_gain = hat_mu_1 * (1.0 - hat_mu_1)
    new_pi_2 = hat_pi_2 + info_gain
    new_mu_2 = hat_mu_2 + delta_1 / jnp.clip(new_pi_2, eps)

    # Surprise: -log p(u | hat_mu_1)
    surprise = -(u * jnp.log(hat_mu_1) + (1.0 - u) * jnp.log(1.0 - hat_mu_1))

    return new_mu_2, new_pi_2, surprise


def binary_hgf(
    observations: jnp.ndarray,
    params: BinaryHGFParams,
) -> HGFResult:
    """Run 2-level binary HGF on a sequence of observations.

    Uses jax.lax.scan for efficient, JIT-compatible sequential processing.

    Args:
        observations: Binary observation sequence, shape (T,).
        params: HGF parameters (omega_2, initial conditions).

    Returns:
        HGFResult with posterior trajectories and surprise.
    """
    eps = 1e-16
    mu_2_0 = params.mu_2_0
    pi_2_0 = 1.0 / jnp.clip(params.sigma_2_0, eps)

    def scan_step(carry, u):
        mu_2, pi_2 = carry
        new_mu_2, new_pi_2, surprise = binary_hgf_update(
            mu_2, pi_2, u, params.omega_2
        )
        # Stack as (1,) arrays for concatenation across levels
        out_mu = jnp.array([new_mu_2])
        out_pi = jnp.array([new_pi_2])
        return (new_mu_2, new_pi_2), (out_mu, out_pi, surprise)

    init_carry = (mu_2_0, pi_2_0)
    _, (mu_traj, pi_traj, surprises) = jax.lax.scan(
        scan_step, init_carry, observations
    )

    return HGFResult(mu=mu_traj, pi=pi_traj, surprise=surprises)


def binary_hgf_surprise(
    observations: jnp.ndarray,
    params: BinaryHGFParams,
) -> jnp.ndarray:
    """Compute total surprise for binary HGF (differentiable NLL).

    Args:
        observations: Binary observation sequence, shape (T,).
        params: HGF parameters.

    Returns:
        Total surprise (scalar, negative log-likelihood).
    """
    result = binary_hgf(observations, params)
    return jnp.sum(result.surprise)


# ---------------------------------------------------------------------------
# 3-level continuous HGF
# ---------------------------------------------------------------------------

def continuous_hgf_update(
    mu_1: jnp.ndarray,
    pi_1: jnp.ndarray,
    mu_2: jnp.ndarray,
    pi_2: jnp.ndarray,
    mu_3: jnp.ndarray,
    pi_3: jnp.ndarray,
    u: jnp.ndarray,
    omega_1: jnp.ndarray,
    omega_2: jnp.ndarray,
    kappa_1: jnp.ndarray,
    kappa_2: jnp.ndarray,
    theta: jnp.ndarray,
    pi_u: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-trial update for 3-level continuous HGF.

    Level 1: tracks the value of the input.
    Level 2: tracks the volatility (log step-size) of level 1.
    Level 3: tracks the meta-volatility of level 2.

    Args:
        mu_1, pi_1: Level 1 posterior mean and precision.
        mu_2, pi_2: Level 2 posterior mean and precision.
        mu_3, pi_3: Level 3 posterior mean and precision.
        u: Continuous observation (scalar).
        omega_1, omega_2: Tonic log-volatilities.
        kappa_1, kappa_2: Volatility coupling strengths.
        theta: Top-level meta-volatility.
        pi_u: Input (observation noise) precision.

    Returns:
        Tuple of (new_mu_1, new_pi_1, new_mu_2, new_pi_2,
                  new_mu_3, new_pi_3, surprise).
    """
    eps = 1e-16

    # --- Prediction step ---
    hat_mu_1 = mu_1
    v_1 = jnp.exp(kappa_1 * mu_2 + omega_1)
    hat_sigma_1 = 1.0 / jnp.clip(pi_1, eps) + v_1
    hat_pi_1 = 1.0 / jnp.clip(hat_sigma_1, eps)

    hat_mu_2 = mu_2
    v_2 = jnp.exp(kappa_2 * mu_3 + omega_2)
    hat_sigma_2 = 1.0 / jnp.clip(pi_2, eps) + v_2
    hat_pi_2 = 1.0 / jnp.clip(hat_sigma_2, eps)

    hat_mu_3 = mu_3
    hat_sigma_3 = 1.0 / jnp.clip(pi_3, eps) + theta
    hat_pi_3 = 1.0 / jnp.clip(hat_sigma_3, eps)

    # --- Level 1 update (value) ---
    delta_1 = u - hat_mu_1
    new_pi_1 = hat_pi_1 + pi_u
    new_sigma_1 = 1.0 / jnp.clip(new_pi_1, eps)
    new_mu_1 = hat_mu_1 + new_sigma_1 * pi_u * delta_1

    # --- Level 2 update (volatility of level 1) ---
    # Volatility prediction error from level 1
    delta_mu_1 = new_mu_1 - hat_mu_1
    vope_1 = hat_pi_1 * (new_sigma_1 + delta_mu_1 ** 2) - 1.0

    # Coupling weight: how much level 2 influences level 1's variance
    w_1 = kappa_1 * v_1 / jnp.clip(hat_sigma_1, eps)

    # Update precision and mean
    new_pi_2 = hat_pi_2 + 0.5 * w_1 ** 2 * (1.0 + vope_1)
    new_pi_2 = jnp.clip(new_pi_2, eps)  # Safeguard against negative precision
    new_mu_2 = hat_mu_2 + 0.5 * w_1 * vope_1 / new_pi_2
    new_sigma_2 = 1.0 / new_pi_2

    # --- Level 3 update (meta-volatility) ---
    # Volatility prediction error from level 2
    delta_mu_2 = new_mu_2 - hat_mu_2
    vope_2 = hat_pi_2 * (new_sigma_2 + delta_mu_2 ** 2) - 1.0

    # Coupling weight
    w_2 = kappa_2 * v_2 / jnp.clip(hat_sigma_2, eps)

    # Update
    new_pi_3 = hat_pi_3 + 0.5 * w_2 ** 2 * (1.0 + vope_2)
    new_pi_3 = jnp.clip(new_pi_3, eps)
    new_mu_3 = hat_mu_3 + 0.5 * w_2 * vope_2 / new_pi_3

    # --- Surprise ---
    # -log N(u; hat_mu_1, 1/pi_u + hat_sigma_1)
    total_var = 1.0 / jnp.clip(pi_u, eps) + hat_sigma_1
    surprise = 0.5 * (jnp.log(2.0 * jnp.pi * total_var) + delta_1 ** 2 / total_var)

    return new_mu_1, new_pi_1, new_mu_2, new_pi_2, new_mu_3, new_pi_3, surprise


def continuous_hgf(
    observations: jnp.ndarray,
    params: ContinuousHGFParams,
) -> HGFResult:
    """Run 3-level continuous HGF on a sequence of observations.

    Uses jax.lax.scan for efficient, JIT-compatible sequential processing.

    Args:
        observations: Continuous observation sequence, shape (T,).
        params: HGF parameters (volatilities, couplings, initial conditions).

    Returns:
        HGFResult with posterior trajectories and surprise at each level.
    """
    eps = 1e-16
    pi_1_0 = 1.0 / jnp.clip(params.sigma_1_0, eps)
    pi_2_0 = 1.0 / jnp.clip(params.sigma_2_0, eps)
    pi_3_0 = 1.0 / jnp.clip(params.sigma_3_0, eps)

    def scan_step(carry, u):
        mu_1, pi_1, mu_2, pi_2, mu_3, pi_3 = carry
        (new_mu_1, new_pi_1, new_mu_2, new_pi_2,
         new_mu_3, new_pi_3, surprise) = continuous_hgf_update(
            mu_1, pi_1, mu_2, pi_2, mu_3, pi_3, u,
            params.omega_1, params.omega_2,
            params.kappa_1, params.kappa_2,
            params.theta, params.pi_u,
        )
        out_mu = jnp.array([new_mu_1, new_mu_2, new_mu_3])
        out_pi = jnp.array([new_pi_1, new_pi_2, new_pi_3])
        new_carry = (new_mu_1, new_pi_1, new_mu_2, new_pi_2,
                     new_mu_3, new_pi_3)
        return new_carry, (out_mu, out_pi, surprise)

    init_carry = (params.mu_1_0, pi_1_0, params.mu_2_0, pi_2_0,
                  params.mu_3_0, pi_3_0)
    _, (mu_traj, pi_traj, surprises) = jax.lax.scan(
        scan_step, init_carry, observations
    )

    return HGFResult(mu=mu_traj, pi=pi_traj, surprise=surprises)


def continuous_hgf_surprise(
    observations: jnp.ndarray,
    params: ContinuousHGFParams,
) -> jnp.ndarray:
    """Compute total surprise for continuous HGF (differentiable NLL).

    Args:
        observations: Continuous observation sequence, shape (T,).
        params: HGF parameters.

    Returns:
        Total surprise (scalar, negative log-likelihood).
    """
    result = continuous_hgf(observations, params)
    return jnp.sum(result.surprise)
