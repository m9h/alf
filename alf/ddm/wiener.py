"""Navarro-Fuss first-passage time density for Wiener diffusion.

Implements the analytical solution for the first-passage time density of a
Wiener diffusion process with absorbing boundaries (Navarro & Fuss, 2009).
This is the core likelihood function for drift-diffusion models (DDMs).

The density p(t, response | v, a, w, tau) is computed via an infinite series
with adaptive truncation for efficiency. Both small-time and large-time
series expansions are available; the implementation automatically selects
the faster-converging variant based on the time argument.

All functions are jax.jit compatible and differentiable via jax.grad,
enabling gradient-based DDM parameter fitting.

References:
    Navarro & Fuss (2009). Fast and accurate calculations for first-passage
        times in Wiener diffusion models. Journal of Mathematical Psychology.
    Ratcliff (1978). A theory of memory retrieval. Psychological Review.
    Blurton, Kesselmeier & Gondan (2012). Fast and accurate calculations for
        cumulative first-passage time distributions in Wiener diffusion models.
        Journal of Mathematical Psychology.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DDMParams(NamedTuple):
    """Parameters for a standard DDM.

    Attributes:
        v: Drift rate (evidence accumulation rate, can be negative).
        a: Boundary separation (threshold, > 0).
        w: Starting point bias as proportion of a (0 < w < 1). w=0.5 is unbiased.
        tau: Non-decision time (encoding + motor, > 0).
    """
    v: jnp.ndarray
    a: jnp.ndarray
    w: jnp.ndarray
    tau: jnp.ndarray


class DDMResult(NamedTuple):
    """Simulated DDM trial data.

    Attributes:
        rt: Reaction times, shape (n_trials,).
        choice: Choices (1 = upper boundary, 0 = lower boundary), shape (n_trials,).
    """
    rt: np.ndarray
    choice: np.ndarray


# ---------------------------------------------------------------------------
# Navarro-Fuss density (core computation)
# ---------------------------------------------------------------------------

def _wiener_small_time(
    t: jnp.ndarray,
    w: jnp.ndarray,
    n_terms: int = 7,
) -> jnp.ndarray:
    """Small-time series expansion of the Wiener FPT density (unnormalized).

    Computes: f_s(t | w) = sum_k (w + 2k) * exp(-(w+2k)^2 / (2t))
    Converges fast when t is small.

    Args:
        t: Decision time (RT - tau), normalized by a^2. Must be > 0.
        w: Starting point (0 < w < 1).
        n_terms: Number of terms in the series.

    Returns:
        Density value (unnormalized by 1/sqrt(2*pi*t^3)).
    """
    ks = jnp.arange(-n_terms, n_terms + 1)
    terms = (w + 2.0 * ks) * jnp.exp(-(w + 2.0 * ks) ** 2 / (2.0 * t))
    return jnp.sum(terms)


def _wiener_large_time(
    t: jnp.ndarray,
    w: jnp.ndarray,
    n_terms: int = 7,
) -> jnp.ndarray:
    """Large-time series expansion of the Wiener FPT density (unnormalized).

    Computes: f_l(t | w) = sum_k k * sin(k*pi*w) * exp(-k^2*pi^2*t/2)
    Converges fast when t is large.

    Args:
        t: Decision time (RT - tau), normalized by a^2. Must be > 0.
        w: Starting point (0 < w < 1).
        n_terms: Number of terms in the series.

    Returns:
        Density value (unnormalized by pi).
    """
    ks = jnp.arange(1, n_terms + 1)
    terms = ks * jnp.sin(ks * jnp.pi * w) * jnp.exp(
        -ks ** 2 * jnp.pi ** 2 * t / 2.0
    )
    return jnp.sum(terms)


def wiener_log_density(
    rt: jnp.ndarray,
    choice: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    w: jnp.ndarray,
    tau: jnp.ndarray,
    n_terms: int = 7,
) -> jnp.ndarray:
    """Log-density of the Wiener first-passage time (single trial).

    Computes log p(RT, choice | v, a, w, tau) using the Navarro-Fuss formula
    with adaptive series selection.

    For an upper-boundary response (choice=1), the density is evaluated
    directly. For a lower-boundary response (choice=0), we flip v and w
    by symmetry: p(t, lower | v, a, w) = p(t, upper | -v, a, 1-w).

    Args:
        rt: Reaction time (scalar, must be > tau).
        choice: Response (1 = upper boundary, 0 = lower boundary).
        v: Drift rate.
        a: Boundary separation (> 0).
        w: Starting point bias (0 < w < 1).
        tau: Non-decision time (> 0).
        n_terms: Series truncation (7 is usually sufficient).

    Returns:
        Log-density (scalar). Returns -inf for invalid rt <= tau.
    """
    eps = 1e-16

    # Flip for lower-boundary responses
    v_eff = jnp.where(choice > 0.5, v, -v)
    w_eff = jnp.where(choice > 0.5, w, 1.0 - w)

    # Decision time (subtract non-decision time)
    dt = rt - tau
    dt = jnp.clip(dt, eps)

    # Normalize time by boundary separation squared
    t_norm = dt / (a ** 2)
    t_norm = jnp.clip(t_norm, eps)

    # Choose series based on which converges faster
    # Navarro & Fuss heuristic: use small-time if t_norm < threshold
    use_small = t_norm < 2.5 / (jnp.pi ** 2)

    # Compute both series
    fs = _wiener_small_time(t_norm, w_eff, n_terms)
    fl = _wiener_large_time(t_norm, w_eff, n_terms)

    # Small-time density: (1/sqrt(2*pi*t^3)) * fs
    log_small = -0.5 * jnp.log(2.0 * jnp.pi) - 1.5 * jnp.log(t_norm) + jnp.log(
        jnp.clip(fs, eps)
    )

    # Large-time density: pi * fl
    log_large = jnp.log(jnp.pi) + jnp.log(jnp.clip(fl, eps))

    log_density_unnorm = jnp.where(use_small, log_small, log_large)

    # Add drift and boundary terms:
    # log p(t, upper | v, a, w) = log f(t_norm | w)
    #   - log(a^2)              [change of variables t -> t_norm]
    #   - v*a*w                 [drift contribution to starting point]
    #   - v^2 * dt / 2          [drift contribution to evolution]
    log_p = (
        log_density_unnorm
        - 2.0 * jnp.log(a)
        + v_eff * a * w_eff
        - (v_eff ** 2) * dt / 2.0
    )

    # Mask invalid trials (rt <= tau)
    valid = rt > tau + eps
    log_p = jnp.where(valid, log_p, -jnp.inf)

    return log_p


def wiener_log_density_batch(
    rt: jnp.ndarray,
    choice: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    w: jnp.ndarray,
    tau: jnp.ndarray,
    n_terms: int = 7,
) -> jnp.ndarray:
    """Vectorized log-density over a batch of trials (vmap).

    Args:
        rt: Reaction times, shape (N,).
        choice: Responses, shape (N,).
        v: Drift rate (scalar, shared across trials).
        a: Boundary separation (scalar).
        w: Starting point bias (scalar).
        tau: Non-decision time (scalar).
        n_terms: Series truncation.

    Returns:
        Log-densities, shape (N,).
    """
    return jax.vmap(
        lambda r, c: wiener_log_density(r, c, v, a, w, tau, n_terms)
    )(rt, choice)


# ---------------------------------------------------------------------------
# Log-likelihood and NLL
# ---------------------------------------------------------------------------

def ddm_log_likelihood(
    params: DDMParams,
    rt: jnp.ndarray,
    choice: jnp.ndarray,
    n_terms: int = 7,
) -> jnp.ndarray:
    """Total log-likelihood of DDM data.

    Args:
        params: DDM parameters.
        rt: Reaction times, shape (N,).
        choice: Responses, shape (N,).
        n_terms: Series truncation.

    Returns:
        Total log-likelihood (scalar).
    """
    log_densities = wiener_log_density_batch(
        rt, choice, params.v, params.a, params.w, params.tau, n_terms
    )
    return jnp.sum(log_densities)


def ddm_nll(
    v: jnp.ndarray,
    a: jnp.ndarray,
    w: jnp.ndarray,
    tau: jnp.ndarray,
    rt: jnp.ndarray,
    choice: jnp.ndarray,
) -> jnp.ndarray:
    """Negative log-likelihood (differentiable w.r.t. v, a, w, tau).

    Convenience wrapper with individual parameter arguments for jax.grad.

    Args:
        v, a, w, tau: DDM parameters (scalars).
        rt: Reaction times, shape (N,).
        choice: Responses, shape (N,).

    Returns:
        Negative log-likelihood (scalar).
    """
    params = DDMParams(v=v, a=a, w=w, tau=tau)
    return -ddm_log_likelihood(params, rt, choice)


# ---------------------------------------------------------------------------
# DDM simulation
# ---------------------------------------------------------------------------

def simulate_ddm(
    v: float,
    a: float,
    w: float = 0.5,
    tau: float = 0.3,
    dt: float = 0.001,
    max_steps: int = 10000,
    n_trials: int = 100,
    seed: int = 42,
) -> DDMResult:
    """Simulate DDM trials via Euler-Maruyama discretization.

    Simulates the Wiener process dx = v*dt + dW with absorbing boundaries
    at 0 and a, starting from z = a*w.

    Args:
        v: Drift rate.
        a: Boundary separation.
        w: Starting point (proportion of a).
        tau: Non-decision time.
        dt: Simulation time step.
        max_steps: Maximum number of steps per trial.
        n_trials: Number of trials to simulate.
        seed: Random seed.

    Returns:
        DDMResult with reaction times and choices.
    """
    rng = np.random.RandomState(seed)
    rts = np.zeros(n_trials)
    choices = np.zeros(n_trials, dtype=np.int32)

    sqrt_dt = np.sqrt(dt)

    for trial in range(n_trials):
        x = a * w  # Starting position
        for step in range(max_steps):
            noise = rng.normal() * sqrt_dt
            x += v * dt + noise

            if x >= a:
                rts[trial] = (step + 1) * dt + tau
                choices[trial] = 1
                break
            elif x <= 0:
                rts[trial] = (step + 1) * dt + tau
                choices[trial] = 0
                break
        else:
            # Did not reach boundary
            rts[trial] = max_steps * dt + tau
            choices[trial] = 1 if x >= a / 2 else 0

    return DDMResult(rt=rts, choice=choices)
