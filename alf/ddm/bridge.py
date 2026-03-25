"""Bridge DDM parameters to/from Active Inference EFE.

Maps between drift-diffusion model parameters and active inference
quantities, enabling DDM as a decision mechanism within the AIF framework.

Key mappings (Schwartenbeck & Friston, 2016):
    - Drift rate v = precision-weighted free energy gradient (gamma * delta_G)
    - Boundary a = policy precision gamma (higher precision -> wider boundary)
    - Starting bias w = prior policy preference E(pi)

References:
    Schwartenbeck & Friston (2016). Computational Phenotyping in Psychiatry:
        A Worked Example Using the Drift Diffusion Model.
    Friston, Schwartenbeck et al. (2014). The anatomy of choice: dopamine
        and decision-making. Phil Trans R Soc B.
"""

import jax.numpy as jnp

from alf.ddm.wiener import DDMParams


def efe_to_ddm(
    G: jnp.ndarray,
    gamma: float = 4.0,
    tau: float = 0.3,
    base_a: float = 1.5,
) -> DDMParams:
    """Map EFE values to DDM parameters for a binary choice.

    For a two-action decision, the EFE difference drives the drift rate
    and the policy precision sets the boundary separation.

    Args:
        G: Expected free energy for each action, shape (2,).
            G[0] = EFE of action 0 (lower boundary).
            G[1] = EFE of action 1 (upper boundary).
        gamma: Policy precision (inverse temperature).
        tau: Non-decision time.
        base_a: Base boundary separation (scaled by sqrt(gamma)).

    Returns:
        DDMParams with v, a, w, tau.
    """
    # Drift = precision-weighted EFE difference
    # Lower G is better, so drift toward the lower-G action
    delta_G = G[0] - G[1]  # Positive -> favor action 1 (upper)
    v = gamma * delta_G

    # Boundary separation scales with precision
    a = base_a * jnp.sqrt(gamma)

    # Unbiased starting point (could be informed by E)
    w = jnp.array(0.5)

    return DDMParams(v=v, a=a, w=w, tau=jnp.array(tau))


def neg_efe_to_ddm(
    neg_efe: jnp.ndarray,
    gamma: float = 4.0,
    tau: float = 0.3,
    base_a: float = 1.5,
) -> DDMParams:
    """Map pymdp neg_efe values to DDM parameters for a binary choice.

    Convenience wrapper for efe_to_ddm that accepts pymdp's neg_efe
    convention (higher = better) instead of ALF's G (lower = better).

    Args:
        neg_efe: Negative EFE for each action, shape (2,) or (1, 2).
            Higher values indicate preferred actions.
        gamma: Policy precision (inverse temperature).
        tau: Non-decision time.
        base_a: Base boundary separation.

    Returns:
        DDMParams with v, a, w, tau.
    """
    if neg_efe.ndim == 2:
        neg_efe = neg_efe[0]
    return efe_to_ddm(-neg_efe, gamma=gamma, tau=tau, base_a=base_a)


def ddm_to_policy_probs(
    params: DDMParams,
) -> jnp.ndarray:
    """Compute choice probabilities from DDM parameters analytically.

    Uses the closed-form solution for the Wiener process absorption
    probability: P(upper) = (1 - exp(-2*v*a*w)) / (1 - exp(-2*v*a)).

    Args:
        params: DDM parameters.

    Returns:
        Policy probabilities, shape (2,). [P(lower), P(upper)].
    """
    eps = 1e-16
    v, a, w = params.v, params.a, params.w

    # Handle v near zero via Taylor expansion
    va = v * a
    exponent_full = -2.0 * va
    exponent_bias = -2.0 * va * w

    # For small |v*a|, P(upper) -> w (unbiased diffusion)
    use_limit = jnp.abs(va) < 1e-6
    p_upper_exact = (
        (1.0 - jnp.exp(exponent_bias))
        / jnp.clip(1.0 - jnp.exp(exponent_full), eps)
    )
    p_upper = jnp.where(use_limit, w, p_upper_exact)
    p_upper = jnp.clip(p_upper, eps, 1.0 - eps)

    return jnp.array([1.0 - p_upper, p_upper])
