"""Variational Free Energy and Expected Free Energy computations.

Provides explicit computation of the Variational Free Energy (VFE) that
belief propagation minimizes, plus decomposed Expected Free Energy (EFE)
and the generalized free energy of Parr & Friston (2019).

The Variational Free Energy is:

    F = E_q[ln q(s) - ln p(o, s)]
      = sum_s q(s) * [ln q(s) - ln p(o|s) - ln p(s)]
      = -ELBO

The Expected Free Energy G(pi) decomposes into:
    G(pi) = -E_q[ln P(o|C)]  (pragmatic value)
            - E_q[H[P(o|s)]]  (epistemic value / ambiguity reduction)

The generalized free energy (Parr & Friston, 2019) unifies past inference
(VFE) and future planning (EFE) into a single objective.

Both numpy and JAX-native (jit/vmap compatible) versions are provided.

References:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
    Parr & Friston (2019). Generalised free energy and active inference.
        Biological Cybernetics.
    Da Costa et al. (2020). Active Inference on Discrete State-Spaces:
        A Synthesis. Journal of Mathematical Psychology.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from alf.generative_model import GenerativeModel
from alf.jax_core import entropy as _entropy, safe_log, safe_normalize


# ---------------------------------------------------------------------------
# Named tuple for EFE decomposition
# ---------------------------------------------------------------------------

class EFEDecomposition(NamedTuple):
    """Decomposed Expected Free Energy.

    Attributes:
        G_total: Total expected free energy G(a).
        pragmatic: Pragmatic value (preference satisfaction).
            E_q[ln P(o|C)] — how well predicted observations match preferences.
        epistemic: Epistemic value (information gain / ambiguity reduction).
            -E_q[H[P(o|s)]] — negative expected ambiguity.
    """
    G_total: float
    pragmatic: float
    epistemic: float


# ---------------------------------------------------------------------------
# 1. Variational Free Energy (VFE) — numpy version
# ---------------------------------------------------------------------------

def variational_free_energy(
    q_s: np.ndarray,
    A: np.ndarray,
    prior_s: np.ndarray,
    observation: int,
) -> float:
    """Compute Variational Free Energy for a single timestep.

    F = sum_s q(s) * [ln q(s) - ln P(o|s) - ln P(s)]

    Args:
        q_s: Approximate posterior over states, shape (num_states,).
        A: Likelihood matrix P(o|s), shape (num_obs, num_states).
        prior_s: Prior distribution over states P(s), shape (num_states,).
        observation: Observed outcome index (integer).

    Returns:
        F: Variational free energy (scalar).
    """
    eps = 1e-16
    q_s = np.asarray(q_s, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    prior_s = np.asarray(prior_s, dtype=np.float64)

    log_q = np.log(np.clip(q_s, eps, None))
    log_likelihood = np.log(np.clip(A[observation, :], eps, None))
    log_prior = np.log(np.clip(prior_s, eps, None))

    F = np.sum(q_s * (log_q - log_likelihood - log_prior))

    return float(F)


# ---------------------------------------------------------------------------
# 2. Free energy from beliefs (wrapper for GenerativeModel)
# ---------------------------------------------------------------------------

def free_energy_from_beliefs(
    gm: GenerativeModel,
    beliefs: list[np.ndarray],
    observations: list[int],
) -> float:
    """Compute VFE given a GenerativeModel, beliefs, and observations.

    Args:
        gm: The generative model with A, D matrices.
        beliefs: Current posterior beliefs (list of arrays, one per factor).
        observations: Observed outcomes, list of indices (one per modality).

    Returns:
        F: Total variational free energy (scalar).
    """
    F_total = 0.0

    for f in range(gm.num_factors):
        q_s = beliefs[f]
        prior_s = gm.D[f]

        for m in range(gm.num_modalities):
            a_matrix = gm.A[m]
            obs = observations[m]

            if a_matrix.ndim == 2:
                F_total += variational_free_energy(q_s, a_matrix, prior_s, obs)
            else:
                raise NotImplementedError(
                    "Multi-factor tensor A matrices not yet supported. "
                    "A matrix has ndim={}, expected 2.".format(a_matrix.ndim)
                )

    return F_total


# ---------------------------------------------------------------------------
# 3. Expected Free Energy decomposition — numpy version
# ---------------------------------------------------------------------------

def expected_free_energy_decomposed(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beliefs: np.ndarray,
    action: int,
) -> EFEDecomposition:
    """Decompose EFE G(a) into pragmatic and epistemic components.

    Args:
        A: Likelihood matrix P(o|s), shape (num_obs, num_states).
        B: Transition matrix P(s'|s, a), shape (num_states, num_states, num_actions).
        C: Log-preferences over observations, shape (num_obs,).
        beliefs: Current beliefs Q(s), shape (num_states,).
        action: Action index.

    Returns:
        EFEDecomposition with G_total, pragmatic, and epistemic components.
    """
    eps = 1e-16
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    beliefs = np.asarray(beliefs, dtype=np.float64)

    # Predicted next state: Q(s') = B[:, :, a] @ Q(s)
    B_a = B[:, :, action]
    predicted_states = B_a @ beliefs
    predicted_states = np.clip(predicted_states, eps, None)
    predicted_states = predicted_states / predicted_states.sum()

    # Predicted observations: Q(o') = A @ Q(s')
    predicted_obs = A @ predicted_states
    predicted_obs = np.clip(predicted_obs, eps, None)
    predicted_obs = predicted_obs / predicted_obs.sum()

    # Pragmatic value: E_Q(o')[C(o')] = sum_o Q(o') * C(o')
    pragmatic = float(np.sum(predicted_obs * C))

    # Epistemic value: -E_Q(s')[H[P(o|s)]]
    log_A = np.log(np.clip(A, eps, None))
    entropy_per_state = -np.sum(A * log_A, axis=0)
    epistemic = float(-np.sum(predicted_states * entropy_per_state))

    G_total = -pragmatic - epistemic

    return EFEDecomposition(
        G_total=G_total,
        pragmatic=pragmatic,
        epistemic=epistemic,
    )


# ---------------------------------------------------------------------------
# 4. Generalized Free Energy (Parr & Friston, 2019)
# ---------------------------------------------------------------------------

def generalized_free_energy(
    q_s: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    prior_s: np.ndarray,
    observation: int,
    policy: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute generalized free energy combining VFE (past) and EFE (future).

    F_gen = F_VFE(past) + gamma * sum_t G(a_t, future)

    Args:
        q_s: Current posterior beliefs Q(s), shape (num_states,).
        A: Likelihood matrix P(o|s), shape (num_obs, num_states).
        B: Transition matrix P(s'|s, a), shape (num_states, num_states, num_actions).
        C: Log-preferences over observations, shape (num_obs,).
        prior_s: Prior over states P(s), shape (num_states,).
        observation: Current observation index.
        policy: Future action sequence, shape (T,) or empty array for T=0.
        gamma: Weight on future (EFE) component. Default 1.0.

    Returns:
        F_gen: Generalized free energy (scalar).
    """
    # Past component: VFE
    F_vfe = variational_free_energy(q_s, A, prior_s, observation)

    # Future component: sum of EFE over policy steps
    policy = np.asarray(policy).flatten()
    T = len(policy)

    if T == 0:
        return F_vfe

    F_efe_total = 0.0
    current_beliefs = np.asarray(q_s, dtype=np.float64).copy()
    eps = 1e-16

    for t in range(T):
        action = int(policy[t])

        decomp = expected_free_energy_decomposed(
            A, B, C, current_beliefs, action
        )
        F_efe_total += decomp.G_total

        # Propagate beliefs forward
        B_a = B[:, :, action]
        current_beliefs = B_a @ current_beliefs
        current_beliefs = np.clip(current_beliefs, eps, None)
        current_beliefs = current_beliefs / current_beliefs.sum()

    return F_vfe + gamma * F_efe_total


# =========================================================================
# 5. JAX-native versions (jit-compatible, vmap-compatible)
# =========================================================================

def jax_variational_free_energy(
    q_s: jnp.ndarray,
    A: jnp.ndarray,
    prior_s: jnp.ndarray,
    observation: jnp.ndarray,
) -> jnp.ndarray:
    """JAX-native VFE computation (jit/vmap compatible).

    Args:
        q_s: Approximate posterior, shape (num_states,).
        A: Likelihood matrix, shape (num_obs, num_states).
        prior_s: Prior over states, shape (num_states,).
        observation: Observation index (scalar integer).

    Returns:
        F: Variational free energy (scalar).
    """
    log_q = safe_log(q_s)
    log_likelihood = safe_log(A[observation, :])
    log_prior = safe_log(prior_s)

    F = jnp.sum(q_s * (log_q - log_likelihood - log_prior))
    return F


def jax_expected_free_energy_decomposed(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    beliefs: jnp.ndarray,
    action: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-native EFE decomposition (jit/vmap compatible).

    Returns (G_total, pragmatic, epistemic) as separate arrays
    rather than a NamedTuple for JAX tracing compatibility.

    Args:
        A: Likelihood matrix, shape (num_obs, num_states).
        B: Transition matrix, shape (num_states, num_states, num_actions).
        C: Log-preferences, shape (num_obs,).
        beliefs: Current beliefs, shape (num_states,).
        action: Action index (scalar integer).

    Returns:
        Tuple of (G_total, pragmatic, epistemic), each a scalar.
    """
    # Predicted next state
    B_a = B[:, :, action]
    predicted_states = safe_normalize(jnp.maximum(B_a @ beliefs, 1e-16))

    # Predicted observations
    predicted_obs = safe_normalize(jnp.maximum(A @ predicted_states, 1e-16))

    # Pragmatic value
    pragmatic = jnp.sum(predicted_obs * C)

    # Epistemic value
    entropy_per_state = _entropy(A, axis=0)
    epistemic = -jnp.sum(predicted_states * entropy_per_state)

    G_total = -pragmatic - epistemic

    return G_total, pragmatic, epistemic


def jax_generalized_free_energy(
    q_s: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    prior_s: jnp.ndarray,
    observation: jnp.ndarray,
    policy: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """JAX-native generalized free energy (jit compatible).

    Uses jax.lax.scan to unroll the future EFE computation.

    Args:
        q_s: Current posterior, shape (num_states,).
        A: Likelihood matrix, shape (num_obs, num_states).
        B: Transition matrix, shape (num_states, num_states, num_actions).
        C: Log-preferences, shape (num_obs,).
        prior_s: Prior over states, shape (num_states,).
        observation: Current observation index (scalar integer).
        policy: Future action sequence, shape (T,).
        gamma: Weight on future component.

    Returns:
        F_gen: Generalized free energy (scalar).
    """
    # Past: VFE
    F_vfe = jax_variational_free_energy(q_s, A, prior_s, observation)

    # Future: sum of EFE over policy using scan
    def scan_step(beliefs, action):
        G, _, _ = jax_expected_free_energy_decomposed(A, B, C, beliefs, action)

        B_a = B[:, :, action]
        next_beliefs = safe_normalize(jnp.maximum(B_a @ beliefs, 1e-16))

        return next_beliefs, G

    _, G_steps = jax.lax.scan(scan_step, q_s, policy)
    F_efe_total = jnp.sum(G_steps)

    return F_vfe + gamma * F_efe_total


# Convenience: vmap-ready VFE over a batch of beliefs
def jax_vfe_batch(
    q_s_batch: jnp.ndarray,
    A: jnp.ndarray,
    prior_s: jnp.ndarray,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Compute VFE for a batch of beliefs (vmap over batch dim).

    Args:
        q_s_batch: Batch of posteriors, shape (batch, num_states).
        A: Likelihood matrix, shape (num_obs, num_states).
        prior_s: Prior, shape (num_states,).
        observations: Observation indices, shape (batch,).

    Returns:
        F_batch: VFE for each element, shape (batch,).
    """
    return jax.vmap(
        jax_variational_free_energy, in_axes=(0, None, None, 0)
    )(q_s_batch, A, prior_s, observations)


# Convenience: vmap-ready EFE decomposition over actions
def jax_efe_all_actions(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    beliefs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute decomposed EFE for all actions (vmap over action dim).

    Args:
        A: Likelihood, shape (num_obs, num_states).
        B: Transitions, shape (num_states, num_states, num_actions).
        C: Preferences, shape (num_obs,).
        beliefs: Beliefs, shape (num_states,).

    Returns:
        Tuple of (G_all, pragmatic_all, epistemic_all), each shape (num_actions,).
    """
    num_actions = B.shape[-1]
    return jax.vmap(
        lambda a: jax_expected_free_energy_decomposed(A, B, C, beliefs, a)
    )(jnp.arange(num_actions))
