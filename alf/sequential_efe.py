"""Sequential Expected Free Energy for multi-step planning.

Implements analytic sequential EFE following Da Costa et al. (2020) and
Smith et al. (2022). Rolls forward through time steps sequentially,
capturing the *value of information* that emerges from conditional planning.

For a T-step policy pi = [a_0, a_1, ..., a_{T-1}]:

    G(pi) = sum_{t=0}^{T-1} G_t(pi)

where:
    G_t(pi) = -E_{Q(o_t|pi)}[ln P(o_t)]        (pragmatic)
              - E_{Q(s_t|pi)}[H[P(o_t|s_t)]]    (epistemic)

and beliefs are propagated SEQUENTIALLY:
    Q(s_0) = D                                    (prior)
    Q(s_{t+1}|pi) = B(a_t) @ Q(s_t|pi)          (predict next state)
    Q(o_t|pi) = A @ Q(s_t|pi)                    (predict observation)

References:
    Da Costa et al. (2020). Active Inference on Discrete State-Spaces:
        A Synthesis. Journal of Mathematical Psychology, 99, 102447.
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology, 107, 102632.
"""

from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# NumPy implementation
# ---------------------------------------------------------------------------


def sequential_efe(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    policy: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute sequential Expected Free Energy for a single policy.

    Args:
        A: Likelihood matrix P(o|s), shape (num_obs, num_states).
        B: Transition matrices P(s'|s,a), shape (num_states, num_states, num_actions).
        C: Log-preferences over observations, shape (num_obs,).
        D: Prior beliefs over initial states, shape (num_states,).
        policy: Action sequence, shape (T,) with integer action indices.
        gamma: Precision scaling (default 1.0).

    Returns:
        G: Total expected free energy (lower = better for the agent).
    """
    policy = np.asarray(policy).flatten()
    T = len(policy)

    log_A = np.log(np.clip(A, 1e-16, None))
    entropy_per_state = -np.sum(A * log_A, axis=0)

    G_total = 0.0
    q_s = D.copy()
    q_s = np.clip(q_s, 1e-16, None)
    q_s = q_s / q_s.sum()

    for t in range(T):
        action = int(policy[t])

        B_a = B[:, :, action]
        q_s_next = B_a @ q_s
        q_s_next = np.clip(q_s_next, 1e-16, None)
        q_s_next = q_s_next / q_s_next.sum()

        q_o = A @ q_s_next
        q_o = np.clip(q_o, 1e-16, None)
        q_o = q_o / q_o.sum()

        pragmatic = np.sum(q_o * C)
        epistemic = np.sum(q_s_next * entropy_per_state)

        G_total += -pragmatic - epistemic
        q_s = q_s_next

    return float(G_total)


def evaluate_all_policies_sequential(
    gm,
    beliefs: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Evaluate all policies using sequential EFE.

    Drop-in replacement for BP-based evaluate_policies() that uses the
    analytic sequential rollout.

    Args:
        gm: GenerativeModel instance with A, B, C, D, policies attributes.
        beliefs: Current beliefs over hidden states. If None, uses gm.D.

    Returns:
        G: Array of shape (num_policies,) with EFE for each policy.
    """
    if gm.num_factors != 1:
        raise ValueError(
            "evaluate_all_policies_sequential only supports single-factor models. "
            f"Got {gm.num_factors} factors."
        )
    if gm.num_modalities != 1:
        raise ValueError(
            "evaluate_all_policies_sequential only supports single-modality models. "
            f"Got {gm.num_modalities} modalities."
        )

    A = gm.A[0]
    B = gm.B[0]
    C = gm.C[0]
    D = beliefs[0] if beliefs is not None else gm.D[0]

    G = np.zeros(gm.num_policies)
    for i, policy in enumerate(gm.policies):
        action_seq = policy[:, 0]
        G[i] = sequential_efe(A, B, C, D, action_seq)

    return G


def select_action_sequential(
    gm,
    beliefs: Optional[list[np.ndarray]] = None,
    gamma: float = 4.0,
    E: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[int, dict]:
    """Select an action using sequential EFE for policy evaluation.

    Args:
        gm: GenerativeModel instance.
        beliefs: Current beliefs. If None, uses gm.D.
        gamma: Policy precision (inverse temperature).
        E: Policy prior (habits). If None, uses gm.E.
        rng: Random state for sampling. If None, uses numpy global RNG.

    Returns:
        Tuple of (action_index, info_dict).
    """
    from alf.policy import select_action

    G = evaluate_all_policies_sequential(gm, beliefs)

    if E is None:
        E = gm.E

    policy_idx, policy_probs = select_action(G, E, gamma=gamma, rng=rng)

    selected_policy = gm.policies[policy_idx]
    action = int(selected_policy[0, 0])

    return action, {
        "G": G,
        "policy_probs": policy_probs,
        "selected_policy_idx": policy_idx,
    }


# ---------------------------------------------------------------------------
# JAX implementation (jit/vmap compatible)
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:

    def jax_sequential_efe(
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        D: jnp.ndarray,
        policy: jnp.ndarray,
        gamma: float = 1.0,
    ) -> jnp.ndarray:
        """Compute sequential EFE using JAX (jit-compatible).

        Args:
            A: Likelihood matrix, shape (num_obs, num_states).
            B: Transition matrices, shape (num_states, num_states, num_actions).
            C: Log-preferences, shape (num_obs,).
            D: Prior beliefs, shape (num_states,).
            policy: Action sequence, shape (T,) with integer action indices.
            gamma: Precision scaling (default 1.0).

        Returns:
            Scalar EFE value.
        """
        log_A = jnp.log(jnp.clip(A, 1e-16))
        entropy_per_state = -jnp.sum(A * log_A, axis=0)

        q_s = jnp.clip(D, 1e-16)
        q_s = q_s / q_s.sum()

        def step_fn(q_s, action):
            B_a = B[:, :, action]
            q_s_next = B_a @ q_s
            q_s_next = jnp.clip(q_s_next, 1e-16)
            q_s_next = q_s_next / q_s_next.sum()

            q_o = A @ q_s_next
            q_o = jnp.clip(q_o, 1e-16)
            q_o = q_o / q_o.sum()

            pragmatic = jnp.sum(q_o * C)
            epistemic = jnp.sum(q_s_next * entropy_per_state)

            G_t = -pragmatic - epistemic

            return q_s_next, G_t

        _, G_per_step = jax.lax.scan(step_fn, q_s, policy)

        return jnp.sum(G_per_step)

    def jax_evaluate_all_policies_sequential(
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        D: jnp.ndarray,
        policies: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate all policies using sequential EFE (JAX, vmappable).

        Args:
            A: Likelihood matrix, shape (num_obs, num_states).
            B: Transition matrices, shape (num_states, num_states, num_actions).
            C: Log-preferences, shape (num_obs,).
            D: Prior beliefs, shape (num_states,).
            policies: All policies, shape (num_policies, T).

        Returns:
            G: EFE for each policy, shape (num_policies,).
        """

        def efe_for_policy(policy):
            return jax_sequential_efe(A, B, C, D, policy)

        return jax.vmap(efe_for_policy)(policies)

    def jax_select_action_sequential(
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        D: jnp.ndarray,
        policies: jnp.ndarray,
        E: jnp.ndarray,
        gamma: float,
        key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Select action using sequential EFE (JAX, jit-compatible).

        Args:
            A: Likelihood matrix, shape (num_obs, num_states).
            B: Transition matrices, shape (num_states, num_states, num_actions).
            C: Log-preferences, shape (num_obs,).
            D: Prior beliefs, shape (num_states,).
            policies: All policies, shape (num_policies, T).
            E: Policy prior, shape (num_policies,).
            gamma: Policy precision.
            key: JAX PRNG key.

        Returns:
            Tuple of (selected_action, policy_probs, G_values).
        """
        G = jax_evaluate_all_policies_sequential(A, B, C, D, policies)

        log_E = jnp.log(jnp.clip(E, 1e-16))
        log_posterior = -gamma * G + log_E
        log_posterior = log_posterior - jnp.max(log_posterior)
        policy_probs = jnp.exp(log_posterior)
        policy_probs = policy_probs / policy_probs.sum()

        selected_idx = jax.random.categorical(key, jnp.log(policy_probs))

        return selected_idx, policy_probs, G
