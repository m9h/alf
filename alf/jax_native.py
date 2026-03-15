"""JAX-native versions of policy functions for jit/vmap compatibility.

Provides pure-JAX implementations of the core Active Inference computations
so they can be compiled with jax.jit and vectorized with jax.vmap.

Usage:
    >>> import jax
    >>> from alf.jax_native import jax_select_action, jax_update_habits
    >>>
    >>> fast_select = jax.jit(jax_select_action)
    >>> action_idx, probs = fast_select(G, E, gamma, key)
    >>>
    >>> batch_select = jax.vmap(jax_select_action, in_axes=(0, 0, None, 0))
    >>> actions, probs = batch_select(G_batch, E_batch, gamma, keys)
"""

import jax
import jax.numpy as jnp
import numpy as np

from alf.sequential_efe import jax_evaluate_all_policies_sequential


def jax_softmax(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable softmax in JAX."""
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()


def jax_select_action(
    G: jnp.ndarray,
    E: jnp.ndarray,
    gamma: float,
    key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Select action from posterior over policies (JAX-native, jit-compatible).

    P(pi) = softmax(-gamma * G(pi) + ln E(pi))

    Args:
        G: Expected free energy, shape (num_policies,).
        E: Policy prior (habits), shape (num_policies,).
        gamma: Policy precision.
        key: JAX PRNG key.

    Returns:
        Tuple of (selected_policy_index, policy_probabilities).
    """
    log_E = jnp.log(jnp.clip(E, 1e-16))
    log_posterior = -gamma * G + log_E
    policy_probs = jax_softmax(log_posterior)
    selected = jax.random.categorical(key, jnp.log(policy_probs))
    return selected, policy_probs


def jax_update_habits(
    E: jnp.ndarray,
    selected_policy: int,
    outcome_valence: float,
    learning_rate: float = 0.1,
) -> jnp.ndarray:
    """Update habits (JAX-native, jit-compatible).

    Args:
        E: Current policy prior, shape (num_policies,).
        selected_policy: Index of executed policy.
        outcome_valence: How good the outcome was.
        learning_rate: Learning rate.

    Returns:
        Updated E vector (normalized).
    """
    E = E.at[selected_policy].add(learning_rate * outcome_valence)
    E = jnp.clip(E, 1e-8)
    return E / E.sum()


def jax_update_precision(
    gamma: float,
    prediction_error: float,
    learning_rate: float = 0.3,
    min_gamma: float = 0.1,
    max_gamma: float = 10.0,
) -> float:
    """Update precision (JAX-native, jit-compatible)."""
    target = 1.0 / (1.0 + prediction_error)
    gamma = (1.0 - learning_rate) * gamma + learning_rate * target
    return jnp.clip(gamma, min_gamma, max_gamma)


def jax_compute_efe_analytic(
    A: jnp.ndarray,
    B_action: jnp.ndarray,
    C: jnp.ndarray,
    beliefs: jnp.ndarray,
) -> float:
    """Compute EFE analytically for single-factor, single-modality models.

    G(a) = -E_Q(o')[ln P(o')] - E_Q(s')[H[P(o'|s')]]

    Args:
        A: Likelihood matrix, shape (num_obs, num_states).
        B_action: Transition matrix for this action, shape (num_states, num_states).
        C: Preferences, shape (num_obs,).
        beliefs: Current beliefs, shape (num_states,).

    Returns:
        G: Expected free energy for this action.
    """
    predicted_states = B_action @ beliefs

    predicted_obs = A @ predicted_states
    predicted_obs = jnp.clip(predicted_obs, 1e-16)
    predicted_obs = predicted_obs / predicted_obs.sum()

    pragmatic = jnp.sum(predicted_obs * C)

    log_A = jnp.log(jnp.clip(A, 1e-16))
    entropy_per_state = -jnp.sum(A * log_A, axis=0)
    epistemic = jnp.sum(predicted_states * entropy_per_state)

    return -pragmatic - epistemic


# Vectorized EFE over all actions at once
def jax_evaluate_all_actions(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    beliefs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute EFE for all actions simultaneously (JAX-native).

    Args:
        A: Likelihood, shape (num_obs, num_states).
        B: Transitions, shape (num_states, num_states, num_actions).
        C: Preferences, shape (num_obs,).
        beliefs: Beliefs, shape (num_states,).

    Returns:
        G: EFE for each action, shape (num_actions,).
    """
    num_actions = B.shape[-1]

    def efe_for_action(action_idx):
        B_a = B[:, :, action_idx]
        return jax_compute_efe_analytic(A, B_a, C, beliefs)

    return jax.vmap(lambda i: efe_for_action(i))(jnp.arange(num_actions))


class BatchAgent:
    """Vectorized Active Inference agent for batch simulation.

    Runs multiple agents with the same generative model but independent
    beliefs, habits, and evidence. Uses jax.vmap for parallel inference.

    Args:
        gm: Shared generative model.
        batch_size: Number of agents.
        gamma: Policy precision (shared).
        seed: Random seed.
    """

    def __init__(self, gm, batch_size: int, gamma: float = 4.0, seed: int = 42):
        self.gm = gm
        self.batch_size = batch_size
        self.gamma = gamma
        self.key = jax.random.PRNGKey(seed)

        self.beliefs = [
            jnp.tile(jnp.array(d), (batch_size, 1))
            for d in gm.D
        ]

        self.E = jnp.tile(jnp.array(gm.E), (batch_size, 1))

        self.A_jax = [jnp.array(a) for a in gm.A]
        self.B_jax = [jnp.array(b) for b in gm.B]
        self.C_jax = [jnp.array(c) for c in gm.C]

        # Pre-compute policies as JAX array for sequential EFE evaluation.
        # policies shape: (num_policies, T, num_factors) -> extract single
        # factor action sequences: (num_policies, T)
        self.policies_jax = jnp.array(gm.policies[:, :, 0])

    def step_analytic(
        self,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Perform one step for all agents using analytic EFE.

        Evaluates all policies (multi-step action sequences) using sequential
        EFE, matching the approach in AnalyticAgent.step(). For T=1 models
        this is equivalent to single-step EFE per action; for T>1 it correctly
        evaluates all T-step policies.

        Args:
            observations: Observation indices, shape (batch_size,).

        Returns:
            Tuple of (actions, info_dict).
        """
        if self.gm.num_factors != 1 or self.gm.num_modalities != 1:
            raise ValueError("step_analytic only supports single-factor, single-modality")

        A = self.A_jax[0]
        B = self.B_jax[0]
        C = self.C_jax[0]
        obs = jnp.array(observations)

        def update_belief(belief, obs_idx):
            likelihood = A[obs_idx, :]
            posterior = belief * likelihood
            posterior = jnp.clip(posterior, 1e-16)
            return posterior / posterior.sum()

        self.beliefs[0] = jax.vmap(update_belief)(self.beliefs[0], obs)

        # Evaluate EFE for all policies (not just single actions).
        # G_batch shape: (batch_size, num_policies)
        G_batch = jax.vmap(
            lambda b: jax_evaluate_all_policies_sequential(
                A, B, C, b, self.policies_jax
            )
        )(self.beliefs[0])

        self.key, *subkeys = jax.random.split(self.key, self.batch_size + 1)
        keys = jnp.stack(subkeys)

        # Select policy from posterior over policies.
        # selected_indices shape: (batch_size,) -- indices into policies
        selected_indices, probs = jax.vmap(
            jax_select_action, in_axes=(0, 0, None, 0)
        )(G_batch, self.E, self.gamma, keys)

        # Extract the first action from each selected policy
        actions = self.policies_jax[selected_indices, 0]

        return np.array(actions), {
            "beliefs": [np.array(b) for b in self.beliefs],
            "G": np.array(G_batch),
            "policy_probs": np.array(probs),
        }

    def learn_batch(self, outcomes: np.ndarray, actions: np.ndarray) -> None:
        """Update habits for all agents.

        Args:
            outcomes: Outcome valences, shape (batch_size,).
            actions: Actions taken, shape (batch_size,).
        """
        self.E = jax.vmap(
            jax_update_habits, in_axes=(0, 0, 0, None)
        )(self.E, jnp.array(actions), jnp.array(outcomes), 0.1)

    def reset(self) -> None:
        """Reset all agents to prior beliefs."""
        self.beliefs = [
            jnp.tile(jnp.array(d), (self.batch_size, 1))
            for d in self.gm.D
        ]
