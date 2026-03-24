"""Differentiable learning of A and B matrices via forward algorithm.

Learn the likelihood (A) and transition (B) matrices from observation-action
data using gradient descent. Because all computations are JAX-native, we can
differentiate through the forward algorithm with jax.grad.

Two learning paths:

1. **Analytic path** (forward filtering): Computes the negative log-likelihood
   of an observation sequence analytically using the HMM forward algorithm.

2. **Deep path** (see deep_aif.py): Uses neural networks to parameterize
   A and B, scaling to high-dimensional observation spaces.

The core idea:
    - Parameterize A and B in unconstrained space (log-space before softmax)
    - Map to valid probability matrices via column-wise softmax
    - Compute NLL of observed data given current parameters
    - Use jax.grad to get gradients of NLL w.r.t. parameters
    - Update parameters via gradient descent (optax or manual SGD)

References:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
    Tschantz, Millidge et al. (2020). Reinforcement Learning through
        Active Inference. arXiv:2002.12636.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from alf.generative_model import GenerativeModel


# ---------------------------------------------------------------------------
# Learnable generative model
# ---------------------------------------------------------------------------


class LearnableParams(NamedTuple):
    """Unconstrained parameters that map to valid A and B matrices."""

    log_A_params: list[jnp.ndarray]
    log_B_params: list[jnp.ndarray]


class LearnableGenerativeModel:
    """Wraps a GenerativeModel with differentiable JAX parameters."""

    def __init__(
        self,
        log_A_params: list[jnp.ndarray],
        log_B_params: list[jnp.ndarray],
        C: list[np.ndarray],
        D: list[np.ndarray],
    ):
        self.params = LearnableParams(
            log_A_params=log_A_params,
            log_B_params=log_B_params,
        )
        self.C = [jnp.array(c) for c in C]
        self.D = [jnp.array(d) for d in D]

        self.num_modalities = len(log_A_params)
        self.num_factors = len(log_B_params)
        self.num_obs = [p.shape[0] for p in log_A_params]
        self.num_states = [p.shape[0] for p in log_B_params]
        self.num_actions = [p.shape[-1] for p in log_B_params]

    @classmethod
    def from_model(cls, gm: GenerativeModel) -> "LearnableGenerativeModel":
        """Initialize learnable parameters from an existing GenerativeModel."""
        log_A_params = [jnp.array(np.log(np.clip(a, 1e-16, None))) for a in gm.A]
        log_B_params = [jnp.array(np.log(np.clip(b, 1e-16, None))) for b in gm.B]
        return cls(
            log_A_params=log_A_params,
            log_B_params=log_B_params,
            C=[c.copy() for c in gm.C],
            D=[d.copy() for d in gm.D],
        )


# ---------------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------------


def params_to_A(log_A_params: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Convert unconstrained log-parameters to valid likelihood matrices."""
    return [jax.nn.softmax(p, axis=0) for p in log_A_params]


def params_to_B(log_B_params: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Convert unconstrained log-parameters to valid transition matrices."""
    return [jax.nn.softmax(p, axis=0) for p in log_B_params]


def params_to_matrices(
    params: LearnableParams,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Convert all parameters to valid probability matrices."""
    return params_to_A(params.log_A_params), params_to_B(params.log_B_params)


# ---------------------------------------------------------------------------
# Analytic negative log-likelihood via forward filtering
# ---------------------------------------------------------------------------


def _forward_filter_step(
    carry: tuple[jnp.ndarray, jnp.ndarray],
    x: tuple[jnp.ndarray, jnp.ndarray],
    A: jnp.ndarray,
    B: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """One step of the HMM forward algorithm."""
    alpha_prev, _ = carry
    obs_idx, action_idx = x

    B_a = B[:, :, action_idx]
    predicted = B_a @ alpha_prev
    predicted = jnp.clip(predicted, 1e-16)

    likelihood = A[obs_idx, :]

    alpha_unnorm = likelihood * predicted
    evidence = jnp.sum(alpha_unnorm)
    evidence = jnp.clip(evidence, 1e-16)
    alpha_new = alpha_unnorm / evidence

    log_evidence = jnp.log(evidence)

    return (alpha_new, jnp.zeros(())), log_evidence


def analytic_nll(
    log_A_params: list[jnp.ndarray],
    log_B_params: list[jnp.ndarray],
    D: list[jnp.ndarray],
    observations: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute negative log-likelihood analytically via forward filtering.

    Args:
        log_A_params: Unconstrained A parameters (list, one per modality).
        log_B_params: Unconstrained B parameters (list, one per factor).
        D: Prior over initial states (list, one per factor).
        observations: Observation indices, shape (T,).
        actions: Action indices, shape (T,).

    Returns:
        Scalar negative log-likelihood.
    """
    A_list = params_to_A(log_A_params)
    B_list = params_to_B(log_B_params)

    A = A_list[0]
    B = B_list[0]
    prior = D[0]

    first_obs = observations[0]
    likelihood_0 = A[first_obs, :]
    alpha_0_unnorm = likelihood_0 * prior
    evidence_0 = jnp.sum(alpha_0_unnorm)
    evidence_0 = jnp.clip(evidence_0, 1e-16)
    alpha_0 = alpha_0_unnorm / evidence_0
    log_evidence_0 = jnp.log(evidence_0)

    def scan_step(carry, x):
        return _forward_filter_step(carry, x, A, B)

    remaining_obs = observations[1:]
    transition_actions = actions[:-1]

    init_carry = (alpha_0, jnp.zeros(()))
    _, log_evidences = jax.lax.scan(
        scan_step,
        init_carry,
        (remaining_obs, transition_actions),
    )

    total_log_likelihood = log_evidence_0 + jnp.sum(log_evidences)

    return -total_log_likelihood


def analytic_nll_single(
    log_A_params: jnp.ndarray,
    log_B_params: jnp.ndarray,
    D: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Convenience wrapper for single-factor, single-modality models."""
    return analytic_nll(
        [log_A_params],
        [log_B_params],
        [D],
        observations,
        actions,
    )


# ---------------------------------------------------------------------------
# Gradient-based learning
# ---------------------------------------------------------------------------


class LearningResult(NamedTuple):
    """Result of gradient-based model learning."""

    learned_A: list[np.ndarray]
    learned_B: list[np.ndarray]
    log_A_params: list[jnp.ndarray]
    log_B_params: list[jnp.ndarray]
    loss_history: list[float]
    generative_model: GenerativeModel


def _try_import_optax():
    """Try to import optax, return None if unavailable."""
    try:
        import optax

        return optax
    except ImportError:
        return None


def learn_model(
    observations: np.ndarray,
    actions: np.ndarray,
    num_obs: int,
    num_states: int,
    num_actions: int,
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    num_epochs: int = 100,
    lr: float = 0.01,
    init_log_A: Optional[jnp.ndarray] = None,
    init_log_B: Optional[jnp.ndarray] = None,
    verbose: bool = False,
) -> LearningResult:
    """Learn A and B matrices from observation-action sequences.

    Args:
        observations: Observation indices, shape (T,).
        actions: Action indices, shape (T,).
        num_obs: Number of possible observations.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        D: Prior over initial states. Default: uniform.
        C: Observation preferences for the returned model. Default: uniform.
        num_epochs: Number of gradient descent steps.
        lr: Learning rate.
        init_log_A: Initial unconstrained A parameters.
        init_log_B: Initial unconstrained B parameters.
        verbose: If True, print loss every 10 epochs.

    Returns:
        LearningResult with learned matrices and training history.
    """
    obs_jnp = jnp.array(observations, dtype=jnp.int32)
    act_jnp = jnp.array(actions, dtype=jnp.int32)

    if D is None:
        D = np.ones(num_states) / num_states
    if C is None:
        C = np.zeros(num_obs)

    D_jnp = jnp.array(D)

    if init_log_A is None:
        key = jax.random.PRNGKey(0)
        init_log_A = 0.1 * jax.random.normal(key, (num_obs, num_states))
    if init_log_B is None:
        key = jax.random.PRNGKey(1)
        init_log_B = 0.1 * jax.random.normal(key, (num_states, num_states, num_actions))

    def loss_fn(log_A, log_B):
        return analytic_nll_single(log_A, log_B, D_jnp, obs_jnp, act_jnp)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    optax = _try_import_optax()

    if optax is not None:
        optimizer = optax.adam(lr)
        opt_state = optimizer.init((init_log_A, init_log_B))

        log_A = init_log_A
        log_B = init_log_B
        loss_history = []

        @jax.jit
        def update_step(log_A, log_B, opt_state):
            loss = loss_fn(log_A, log_B)
            grads = grad_fn(log_A, log_B)
            updates, new_opt_state = optimizer.update(grads, opt_state, (log_A, log_B))
            new_log_A = log_A + updates[0]
            new_log_B = log_B + updates[1]
            return new_log_A, new_log_B, new_opt_state, loss

        for epoch in range(num_epochs):
            log_A, log_B, opt_state, loss = update_step(log_A, log_B, opt_state)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    else:
        log_A = init_log_A
        log_B = init_log_B
        loss_history = []

        @jax.jit
        def sgd_step(log_A, log_B):
            loss = loss_fn(log_A, log_B)
            grad_A, grad_B = grad_fn(log_A, log_B)
            new_log_A = log_A - lr * grad_A
            new_log_B = log_B - lr * grad_B
            return new_log_A, new_log_B, loss

        for epoch in range(num_epochs):
            log_A, log_B, loss = sgd_step(log_A, log_B)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    A_learned = params_to_A([log_A])
    B_learned = params_to_B([log_B])

    gm = GenerativeModel(
        A=[np.array(A_learned[0])],
        B=[np.array(B_learned[0])],
        C=[np.array(C)],
        D=[np.array(D)],
        T=1,
    )

    return LearningResult(
        learned_A=[np.array(a) for a in A_learned],
        learned_B=[np.array(b) for b in B_learned],
        log_A_params=[log_A],
        log_B_params=[log_B],
        loss_history=loss_history,
        generative_model=gm,
    )


def learn_from_agent_data(
    observation_history: list[list[int]],
    action_history: list[int],
    num_obs: int,
    num_states: int,
    num_actions: int,
    D: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> LearningResult:
    """Learn A/B matrices from an agent's history.

    Args:
        observation_history: List of observation lists (one per timestep).
        action_history: List of action indices taken at each timestep.
        num_obs: Number of possible observations (first modality).
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        D: Prior over initial states. Default: uniform.
        C: Observation preferences. Default: zeros.
        num_epochs: Number of gradient descent steps.
        lr: Learning rate.
        verbose: If True, print training progress.

    Returns:
        LearningResult with learned model.
    """
    if isinstance(observation_history[0], (list, tuple)):
        obs = np.array([o[0] for o in observation_history])
    else:
        obs = np.array(observation_history)

    acts = np.array(action_history)

    min_len = min(len(obs), len(acts) + 1)
    obs = obs[:min_len]
    acts = acts[: min_len - 1]

    acts_padded = np.concatenate([acts, [0]])

    return learn_model(
        observations=obs,
        actions=acts_padded,
        num_obs=num_obs,
        num_states=num_states,
        num_actions=num_actions,
        D=D,
        C=C,
        num_epochs=num_epochs,
        lr=lr,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Data generation utilities
# ---------------------------------------------------------------------------


def generate_data(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    actions: np.ndarray,
    key: Optional[jax.random.PRNGKey] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate observation data from a known POMDP model.

    Args:
        A: Likelihood matrix, shape (num_obs, num_states).
        B: Transition matrix, shape (num_states, num_states, num_actions).
        D: Initial state distribution, shape (num_states,).
        actions: Action sequence, shape (T,).
        key: JAX PRNG key. If None, uses seed.
        seed: Random seed (used if key is None).

    Returns:
        Tuple of (observations, states) arrays, each shape (T+1,).
    """
    if key is None:
        key = jax.random.PRNGKey(seed)

    T = len(actions)

    states = np.zeros(T + 1, dtype=np.int32)
    observations = np.zeros(T + 1, dtype=np.int32)

    key, subkey = jax.random.split(key)
    states[0] = int(jax.random.categorical(subkey, jnp.log(jnp.array(D))))

    key, subkey = jax.random.split(key)
    observations[0] = int(
        jax.random.categorical(subkey, jnp.log(jnp.array(A[:, states[0]])))
    )

    for t in range(T):
        key, subkey = jax.random.split(key)
        trans_probs = B[:, states[t], actions[t]]
        states[t + 1] = int(
            jax.random.categorical(subkey, jnp.log(jnp.array(trans_probs)))
        )

        key, subkey = jax.random.split(key)
        obs_probs = A[:, states[t + 1]]
        observations[t + 1] = int(
            jax.random.categorical(subkey, jnp.log(jnp.array(obs_probs)))
        )

    return observations, states


# ---------------------------------------------------------------------------
# Log-likelihood wrapper
# ---------------------------------------------------------------------------


def compute_observation_log_likelihood_analytic(
    params: LearnableParams,
    D: list[jnp.ndarray],
    observations: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log P(observations | model) using the analytic forward filter."""
    nll = analytic_nll(
        params.log_A_params,
        params.log_B_params,
        D,
        observations,
        actions,
    )
    return -nll
