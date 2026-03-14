"""Deep Active Inference: neural network parameterized A and B matrices.

Replace flat log-parameter arrays with neural network encoders, enabling
Active Inference to scale to high-dimensional observation spaces.

Architecture overview:

    Encoder:  obs_raw  -->  MLP  -->  state_logits  -->  softmax  -->  Q(s|o)
    Transition:  (state_beliefs, action_onehot)  -->  MLP  -->  next_state_logits

Key design decisions:
    - Pure JAX (jax.numpy, jax.random) — no Flax, Haiku, or Equinox dependency.
    - Network parameters stored as lists of (weight, bias) tuples for
      transparent pytree handling by jax.grad and jax.jit.
    - Compatible with the existing analytic forward-filter NLL framework
      from learning.py.

References:
    Fountas et al. (2020). Deep Active Inference Agents Using Monte-Carlo
        Methods. NeurIPS.
    Millidge, Tschantz & Buckley (2020). Whence the Expected Free Energy?
        Neural Computation.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases for network parameters
# ---------------------------------------------------------------------------

LayerParams = tuple[jnp.ndarray, jnp.ndarray]
NetworkParams = list[LayerParams]


# ---------------------------------------------------------------------------
# Encoder Network: observations -> state belief logits
# ---------------------------------------------------------------------------

def init_encoder(
    key: jax.random.PRNGKey,
    obs_dim: int,
    hidden_dims: list[int],
    num_states: int,
) -> NetworkParams:
    """Initialize an encoder MLP that maps observations to state logits.

    Args:
        key: JAX PRNG key for random initialization.
        obs_dim: Dimensionality of the raw observation vector.
        hidden_dims: List of hidden layer widths.
        num_states: Number of hidden states (output dimension).

    Returns:
        NetworkParams: list of (weight, bias) tuples.
    """
    params = []
    dims = [obs_dim] + list(hidden_dims) + [num_states]

    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in = dims[i]
        fan_out = dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = std * jax.random.normal(subkey, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((w, b))

    return params


def encode(params: NetworkParams, obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through the encoder network.

    Args:
        params: Encoder network parameters from init_encoder.
        obs: Observation vector, shape (obs_dim,).

    Returns:
        State logits, shape (num_states,). Unnormalized.
    """
    x = obs
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


# ---------------------------------------------------------------------------
# Transition Network: (state_beliefs, action_onehot) -> next_state logits
# ---------------------------------------------------------------------------

def init_transition(
    key: jax.random.PRNGKey,
    num_states: int,
    num_actions: int,
    hidden_dims: list[int],
) -> NetworkParams:
    """Initialize a transition MLP.

    Args:
        key: JAX PRNG key.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        hidden_dims: List of hidden layer widths.

    Returns:
        NetworkParams: list of (weight, bias) tuples.
    """
    input_dim = num_states + num_actions
    params = []
    dims = [input_dim] + list(hidden_dims) + [num_states]

    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in = dims[i]
        fan_out = dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = std * jax.random.normal(subkey, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((w, b))

    return params


def predict_transition(
    params: NetworkParams,
    state_beliefs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> jnp.ndarray:
    """Predict next-state logits from current beliefs and action.

    Args:
        params: Transition network parameters from init_transition.
        state_beliefs: Current state beliefs, shape (num_states,).
        action_onehot: One-hot action vector, shape (num_actions,).

    Returns:
        Next-state logits, shape (num_states,). Unnormalized.
    """
    x = jnp.concatenate([state_beliefs, action_onehot])
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


# ---------------------------------------------------------------------------
# Deep Generative Model
# ---------------------------------------------------------------------------

class DeepGenerativeModel:
    """Wraps encoder and transition networks for Active Inference.

    Instead of explicit A and B matrices, this produces equivalent
    quantities on-the-fly using neural networks.
    """

    def __init__(
        self,
        obs_dim: int,
        num_states: int,
        num_actions: int,
        encoder_hidden: list[int] = [32, 32],
        transition_hidden: list[int] = [32, 32],
        seed: int = 0,
    ):
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)

        self.obs_dim = obs_dim
        self.num_states = num_states
        self.num_actions = num_actions

        self.encoder_params = init_encoder(key1, obs_dim, encoder_hidden, num_states)
        self.transition_params = init_transition(
            key2, num_states, num_actions, transition_hidden
        )

    @staticmethod
    def get_likelihood(
        encoder_params: NetworkParams,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute P(s|o) from the encoder network.

        Args:
            encoder_params: Encoder network parameters.
            obs: Observation vector, shape (obs_dim,).

        Returns:
            State belief distribution P(s|o), shape (num_states,).
        """
        logits = encode(encoder_params, obs)
        return jax.nn.softmax(logits)

    @staticmethod
    def get_transition(
        transition_params: NetworkParams,
        state_beliefs: jnp.ndarray,
        action: int,
        num_actions: int,
    ) -> jnp.ndarray:
        """Compute P(s'|beliefs, a) from the transition network.

        Args:
            transition_params: Transition network parameters.
            state_beliefs: Current state beliefs, shape (num_states,).
            action: Action index (integer).
            num_actions: Total number of actions (for one-hot encoding).

        Returns:
            Next-state distribution, shape (num_states,).
        """
        action_onehot = jax.nn.one_hot(action, num_actions)
        logits = predict_transition(transition_params, state_beliefs, action_onehot)
        return jax.nn.softmax(logits)


# ---------------------------------------------------------------------------
# Deep Analytic NLL
# ---------------------------------------------------------------------------

def _deep_forward_step(
    carry: tuple[jnp.ndarray, jnp.ndarray],
    x: tuple[jnp.ndarray, jnp.ndarray],
    encoder_params: NetworkParams,
    transition_params: NetworkParams,
    num_actions: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """One step of forward filtering with deep likelihood/transition."""
    alpha_prev, _ = carry
    obs_vec, action_idx = x

    action_onehot = jax.nn.one_hot(action_idx, num_actions)
    trans_logits = predict_transition(transition_params, alpha_prev, action_onehot)
    predicted = jax.nn.softmax(trans_logits)
    predicted = jnp.clip(predicted, 1e-16)

    enc_logits = encode(encoder_params, obs_vec)
    likelihood = jax.nn.softmax(enc_logits)

    alpha_unnorm = likelihood * predicted
    evidence = jnp.sum(alpha_unnorm)
    evidence = jnp.clip(evidence, 1e-16)
    alpha_new = alpha_unnorm / evidence

    log_evidence = jnp.log(evidence)

    return (alpha_new, jnp.zeros(())), log_evidence


def deep_analytic_nll(
    encoder_params: NetworkParams,
    transition_params: NetworkParams,
    D: jnp.ndarray,
    observations_raw: jnp.ndarray,
    actions: jnp.ndarray,
    num_actions: int,
) -> jnp.ndarray:
    """Compute NLL using neural network encoder and transition models.

    Args:
        encoder_params: Encoder network parameters.
        transition_params: Transition network parameters.
        D: Prior over initial states, shape (num_states,).
        observations_raw: Raw observation vectors, shape (T, obs_dim).
        actions: Action indices, shape (T,). Integer-valued.
        num_actions: Number of possible actions.

    Returns:
        Scalar negative log-likelihood.
    """
    first_obs = observations_raw[0]
    enc_logits_0 = encode(encoder_params, first_obs)
    likelihood_0 = jax.nn.softmax(enc_logits_0)

    alpha_0_unnorm = likelihood_0 * D
    evidence_0 = jnp.sum(alpha_0_unnorm)
    evidence_0 = jnp.clip(evidence_0, 1e-16)
    alpha_0 = alpha_0_unnorm / evidence_0
    log_evidence_0 = jnp.log(evidence_0)

    def scan_step(carry, x):
        return _deep_forward_step(
            carry, x, encoder_params, transition_params, num_actions
        )

    remaining_obs = observations_raw[1:]
    transition_actions = actions[:-1]

    init_carry = (alpha_0, jnp.zeros(()))
    _, log_evidences = jax.lax.scan(
        scan_step,
        init_carry,
        (remaining_obs, transition_actions),
    )

    total_log_likelihood = log_evidence_0 + jnp.sum(log_evidences)
    return -total_log_likelihood


# ---------------------------------------------------------------------------
# Deep learning result and training loop
# ---------------------------------------------------------------------------

class DeepLearningResult(NamedTuple):
    """Result of deep model learning."""
    encoder_params: NetworkParams
    transition_params: NetworkParams
    loss_history: list[float]


def learn_deep_model(
    observations_raw: np.ndarray,
    actions: np.ndarray,
    obs_dim: int,
    num_states: int,
    num_actions: int,
    D: Optional[np.ndarray] = None,
    encoder_hidden: list[int] = [32, 32],
    transition_hidden: list[int] = [32, 32],
    num_epochs: int = 200,
    lr: float = 0.001,
    seed: int = 0,
    verbose: bool = False,
) -> DeepLearningResult:
    """Train deep encoder and transition networks on observation-action data.

    Args:
        observations_raw: Raw observation vectors, shape (T, obs_dim).
        actions: Action indices, shape (T,). Integer-valued.
        obs_dim: Observation vector dimensionality.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        D: Prior over initial states. Default: uniform.
        encoder_hidden: Hidden layer widths for encoder MLP.
        transition_hidden: Hidden layer widths for transition MLP.
        num_epochs: Number of gradient descent steps.
        lr: Learning rate.
        seed: Random seed for network initialization.
        verbose: If True, print loss every 20 epochs.

    Returns:
        DeepLearningResult with learned parameters and loss history.
    """
    obs_jnp = jnp.array(observations_raw, dtype=jnp.float32)
    act_jnp = jnp.array(actions, dtype=jnp.int32)

    if D is None:
        D = np.ones(num_states) / num_states
    D_jnp = jnp.array(D, dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    enc_params = init_encoder(key1, obs_dim, encoder_hidden, num_states)
    trans_params = init_transition(key2, num_states, num_actions, transition_hidden)

    def loss_fn(enc_p, trans_p):
        return deep_analytic_nll(
            enc_p, trans_p, D_jnp, obs_jnp, act_jnp, num_actions
        )

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    def _apply_grads(params, grads, lr):
        return [
            (w - lr * gw, b - lr * gb)
            for (w, b), (gw, gb) in zip(params, grads)
        ]

    @jax.jit
    def sgd_step(enc_p, trans_p):
        loss = loss_fn(enc_p, trans_p)
        grad_enc, grad_trans = grad_fn(enc_p, trans_p)
        new_enc = _apply_grads(enc_p, grad_enc, lr)
        new_trans = _apply_grads(trans_p, grad_trans, lr)
        return new_enc, new_trans, loss

    loss_history = []
    for epoch in range(num_epochs):
        enc_params, trans_params, loss = sgd_step(enc_params, trans_params)
        loss_val = float(loss)
        loss_history.append(loss_val)
        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    return DeepLearningResult(
        encoder_params=enc_params,
        transition_params=trans_params,
        loss_history=loss_history,
    )


# ---------------------------------------------------------------------------
# Utility: extract approximate A matrix from encoder
# ---------------------------------------------------------------------------

def extract_A_matrix(
    encoder_params: NetworkParams,
    obs_dim: int,
) -> jnp.ndarray:
    """Extract an approximate A matrix from the encoder network.

    Args:
        encoder_params: Trained encoder parameters.
        obs_dim: Number of discrete observations (= obs vector dim).

    Returns:
        Approximate A matrix, shape (obs_dim, num_states).
    """
    identity = jnp.eye(obs_dim)

    def encode_one(obs_vec):
        logits = encode(encoder_params, obs_vec)
        return jax.nn.softmax(logits)

    return jax.vmap(encode_one)(identity)
