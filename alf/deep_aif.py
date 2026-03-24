"""Deep Active Inference: neural network parameterized A and B matrices.

Replace flat log-parameter arrays with neural network encoders, enabling
Active Inference to scale to high-dimensional observation spaces.

Architecture overview:

    Encoder:  obs_raw  -->  MLP  -->  state_logits  -->  softmax  -->  Q(s|o)
    Decoder:  state_onehot  -->  MLP  -->  obs_logits  -->  softmax  -->  P(o|s)
    Transition:  (state_beliefs, action_onehot)  -->  MLP  -->  next_state_logits

Two architectures:

1. **Encoder-based** (recognition model): Learns Q(s|o) directly. Fast
   inference but susceptible to a degenerate optimum where the encoder
   maps all observations to a single state.

2. **Decoder-based** (generative model): Learns P(o|s) and uses Bayes'
   rule for inference: Q(s|o) proportional to P(o|s) * P(s). Avoids
   the degenerate optimum because each state must explain its
   observations, and the likelihood matrix is constructed explicitly.

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
        return deep_analytic_nll(enc_p, trans_p, D_jnp, obs_jnp, act_jnp, num_actions)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    def _apply_grads(params, grads, lr):
        return [(w - lr * gw, b - lr * gb) for (w, b), (gw, gb) in zip(params, grads)]

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


# ===========================================================================
# Decoder-based architecture: P(o|s) instead of Q(s|o)
# ===========================================================================

# ---------------------------------------------------------------------------
# Decoder Network: state_onehot -> observation distribution params
# ---------------------------------------------------------------------------


def init_decoder(
    key: jax.random.PRNGKey,
    num_states: int,
    hidden_dims: list[int],
    obs_dim: int,
) -> NetworkParams:
    """Initialize a decoder MLP: state_onehot -> obs_distribution_params.

    The decoder maps a one-hot state vector to observation logits, producing
    a categorical distribution P(o|s) over observations.

    Args:
        key: JAX PRNG key for random initialization.
        num_states: Number of hidden states (input dimension).
        hidden_dims: List of hidden layer widths.
        obs_dim: Dimensionality of the observation vector (output dimension).

    Returns:
        NetworkParams: list of (weight, bias) tuples.
    """
    params = []
    dims = [num_states] + list(hidden_dims) + [obs_dim]

    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in = dims[i]
        fan_out = dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = std * jax.random.normal(subkey, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((w, b))

    return params


def decode(params: NetworkParams, state_onehot: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through the decoder network.

    Args:
        params: Decoder network parameters from init_decoder.
        state_onehot: One-hot state vector, shape (num_states,).

    Returns:
        Observation logits, shape (obs_dim,). Unnormalized.
    """
    x = state_onehot
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def decoder_log_likelihood(
    decoder_params: NetworkParams,
    obs: jnp.ndarray,
    state_idx: jnp.ndarray,
    num_states: int,
) -> jnp.ndarray:
    """Compute log P(o|s) for a specific state using the decoder.

    For categorical (one-hot) observations, this computes the log probability
    of the observation under the decoder's predicted distribution for the
    given state.

    Args:
        decoder_params: Decoder network parameters.
        obs: Observation vector, shape (obs_dim,). For discrete obs, one-hot.
        state_idx: State index (scalar integer).
        num_states: Total number of states (for one-hot encoding).

    Returns:
        Scalar log-likelihood log P(o|s).
    """
    state_onehot = jax.nn.one_hot(state_idx, num_states)
    logits = decode(decoder_params, state_onehot)
    log_probs = jax.nn.log_softmax(logits)
    return jnp.sum(obs * log_probs)


# ---------------------------------------------------------------------------
# Gaussian Decoder: state_onehot -> (mean, log_var) for continuous obs
# ---------------------------------------------------------------------------


def init_gaussian_decoder(
    key: jax.random.PRNGKey,
    num_states: int,
    hidden_dims: list[int],
    obs_dim: int,
) -> NetworkParams:
    """Initialize a Gaussian decoder MLP: state_onehot -> (mean, log_var).

    The output dimension is 2 * obs_dim: the first obs_dim elements are
    the mean, the remaining obs_dim elements are the log-variance.

    Args:
        key: JAX PRNG key for random initialization.
        num_states: Number of hidden states (input dimension).
        hidden_dims: List of hidden layer widths.
        obs_dim: Dimensionality of the continuous observation vector.

    Returns:
        NetworkParams: list of (weight, bias) tuples.
    """
    output_dim = 2 * obs_dim
    params = []
    dims = [num_states] + list(hidden_dims) + [output_dim]

    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        fan_in = dims[i]
        fan_out = dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        w = std * jax.random.normal(subkey, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((w, b))

    return params


def gaussian_decode(
    params: NetworkParams,
    state_onehot: jnp.ndarray,
    obs_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass through the Gaussian decoder.

    Args:
        params: Gaussian decoder network parameters.
        state_onehot: One-hot state vector, shape (num_states,).
        obs_dim: Dimensionality of the observation vector.

    Returns:
        Tuple of (mean, log_var), each shape (obs_dim,).
    """
    x = state_onehot
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    mean = x[:obs_dim]
    log_var = x[obs_dim:]
    return mean, log_var


def gaussian_log_likelihood(
    decoder_params: NetworkParams,
    obs: jnp.ndarray,
    state_idx: jnp.ndarray,
    num_states: int,
) -> jnp.ndarray:
    """Compute log N(obs; mu(s), sigma(s)^2) — Gaussian observation model.

    Args:
        decoder_params: Gaussian decoder network parameters.
        obs: Continuous observation vector, shape (obs_dim,).
        state_idx: State index (scalar integer).
        num_states: Total number of states (for one-hot encoding).

    Returns:
        Scalar log-likelihood log P(o|s).
    """
    obs_dim = obs.shape[0]
    state_onehot = jax.nn.one_hot(state_idx, num_states)
    mean, log_var = gaussian_decode(decoder_params, state_onehot, obs_dim)
    # log N(x; mu, sigma^2) = -0.5 * [log(2pi) + log_var + (x-mu)^2/var]
    var = jnp.exp(log_var)
    var = jnp.clip(var, 1e-8)
    log_prob = -0.5 * (jnp.log(2.0 * jnp.pi) + log_var + (obs - mean) ** 2 / var)
    return jnp.sum(log_prob)


# ---------------------------------------------------------------------------
# Decoder Generative Model
# ---------------------------------------------------------------------------


class DecoderGenerativeModel:
    """Decoder-based deep generative model P(o|s).

    Instead of learning Q(s|o) directly (encoder), learns P(o|s) (decoder)
    and uses Bayes' rule for inference: Q(s|o) proportional to P(o|s) * P(s).

    This avoids the degenerate optimum of the encoder architecture where
    the encoder maps all observations to a single state. The decoder must
    explain each state's observations independently, so the likelihood
    matrix is well-conditioned.

    Attributes:
        obs_dim: Observation vector dimensionality.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        decoder_params: Decoder network parameters.
        transition_params: Transition network parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        num_states: int,
        num_actions: int,
        decoder_hidden: list[int] = [32, 32],
        transition_hidden: list[int] = [32, 32],
        seed: int = 0,
    ):
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)

        self.obs_dim = obs_dim
        self.num_states = num_states
        self.num_actions = num_actions

        self.decoder_params = init_decoder(key1, num_states, decoder_hidden, obs_dim)
        self.transition_params = init_transition(
            key2, num_states, num_actions, transition_hidden
        )

    @staticmethod
    def get_likelihood_matrix(
        decoder_params: NetworkParams,
        num_states: int,
    ) -> jnp.ndarray:
        """Construct full A matrix P(o|s) by evaluating decoder at each state.

        Evaluates the decoder network at each one-hot state vector and
        applies softmax to get a proper categorical distribution for each
        state. The resulting matrix is compatible with ALF's GenerativeModel.

        Args:
            decoder_params: Decoder network parameters.
            num_states: Number of hidden states.

        Returns:
            A matrix of shape (obs_dim, num_states), where each column
            is a valid probability distribution P(o|s) for state s.
        """
        state_onehots = jnp.eye(num_states)

        def decode_one(state_oh):
            logits = decode(decoder_params, state_oh)
            return jax.nn.softmax(logits)

        # Each row of the result is P(o|s) for one state -> shape (num_states, obs_dim)
        # Transpose to get (obs_dim, num_states) to match ALF's A matrix convention
        probs = jax.vmap(decode_one)(state_onehots)
        return probs.T

    @staticmethod
    def infer_states(
        decoder_params: NetworkParams,
        obs: jnp.ndarray,
        prior: jnp.ndarray,
        num_states: int,
    ) -> jnp.ndarray:
        """Bayesian inference: Q(s|o) proportional to P(o|s) * P(s).

        Uses the decoder to compute P(o|s) for each state, then
        applies Bayes' rule with the given prior to produce a posterior.

        Args:
            decoder_params: Decoder network parameters.
            obs: Observation vector, shape (obs_dim,). For discrete obs, one-hot.
            prior: Prior distribution over states P(s), shape (num_states,).
            num_states: Number of hidden states.

        Returns:
            Posterior Q(s|o), shape (num_states,). Normalized.
        """

        # Compute log P(o|s) for each state
        def log_lik_for_state(s_idx):
            return decoder_log_likelihood(decoder_params, obs, s_idx, num_states)

        log_liks = jax.vmap(log_lik_for_state)(jnp.arange(num_states))
        # P(o|s) for each state
        liks = jnp.exp(log_liks)
        liks = jnp.clip(liks, 1e-16)

        # Bayes' rule: Q(s|o) proportional to P(o|s) * P(s)
        unnorm = liks * prior
        evidence = jnp.sum(unnorm)
        evidence = jnp.clip(evidence, 1e-16)
        return unnorm / evidence

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
# Decoder-based Analytic NLL
# ---------------------------------------------------------------------------


def _decoder_likelihood_vector(
    decoder_params: NetworkParams,
    obs: jnp.ndarray,
    num_states: int,
) -> jnp.ndarray:
    """Compute P(o_t|s) for all states as a vector.

    Args:
        decoder_params: Decoder network parameters.
        obs: Observation vector, shape (obs_dim,).
        num_states: Number of hidden states.

    Returns:
        Likelihood vector, shape (num_states,), where entry s is P(o|s).
    """

    def log_lik_for_state(s_idx):
        return decoder_log_likelihood(decoder_params, obs, s_idx, num_states)

    log_liks = jax.vmap(log_lik_for_state)(jnp.arange(num_states))
    liks = jnp.exp(log_liks)
    return jnp.clip(liks, 1e-16)


def _decoder_forward_step(
    carry: tuple[jnp.ndarray, jnp.ndarray],
    x: tuple[jnp.ndarray, jnp.ndarray],
    decoder_params: NetworkParams,
    transition_params: NetworkParams,
    num_states: int,
    num_actions: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """One step of forward filtering with decoder likelihood/transition.

    At each step:
    1. Predict: P(s_t) = transition(alpha_{t-1}, a_{t-1})
    2. Update: alpha_t proportional to P(o_t|s_t) * P(s_t),
       where P(o_t|s_t) comes from the decoder
    3. Evidence: P(o_t) = sum_s P(o_t|s_t) * P(s_t)
    """
    alpha_prev, _ = carry
    obs_vec, action_idx = x

    # Predict step: transition network
    action_onehot = jax.nn.one_hot(action_idx, num_actions)
    trans_logits = predict_transition(transition_params, alpha_prev, action_onehot)
    predicted = jax.nn.softmax(trans_logits)
    predicted = jnp.clip(predicted, 1e-16)

    # Update step: decoder likelihood
    likelihood = _decoder_likelihood_vector(decoder_params, obs_vec, num_states)

    alpha_unnorm = likelihood * predicted
    evidence = jnp.sum(alpha_unnorm)
    evidence = jnp.clip(evidence, 1e-16)
    alpha_new = alpha_unnorm / evidence

    log_evidence = jnp.log(evidence)

    return (alpha_new, jnp.zeros(())), log_evidence


def decoder_analytic_nll(
    decoder_params: NetworkParams,
    transition_params: NetworkParams,
    D: jnp.ndarray,
    observations_raw: jnp.ndarray,
    actions: jnp.ndarray,
    num_states: int,
    num_actions: int,
) -> jnp.ndarray:
    """Compute NLL using decoder P(o|s) instead of encoder P(s|o).

    Forward-filter NLL using the decoder-based architecture. At each step:
    1. Predict: P(s_t) = B(a_{t-1}) @ alpha_{t-1}
    2. Update: alpha_t proportional to P(o_t|s_t) * P(s_t),
       where P(o_t|s_t) comes from the decoder
    3. Evidence: P(o_t) = sum_s P(o_t|s_t) * P(s_t)

    Args:
        decoder_params: Decoder network parameters.
        transition_params: Transition network parameters.
        D: Prior over initial states, shape (num_states,).
        observations_raw: Raw observation vectors, shape (T, obs_dim).
        actions: Action indices, shape (T,). Integer-valued.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.

    Returns:
        Scalar negative log-likelihood.
    """
    # First observation: P(o_0|s) * D(s)
    first_obs = observations_raw[0]
    likelihood_0 = _decoder_likelihood_vector(decoder_params, first_obs, num_states)

    alpha_0_unnorm = likelihood_0 * D
    evidence_0 = jnp.sum(alpha_0_unnorm)
    evidence_0 = jnp.clip(evidence_0, 1e-16)
    alpha_0 = alpha_0_unnorm / evidence_0
    log_evidence_0 = jnp.log(evidence_0)

    def scan_step(carry, x):
        return _decoder_forward_step(
            carry,
            x,
            decoder_params,
            transition_params,
            num_states,
            num_actions,
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
# Decoder learning result and training loop
# ---------------------------------------------------------------------------


class DecoderLearningResult(NamedTuple):
    """Result of decoder-based model learning."""

    decoder_params: NetworkParams
    transition_params: NetworkParams
    loss_history: list[float]


def learn_decoder_model(
    observations_raw: np.ndarray,
    actions: np.ndarray,
    obs_dim: int,
    num_states: int,
    num_actions: int,
    D: Optional[np.ndarray] = None,
    decoder_hidden: list[int] = [32, 32],
    transition_hidden: list[int] = [32, 32],
    num_epochs: int = 200,
    lr: float = 0.001,
    seed: int = 0,
    verbose: bool = False,
) -> DecoderLearningResult:
    """Train decoder + transition networks on observation-action data.

    Uses the decoder-based architecture P(o|s) instead of the encoder-based
    Q(s|o). This avoids the degenerate optimum where all observations
    map to a single state.

    Args:
        observations_raw: Raw observation vectors, shape (T, obs_dim).
        actions: Action indices, shape (T,). Integer-valued.
        obs_dim: Observation vector dimensionality.
        num_states: Number of hidden states.
        num_actions: Number of possible actions.
        D: Prior over initial states. Default: uniform.
        decoder_hidden: Hidden layer widths for decoder MLP.
        transition_hidden: Hidden layer widths for transition MLP.
        num_epochs: Number of gradient descent steps.
        lr: Learning rate.
        seed: Random seed for network initialization.
        verbose: If True, print loss every 20 epochs.

    Returns:
        DecoderLearningResult with learned parameters and loss history.
    """
    obs_jnp = jnp.array(observations_raw, dtype=jnp.float32)
    act_jnp = jnp.array(actions, dtype=jnp.int32)

    if D is None:
        D = np.ones(num_states) / num_states
    D_jnp = jnp.array(D, dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    dec_params = init_decoder(key1, num_states, decoder_hidden, obs_dim)
    trans_params = init_transition(key2, num_states, num_actions, transition_hidden)

    def loss_fn(dec_p, trans_p):
        return decoder_analytic_nll(
            dec_p, trans_p, D_jnp, obs_jnp, act_jnp, num_states, num_actions
        )

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    def _apply_grads(params, grads, lr):
        return [(w - lr * gw, b - lr * gb) for (w, b), (gw, gb) in zip(params, grads)]

    @jax.jit
    def sgd_step(dec_p, trans_p):
        loss = loss_fn(dec_p, trans_p)
        grad_dec, grad_trans = grad_fn(dec_p, trans_p)
        new_dec = _apply_grads(dec_p, grad_dec, lr)
        new_trans = _apply_grads(trans_p, grad_trans, lr)
        return new_dec, new_trans, loss

    loss_history = []
    for epoch in range(num_epochs):
        dec_params, trans_params, loss = sgd_step(dec_params, trans_params)
        loss_val = float(loss)
        loss_history.append(loss_val)
        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    return DecoderLearningResult(
        decoder_params=dec_params,
        transition_params=trans_params,
        loss_history=loss_history,
    )
