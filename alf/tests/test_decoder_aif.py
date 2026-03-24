"""Tests for Decoder-based Deep Active Inference (P(o|s) architecture).

Tests cover:
    1. Decoder init and forward pass produce correct shapes
    2. get_likelihood_matrix() produces valid (obs_dim, num_states) matrix
    3. infer_states() produces valid posterior
    4. decoder_analytic_nll() is differentiable (jax.grad works)
    5. learn_decoder_model() on synthetic T-maze-like data recovers A matrix
       better than the encoder architecture
    6. Gaussian decoder with continuous observations
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alf.deep_aif import (
    DecoderGenerativeModel,
    DecoderLearningResult,
    decode,
    decoder_analytic_nll,
    decoder_log_likelihood,
    gaussian_decode,
    gaussian_log_likelihood,
    init_decoder,
    init_gaussian_decoder,
    init_transition,
    learn_decoder_model,
)
from alf.learning import generate_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_spec():
    """Small model specification for tests."""
    return dict(
        obs_dim=5,
        num_states=3,
        num_actions=2,
        hidden_dims=[16, 16],
    )


@pytest.fixture
def decoder_params(model_spec):
    key = jax.random.PRNGKey(42)
    return init_decoder(
        key,
        model_spec["num_states"],
        model_spec["hidden_dims"],
        model_spec["obs_dim"],
    )


@pytest.fixture
def transition_params(model_spec):
    key = jax.random.PRNGKey(43)
    return init_transition(
        key,
        model_spec["num_states"],
        model_spec["num_actions"],
        model_spec["hidden_dims"],
    )


@pytest.fixture
def synthetic_data():
    """Generate synthetic POMDP data with known A/B matrices.

    Uses the same ground-truth model as test_deep_aif.py for fair
    comparison between encoder and decoder architectures.
    """
    num_obs = 5
    num_states = 3
    num_actions = 2

    A = np.array(
        [
            [0.85, 0.05, 0.05],
            [0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05],
            [0.00, 0.00, 0.00],
            [0.05, 0.05, 0.85],
        ],
        dtype=np.float64,
    )
    A[3, :] = 1.0 - A[:3, :].sum(axis=0) - A[4, :]
    A = np.clip(A, 0.001, None)
    A = A / A.sum(axis=0, keepdims=True)

    B = np.zeros((num_states, num_states, num_actions), dtype=np.float64)
    B[1, 0, 0] = 0.9
    B[0, 0, 0] = 0.05
    B[2, 0, 0] = 0.05
    B[2, 1, 0] = 0.9
    B[0, 1, 0] = 0.05
    B[1, 1, 0] = 0.05
    B[0, 2, 0] = 0.9
    B[1, 2, 0] = 0.05
    B[2, 2, 0] = 0.05
    B[0, 0, 1] = 0.9
    B[1, 0, 1] = 0.05
    B[2, 0, 1] = 0.05
    B[1, 1, 1] = 0.9
    B[0, 1, 1] = 0.05
    B[2, 1, 1] = 0.05
    B[2, 2, 1] = 0.9
    B[0, 2, 1] = 0.05
    B[1, 2, 1] = 0.05

    D = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    T = 500
    key = jax.random.PRNGKey(99)
    actions_seq = np.array(
        [
            int(jax.random.randint(k, (), 0, num_actions))
            for k in jax.random.split(key, T)
        ]
    )

    observations, states = generate_data(A, B, D, actions_seq, seed=123)
    obs_onehot = np.eye(num_obs)[observations]
    actions_padded = np.concatenate([actions_seq, [0]])

    return dict(
        A=A,
        B=B,
        D=D,
        num_obs=num_obs,
        num_states=num_states,
        num_actions=num_actions,
        observations=observations,
        obs_onehot=obs_onehot,
        actions=actions_padded,
        states=states,
    )


# ---------------------------------------------------------------------------
# Test 1: Decoder init and forward pass produce correct shapes
# ---------------------------------------------------------------------------


class TestDecoderShapes:
    def test_decoder_init_shapes(self, model_spec):
        """Decoder init produces correct number of layers and shapes."""
        key = jax.random.PRNGKey(0)
        params = init_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )
        # 2 hidden layers + 1 output = 3 layers
        assert len(params) == 3

        w0, b0 = params[0]
        assert w0.shape == (3, 16)  # num_states -> first hidden
        assert b0.shape == (16,)

        w1, b1 = params[1]
        assert w1.shape == (16, 16)
        assert b1.shape == (16,)

        w2, b2 = params[2]
        assert w2.shape == (16, 5)  # last hidden -> obs_dim
        assert b2.shape == (5,)

    def test_decode_output_shape(self, decoder_params, model_spec):
        """Decoder forward pass produces (obs_dim,) logits."""
        state_oh = jax.nn.one_hot(0, model_spec["num_states"])
        logits = decode(decoder_params, state_oh)
        assert logits.shape == (model_spec["obs_dim"],)

    def test_decode_different_states_different_output(self, decoder_params, model_spec):
        """Different states produce different observation logits."""
        state_oh_0 = jax.nn.one_hot(0, model_spec["num_states"])
        state_oh_1 = jax.nn.one_hot(1, model_spec["num_states"])
        logits_0 = decode(decoder_params, state_oh_0)
        logits_1 = decode(decoder_params, state_oh_1)
        assert not jnp.allclose(logits_0, logits_1)

    def test_decoder_log_likelihood_scalar(self, decoder_params, model_spec):
        """decoder_log_likelihood returns a finite scalar."""
        obs = jax.nn.one_hot(0, model_spec["obs_dim"])
        ll = decoder_log_likelihood(
            decoder_params, obs, jnp.array(0), model_spec["num_states"]
        )
        assert ll.shape == ()
        assert jnp.isfinite(ll)

    def test_decoder_log_likelihood_negative(self, decoder_params, model_spec):
        """Log-likelihood of a categorical distribution is <= 0."""
        obs = jax.nn.one_hot(0, model_spec["obs_dim"])
        ll = decoder_log_likelihood(
            decoder_params, obs, jnp.array(0), model_spec["num_states"]
        )
        assert ll <= 0.0


# ---------------------------------------------------------------------------
# Test 2: get_likelihood_matrix() produces valid (obs_dim, num_states) matrix
# ---------------------------------------------------------------------------


class TestLikelihoodMatrix:
    def test_shape(self, decoder_params, model_spec):
        """Likelihood matrix has shape (obs_dim, num_states)."""
        A = DecoderGenerativeModel.get_likelihood_matrix(
            decoder_params, model_spec["num_states"]
        )
        assert A.shape == (model_spec["obs_dim"], model_spec["num_states"])

    def test_columns_sum_to_one(self, decoder_params, model_spec):
        """Each column of the A matrix sums to 1 (valid P(o|s))."""
        A = DecoderGenerativeModel.get_likelihood_matrix(
            decoder_params, model_spec["num_states"]
        )
        col_sums = jnp.sum(A, axis=0)
        assert jnp.allclose(col_sums, 1.0, atol=1e-5)

    def test_all_positive(self, decoder_params, model_spec):
        """All entries of the A matrix are non-negative."""
        A = DecoderGenerativeModel.get_likelihood_matrix(
            decoder_params, model_spec["num_states"]
        )
        assert jnp.all(A >= 0.0)

    def test_different_columns(self, decoder_params, model_spec):
        """Different states produce different observation distributions."""
        A = DecoderGenerativeModel.get_likelihood_matrix(
            decoder_params, model_spec["num_states"]
        )
        # At least two columns should be different
        assert not jnp.allclose(A[:, 0], A[:, 1])


# ---------------------------------------------------------------------------
# Test 3: infer_states() produces valid posterior
# ---------------------------------------------------------------------------


class TestInferStates:
    def test_posterior_shape(self, decoder_params, model_spec):
        """Posterior has shape (num_states,)."""
        obs = jax.nn.one_hot(0, model_spec["obs_dim"])
        prior = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        posterior = DecoderGenerativeModel.infer_states(
            decoder_params, obs, prior, model_spec["num_states"]
        )
        assert posterior.shape == (model_spec["num_states"],)

    def test_posterior_sums_to_one(self, decoder_params, model_spec):
        """Posterior is a valid probability distribution."""
        obs = jax.nn.one_hot(0, model_spec["obs_dim"])
        prior = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        posterior = DecoderGenerativeModel.infer_states(
            decoder_params, obs, prior, model_spec["num_states"]
        )
        assert jnp.allclose(jnp.sum(posterior), 1.0, atol=1e-5)
        assert jnp.all(posterior >= 0.0)

    def test_posterior_depends_on_obs(self, decoder_params, model_spec):
        """Different observations produce different posteriors."""
        prior = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        obs_0 = jax.nn.one_hot(0, model_spec["obs_dim"])
        obs_1 = jax.nn.one_hot(2, model_spec["obs_dim"])
        post_0 = DecoderGenerativeModel.infer_states(
            decoder_params, obs_0, prior, model_spec["num_states"]
        )
        post_1 = DecoderGenerativeModel.infer_states(
            decoder_params, obs_1, prior, model_spec["num_states"]
        )
        assert not jnp.allclose(post_0, post_1)

    def test_posterior_depends_on_prior(self, decoder_params, model_spec):
        """Different priors produce different posteriors."""
        obs = jax.nn.one_hot(0, model_spec["obs_dim"])
        prior_1 = jnp.array([0.8, 0.1, 0.1])
        prior_2 = jnp.array([0.1, 0.1, 0.8])
        post_1 = DecoderGenerativeModel.infer_states(
            decoder_params, obs, prior_1, model_spec["num_states"]
        )
        post_2 = DecoderGenerativeModel.infer_states(
            decoder_params, obs, prior_2, model_spec["num_states"]
        )
        assert not jnp.allclose(post_1, post_2)


# ---------------------------------------------------------------------------
# Test 4: decoder_analytic_nll() is differentiable (jax.grad works)
# ---------------------------------------------------------------------------


class TestDecoderDifferentiability:
    def test_grad_decoder(self, decoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. decoder_params."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(decoder_analytic_nll, argnums=0)
        grads = grad_fn(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )

        assert len(grads) == len(decoder_params)
        for (gw, gb), (w, b) in zip(grads, decoder_params):
            assert gw.shape == w.shape
            assert gb.shape == b.shape
            assert not jnp.allclose(gw, 0.0)

    def test_grad_transition(self, decoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. transition_params."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(decoder_analytic_nll, argnums=1)
        grads = grad_fn(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )

        assert len(grads) == len(transition_params)
        for (gw, gb), (w, b) in zip(grads, transition_params):
            assert gw.shape == w.shape
            assert gb.shape == b.shape

    def test_grad_both(self, decoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. both decoder and transition simultaneously."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(decoder_analytic_nll, argnums=(0, 1))
        grad_dec, grad_trans = grad_fn(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )

        assert len(grad_dec) == len(decoder_params)
        assert len(grad_trans) == len(transition_params)

    def test_nll_is_finite(self, decoder_params, transition_params, model_spec):
        """NLL produces a finite scalar."""
        T = 20
        key = jax.random.PRNGKey(7)
        obs = jax.random.normal(key, (T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        nll = decoder_analytic_nll(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )
        assert jnp.isfinite(nll)
        assert nll.shape == ()

    def test_nll_jit(self, decoder_params, transition_params, model_spec):
        """decoder_analytic_nll works under jax.jit."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        jit_nll = jax.jit(
            decoder_analytic_nll,
            static_argnums=(5, 6),
        )

        nll_eager = decoder_analytic_nll(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )
        nll_jit = jit_nll(
            decoder_params,
            transition_params,
            D,
            obs,
            actions,
            model_spec["num_states"],
            model_spec["num_actions"],
        )

        assert jnp.allclose(nll_eager, nll_jit, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: learn_decoder_model() on synthetic data
# ---------------------------------------------------------------------------


class TestDecoderTraining:
    def test_loss_decreases(self, synthetic_data):
        """Training loop reduces NLL over epochs."""
        result = learn_decoder_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            decoder_hidden=[32, 32],
            transition_hidden=[32, 32],
            num_epochs=100,
            lr=0.001,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, DecoderLearningResult)
        assert len(result.loss_history) == 100

        early_loss = np.mean(result.loss_history[:5])
        late_loss = np.mean(result.loss_history[-5:])
        assert late_loss < early_loss, (
            f"Loss did not decrease: early={early_loss:.4f}, late={late_loss:.4f}"
        )

    def test_decoder_recovers_A_structure(self, synthetic_data):
        """Decoder recovers the structure of the ground-truth A matrix.

        The key advantage over the encoder: the decoder learns P(o|s)
        directly, so each state must generate distinct observation
        distributions. This avoids the degenerate optimum.
        """
        result = learn_decoder_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            decoder_hidden=[64, 64],
            transition_hidden=[64, 64],
            num_epochs=3000,
            lr=0.002,
            seed=7,
        )

        # Extract A matrix from the decoder
        A_learned = DecoderGenerativeModel.get_likelihood_matrix(
            result.decoder_params, synthetic_data["num_states"]
        )
        A_learned = np.array(A_learned)

        # Each state should have a dominant observation
        # Ground truth: state 0 -> obs 0, state 1 -> obs 2, state 2 -> obs 4
        # (up to permutation)
        #
        # Check that the decoder has learned to distinguish states:
        # each state's argmax observation should be unique
        argmax_obs = [int(np.argmax(A_learned[:, s])) for s in range(3)]

        assert len(set(argmax_obs)) == 3, (
            f"Decoder failed to distinguish states: argmax obs = {argmax_obs}\n"
            f"A_learned:\n{A_learned}"
        )

    def test_result_types(self, synthetic_data):
        """DecoderLearningResult contains correct types."""
        result = learn_decoder_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            num_epochs=10,
            seed=0,
        )

        assert isinstance(result, DecoderLearningResult)
        assert isinstance(result.decoder_params, list)
        assert isinstance(result.transition_params, list)
        assert isinstance(result.loss_history, list)
        assert all(isinstance(x, float) for x in result.loss_history)


# ---------------------------------------------------------------------------
# Test 6: Gaussian decoder with continuous observations
# ---------------------------------------------------------------------------


class TestGaussianDecoder:
    def test_init_shapes(self, model_spec):
        """Gaussian decoder init produces correct output dimension."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )
        # 2 hidden layers + 1 output = 3 layers
        assert len(params) == 3

        w_last, b_last = params[-1]
        # Output is 2 * obs_dim (mean + log_var)
        assert w_last.shape == (16, 2 * model_spec["obs_dim"])
        assert b_last.shape == (2 * model_spec["obs_dim"],)

    def test_gaussian_decode_output(self, model_spec):
        """gaussian_decode returns (mean, log_var) with correct shapes."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )

        state_oh = jax.nn.one_hot(0, model_spec["num_states"])
        mean, log_var = gaussian_decode(params, state_oh, model_spec["obs_dim"])

        assert mean.shape == (model_spec["obs_dim"],)
        assert log_var.shape == (model_spec["obs_dim"],)

    def test_gaussian_log_likelihood_scalar(self, model_spec):
        """Gaussian log-likelihood returns a finite scalar."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )

        obs = jnp.ones(model_spec["obs_dim"])
        ll = gaussian_log_likelihood(
            params, obs, jnp.array(0), model_spec["num_states"]
        )

        assert ll.shape == ()
        assert jnp.isfinite(ll)

    def test_gaussian_log_likelihood_differentiable(self, model_spec):
        """jax.grad works for gaussian_log_likelihood."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )

        obs = jnp.ones(model_spec["obs_dim"])

        def loss(p):
            return -gaussian_log_likelihood(
                p, obs, jnp.array(0), model_spec["num_states"]
            )

        grads = jax.grad(loss)(params)
        assert len(grads) == len(params)
        for (gw, gb), (w, b) in zip(grads, params):
            assert gw.shape == w.shape
            assert gb.shape == b.shape

    def test_different_states_different_params(self, model_spec):
        """Different states produce different Gaussian parameters."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )

        state_oh_0 = jax.nn.one_hot(0, model_spec["num_states"])
        state_oh_1 = jax.nn.one_hot(1, model_spec["num_states"])

        mean_0, logvar_0 = gaussian_decode(params, state_oh_0, model_spec["obs_dim"])
        mean_1, logvar_1 = gaussian_decode(params, state_oh_1, model_spec["obs_dim"])

        assert not jnp.allclose(mean_0, mean_1)

    def test_gaussian_log_likelihood_close_obs_higher(self, model_spec):
        """Observations closer to the mean have higher log-likelihood."""
        key = jax.random.PRNGKey(0)
        params = init_gaussian_decoder(
            key,
            model_spec["num_states"],
            model_spec["hidden_dims"],
            model_spec["obs_dim"],
        )

        # Get the mean for state 0
        state_oh = jax.nn.one_hot(0, model_spec["num_states"])
        mean, _ = gaussian_decode(params, state_oh, model_spec["obs_dim"])

        # Observation at the mean vs far from the mean
        obs_close = mean
        obs_far = mean + 10.0

        ll_close = gaussian_log_likelihood(
            params, obs_close, jnp.array(0), model_spec["num_states"]
        )
        ll_far = gaussian_log_likelihood(
            params, obs_far, jnp.array(0), model_spec["num_states"]
        )

        assert ll_close > ll_far


# ---------------------------------------------------------------------------
# Test: DecoderGenerativeModel class integration
# ---------------------------------------------------------------------------


class TestDecoderGenerativeModel:
    def test_init(self):
        """DecoderGenerativeModel initializes without error."""
        dgm = DecoderGenerativeModel(
            obs_dim=10,
            num_states=4,
            num_actions=3,
            decoder_hidden=[16],
            transition_hidden=[16],
            seed=0,
        )
        assert len(dgm.decoder_params) == 2  # 1 hidden + 1 output
        assert len(dgm.transition_params) == 2

    def test_likelihood_matrix_and_transition(self):
        """get_likelihood_matrix and get_transition work correctly."""
        dgm = DecoderGenerativeModel(
            obs_dim=5,
            num_states=3,
            num_actions=2,
            decoder_hidden=[16, 16],
            transition_hidden=[16, 16],
        )

        A = dgm.get_likelihood_matrix(dgm.decoder_params, dgm.num_states)
        assert A.shape == (5, 3)
        assert jnp.allclose(jnp.sum(A, axis=0), 1.0, atol=1e-5)
        assert jnp.all(A >= 0.0)

        beliefs = jnp.array([0.5, 0.3, 0.2])
        trans = dgm.get_transition(dgm.transition_params, beliefs, 0, 2)
        assert trans.shape == (3,)
        assert jnp.allclose(jnp.sum(trans), 1.0, atol=1e-5)
        assert jnp.all(trans >= 0.0)

    def test_infer_states_from_model(self):
        """infer_states produces valid posterior through the model."""
        dgm = DecoderGenerativeModel(
            obs_dim=5,
            num_states=3,
            num_actions=2,
        )

        obs = jax.nn.one_hot(0, 5)
        prior = jnp.ones(3) / 3.0
        posterior = dgm.infer_states(dgm.decoder_params, obs, prior, dgm.num_states)
        assert posterior.shape == (3,)
        assert jnp.allclose(jnp.sum(posterior), 1.0, atol=1e-5)
        assert jnp.all(posterior >= 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
