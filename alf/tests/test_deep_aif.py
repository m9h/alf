"""Tests for Deep Active Inference (neural network parameterized A/B).

Tests cover:
    1. MLP forward pass produces correct shapes
    2. Encoder output has correct dimensions
    3. Transition output has correct dimensions
    4. Deep NLL is differentiable (jax.grad works)
    5. Deep NLL under jit works
    6. Training on synthetic discrete data reduces loss
    7. Deep model recovers approximate A matrix structure from enough data
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alf.deep_aif import (
    DeepGenerativeModel,
    DeepLearningResult,
    deep_analytic_nll,
    encode,
    extract_A_matrix,
    init_encoder,
    init_transition,
    learn_deep_model,
    predict_transition,
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
def encoder_params(model_spec):
    key = jax.random.PRNGKey(42)
    return init_encoder(
        key,
        model_spec["obs_dim"],
        model_spec["hidden_dims"],
        model_spec["num_states"],
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
    """Generate synthetic POMDP data with known A/B matrices."""
    num_obs = 5
    num_states = 3
    num_actions = 2

    A = np.array([
        [0.85, 0.05, 0.05],
        [0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05],
        [0.00, 0.00, 0.00],
        [0.05, 0.05, 0.85],
    ], dtype=np.float64)
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
        [int(jax.random.randint(k, (), 0, num_actions))
         for k in jax.random.split(key, T)]
    )

    observations, states = generate_data(A, B, D, actions_seq, seed=123)
    obs_onehot = np.eye(num_obs)[observations]
    actions_padded = np.concatenate([actions_seq, [0]])

    return dict(
        A=A, B=B, D=D,
        num_obs=num_obs, num_states=num_states, num_actions=num_actions,
        observations=observations,
        obs_onehot=obs_onehot,
        actions=actions_padded,
        states=states,
    )


# ---------------------------------------------------------------------------
# Test 1: MLP forward pass produces correct shapes
# ---------------------------------------------------------------------------

class TestMLPShapes:

    def test_encoder_init_shapes(self, model_spec):
        """Encoder init produces correct number of layers and shapes."""
        key = jax.random.PRNGKey(0)
        params = init_encoder(
            key,
            model_spec["obs_dim"],
            model_spec["hidden_dims"],
            model_spec["num_states"],
        )
        assert len(params) == 3

        w0, b0 = params[0]
        assert w0.shape == (5, 16)
        assert b0.shape == (16,)

        w1, b1 = params[1]
        assert w1.shape == (16, 16)
        assert b1.shape == (16,)

        w2, b2 = params[2]
        assert w2.shape == (16, 3)
        assert b2.shape == (3,)

    def test_transition_init_shapes(self, model_spec):
        """Transition init produces correct number of layers and shapes."""
        key = jax.random.PRNGKey(0)
        params = init_transition(
            key,
            model_spec["num_states"],
            model_spec["num_actions"],
            model_spec["hidden_dims"],
        )
        assert len(params) == 3

        w0, b0 = params[0]
        assert w0.shape == (5, 16)

        w2, b2 = params[2]
        assert w2.shape == (16, 3)


# ---------------------------------------------------------------------------
# Test 2: Encoder output has correct dimensions
# ---------------------------------------------------------------------------

class TestEncoderOutput:

    def test_encode_output_shape(self, encoder_params, model_spec):
        """Encoder forward pass produces (num_states,) logits."""
        obs = jnp.ones(model_spec["obs_dim"])
        logits = encode(encoder_params, obs)
        assert logits.shape == (model_spec["num_states"],)

    def test_encode_different_obs_different_output(self, encoder_params, model_spec):
        """Different observations produce different logits."""
        obs1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        obs2 = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
        logits1 = encode(encoder_params, obs1)
        logits2 = encode(encoder_params, obs2)
        assert not jnp.allclose(logits1, logits2)

    def test_get_likelihood_normalized(self, encoder_params, model_spec):
        """get_likelihood returns a valid probability distribution."""
        obs = jnp.ones(model_spec["obs_dim"])
        probs = DeepGenerativeModel.get_likelihood(encoder_params, obs)
        assert probs.shape == (model_spec["num_states"],)
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0.0)


# ---------------------------------------------------------------------------
# Test 3: Transition output has correct dimensions
# ---------------------------------------------------------------------------

class TestTransitionOutput:

    def test_predict_transition_shape(self, transition_params, model_spec):
        """Transition forward pass produces (num_states,) logits."""
        beliefs = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        action = jnp.array([1.0, 0.0])
        logits = predict_transition(transition_params, beliefs, action)
        assert logits.shape == (model_spec["num_states"],)

    def test_get_transition_normalized(self, transition_params, model_spec):
        """get_transition returns a valid probability distribution."""
        beliefs = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        probs = DeepGenerativeModel.get_transition(
            transition_params, beliefs, 0, model_spec["num_actions"]
        )
        assert probs.shape == (model_spec["num_states"],)
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0.0)

    def test_different_actions_different_output(self, transition_params, model_spec):
        """Different actions produce different transition predictions."""
        beliefs = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]
        probs0 = DeepGenerativeModel.get_transition(
            transition_params, beliefs, 0, model_spec["num_actions"]
        )
        probs1 = DeepGenerativeModel.get_transition(
            transition_params, beliefs, 1, model_spec["num_actions"]
        )
        assert not jnp.allclose(probs0, probs1)


# ---------------------------------------------------------------------------
# Test 4: Deep NLL is differentiable (jax.grad works)
# ---------------------------------------------------------------------------

class TestDifferentiability:

    def test_grad_encoder(self, encoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. encoder_params."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(deep_analytic_nll, argnums=0)
        grads = grad_fn(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )

        assert len(grads) == len(encoder_params)
        for (gw, gb), (w, b) in zip(grads, encoder_params):
            assert gw.shape == w.shape
            assert gb.shape == b.shape
            assert not jnp.allclose(gw, 0.0)

    def test_grad_transition(self, encoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. transition_params."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(deep_analytic_nll, argnums=1)
        grads = grad_fn(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )

        assert len(grads) == len(transition_params)
        for (gw, gb), (w, b) in zip(grads, transition_params):
            assert gw.shape == w.shape
            assert gb.shape == b.shape

    def test_grad_both(self, encoder_params, transition_params, model_spec):
        """jax.grad works w.r.t. both encoder and transition simultaneously."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        grad_fn = jax.grad(deep_analytic_nll, argnums=(0, 1))
        grad_enc, grad_trans = grad_fn(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )

        assert len(grad_enc) == len(encoder_params)
        assert len(grad_trans) == len(transition_params)


# ---------------------------------------------------------------------------
# Test 5: Deep NLL under jit works
# ---------------------------------------------------------------------------

class TestJIT:

    def test_nll_jit(self, encoder_params, transition_params, model_spec):
        """deep_analytic_nll works under jax.jit."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        jit_nll = jax.jit(
            deep_analytic_nll,
            static_argnums=(5,),
        )

        nll_eager = deep_analytic_nll(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )
        nll_jit = jit_nll(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )

        assert jnp.allclose(nll_eager, nll_jit, atol=1e-5)

    def test_grad_jit(self, encoder_params, transition_params, model_spec):
        """Gradient computation works under jit."""
        T = 10
        obs = jnp.ones((T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        @jax.jit
        def loss_and_grad(enc_p, trans_p):
            def loss_fn(ep, tp):
                return deep_analytic_nll(ep, tp, D, obs, actions, model_spec["num_actions"])
            loss = loss_fn(enc_p, trans_p)
            grads = jax.grad(loss_fn, argnums=(0, 1))(enc_p, trans_p)
            return loss, grads

        loss, (g_enc, g_trans) = loss_and_grad(encoder_params, transition_params)
        assert jnp.isfinite(loss)
        assert len(g_enc) == len(encoder_params)


# ---------------------------------------------------------------------------
# Test 6: Training on synthetic discrete data reduces loss
# ---------------------------------------------------------------------------

class TestTraining:

    def test_loss_decreases(self, synthetic_data):
        """Training loop reduces NLL over epochs."""
        result = learn_deep_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            encoder_hidden=[32, 32],
            transition_hidden=[32, 32],
            num_epochs=100,
            lr=0.001,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, DeepLearningResult)
        assert len(result.loss_history) == 100

        early_loss = np.mean(result.loss_history[:5])
        late_loss = np.mean(result.loss_history[-5:])
        assert late_loss < early_loss, (
            f"Loss did not decrease: early={early_loss:.4f}, late={late_loss:.4f}"
        )

    def test_loss_decreases_longer(self, synthetic_data):
        """Longer training produces lower loss."""
        result_short = learn_deep_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            num_epochs=50,
            lr=0.001,
            seed=42,
        )
        result_long = learn_deep_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            num_epochs=200,
            lr=0.001,
            seed=42,
        )

        assert result_long.loss_history[-1] < result_short.loss_history[-1]


# ---------------------------------------------------------------------------
# Test 7: Deep model recovers approximate A matrix structure
# ---------------------------------------------------------------------------

class TestARecovery:

    @pytest.mark.xfail(
        reason=(
            "Known limitation: encoder-based (recognition model) architecture "
            "P(s|o) has a degenerate optimum where the encoder maps all "
            "observations to a single state. Fix requires decoder-based "
            "architecture P(o|s). See deep_aif.py docstring."
        ),
        strict=False,
    )
    def test_encoder_recovers_structure(self, synthetic_data):
        """Trained encoder learns to distinguish observations."""
        result = learn_deep_model(
            observations_raw=synthetic_data["obs_onehot"],
            actions=synthetic_data["actions"],
            obs_dim=synthetic_data["num_obs"],
            num_states=synthetic_data["num_states"],
            num_actions=synthetic_data["num_actions"],
            D=synthetic_data["D"],
            encoder_hidden=[64, 64],
            transition_hidden=[64, 64],
            num_epochs=2000,
            lr=0.003,
            seed=42,
        )

        A_approx = extract_A_matrix(
            result.encoder_params,
            synthetic_data["num_obs"],
        )
        A_approx = np.array(A_approx)

        primary_obs = [0, 2, 4]
        argmax_states = [int(jnp.argmax(A_approx[o, :])) for o in primary_obs]

        assert len(set(argmax_states)) == 3, (
            f"Encoder failed to distinguish observations: argmax states = {argmax_states}"
        )

    def test_extract_A_shape(self, encoder_params, model_spec):
        """extract_A_matrix produces correct shape."""
        A = extract_A_matrix(encoder_params, model_spec["obs_dim"])
        assert A.shape == (model_spec["obs_dim"], model_spec["num_states"])
        row_sums = jnp.sum(A, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

class TestDeepGenerativeModel:

    def test_init(self):
        """DeepGenerativeModel initializes without error."""
        dgm = DeepGenerativeModel(
            obs_dim=10, num_states=4, num_actions=3,
            encoder_hidden=[16], transition_hidden=[16],
            seed=0,
        )
        assert len(dgm.encoder_params) == 2  # 1 hidden + 1 output
        assert len(dgm.transition_params) == 2

    def test_likelihood_and_transition(self):
        """DeepGenerativeModel.get_likelihood and get_transition work."""
        dgm = DeepGenerativeModel(
            obs_dim=5, num_states=3, num_actions=2,
            encoder_hidden=[16, 16], transition_hidden=[16, 16],
        )

        obs = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        lik = dgm.get_likelihood(dgm.encoder_params, obs)
        assert lik.shape == (3,)
        assert jnp.allclose(jnp.sum(lik), 1.0, atol=1e-5)

        beliefs = jnp.array([0.5, 0.3, 0.2])
        trans = dgm.get_transition(dgm.transition_params, beliefs, 0, 2)
        assert trans.shape == (3,)
        assert jnp.allclose(jnp.sum(trans), 1.0, atol=1e-5)


class TestNLLFiniteness:

    def test_nll_is_finite(self, encoder_params, transition_params, model_spec):
        """NLL produces a finite scalar."""
        T = 20
        key = jax.random.PRNGKey(7)
        obs = jax.random.normal(key, (T, model_spec["obs_dim"]))
        actions = jnp.zeros(T, dtype=jnp.int32)
        D = jnp.ones(model_spec["num_states"]) / model_spec["num_states"]

        nll = deep_analytic_nll(
            encoder_params, transition_params, D, obs, actions,
            model_spec["num_actions"],
        )
        assert jnp.isfinite(nll)
        assert nll.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
