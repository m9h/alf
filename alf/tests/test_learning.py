"""Tests for differentiable learning of A and B matrices.

Tests the learning module, verifying:
1. Parameter conversion produces valid probability matrices
2. Analytic NLL is differentiable (jax.grad works)
3. Learning recovers known A/B matrices from generated data
4. Integration with the simple 2-state model
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.generative_model import GenerativeModel
from alf.learning import (
    LearnableGenerativeModel,
    LearnableParams,
    params_to_A,
    params_to_B,
    params_to_matrices,
    analytic_nll,
    analytic_nll_single,
    learn_model,
    learn_from_agent_data,
    generate_data,
    compute_observation_log_likelihood_analytic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_simple_model():
    """Build the same 2-state, 2-obs, 2-action model."""
    A = [
        np.array(
            [
                [0.9, 0.1],  # P(o=0 | s=0)=0.9, P(o=0 | s=1)=0.1
                [0.1, 0.9],  # P(o=1 | s=0)=0.1, P(o=1 | s=1)=0.9
            ]
        )
    ]
    B = [
        np.array(
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.1, 0.9], [0.9, 0.1]],
            ]
        ).transpose(1, 2, 0)
    ]
    C = [np.array([2.0, -2.0])]
    D = [np.array([0.5, 0.5])]
    return GenerativeModel(A=A, B=B, C=C, D=D, T=1)


def build_distinctive_model():
    """Build a 3-state, 3-obs, 2-action model with distinctive parameters."""
    A = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    # Action 0: rotate states forward (0->1->2->0)
    B_a0 = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    # Action 1: stay
    B_a1 = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    B = np.stack([B_a0, B_a1], axis=-1)  # (3, 3, 2)
    D = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return A, B, D


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_params_to_A_valid():
    """Test that params_to_A produces a valid probability matrix."""
    log_A = [
        jnp.array(
            [
                [1.0, -1.0],
                [-0.5, 2.0],
            ]
        )
    ]

    A_list = params_to_A(log_A)
    A = A_list[0]

    assert A.shape == (2, 2), f"Expected shape (2,2), got {A.shape}"
    assert jnp.all(A >= 0), "A has negative entries"

    col_sums = jnp.sum(A, axis=0)
    np.testing.assert_allclose(
        np.array(col_sums), np.ones(2), atol=1e-6, err_msg="Columns of A don't sum to 1"
    )


def test_params_to_B_valid():
    """Test that params_to_B produces a valid transition matrix."""
    log_B = [
        jnp.array(
            [
                [[0.5, -0.5], [-1.0, 1.0]],
                [[-0.5, 0.5], [1.0, -1.0]],
            ]
        )
    ]

    B_list = params_to_B(log_B)
    B = B_list[0]

    assert B.shape == (2, 2, 2), f"Expected shape (2,2,2), got {B.shape}"
    assert jnp.all(B >= 0), "B has negative entries"

    for a in range(2):
        col_sums = jnp.sum(B[:, :, a], axis=0)
        np.testing.assert_allclose(
            np.array(col_sums),
            np.ones(2),
            atol=1e-6,
            err_msg=f"Columns of B[:,:,{a}] don't sum to 1",
        )


def test_params_to_A_zeros_gives_uniform():
    """Test that zero parameters produce a uniform distribution."""
    num_obs, num_states = 3, 2
    log_A = [jnp.zeros((num_obs, num_states))]
    A = params_to_A(log_A)[0]

    expected = 1.0 / num_obs
    np.testing.assert_allclose(
        np.array(A),
        np.full((num_obs, num_states), expected),
        atol=1e-6,
    )


def test_from_model_roundtrip():
    """Test that from_model + params_to_A/B approximately recovers original."""
    gm = build_simple_model()
    lgm = LearnableGenerativeModel.from_model(gm)

    A_recovered = params_to_A(lgm.params.log_A_params)
    B_recovered = params_to_B(lgm.params.log_B_params)

    np.testing.assert_allclose(
        np.array(A_recovered[0]),
        gm.A[0],
        atol=1e-4,
        err_msg="A matrix not recovered from log-params",
    )
    np.testing.assert_allclose(
        np.array(B_recovered[0]),
        gm.B[0],
        atol=1e-4,
        err_msg="B matrix not recovered from log-params",
    )


def test_analytic_nll_differentiable():
    """Test that analytic NLL is differentiable with jax.grad."""
    gm = build_simple_model()
    lgm = LearnableGenerativeModel.from_model(gm)

    obs = jnp.array([0, 0, 1, 0], dtype=jnp.int32)
    acts = jnp.array([0, 1, 0, 0], dtype=jnp.int32)

    log_A = lgm.params.log_A_params[0]
    log_B = lgm.params.log_B_params[0]
    D = lgm.D[0]

    # Compute NLL
    nll = analytic_nll_single(log_A, log_B, D, obs, acts)
    assert jnp.isfinite(nll), f"NLL is not finite: {nll}"

    # Compute gradient w.r.t. log_A
    grad_A_fn = jax.grad(analytic_nll_single, argnums=0)
    grad_A = grad_A_fn(log_A, log_B, D, obs, acts)
    assert grad_A.shape == log_A.shape, "Gradient shape mismatch for A"
    assert jnp.all(jnp.isfinite(grad_A)), "Gradient of A contains non-finite values"

    # Compute gradient w.r.t. log_B
    grad_B_fn = jax.grad(analytic_nll_single, argnums=1)
    grad_B = grad_B_fn(log_A, log_B, D, obs, acts)
    assert grad_B.shape == log_B.shape, "Gradient shape mismatch for B"
    assert jnp.all(jnp.isfinite(grad_B)), "Gradient of B contains non-finite values"

    # Compute both gradients simultaneously
    grad_both_fn = jax.grad(analytic_nll_single, argnums=(0, 1))
    grad_A2, grad_B2 = grad_both_fn(log_A, log_B, D, obs, acts)
    np.testing.assert_allclose(np.array(grad_A), np.array(grad_A2), atol=1e-6)
    np.testing.assert_allclose(np.array(grad_B), np.array(grad_B2), atol=1e-6)


def test_analytic_nll_jit():
    """Test that analytic NLL works under jax.jit."""
    gm = build_simple_model()
    lgm = LearnableGenerativeModel.from_model(gm)

    obs = jnp.array([0, 0, 1, 0], dtype=jnp.int32)
    acts = jnp.array([0, 1, 0, 0], dtype=jnp.int32)

    log_A = lgm.params.log_A_params[0]
    log_B = lgm.params.log_B_params[0]
    D = lgm.D[0]

    nll_jit = jax.jit(analytic_nll_single)
    nll_val = nll_jit(log_A, log_B, D, obs, acts)
    nll_ref = analytic_nll_single(log_A, log_B, D, obs, acts)
    np.testing.assert_allclose(float(nll_val), float(nll_ref), atol=1e-6)

    grad_jit = jax.jit(jax.grad(analytic_nll_single, argnums=(0, 1)))
    grad_A, grad_B = grad_jit(log_A, log_B, D, obs, acts)
    assert jnp.all(jnp.isfinite(grad_A))
    assert jnp.all(jnp.isfinite(grad_B))


def test_nll_prefers_true_model():
    """Test that NLL is lower for the true model than a random one."""
    gm = build_simple_model()

    np.random.seed(42)
    T = 200
    actions = np.random.randint(0, 2, size=T)
    obs, states = generate_data(gm.A[0], gm.B[0], gm.D[0], actions, seed=42)

    D = jnp.array(gm.D[0])
    obs_jnp = jnp.array(obs, dtype=jnp.int32)
    acts_jnp = jnp.array(np.concatenate([actions, [0]]), dtype=jnp.int32)

    # NLL with true parameters
    log_A_true = jnp.log(jnp.clip(jnp.array(gm.A[0]), 1e-16))
    log_B_true = jnp.log(jnp.clip(jnp.array(gm.B[0]), 1e-16))
    nll_true = float(analytic_nll_single(log_A_true, log_B_true, D, obs_jnp, acts_jnp))

    # NLL with random (uniform) parameters
    log_A_random = jnp.zeros((2, 2))
    log_B_random = jnp.zeros((2, 2, 2))
    nll_random = float(
        analytic_nll_single(log_A_random, log_B_random, D, obs_jnp, acts_jnp)
    )

    assert nll_true < nll_random, (
        f"True model should have lower NLL: {nll_true:.4f} vs {nll_random:.4f}"
    )


def test_generate_data():
    """Test that data generation produces valid outputs."""
    gm = build_simple_model()
    actions = np.array([0, 1, 0, 0, 1])
    obs, states = generate_data(gm.A[0], gm.B[0], gm.D[0], actions, seed=42)

    assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"
    assert states.shape == (6,), f"Expected states shape (6,), got {states.shape}"
    assert np.all((obs >= 0) & (obs < 2)), "Observations out of range"
    assert np.all((states >= 0) & (states < 2)), "States out of range"


def _best_permutation_error(learned, true, axis=1):
    """Compute min error over state permutations (handles label switching)."""
    from itertools import permutations

    n = true.shape[axis]
    best_error = np.inf
    best_perm = None
    for perm in permutations(range(n)):
        perm = list(perm)
        if axis == 1 and true.ndim == 2:
            permuted = learned[:, perm]
        elif axis == 1 and true.ndim == 3:
            permuted = learned[np.ix_(perm, perm, range(true.shape[2]))]
        else:
            permuted = learned
        err = np.max(np.abs(permuted - true))
        if err < best_error:
            best_error = err
            best_perm = perm
    return best_error, best_perm


def test_learn_model_simple():
    """Test that learning recovers known A/B from generated data."""
    gm = build_simple_model()
    true_A = gm.A[0]
    true_B = gm.B[0]

    T = 200
    np.random.seed(42)
    actions = np.random.randint(0, 2, size=T)
    obs, states = generate_data(true_A, true_B, gm.D[0], actions, seed=42)

    actions_padded = np.concatenate([actions, [0]])

    result = learn_model(
        observations=obs,
        actions=actions_padded,
        num_obs=2,
        num_states=2,
        num_actions=2,
        D=gm.D[0],
        num_epochs=300,
        lr=0.01,
        verbose=False,
    )

    learned_A = result.learned_A[0]
    learned_B = result.learned_B[0]

    # Check that loss decreased
    assert result.loss_history[-1] < result.loss_history[0], (
        f"Loss should decrease: {result.loss_history[0]:.4f} -> "
        f"{result.loss_history[-1]:.4f}"
    )

    # Check A recovery up to state permutation (label switching)
    a_error, best_perm = _best_permutation_error(learned_A, true_A)
    assert a_error < 0.15, f"A matrix error too large (best perm): {a_error:.4f}"

    # Check B recovery under same permutation
    perm = best_perm
    learned_B_perm = learned_B[np.ix_(perm, perm, range(true_B.shape[2]))]
    b_error = np.max(np.abs(learned_B_perm - true_B))
    assert b_error < 0.25, f"B matrix error too large: {b_error:.4f}"

    # Verify the returned GenerativeModel is valid
    assert result.generative_model.num_obs == [2]
    assert result.generative_model.num_states == [2]
    assert result.generative_model.num_actions == [2]


def test_learn_model_3state():
    """Test learning on a 3-state model with distinctive structure."""
    A_true, B_true, D = build_distinctive_model()

    T = 500
    np.random.seed(123)
    actions = np.random.randint(0, 2, size=T)
    obs, states = generate_data(A_true, B_true, D, actions, seed=123)
    actions_padded = np.concatenate([actions, [0]])

    result = learn_model(
        observations=obs,
        actions=actions_padded,
        num_obs=3,
        num_states=3,
        num_actions=2,
        D=D,
        num_epochs=500,
        lr=0.01,
        verbose=False,
    )

    learned_A = result.learned_A[0]

    # Loss should decrease
    assert result.loss_history[-1] < result.loss_history[0], "Loss should decrease"

    # Check A recovery up to state permutation (label switching)
    a_error, best_perm = _best_permutation_error(learned_A, A_true)
    assert a_error < 0.20, f"A matrix error too large (best perm): {a_error:.4f}"

    # Under best permutation, A should have dominant diagonal
    learned_A_perm = learned_A[:, best_perm]
    for s in range(3):
        assert learned_A_perm[s, s] > 0.4, (
            f"A_perm[{s},{s}] should be dominant: {learned_A_perm[s, s]:.3f}"
        )


def test_learn_from_agent_data():
    """Test learning from agent history format."""
    gm = build_simple_model()

    T = 100
    np.random.seed(42)
    true_state = 0
    observation_history = []
    action_history = []

    for t in range(T):
        obs_probs = gm.A[0][:, true_state]
        obs = int(np.random.choice(2, p=obs_probs))
        observation_history.append([obs])

        action = int(np.random.choice(2))
        action_history.append(action)

        trans_probs = gm.B[0][:, true_state, action]
        true_state = int(np.random.choice(2, p=trans_probs))

    result = learn_from_agent_data(
        observation_history=observation_history,
        action_history=action_history,
        num_obs=2,
        num_states=2,
        num_actions=2,
        D=gm.D[0],
        num_epochs=200,
        lr=0.01,
        verbose=False,
    )

    # Loss should decrease
    assert result.loss_history[-1] < result.loss_history[0], "Loss should decrease"

    # Should produce valid matrices
    learned_A = result.learned_A[0]
    col_sums = np.sum(learned_A, axis=0)
    np.testing.assert_allclose(col_sums, np.ones(2), atol=1e-5)


def test_learnable_params_namedtuple():
    """Test that LearnableParams works as a NamedTuple for JAX tree ops."""
    log_A = [jnp.zeros((2, 2))]
    log_B = [jnp.zeros((2, 2, 2))]
    params = LearnableParams(log_A_params=log_A, log_B_params=log_B)

    assert len(params.log_A_params) == 1
    assert len(params.log_B_params) == 1
    assert params.log_A_params[0].shape == (2, 2)
    assert params.log_B_params[0].shape == (2, 2, 2)


def test_compute_observation_log_likelihood():
    """Test the log-likelihood wrapper function."""
    gm = build_simple_model()
    lgm = LearnableGenerativeModel.from_model(gm)

    obs = jnp.array([0, 0, 1, 0], dtype=jnp.int32)
    acts = jnp.array([0, 1, 0, 0], dtype=jnp.int32)

    ll = compute_observation_log_likelihood_analytic(lgm.params, lgm.D, obs, acts)
    nll = analytic_nll(
        lgm.params.log_A_params,
        lgm.params.log_B_params,
        lgm.D,
        obs,
        acts,
    )

    np.testing.assert_allclose(float(ll), -float(nll), atol=1e-6)
    assert ll <= 0, f"Log-likelihood should be <= 0, got {float(ll)}"


def test_gradient_descent_direction():
    """Test that one gradient step actually reduces the NLL."""
    gm = build_simple_model()

    actions = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    obs, _ = generate_data(gm.A[0], gm.B[0], gm.D[0], actions, seed=42)
    acts_padded = np.concatenate([actions, [0]])

    obs_jnp = jnp.array(obs, dtype=jnp.int32)
    acts_jnp = jnp.array(acts_padded, dtype=jnp.int32)
    D = jnp.array(gm.D[0])

    # Start from uniform (zeros in log-space)
    log_A = jnp.zeros((2, 2))
    log_B = jnp.zeros((2, 2, 2))

    nll_before = float(analytic_nll_single(log_A, log_B, D, obs_jnp, acts_jnp))

    grad_fn = jax.grad(analytic_nll_single, argnums=(0, 1))
    grad_A, grad_B = grad_fn(log_A, log_B, D, obs_jnp, acts_jnp)

    lr = 0.1
    log_A_new = log_A - lr * grad_A
    log_B_new = log_B - lr * grad_B

    nll_after = float(analytic_nll_single(log_A_new, log_B_new, D, obs_jnp, acts_jnp))

    assert nll_after < nll_before, (
        f"NLL should decrease after gradient step: {nll_before:.4f} -> {nll_after:.4f}"
    )


def test_params_to_matrices():
    """Test the combined conversion function."""
    log_A = [jnp.array([[1.0, -1.0], [-1.0, 1.0]])]
    log_B = [
        jnp.array(
            [
                [[0.5, -0.5], [-1.0, 1.0]],
                [[-0.5, 0.5], [1.0, -1.0]],
            ]
        )
    ]
    params = LearnableParams(log_A_params=log_A, log_B_params=log_B)

    A_list, B_list = params_to_matrices(params)
    assert len(A_list) == 1
    assert len(B_list) == 1

    col_sums = jnp.sum(A_list[0], axis=0)
    np.testing.assert_allclose(np.array(col_sums), np.ones(2), atol=1e-6)

    for a in range(2):
        col_sums = jnp.sum(B_list[0][:, :, a], axis=0)
        np.testing.assert_allclose(np.array(col_sums), np.ones(2), atol=1e-6)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
