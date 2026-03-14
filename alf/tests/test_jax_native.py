"""Tests for JAX-native inference functions and BatchAgent."""

import numpy as np
import jax
import jax.numpy as jnp

from alf.generative_model import GenerativeModel
from alf.jax_native import (
    jax_softmax, jax_select_action, jax_update_habits,
    jax_update_precision, jax_compute_efe_analytic,
    jax_evaluate_all_actions, BatchAgent,
)
from alf import policy as np_policy


def build_simple_model():
    """Build a minimal model for testing."""
    A = [np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])]
    B = [np.array([
        [[0.9, 0.1],
         [0.1, 0.9]],
        [[0.1, 0.9],
         [0.9, 0.1]],
    ]).transpose(1, 2, 0)]
    C = [np.array([2.0, -2.0])]
    D = [np.array([0.5, 0.5])]
    return GenerativeModel(A=A, B=B, C=C, D=D, T=1)


def test_jax_softmax():
    """Test JAX softmax matches numpy softmax."""
    x = jnp.array([1.0, 2.0, 3.0])
    result = jax_softmax(x)
    expected = np_policy._softmax(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


def test_jax_select_action_matches_numpy():
    """Test that JAX action selection produces valid distributions."""
    G = jnp.array([-1.0, 0.5])
    E = jnp.array([0.5, 0.5])
    key = jax.random.PRNGKey(42)

    action, probs = jax_select_action(G, E, 4.0, key)

    # Probs should sum to 1
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    # Policy 0 should be more probable (lower G)
    assert float(probs[0]) > float(probs[1])


def test_jax_select_action_jit():
    """Test that jax_select_action works under jit."""
    G = jnp.array([-1.0, 0.5])
    E = jnp.array([0.5, 0.5])
    key = jax.random.PRNGKey(42)

    jit_select = jax.jit(jax_select_action, static_argnums=(2,))
    action, probs = jit_select(G, E, 4.0, key)

    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_jax_update_habits():
    """Test JAX habit update."""
    E = jnp.array([0.5, 0.5])
    E_new = jax_update_habits(E, 0, 1.0, learning_rate=0.1)
    assert float(E_new[0]) > float(E_new[1])
    assert abs(float(E_new.sum()) - 1.0) < 1e-6


def test_analytic_efe():
    """Test analytic EFE computation."""
    gm = build_simple_model()
    A = jnp.array(gm.A[0])
    B = jnp.array(gm.B[0])
    C = jnp.array(gm.C[0])
    beliefs = jnp.array([0.9, 0.1])

    G = jax_evaluate_all_actions(A, B, C, beliefs)

    # When believing state 0 (preferred), stay should be better
    assert float(G[0]) < float(G[1]), "Stay should have lower EFE when in preferred state"


def test_analytic_efe_jit():
    """Test that analytic EFE works under jit."""
    gm = build_simple_model()
    A = jnp.array(gm.A[0])
    B = jnp.array(gm.B[0])
    C = jnp.array(gm.C[0])
    beliefs = jnp.array([0.9, 0.1])

    jit_efe = jax.jit(jax_evaluate_all_actions)
    G = jit_efe(A, B, C, beliefs)
    assert float(G[0]) < float(G[1])


def test_vmap_select_action():
    """Test vmap over a batch of agents."""
    batch_size = 4
    G_batch = jnp.array([
        [-1.0, 0.5],
        [0.5, -1.0],
        [-0.5, -0.5],
        [-2.0, 1.0],
    ])
    E_batch = jnp.ones((batch_size, 2)) * 0.5
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

    batch_select = jax.vmap(jax_select_action, in_axes=(0, 0, None, 0))
    actions, probs = batch_select(G_batch, E_batch, 4.0, keys)

    assert actions.shape == (batch_size,)
    assert probs.shape == (batch_size, 2)
    # Each agent's probs should sum to 1
    for i in range(batch_size):
        assert abs(float(probs[i].sum()) - 1.0) < 1e-6


def test_batch_agent():
    """Test BatchAgent with step_analytic."""
    gm = build_simple_model()
    batch_size = 4
    agent = BatchAgent(gm, batch_size=batch_size, gamma=4.0, seed=42)

    # All agents observe observation 0
    observations = np.array([0, 0, 1, 1])
    actions, info = agent.step_analytic(observations)

    assert actions.shape == (batch_size,)
    assert info["beliefs"][0].shape == (batch_size, 2)
    assert info["G"].shape == (batch_size, 2)
    assert info["policy_probs"].shape == (batch_size, 2)

    # Agents observing 0 should believe state 0 more
    assert info["beliefs"][0][0, 0] > info["beliefs"][0][0, 1]
    # Agents observing 1 should believe state 1 more
    assert info["beliefs"][0][2, 1] > info["beliefs"][0][2, 0]


def test_batch_agent_learning():
    """Test BatchAgent habit learning."""
    gm = build_simple_model()
    agent = BatchAgent(gm, batch_size=2, gamma=4.0, seed=42)

    observations = np.array([0, 1])
    actions, _ = agent.step_analytic(observations)

    # Learn from outcomes
    outcomes = np.array([1.0, -1.0])
    agent.learn_batch(outcomes, actions)

    # Habits should have diverged
    assert not np.allclose(agent.E[0], agent.E[1])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
