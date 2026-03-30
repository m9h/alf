"""Tests for AnalyticAgent."""

import numpy as np

from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


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


def test_init():
    """Test that the agent initializes properly."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, gamma=4.0, learning_rate=0.1, seed=42)

    assert agent.gamma == 4.0
    assert agent.learning_rate == 0.1

    # Beliefs initialized from prior
    assert len(agent.beliefs) == gm.num_factors
    np.testing.assert_array_almost_equal(agent.beliefs[0], gm.D[0])

    # Habits initialized from prior
    np.testing.assert_array_almost_equal(agent.E, gm.E)

    # Histories empty
    assert len(agent.belief_history) == 0
    assert len(agent.action_history) == 0
    assert len(agent.efe_history) == 0
    assert len(agent.policy_prob_history) == 0


def test_step():
    """Test step updates histories and returns an action."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, seed=42)

    obs = [0]
    action, info = agent.step(obs)

    # Check histories updated
    assert len(agent.belief_history) == 1
    assert len(agent.action_history) == 1
    assert len(agent.efe_history) == 1
    assert len(agent.policy_prob_history) == 1

    # Check return values
    assert isinstance(action, int)
    assert 0 <= action < gm.num_actions[0]

    assert "beliefs" in info
    assert "G" in info
    assert "policy_probs" in info
    assert "selected_policy" in info

    # Beliefs should have shifted towards state 0 based on observation 0
    assert info["beliefs"][0][0] > info["beliefs"][0][1]


def test_learn():
    """Test habit learning."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, seed=42)

    # Initial state
    initial_E = agent.E.copy()

    # Step to populate histories
    agent.step([0])

    # Learn from positive outcome
    agent.learn(outcome_valence=1.0)

    # Habits should change
    assert not np.allclose(agent.E, initial_E)

    # Selected policy probability should increase
    last_policy_idx = agent.policy_prob_history[-1].argmax()
    assert agent.E[last_policy_idx] > initial_E[last_policy_idx]


def test_update_precision():
    """Test precision updating."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, gamma=4.0)

    # With positive PE, precision should decrease
    agent.update_precision(prediction_error=1.0)
    assert agent.gamma < 4.0


def test_reset():
    """Test resetting beliefs and histories."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, seed=42)

    # Step and learn to change state
    agent.step([0])
    agent.learn(1.0)

    modified_E = agent.E.copy()

    # Reset
    agent.reset()

    # Histories empty
    assert len(agent.belief_history) == 0
    assert len(agent.action_history) == 0
    assert len(agent.efe_history) == 0
    assert len(agent.policy_prob_history) == 0

    # Beliefs back to prior
    np.testing.assert_array_almost_equal(agent.beliefs[0], gm.D[0])

    # Habits (E) should NOT reset
    np.testing.assert_array_almost_equal(agent.E, modified_E)


def test_get_state_summary():
    """Test getting state summary."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, seed=42)

    agent.step([0])

    summary = agent.get_state_summary()

    assert "beliefs" in summary
    assert "factor_0" in summary["beliefs"]
    assert "gamma" in summary
    assert "E" in summary
    assert "num_actions_taken" in summary
    assert summary["num_actions_taken"] == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])