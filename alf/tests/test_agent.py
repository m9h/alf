"""Tests for the AnalyticAgent class."""

import numpy as np

from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent
import alf.policy as alf_policy


def build_simple_model():
    """Build a minimal model for testing."""
    A = [
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
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


def test_analytic_agent_update_precision():
    """Test that AnalyticAgent.update_precision correctly updates gamma."""
    gm = build_simple_model()
    agent = AnalyticAgent(gm, gamma=4.0)

    # 1. Normal prediction error
    prediction_error = 0.5
    agent.update_precision(prediction_error)
    expected_gamma = alf_policy.update_precision(4.0, prediction_error)
    assert np.isclose(agent.gamma, expected_gamma)
    assert np.isclose(agent.gamma, 3.0), f"Expected 3.0, got {agent.gamma}"

    # 2. Zero prediction error
    agent.gamma = 4.0
    prediction_error = 0.0
    agent.update_precision(prediction_error)
    expected_gamma = alf_policy.update_precision(4.0, prediction_error)
    assert np.isclose(agent.gamma, expected_gamma)
    assert np.isclose(agent.gamma, 3.1), f"Expected 3.1, got {agent.gamma}"

    # 3. Large prediction error
    agent.gamma = 4.0
    prediction_error = 9.0
    agent.update_precision(prediction_error)
    expected_gamma = alf_policy.update_precision(4.0, prediction_error)
    assert np.isclose(agent.gamma, expected_gamma)
    assert np.isclose(agent.gamma, 2.83), f"Expected 2.83, got {agent.gamma}"

    # 4. Forcing a clip below 0.1
    agent.gamma = 0.05
    prediction_error = 99.0
    agent.update_precision(prediction_error)
    expected_gamma = alf_policy.update_precision(0.05, prediction_error)
    assert np.isclose(agent.gamma, expected_gamma)
    assert np.isclose(agent.gamma, 0.1), f"Expected 0.1, got {agent.gamma}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
