"""Closed-loop filtering and habit updates must follow executed actions."""

import numpy as np

from alf.agent import AnalyticAgent
from alf.generative_model import GenerativeModel


def test_filter_predicts_through_previous_action_and_reset_clears_it():
    gm = GenerativeModel(
        A=[np.ones((1, 2))],
        B=[np.array([[0, 1], [1, 0]])[:, :, None]],
        C=[np.zeros(1)],
        D=[np.array([1.0, 0.0])],
    )
    agent = AnalyticAgent(gm)
    agent.step([0])
    np.testing.assert_allclose(agent.beliefs[0], [1, 0], atol=1e-12)
    agent.step([0])
    np.testing.assert_allclose(agent.beliefs[0], [0, 1], atol=1e-12)
    agent.reset()
    agent.step([0])
    np.testing.assert_allclose(agent.beliefs[0], [1, 0], atol=1e-12)


def test_habit_learning_credits_sampled_policy():
    gm = GenerativeModel(
        A=[np.ones((1, 1))], B=[np.ones((1, 1, 2))], C=[np.zeros(1)], D=[np.ones(1)]
    )
    agent = AnalyticAgent(gm, gamma=0, seed=0)
    _, info = agent.step([0])
    assert info["selected_policy"] == 1
    assert info["policy_probs"].argmax() == 0
    agent.learn(1.0)
    assert agent.E[1] > agent.E[0]
