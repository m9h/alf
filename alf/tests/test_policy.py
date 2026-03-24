import numpy as np
import pytest

from alf.policy import (
    _softmax,
    select_action,
    update_habits,
    update_precision,
)


def test_softmax():
    # Test standard output
    x1 = np.array([1.0, 2.0, 3.0])
    res1 = _softmax(x1)
    assert np.allclose(res1.sum(), 1.0)
    assert res1[2] > res1[1] > res1[0]

    # Test numerical stability with large numbers
    x2 = np.array([1000.0, 1001.0, 1002.0])
    res2 = _softmax(x2)
    assert np.allclose(res2.sum(), 1.0)
    assert np.allclose(res1, res2)

    # Test zeros
    x3 = np.array([0.0, 0.0, 0.0])
    res3 = _softmax(x3)
    assert np.allclose(res3, np.array([1/3, 1/3, 1/3]))


def test_select_action():
    # G mapping to policy_probs (lowest G should have highest probability if E is uniform)
    G1 = np.array([3.0, 1.0, 2.0])
    E1 = np.array([1/3, 1/3, 1/3])
    idx1, probs1 = select_action(G1, E1, gamma=1.0)
    assert probs1[1] > probs1[2] > probs1[0]
    assert np.allclose(probs1.sum(), 1.0)

    # E array (habits) influencing the policy_probs (higher E should increase probability)
    G2 = np.array([1.0, 1.0, 1.0])
    E2 = np.array([0.1, 0.8, 0.1])
    idx2, probs2 = select_action(G2, E2, gamma=1.0)
    assert probs2[1] > probs2[0]
    assert np.allclose(probs2[0], probs2[2])
    assert np.allclose(probs2.sum(), 1.0)

    # Gamma adjusting the sharpness
    G3 = np.array([2.0, 1.0])
    E3 = np.array([0.5, 0.5])
    _, probs3_low_gamma = select_action(G3, E3, gamma=0.1)
    _, probs3_high_gamma = select_action(G3, E3, gamma=10.0)
    # High gamma makes the distribution sharper (more deterministic)
    assert probs3_high_gamma[1] > probs3_low_gamma[1]

    # E values clip
    G4 = np.array([1.0, 1.0])
    E4 = np.array([1.0, 0.0])
    _, probs4 = select_action(G4, E4, gamma=1.0)
    assert probs4[1] > 0.0 # Should be very small, but not exactly 0 or nan

    # Reproducible sampling using rng
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    G5 = np.array([1.0, 2.0, 1.5, 3.0])
    E5 = np.array([0.25, 0.25, 0.25, 0.25])
    idx5_1, probs5_1 = select_action(G5, E5, gamma=1.0, rng=rng1)
    idx5_2, probs5_2 = select_action(G5, E5, gamma=1.0, rng=rng2)
    assert idx5_1 == idx5_2
    assert np.allclose(probs5_1, probs5_2)


def test_update_habits():
    E = np.array([0.2, 0.5, 0.3])

    # positive outcome_valence
    E_new_pos = update_habits(E, selected_policy=1, outcome_valence=1.0, learning_rate=0.1)
    assert E_new_pos[1] > E[1] # Should increase
    assert np.allclose(E_new_pos.sum(), 1.0) # Normalized

    # negative outcome_valence
    E_new_neg = update_habits(E, selected_policy=1, outcome_valence=-1.0, learning_rate=0.1)
    assert E_new_neg[1] < E[1] # Should decrease
    assert np.allclose(E_new_neg.sum(), 1.0) # Normalized

    # verify 1e-8 clip bounds
    E_zero = np.array([0.5, 0.5])
    E_clipped = update_habits(E_zero, selected_policy=1, outcome_valence=-1000.0, learning_rate=1.0)
    assert E_clipped[1] > 0.0 # Bounded to 1e-8 before norm


def test_update_precision():
    gamma = 1.0

    # adapting based on prediction error correctly updates the value
    # target = 1 / (1 + pred_err)
    gamma_new = update_precision(gamma, prediction_error=0.0, learning_rate=0.3)
    # target is 1.0, gamma is 1.0, so no change
    assert np.isclose(gamma_new, 1.0)

    gamma_new2 = update_precision(gamma, prediction_error=1.0, learning_rate=0.3)
    # target is 0.5, gamma is 1.0 => (0.7)*1 + (0.3)*0.5 = 0.85
    assert np.isclose(gamma_new2, 0.85)

    # respects min_gamma and max_gamma boundaries
    gamma_min = update_precision(0.1, prediction_error=100.0, learning_rate=1.0, min_gamma=0.5)
    assert np.isclose(gamma_min, 0.5)

    gamma_max = update_precision(10.0, prediction_error=0.0, learning_rate=0.0, max_gamma=5.0)
    assert np.isclose(gamma_max, 5.0)
