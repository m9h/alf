"""Tests for the T-maze benchmark (ALF version).

Tests model construction, environment, cue inference via Bayesian update,
and sequential epistemic planning. BP-specific tests are excluded (they
remain in pgmax/aif).
"""

import numpy as np
from alf.benchmarks.t_maze import (
    build_t_maze_model, TMazeEnv, run_t_maze,
    NUM_STATES, NUM_OBS, NUM_ACTIONS, ACT_STAY, ACT_CUE, ACT_LEFT, ACT_RIGHT,
    OBS_NULL, OBS_CUE_LEFT, OBS_CUE_RIGHT, OBS_REWARD, OBS_PUNISHMENT,
    _build_A, _build_B, _build_D,
)


def test_model_construction():
    """Test that the T-maze generative model builds correctly."""
    gm = build_t_maze_model()
    assert gm.num_modalities == 1
    assert gm.num_factors == 1
    assert gm.num_obs == [NUM_OBS]
    assert gm.num_states == [NUM_STATES]
    assert gm.num_actions == [NUM_ACTIONS]
    assert gm.T == 2

    # A matrix: columns should sum to 1 (probability distributions)
    A = gm.A[0]
    col_sums = A.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    # B matrix: columns should sum to 1 for each action
    B = gm.B[0]
    for a in range(NUM_ACTIONS):
        col_sums = B[:, :, a].sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    # D: should sum to 1
    np.testing.assert_allclose(gm.D[0].sum(), 1.0, atol=1e-10)

    # Policies: T=2 with 4 actions = 4^2 = 16 policies
    assert gm.num_policies == 16


def test_environment():
    """Test the T-maze environment."""
    env = TMazeEnv(reward_side="left", cue_reliability=1.0, seed=0)
    obs = env.reset()
    assert obs == OBS_NULL
    assert env.location == "center"

    # Go to cue -- with reliability=1.0, should see cue_left
    obs, reward, done = env.step(ACT_CUE)
    assert obs == OBS_CUE_LEFT
    assert reward == 0.0
    assert not done

    # Go to left arm -- should get reward (reward is left)
    obs, reward, done = env.step(ACT_LEFT)
    assert obs == OBS_REWARD
    assert reward == 1.0
    assert done

    # Test wrong arm
    env2 = TMazeEnv(reward_side="left", cue_reliability=1.0, seed=0)
    env2.reset()
    env2.step(ACT_CUE)
    obs, reward, done = env2.step(ACT_RIGHT)
    assert obs == OBS_PUNISHMENT
    assert reward == -1.0
    assert done


def test_cue_inference_bayesian():
    """Test that observing the cue updates beliefs correctly via Bayesian update.

    This is the analytic (non-BP) version of cue inference.
    """
    A = _build_A(0.9)
    B = _build_B()
    D = _build_D()

    # Transition to cue location
    predicted = B[:, :, ACT_CUE] @ D

    # Bayesian update with cue_left observation
    likelihood = A[OBS_CUE_LEFT, :]
    updated = predicted * likelihood
    updated = updated / updated.sum()

    # States 0,2,4,6 are reward-left; states 1,3,5,7 are reward-right
    left_prob = sum(updated[i] for i in [0, 2, 4, 6])
    right_prob = sum(updated[i] for i in [1, 3, 5, 7])
    assert left_prob > right_prob, (
        f"After cue_left: P(left)={left_prob:.3f} should > P(right)={right_prob:.3f}"
    )

    # After seeing cue_right, should believe reward is right
    likelihood_right = A[OBS_CUE_RIGHT, :]
    updated_right = predicted * likelihood_right
    updated_right = updated_right / updated_right.sum()
    left_prob_r = sum(updated_right[i] for i in [0, 2, 4, 6])
    right_prob_r = sum(updated_right[i] for i in [1, 3, 5, 7])
    assert right_prob_r > left_prob_r, (
        f"After cue_right: P(right)={right_prob_r:.3f} should > P(left)={left_prob_r:.3f}"
    )


def test_analytic_efe():
    """Test that analytic EFE differentiates actions at T=1."""
    from alf.jax_native import jax_compute_efe_analytic
    import jax.numpy as jnp

    gm = build_t_maze_model(cue_reliability=0.9, T=1)
    A = jnp.array(gm.A[0])
    B = jnp.array(gm.B[0])
    C = jnp.array(gm.C[0])
    b = jnp.array(gm.D[0])

    G_analytic = np.array([
        float(jax_compute_efe_analytic(A, B[:, :, a], C, b))
        for a in range(NUM_ACTIONS)
    ])

    assert all(np.isfinite(G_analytic)), "Analytic EFE should be finite"
    assert G_analytic[ACT_CUE] != G_analytic[ACT_STAY], (
        "Cue and stay should have different analytic EFE"
    )
    # Left and right should be symmetric (agent doesn't know reward side)
    assert abs(G_analytic[ACT_LEFT] - G_analytic[ACT_RIGHT]) < 0.1, (
        f"Left ({G_analytic[ACT_LEFT]:.3f}) and right ({G_analytic[ACT_RIGHT]:.3f}) "
        f"should be symmetric"
    )


def test_cue_reliability_affects_model():
    """Test that cue reliability changes the A-matrix observation structure."""
    gm_high = build_t_maze_model(cue_reliability=0.95)
    gm_low = build_t_maze_model(cue_reliability=0.55)

    A_high = gm_high.A[0]
    A_low = gm_low.A[0]

    assert not np.allclose(A_high[:, 2], A_low[:, 2]), (
        "Cue reliability should affect A-matrix at cue states"
    )

    # High reliability: more peaked distribution at cue
    entropy_high = -np.sum(A_high[:, 2] * np.log(A_high[:, 2] + 1e-16))
    entropy_low = -np.sum(A_low[:, 2] * np.log(A_low[:, 2] + 1e-16))
    assert entropy_high < entropy_low, (
        f"High reliability cue should have lower entropy: {entropy_high:.3f} vs {entropy_low:.3f}"
    )


def test_full_benchmark():
    """Run a short benchmark and check it completes."""
    results = run_t_maze(num_trials=3, gamma=4.0, seed=42, verbose=False)
    assert "reward_rate" in results
    assert "cue_visit_rate" in results
    assert len(results["trial_log"]) == 3


def test_sequential_epistemic_drive():
    """Test that sequential EFE captures the value of information at T=2."""
    from alf.sequential_efe import (
        sequential_efe,
        evaluate_all_policies_sequential,
    )

    gm = build_t_maze_model(cue_reliability=0.9, reward_magnitude=3.0, T=2)
    A = gm.A[0]
    B = gm.B[0]
    C = gm.C[0]
    D = gm.D[0]

    cue_then_left = np.array([ACT_CUE, ACT_LEFT])
    cue_then_right = np.array([ACT_CUE, ACT_RIGHT])
    left_then_stay = np.array([ACT_LEFT, ACT_STAY])
    right_then_stay = np.array([ACT_RIGHT, ACT_STAY])

    g_cue_left = sequential_efe(A, B, C, D, cue_then_left)
    g_cue_right = sequential_efe(A, B, C, D, cue_then_right)
    g_left = sequential_efe(A, B, C, D, left_then_stay)
    g_right = sequential_efe(A, B, C, D, right_then_stay)

    # Core assertion: cue-first policies dominate
    best_cue = min(g_cue_left, g_cue_right)
    best_arm = min(g_left, g_right)
    assert best_cue < best_arm, (
        f"Cue-first ({best_cue:.4f}) should have lower G than "
        f"arm-first ({best_arm:.4f})"
    )

    # Symmetry checks
    np.testing.assert_allclose(
        g_cue_left, g_cue_right, atol=1e-10,
        err_msg="Cue-then-left and cue-then-right should be symmetric",
    )
    np.testing.assert_allclose(
        g_left, g_right, atol=1e-10,
        err_msg="Direct left and direct right should be symmetric",
    )

    # Full policy evaluation should also show cue-first dominance
    G_all = evaluate_all_policies_sequential(gm, [D])
    cue_first_Gs = []
    arm_first_Gs = []
    for i, policy in enumerate(gm.policies):
        first_action = policy[0, 0]
        if first_action == ACT_CUE:
            cue_first_Gs.append(G_all[i])
        elif first_action in (ACT_LEFT, ACT_RIGHT):
            arm_first_Gs.append(G_all[i])

    assert min(cue_first_Gs) < min(arm_first_Gs), (
        "Across all policies, best cue-first should beat best arm-first"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
