"""Tests for sequential Expected Free Energy.

Verifies that:
1. Sequential EFE matches single-step analytic EFE for T=1
2. T-maze scores reflect information gain under open-loop planning
3. JAX version matches numpy version
4. JIT compilation works correctly
"""

import numpy as np

from alf.sequential_efe import (
    sequential_efe,
    evaluate_all_policies_sequential,
    select_action_sequential,
)
from alf.benchmarks.t_maze import (
    build_t_maze_model,
    ACT_STAY,
    ACT_CUE,
    ACT_LEFT,
    ACT_RIGHT,
    ACTION_NAMES,
    NUM_ACTIONS,
)


def test_single_step_matches_analytic():
    """Sequential EFE with T=1 should match jax_compute_efe_analytic."""
    from alf.jax_native import jax_compute_efe_analytic
    import jax.numpy as jnp

    gm = build_t_maze_model(cue_reliability=0.9, T=1)
    A = gm.A[0]
    B = gm.B[0]
    C = gm.C[0]
    D = gm.D[0]

    for action in range(NUM_ACTIONS):
        # Analytic single-step EFE
        g_analytic = float(jax_compute_efe_analytic(
            jnp.array(A), jnp.array(B[:, :, action]),
            jnp.array(C), jnp.array(D),
        ))

        # Sequential EFE with T=1
        g_sequential = sequential_efe(A, B, C, D, np.array([action]))

        np.testing.assert_allclose(
            g_sequential, g_analytic, atol=1e-10,
            err_msg=(
                f"Action {ACTION_NAMES[action]}: sequential={g_sequential:.6f} "
                f"vs analytic={g_analytic:.6f}"
            ),
        )


def test_t_maze_open_loop_information_scores():
    """Arms reveal the hidden condition perfectly; the cue is noisy.

    Fixed action sequences do not model choosing an arm conditional on a cue.
    Expected utility is zero by symmetry for these policies.
    """
    gm = build_t_maze_model(cue_reliability=0.9, reward_magnitude=3.0, T=2)
    A = gm.A[0]
    B = gm.B[0]
    C = gm.C[0]
    D = gm.D[0]

    # Cue-first policies
    cue_then_left = np.array([ACT_CUE, ACT_LEFT])
    cue_then_right = np.array([ACT_CUE, ACT_RIGHT])

    # Arm-first policies (uninformed)
    left_direct = np.array([ACT_LEFT, ACT_STAY])
    right_direct = np.array([ACT_RIGHT, ACT_STAY])
    left_then_stay = np.array([ACT_LEFT, ACT_LEFT])
    right_then_stay = np.array([ACT_RIGHT, ACT_RIGHT])

    g_cue_left = sequential_efe(A, B, C, D, cue_then_left)
    g_cue_right = sequential_efe(A, B, C, D, cue_then_right)
    g_left_direct = sequential_efe(A, B, C, D, left_direct)
    g_right_direct = sequential_efe(A, B, C, D, right_direct)
    g_left_stay = sequential_efe(A, B, C, D, left_then_stay)
    g_right_stay = sequential_efe(A, B, C, D, right_then_stay)

    cue_ambiguity = -0.9 * np.log(0.9) - 0.1 * np.log(0.1)
    np.testing.assert_allclose([g_cue_left, g_cue_right],
                               -2 * np.log(2) + cue_ambiguity, atol=1e-10)
    np.testing.assert_allclose([g_left_direct, g_right_direct, g_left_stay, g_right_stay],
                               -2 * np.log(2), atol=1e-10)

    # Cue-then-left and cue-then-right should be symmetric
    np.testing.assert_allclose(
        g_cue_left, g_cue_right, atol=1e-10,
        err_msg="Cue-then-left and cue-then-right should be symmetric",
    )

    # Left-direct and right-direct should also be symmetric
    np.testing.assert_allclose(
        g_left_direct, g_right_direct, atol=1e-10,
        err_msg="Direct left and direct right should be symmetric",
    )


def test_evaluate_all_policies():
    """Test evaluate_all_policies_sequential gives consistent results."""
    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    beliefs = [gm.D[0].copy()]

    G = evaluate_all_policies_sequential(gm, beliefs)

    assert G.shape == (gm.num_policies,), (
        f"Expected shape ({gm.num_policies},), got {G.shape}"
    )
    assert np.all(np.isfinite(G)), "All EFE values should be finite"

    # Verify against manual computation for a specific policy
    A, B, C, D = gm.A[0], gm.B[0], gm.C[0], gm.D[0]
    # Find the policy index for [cue, left]
    for i, policy in enumerate(gm.policies):
        if policy[0, 0] == ACT_CUE and policy[1, 0] == ACT_LEFT:
            g_manual = sequential_efe(A, B, C, D, policy[:, 0])
            np.testing.assert_allclose(
                G[i], g_manual, atol=1e-12,
                err_msg=f"Policy {i} mismatch: {G[i]} vs {g_manual}",
            )
            break



def test_select_action():
    """Selected actions and probabilities agree with the policy posterior."""
    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    beliefs = [gm.D[0].copy()]
    rng = np.random.RandomState(42)
    scores = evaluate_all_policies_sequential(gm, beliefs)
    weights = np.exp(-4.0 * (scores - scores.min())) * gm.E
    expected = weights / weights.sum()
    for _ in range(10):
        action, info = select_action_sequential(gm, beliefs, gamma=4.0, rng=rng)
        np.testing.assert_allclose(info["policy_probs"], expected, atol=1e-12)
        assert action == gm.policies[info["selected_policy_idx"], 0, 0]


def test_cue_reliability_effect():
    """Higher cue reliability should make cue-first even more preferred."""
    gm_high = build_t_maze_model(cue_reliability=0.99, T=2)
    gm_low = build_t_maze_model(cue_reliability=0.55, T=2)

    A_h, B_h, C_h, D_h = gm_high.A[0], gm_high.B[0], gm_high.C[0], gm_high.D[0]
    A_l, B_l, C_l, D_l = gm_low.A[0], gm_low.B[0], gm_low.C[0], gm_low.D[0]

    policy_cue_left = np.array([ACT_CUE, ACT_LEFT])

    g_high = sequential_efe(A_h, B_h, C_h, D_h, policy_cue_left)
    g_low = sequential_efe(A_l, B_l, C_l, D_l, policy_cue_left)

    assert g_high < g_low, "A more reliable cue must provide more information"


def test_jax_matches_numpy():
    """JAX sequential EFE should match numpy version."""
    import jax.numpy as jnp
    from alf.sequential_efe import jax_sequential_efe

    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    A = gm.A[0]
    B = gm.B[0]
    C = gm.C[0]
    D = gm.D[0]

    for i, policy in enumerate(gm.policies):
        action_seq = policy[:, 0]
        g_np = sequential_efe(A, B, C, D, action_seq)
        g_jax = float(jax_sequential_efe(
            jnp.array(A), jnp.array(B), jnp.array(C),
            jnp.array(D), jnp.array(action_seq),
        ))

        np.testing.assert_allclose(
            g_jax, g_np, atol=1e-6,
            err_msg=f"Policy {i} ({action_seq}): JAX={g_jax:.6f} vs numpy={g_np:.6f}",
        )


def test_jax_jit_works():
    """Test that JAX sequential EFE can be JIT-compiled."""
    import jax
    import jax.numpy as jnp
    from alf.sequential_efe import jax_sequential_efe

    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    A_j = jnp.array(gm.A[0])
    B_j = jnp.array(gm.B[0])
    C_j = jnp.array(gm.C[0])
    D_j = jnp.array(gm.D[0])

    # JIT compile
    jitted_efe = jax.jit(jax_sequential_efe, static_argnums=(5,))

    policy = jnp.array([ACT_CUE, ACT_LEFT])

    # First call (compilation)
    g1 = float(jitted_efe(A_j, B_j, C_j, D_j, policy))

    # Second call (cached)
    g2 = float(jitted_efe(A_j, B_j, C_j, D_j, policy))

    np.testing.assert_allclose(g1, g2, atol=1e-10)

    # Verify it matches non-jitted
    g_nojit = float(jax_sequential_efe(A_j, B_j, C_j, D_j, policy))
    np.testing.assert_allclose(g1, g_nojit, atol=1e-10)


def test_jax_vmap_over_policies():
    """Test that JAX version can be vmapped over policies."""
    import jax
    import jax.numpy as jnp
    from alf.sequential_efe import (
        jax_sequential_efe,
        jax_evaluate_all_policies_sequential,
    )

    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    A_j = jnp.array(gm.A[0])
    B_j = jnp.array(gm.B[0])
    C_j = jnp.array(gm.C[0])
    D_j = jnp.array(gm.D[0])

    # Build policies array: shape (num_policies, T)
    policies_j = jnp.array(gm.policies[:, :, 0])  # (num_policies, T)

    # Vmapped evaluation
    G_vmap = jax_evaluate_all_policies_sequential(A_j, B_j, C_j, D_j, policies_j)

    # Compare with loop
    G_loop = []
    for i in range(gm.num_policies):
        g = float(jax_sequential_efe(A_j, B_j, C_j, D_j, policies_j[i]))
        G_loop.append(g)
    G_loop = np.array(G_loop)

    np.testing.assert_allclose(
        np.array(G_vmap), G_loop, atol=1e-6,
        err_msg="Vmapped results should match loop results",
    )


def test_jax_select_action_sequential():
    """Test JAX action selection with sequential EFE."""
    import jax
    import jax.numpy as jnp
    from alf.sequential_efe import jax_select_action_sequential

    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    A_j = jnp.array(gm.A[0])
    B_j = jnp.array(gm.B[0])
    C_j = jnp.array(gm.C[0])
    D_j = jnp.array(gm.D[0])
    E_j = jnp.array(gm.E)
    policies_j = jnp.array(gm.policies[:, :, 0])

    key = jax.random.PRNGKey(42)

    selected_idx, probs, G = jax_select_action_sequential(
        A_j, B_j, C_j, D_j, policies_j, E_j, 4.0, key,
    )

    assert probs.shape == (gm.num_policies,)
    assert G.shape == (gm.num_policies,)
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)

    # Compare all policies to the independently normalized Boltzmann weights.
    weights = np.exp(-4 * (np.asarray(G) - np.min(G))) * gm.E
    np.testing.assert_allclose(probs, weights / weights.sum(), atol=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
