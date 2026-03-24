"""Tests for sequential Expected Free Energy.

Verifies that:
1. Sequential EFE matches single-step analytic EFE for T=1
2. Cue-first policies dominate on T-maze with T=2
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
        g_analytic = float(
            jax_compute_efe_analytic(
                jnp.array(A),
                jnp.array(B[:, :, action]),
                jnp.array(C),
                jnp.array(D),
            )
        )

        # Sequential EFE with T=1
        g_sequential = sequential_efe(A, B, C, D, np.array([action]))

        np.testing.assert_allclose(
            g_sequential,
            g_analytic,
            atol=1e-10,
            err_msg=(
                f"Action {ACTION_NAMES[action]}: sequential={g_sequential:.6f} "
                f"vs analytic={g_analytic:.6f}"
            ),
        )


def test_cue_first_dominates_t2():
    """With T=2, cue-first policies should have lower G than arm-first."""
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

    # Cue-first should be better (lower G) than direct arm choices
    best_cue_first = min(g_cue_left, g_cue_right)
    worst_arm_first = min(g_left_direct, g_right_direct, g_left_stay, g_right_stay)

    assert best_cue_first < worst_arm_first, (
        f"Cue-first ({best_cue_first:.4f}) should have lower G than "
        f"arm-first ({worst_arm_first:.4f})"
    )

    # Cue-then-left and cue-then-right should be symmetric
    np.testing.assert_allclose(
        g_cue_left,
        g_cue_right,
        atol=1e-10,
        err_msg="Cue-then-left and cue-then-right should be symmetric",
    )

    # Left-direct and right-direct should also be symmetric
    np.testing.assert_allclose(
        g_left_direct,
        g_right_direct,
        atol=1e-10,
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
                G[i],
                g_manual,
                atol=1e-12,
                err_msg=f"Policy {i} mismatch: {G[i]} vs {g_manual}",
            )
            break

    # Check that cue-first policies dominate in the full evaluation
    cue_first_indices = []
    arm_first_indices = []
    for i, policy in enumerate(gm.policies):
        first_action = policy[0, 0]
        if first_action == ACT_CUE:
            cue_first_indices.append(i)
        elif first_action in (ACT_LEFT, ACT_RIGHT):
            arm_first_indices.append(i)

    best_cue = min(G[i] for i in cue_first_indices)
    best_arm = min(G[i] for i in arm_first_indices)
    assert best_cue < best_arm, (
        f"Best cue-first G ({best_cue:.4f}) should be lower than "
        f"best arm-first G ({best_arm:.4f})"
    )


def test_select_action():
    """Test that select_action_sequential prefers cue-first."""
    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    beliefs = [gm.D[0].copy()]
    rng = np.random.RandomState(42)

    # Run multiple selections to check distribution
    cue_count = 0
    num_samples = 100
    for _ in range(num_samples):
        action, info = select_action_sequential(
            gm,
            beliefs,
            gamma=4.0,
            rng=rng,
        )
        if action == ACT_CUE:
            cue_count += 1

    cue_rate = cue_count / num_samples
    assert cue_rate > 0.5, (
        f"Agent should prefer cue action, but cue_rate={cue_rate:.2f}"
    )


def test_cue_reliability_effect():
    """Higher cue reliability should make cue-first even more preferred."""
    gm_high = build_t_maze_model(cue_reliability=0.99, T=2)
    gm_low = build_t_maze_model(cue_reliability=0.55, T=2)

    A_h, B_h, C_h, D_h = gm_high.A[0], gm_high.B[0], gm_high.C[0], gm_high.D[0]
    A_l, B_l, C_l, D_l = gm_low.A[0], gm_low.B[0], gm_low.C[0], gm_low.D[0]

    policy_cue_left = np.array([ACT_CUE, ACT_LEFT])

    g_high = sequential_efe(A_h, B_h, C_h, D_h, policy_cue_left)
    g_low = sequential_efe(A_l, B_l, C_l, D_l, policy_cue_left)

    assert g_high != g_low, "Different reliability should give different G"


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
        g_jax = float(
            jax_sequential_efe(
                jnp.array(A),
                jnp.array(B),
                jnp.array(C),
                jnp.array(D),
                jnp.array(action_seq),
            )
        )

        np.testing.assert_allclose(
            g_jax,
            g_np,
            atol=1e-6,
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
        np.array(G_vmap),
        G_loop,
        atol=1e-6,
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
        A_j,
        B_j,
        C_j,
        D_j,
        policies_j,
        E_j,
        4.0,
        key,
    )

    assert probs.shape == (gm.num_policies,)
    assert G.shape == (gm.num_policies,)
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)

    # Cue-first policies should have highest probability mass collectively
    cue_first_prob = 0.0
    arm_first_prob = 0.0
    for i, policy in enumerate(gm.policies):
        if policy[0, 0] == ACT_CUE:
            cue_first_prob += float(probs[i])
        elif policy[0, 0] in (ACT_LEFT, ACT_RIGHT):
            arm_first_prob += float(probs[i])

    assert cue_first_prob > arm_first_prob, (
        f"Cue-first prob ({cue_first_prob:.4f}) should exceed "
        f"arm-first prob ({arm_first_prob:.4f})"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
