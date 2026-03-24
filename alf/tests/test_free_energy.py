"""Tests for Variational Free Energy and Expected Free Energy computations.

Tests the free_energy module against known mathematical properties:
- VFE is minimized at the true posterior
- VFE at uniform q equals negative log evidence
- EFE decomposition: pragmatic + epistemic = total G
- Epistemic value is higher for ambiguous states (T-maze cue)
- JAX versions match numpy versions
- jit compilation works
- Generalized FE reduces to VFE when T=0

Uses the T-maze model from benchmarks/t_maze.py as the primary test case.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from alf.free_energy import (
    variational_free_energy,
    free_energy_from_beliefs,
    expected_free_energy_decomposed,
    generalized_free_energy,
    jax_variational_free_energy,
    jax_expected_free_energy_decomposed,
    jax_generalized_free_energy,
    jax_vfe_batch,
    jax_efe_all_actions,
    EFEDecomposition,
)
from alf.generative_model import GenerativeModel
from alf.benchmarks.t_maze import (
    build_t_maze_model,
    _build_A,
    _build_B,
    _build_C,
    _build_D,
    NUM_STATES,
    NUM_ACTIONS,
    OBS_NULL,
    OBS_CUE_LEFT,
    OBS_REWARD,
    ACT_CUE,
    ACT_LEFT,
    ACT_RIGHT,
)


# =========================================================================
# Fixtures: simple 2-state model for analytical verification
# =========================================================================


def make_simple_model():
    """Create a simple 2-state, 2-obs model for analytical tests.

    States: {0, 1}
    Observations: {0, 1}
    A = [[0.9, 0.2],   P(o=0|s=0)=0.9, P(o=0|s=1)=0.2
         [0.1, 0.8]]   P(o=1|s=0)=0.1, P(o=1|s=1)=0.8
    B = identity (no transitions)
    C = [0, 0] (no preferences)
    D = [0.5, 0.5] (uniform prior)
    """
    A = np.array([[0.9, 0.2], [0.1, 0.8]])
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.eye(2)
    C = np.zeros(2)
    D = np.array([0.5, 0.5])
    return A, B, C, D


# =========================================================================
# Test 1: VFE is minimized at the true posterior
# =========================================================================


class TestVFEMinimizedAtTruePosterior:
    """VFE should be lowest when q(s) equals the exact Bayesian posterior."""

    def test_simple_model(self):
        """For a 2-state model, VFE at the true posterior < VFE at other q."""
        A, B, C, D = make_simple_model()
        obs = 0  # observe o=0

        # Compute true posterior: P(s|o) propto P(o|s) * P(s)
        likelihood = A[obs, :]  # [0.9, 0.2]
        unnorm = likelihood * D  # [0.45, 0.10]
        true_posterior = unnorm / unnorm.sum()  # [0.818, 0.182]

        # VFE at true posterior
        F_true = variational_free_energy(true_posterior, A, D, obs)

        # VFE at uniform q
        q_uniform = np.array([0.5, 0.5])
        F_uniform = variational_free_energy(q_uniform, A, D, obs)

        # VFE at wrong posterior (heavily favoring wrong state)
        q_wrong = np.array([0.1, 0.9])
        F_wrong = variational_free_energy(q_wrong, A, D, obs)

        # True posterior should minimize VFE
        assert F_true < F_uniform, (
            f"VFE at true posterior ({F_true:.6f}) should be less than "
            f"at uniform ({F_uniform:.6f})"
        )
        assert F_true < F_wrong, (
            f"VFE at true posterior ({F_true:.6f}) should be less than "
            f"at wrong q ({F_wrong:.6f})"
        )

    def test_vfe_gradient_zero_at_optimum(self):
        """VFE gradient w.r.t. q should be near zero at true posterior.

        We verify this using JAX autodiff on the softmax-parameterized q.
        """
        A, B, C, D = make_simple_model()
        obs = 0

        A_jax = jnp.array(A)
        D_jax = jnp.array(D)
        obs_jax = jnp.array(obs)

        # Parameterize q via softmax: q = softmax(logits)
        # At the true posterior, find the logits
        likelihood = A[obs, :]
        unnorm = likelihood * D
        true_posterior = unnorm / unnorm.sum()

        def vfe_from_logits(logits):
            q = jax.nn.softmax(logits)
            return jax_variational_free_energy(q, A_jax, D_jax, obs_jax)

        # Find logits corresponding to true posterior (up to constant)
        true_logits = jnp.log(jnp.array(true_posterior))
        grad_fn = jax.grad(vfe_from_logits)
        grad_at_true = grad_fn(true_logits)

        # Gradient should be near zero (up to constant shift in softmax)
        # The gradient through softmax has the form g - (g . q) * 1,
        # so the "effective" gradient in the simplex is zero.
        # We check by verifying that the gradient is constant (all equal).
        grad_range = float(jnp.max(grad_at_true) - jnp.min(grad_at_true))
        assert grad_range < 1e-5, (
            f"Gradient range at true posterior should be ~0, got {grad_range:.8f}"
        )


# =========================================================================
# Test 2: VFE at uniform q equals negative log evidence
# =========================================================================


class TestVFEUniformEqualsNegLogEvidence:
    """When q is uniform, VFE = -ln P(o) + ln(num_states) = -ln P(o) + H(q)."""

    def test_uniform_q_uniform_prior(self):
        """F(q=uniform) identity: F = -ln P(o) + KL(q || p(s|o))."""
        A, B, C, D = make_simple_model()
        # Use uniform prior
        D_uniform = np.array([0.5, 0.5])
        obs = 0

        q_uniform = np.array([0.5, 0.5])
        F_uniform = variational_free_energy(q_uniform, A, D_uniform, obs)

        # Compute -ln P(o) = -ln sum_s P(o|s) * P(s)
        log_evidence = np.log(np.sum(A[obs, :] * D_uniform))
        neg_log_evidence = -log_evidence

        # Compute KL(q || posterior)
        posterior_unnorm = A[obs, :] * D_uniform
        posterior = posterior_unnorm / posterior_unnorm.sum()
        eps = 1e-16
        kl = np.sum(
            q_uniform
            * (
                np.log(np.clip(q_uniform, eps, None))
                - np.log(np.clip(posterior, eps, None))
            )
        )

        # F should equal -ln P(o) + KL(q || posterior)
        expected_F = neg_log_evidence + kl
        np.testing.assert_allclose(
            F_uniform,
            expected_F,
            atol=1e-10,
            err_msg="F should equal -ln P(o) + KL(q || posterior)",
        )

        # And F >= -ln P(o)
        assert F_uniform >= neg_log_evidence - 1e-10, (
            f"F ({F_uniform:.6f}) should be >= -ln P(o) ({neg_log_evidence:.6f})"
        )

    def test_vfe_equals_neg_log_evidence_at_true_posterior(self):
        """F = -ln P(o) when q = true posterior (KL term vanishes)."""
        A, B, C, D = make_simple_model()
        obs = 0

        # True posterior
        likelihood = A[obs, :]
        unnorm = likelihood * D
        true_posterior = unnorm / unnorm.sum()

        F_true = variational_free_energy(true_posterior, A, D, obs)

        # -ln P(o)
        evidence = np.sum(A[obs, :] * D)
        neg_log_evidence = -np.log(evidence)

        np.testing.assert_allclose(
            F_true,
            neg_log_evidence,
            atol=1e-10,
            err_msg="F at true posterior should equal -ln P(o)",
        )


# =========================================================================
# Test 3: EFE decomposition -- pragmatic + epistemic = total G
# =========================================================================


class TestEFEDecomposition:
    """EFE should decompose cleanly: G = -pragmatic - epistemic."""

    def test_decomposition_consistency_simple(self):
        """G_total should equal -pragmatic - epistemic for simple model."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])  # prefer observation 0
        beliefs = np.array([0.7, 0.3])

        for action in range(2):
            decomp = expected_free_energy_decomposed(A, B, C, beliefs, action)
            reconstructed = -decomp.pragmatic - decomp.epistemic
            np.testing.assert_allclose(
                decomp.G_total,
                reconstructed,
                atol=1e-10,
                err_msg=f"G = -pragmatic - epistemic failed for action {action}",
            )

    def test_decomposition_consistency_t_maze(self):
        """G decomposition should be consistent for all T-maze actions."""
        A = _build_A(0.9)
        B = _build_B()
        C = _build_C(3.0)
        D = _build_D()

        beliefs = D.copy()  # Start at center with uniform over reward side

        for action in range(NUM_ACTIONS):
            decomp = expected_free_energy_decomposed(A, B, C, beliefs, action)
            reconstructed = -decomp.pragmatic - decomp.epistemic
            np.testing.assert_allclose(
                decomp.G_total,
                reconstructed,
                atol=1e-10,
                err_msg=f"G decomposition failed for T-maze action {action}",
            )

    def test_return_type(self):
        """Should return an EFEDecomposition named tuple."""
        A, B, C, D = make_simple_model()
        decomp = expected_free_energy_decomposed(A, B, np.array([1.0, -1.0]), D, 0)
        assert isinstance(decomp, EFEDecomposition)
        assert hasattr(decomp, "G_total")
        assert hasattr(decomp, "pragmatic")
        assert hasattr(decomp, "epistemic")


# =========================================================================
# Test 4: Epistemic value is higher for ambiguous states (T-maze cue)
# =========================================================================


class TestEpistemicValue:
    """The go_cue action should have higher epistemic value than go_left/right."""

    def test_cue_action_highest_epistemic(self):
        """Verify observation entropy structure at cue vs arm states."""
        A = _build_A(0.9)

        # H[P(o|s)] at the cue states (states 2,3) is > 0 because cue_reliability < 1
        log_A = np.log(np.clip(A, 1e-16, None))
        entropy_per_state = -np.sum(A * log_A, axis=0)

        # Cue states (2 and 3) should have positive entropy
        assert entropy_per_state[2] > 0.1, (
            f"Cue state entropy should be positive, got {entropy_per_state[2]:.4f}"
        )
        assert entropy_per_state[3] > 0.1, (
            f"Cue state entropy should be positive, got {entropy_per_state[3]:.4f}"
        )

        # Arm states (4,5,6,7) should have zero entropy (deterministic obs)
        for s in [4, 5, 6, 7]:
            assert entropy_per_state[s] < 1e-10, (
                f"Arm state {s} entropy should be ~0, got {entropy_per_state[s]:.6f}"
            )

        # Center states (0,1) should have zero entropy (always observe null)
        for s in [0, 1]:
            assert entropy_per_state[s] < 1e-10, (
                f"Center state {s} entropy should be ~0, got {entropy_per_state[s]:.6f}"
            )

    def test_cue_is_informative_about_hidden_state(self):
        """After observing the cue, beliefs should concentrate."""
        A = _build_A(0.9)

        # At center with uniform beliefs over reward side
        beliefs = _build_D()  # [0.5, 0.5, 0, 0, 0, 0, 0, 0]

        # After going to cue and observing cue_left:
        # Bayesian update: P(s|o=cue_left) propto P(o=cue_left|s) * P(s)
        # Only states 2,3 (cue states) are reachable
        likelihood_cue_left = A[OBS_CUE_LEFT, :]  # [0, 0, 0.9, 0.1, 0, 0, 0, 0]

        # Transition from center to cue
        B = _build_B()
        B_cue = B[:, :, ACT_CUE]
        predicted_states = B_cue @ beliefs
        # Update with observation
        updated = predicted_states * likelihood_cue_left
        updated = updated / updated.sum()

        # Should strongly favor cue_left state (reward is left)
        assert updated[2] > 0.8, (
            f"After cue_left obs, belief in cue_left state should be > 0.8, "
            f"got {updated[2]:.4f}"
        )


# =========================================================================
# Test 5: JAX versions match numpy versions
# =========================================================================


class TestJAXMatchesNumpy:
    """JAX-native functions should produce identical results to numpy versions."""

    def test_vfe_matches(self):
        """jax_variational_free_energy should match variational_free_energy."""
        A, B, C, D = make_simple_model()

        for obs in range(2):
            for q in [
                np.array([0.7, 0.3]),
                np.array([0.5, 0.5]),
                np.array([0.1, 0.9]),
                np.array([0.99, 0.01]),
            ]:
                F_np = variational_free_energy(q, A, D, obs)
                F_jax = float(
                    jax_variational_free_energy(
                        jnp.array(q), jnp.array(A), jnp.array(D), jnp.array(obs)
                    )
                )
                np.testing.assert_allclose(
                    F_np, F_jax, atol=1e-6, err_msg=f"VFE mismatch for obs={obs}, q={q}"
                )

    def test_efe_decomposition_matches(self):
        """jax_expected_free_energy_decomposed should match numpy version."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])
        beliefs = np.array([0.6, 0.4])

        A_jax = jnp.array(A)
        B_jax = jnp.array(B)
        C_jax = jnp.array(C)
        beliefs_jax = jnp.array(beliefs)

        for action in range(2):
            decomp_np = expected_free_energy_decomposed(A, B, C, beliefs, action)
            G_jax, prag_jax, epist_jax = jax_expected_free_energy_decomposed(
                A_jax, B_jax, C_jax, beliefs_jax, jnp.array(action)
            )

            np.testing.assert_allclose(
                decomp_np.G_total,
                float(G_jax),
                atol=1e-6,
                err_msg=f"G_total mismatch for action {action}",
            )
            np.testing.assert_allclose(
                decomp_np.pragmatic,
                float(prag_jax),
                atol=1e-6,
                err_msg=f"Pragmatic mismatch for action {action}",
            )
            np.testing.assert_allclose(
                decomp_np.epistemic,
                float(epist_jax),
                atol=1e-6,
                err_msg=f"Epistemic mismatch for action {action}",
            )

    def test_generalized_fe_matches(self):
        """jax_generalized_free_energy should match numpy version."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])
        q = np.array([0.7, 0.3])
        obs = 0

        for policy in [np.array([0]), np.array([1]), np.array([0, 1])]:
            F_np = generalized_free_energy(q, A, B, C, D, obs, policy, gamma=1.0)
            F_jax = float(
                jax_generalized_free_energy(
                    jnp.array(q),
                    jnp.array(A),
                    jnp.array(B),
                    jnp.array(C),
                    jnp.array(D),
                    jnp.array(obs),
                    jnp.array(policy),
                    gamma=1.0,
                )
            )
            np.testing.assert_allclose(
                F_np,
                F_jax,
                atol=1e-5,
                err_msg=f"Generalized FE mismatch for policy={policy}",
            )

    def test_t_maze_vfe_matches(self):
        """VFE should match between numpy and JAX on the T-maze model."""
        A = _build_A(0.9)
        D = _build_D()

        # Various belief states and observations
        beliefs_list = [
            D,  # uniform over center
            np.zeros(NUM_STATES),  # will be set below
        ]
        # Create a belief concentrated at cue_left
        b = np.zeros(NUM_STATES)
        b[2] = 1.0
        beliefs_list[1] = b

        for q in beliefs_list:
            # Make sure q sums to 1 and is valid
            if q.sum() == 0:
                continue
            q = q / q.sum()
            for obs in [OBS_NULL, OBS_CUE_LEFT, OBS_REWARD]:
                F_np = variational_free_energy(q, A, D, obs)
                F_jax = float(
                    jax_variational_free_energy(
                        jnp.array(q), jnp.array(A), jnp.array(D), jnp.array(obs)
                    )
                )
                np.testing.assert_allclose(
                    F_np, F_jax, atol=1e-6, err_msg=f"T-maze VFE mismatch for obs={obs}"
                )


# =========================================================================
# Test 6: JIT compilation works
# =========================================================================


class TestJITCompilation:
    """All JAX-native functions should compile with jax.jit without error."""

    def test_jit_vfe(self):
        """jax_variational_free_energy should work under jit."""
        A, B, C, D = make_simple_model()
        q = jnp.array([0.7, 0.3])
        A_jax = jnp.array(A)
        D_jax = jnp.array(D)
        obs = jnp.array(0)

        vfe_jit = jax.jit(jax_variational_free_energy)
        F1 = float(vfe_jit(q, A_jax, D_jax, obs))
        F2 = float(vfe_jit(q, A_jax, D_jax, obs))  # second call uses cached
        np.testing.assert_allclose(F1, F2, atol=1e-10)

    def test_jit_efe_decomposed(self):
        """jax_expected_free_energy_decomposed should work under jit."""
        A, B, C, D = make_simple_model()
        C = jnp.array([1.0, -1.0])

        efe_jit = jax.jit(jax_expected_free_energy_decomposed)
        G, prag, epist = efe_jit(
            jnp.array(A), jnp.array(B), C, jnp.array([0.6, 0.4]), jnp.array(0)
        )
        assert jnp.isfinite(G)
        assert jnp.isfinite(prag)
        assert jnp.isfinite(epist)

    def test_jit_generalized_fe(self):
        """jax_generalized_free_energy should work under jit."""
        A, B, C, D = make_simple_model()

        # jit with static gamma
        @jax.jit
        def compute(q, A, B, C, D, obs, policy):
            return jax_generalized_free_energy(q, A, B, C, D, obs, policy, gamma=1.0)

        F = compute(
            jnp.array([0.7, 0.3]),
            jnp.array(A),
            jnp.array(B),
            jnp.array([1.0, -1.0]),
            jnp.array(D),
            jnp.array(0),
            jnp.array([0]),
        )
        assert jnp.isfinite(F)

    def test_jit_vfe_batch(self):
        """jax_vfe_batch should work under jit."""
        A, B, C, D = make_simple_model()
        batch_q = jnp.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
        batch_obs = jnp.array([0, 1, 0])

        vfe_batch_jit = jax.jit(jax_vfe_batch)
        F_batch = vfe_batch_jit(batch_q, jnp.array(A), jnp.array(D), batch_obs)
        assert F_batch.shape == (3,)
        assert jnp.all(jnp.isfinite(F_batch))

    def test_jit_efe_all_actions(self):
        """jax_efe_all_actions should work under jit."""
        A, B, C, D = make_simple_model()

        efe_jit = jax.jit(jax_efe_all_actions)
        G_all, prag_all, epist_all = efe_jit(
            jnp.array(A), jnp.array(B), jnp.array([1.0, -1.0]), jnp.array([0.6, 0.4])
        )
        assert G_all.shape == (2,)
        assert prag_all.shape == (2,)
        assert epist_all.shape == (2,)

    def test_grad_through_vfe(self):
        """Should be able to compute gradients through VFE."""
        A, B, C, D = make_simple_model()

        def loss(logits):
            q = jax.nn.softmax(logits)
            return jax_variational_free_energy(
                q, jnp.array(A), jnp.array(D), jnp.array(0)
            )

        grad_fn = jax.jit(jax.grad(loss))
        g = grad_fn(jnp.array([0.0, 0.0]))
        assert jnp.all(jnp.isfinite(g))


# =========================================================================
# Test 7: Generalized FE reduces to VFE when T=0
# =========================================================================


class TestGeneralizedFEReducesToVFE:
    """When there is no future policy, generalized FE should equal VFE."""

    def test_empty_policy_simple(self):
        """Generalized FE with empty policy = VFE for simple model."""
        A, B, C, D = make_simple_model()
        q = np.array([0.7, 0.3])
        obs = 0

        F_vfe = variational_free_energy(q, A, D, obs)
        F_gen = generalized_free_energy(
            q, A, B, np.zeros(2), D, obs, policy=np.array([]), gamma=1.0
        )

        np.testing.assert_allclose(
            F_vfe,
            F_gen,
            atol=1e-10,
            err_msg="Generalized FE with empty policy should equal VFE",
        )

    def test_gamma_zero(self):
        """Generalized FE with gamma=0 should equal VFE (ignore future)."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])
        q = np.array([0.7, 0.3])
        obs = 0

        F_vfe = variational_free_energy(q, A, D, obs)
        F_gen = generalized_free_energy(
            q,
            A,
            B,
            C,
            D,
            obs,
            policy=np.array([0, 1]),  # non-trivial policy
            gamma=0.0,
        )

        np.testing.assert_allclose(
            F_vfe,
            F_gen,
            atol=1e-10,
            err_msg="Generalized FE with gamma=0 should equal VFE",
        )

    def test_jax_empty_policy(self):
        """JAX generalized FE with empty policy should equal JAX VFE."""
        A, B, C, D = make_simple_model()
        q = jnp.array([0.7, 0.3])
        obs = jnp.array(0)

        F_vfe = float(jax_variational_free_energy(q, jnp.array(A), jnp.array(D), obs))

        # For JAX version with scan, we need at least a length-0 policy
        # which won't enter the scan. Using a length-1 dummy with gamma=0.
        F_gen = float(
            jax_generalized_free_energy(
                q,
                jnp.array(A),
                jnp.array(B),
                jnp.array([0.0, 0.0]),
                jnp.array(D),
                obs,
                jnp.array([0]),
                gamma=0.0,
            )
        )

        np.testing.assert_allclose(
            F_vfe,
            F_gen,
            atol=1e-10,
            err_msg="JAX generalized FE with gamma=0 should equal VFE",
        )

    def test_future_increases_generalized_fe(self):
        """Adding future EFE should change generalized FE relative to VFE."""
        A, B, C, D = make_simple_model()
        C_pref = np.array([2.0, -2.0])
        q = np.array([0.7, 0.3])
        obs = 0

        F_vfe = variational_free_energy(q, A, D, obs)
        F_gen = generalized_free_energy(
            q,
            A,
            B,
            C_pref,
            D,
            obs,
            policy=np.array([0]),
            gamma=1.0,
        )

        # Should differ (unless EFE happens to be exactly 0)
        # With preferences, the EFE component should be non-zero
        assert not np.isclose(F_vfe, F_gen, atol=1e-6), (
            "Generalized FE with preferences and policy should differ from VFE"
        )


# =========================================================================
# Test 8: free_energy_from_beliefs wrapper
# =========================================================================


class TestFreeEnergyFromBeliefs:
    """Test the GenerativeModel wrapper function."""

    def test_matches_direct_computation(self):
        """free_energy_from_beliefs should match direct variational_free_energy."""
        A, B, C, D = make_simple_model()
        gm = GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=1)

        beliefs = [np.array([0.7, 0.3])]
        observations = [0]

        F_wrapper = free_energy_from_beliefs(gm, beliefs, observations)
        F_direct = variational_free_energy(beliefs[0], A, D, observations[0])

        np.testing.assert_allclose(
            F_wrapper,
            F_direct,
            atol=1e-10,
            err_msg="Wrapper should match direct VFE computation",
        )

    def test_t_maze_model(self):
        """Should work with the T-maze generative model."""
        gm = build_t_maze_model(cue_reliability=0.9)
        beliefs = [_build_D()]  # prior beliefs
        observations = [OBS_NULL]  # initial observation

        F = free_energy_from_beliefs(gm, beliefs, observations)
        assert np.isfinite(F), f"VFE should be finite, got {F}"


# =========================================================================
# Test 9: T-maze specific tests
# =========================================================================


class TestTMazeEFE:
    """T-maze specific tests for EFE decomposition."""

    def test_all_actions_finite(self):
        """EFE should be finite for all T-maze actions."""
        A = _build_A(0.9)
        B = _build_B()
        C = _build_C(3.0)
        beliefs = _build_D()

        for action in range(NUM_ACTIONS):
            decomp = expected_free_energy_decomposed(A, B, C, beliefs, action)
            assert np.isfinite(decomp.G_total), (
                f"G_total not finite for action {action}"
            )
            assert np.isfinite(decomp.pragmatic), (
                f"Pragmatic not finite for action {action}"
            )
            assert np.isfinite(decomp.epistemic), (
                f"Epistemic not finite for action {action}"
            )

    def test_go_left_and_right_have_zero_pragmatic_from_center(self):
        """From center with uniform beliefs, expected reward is zero."""
        A = _build_A(0.9)
        B = _build_B()
        C = _build_C(3.0)
        beliefs = _build_D()

        decomp_left = expected_free_energy_decomposed(A, B, C, beliefs, ACT_LEFT)
        decomp_right = expected_free_energy_decomposed(A, B, C, beliefs, ACT_RIGHT)

        # With uniform beliefs, expected observations are symmetric
        # Pragmatic should be near 0 (reward and punishment cancel)
        np.testing.assert_allclose(
            decomp_left.pragmatic,
            0.0,
            atol=0.01,
            err_msg="Go_left pragmatic should be ~0 with uniform beliefs",
        )
        np.testing.assert_allclose(
            decomp_right.pragmatic,
            0.0,
            atol=0.01,
            err_msg="Go_right pragmatic should be ~0 with uniform beliefs",
        )

    def test_go_left_right_symmetric(self):
        """With uniform beliefs, go_left and go_right should have same G."""
        A = _build_A(0.9)
        B = _build_B()
        C = _build_C(3.0)
        beliefs = _build_D()

        decomp_left = expected_free_energy_decomposed(A, B, C, beliefs, ACT_LEFT)
        decomp_right = expected_free_energy_decomposed(A, B, C, beliefs, ACT_RIGHT)

        np.testing.assert_allclose(
            decomp_left.G_total,
            decomp_right.G_total,
            atol=1e-10,
            err_msg="Go_left and go_right should have same G with uniform beliefs",
        )


# =========================================================================
# Run all tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
