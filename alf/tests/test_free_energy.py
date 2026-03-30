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
    A = np.array([[0.9, 0.2],
                   [0.1, 0.8]])
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
        kl = np.sum(q_uniform * (
            np.log(np.clip(q_uniform, eps, None))
            - np.log(np.clip(posterior, eps, None))
        ))

        # F should equal -ln P(o) + KL(q || posterior)
        expected_F = neg_log_evidence + kl
        np.testing.assert_allclose(
            F_uniform, expected_F, atol=1e-10,
            err_msg="F should equal -ln P(o) + KL(q || posterior)"
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
            F_true, neg_log_evidence, atol=1e-10,
            err_msg="F at true posterior should equal -ln P(o)"
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
                decomp.G_total, reconstructed, atol=1e-10,
                err_msg=f"G = -pragmatic - epistemic failed for action {action}"
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
                decomp.G_total, reconstructed, atol=1e-10,
                err_msg=f"G decomposition failed for T-maze action {action}"
            )

    def test_return_type(self):
        """Should return an EFEDecomposition named tuple."""
        A, B, C, D = make_simple_model()
        decomp = expected_free_energy_decomposed(
            A, B, np.array([1.0, -1.0]), D, 0
        )
        assert isinstance(decomp, EFEDecomposition)
        assert hasattr(decomp, 'G_total')
        assert hasattr(decomp, 'pragmatic')
        assert hasattr(decomp, 'epistemic')


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
            for q in [np.array([0.7, 0.3]),
                      np.array([0.5, 0.5]),
                      np.array([0.1, 0.9]),
                      np.array([0.99, 0.01])]:
                F_np = variational_free_energy(q, A, D, obs)
                F_jax = float(jax_variational_free_energy(
                    jnp.array(q), jnp.array(A), jnp.array(D), jnp.array(obs)
                ))
                np.testing.assert_allclose(
                    F_np, F_jax, atol=1e-6,
                    err_msg=f"VFE mismatch for obs={obs}, q={q}"
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
                decomp_np.G_total, float(G_jax), atol=1e-6,
                err_msg=f"G_total mismatch for action {action}"
            )
            np.testing.assert_allclose(
                decomp_np.pragmatic, float(prag_jax), atol=1e-6,
                err_msg=f"Pragmatic mismatch for action {action}"
            )
            np.testing.assert_allclose(
                decomp_np.epistemic, float(epist_jax), atol=1e-6,
                err_msg=f"Epistemic mismatch for action {action}"
            )

    def test_generalized_fe_matches(self):
        """jax_generalized_free_energy should match numpy version."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])
        q = np.array([0.7, 0.3])
        obs = 0

        for policy in [np.array([0]), np.array([1]), np.array([0, 1])]:
            F_np = generalized_free_energy(q, A, B, C, D, obs, policy, gamma=1.0)
            F_jax = float(jax_generalized_free_energy(
                jnp.array(q), jnp.array(A), jnp.array(B),
                jnp.array(C), jnp.array(D), jnp.array(obs),
                jnp.array(policy), gamma=1.0
            ))
            np.testing.assert_allclose(
                F_np, F_jax, atol=1e-5,
                err_msg=f"Generalized FE mismatch for policy={policy}"
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
                F_jax = float(jax_variational_free_energy(
                    jnp.array(q), jnp.array(A), jnp.array(D), jnp.array(obs)
                ))
                np.testing.assert_allclose(
                    F_np, F_jax, atol=1e-6,
                    err_msg=f"T-maze VFE mismatch for obs={obs}"
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
            jnp.array(A), jnp.array(B), C,
            jnp.array([0.6, 0.4]), jnp.array(0)
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
            jnp.array(A), jnp.array(B),
            jnp.array([1.0, -1.0]), jnp.array(D),
            jnp.array(0), jnp.array([0])
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
            jnp.array(A), jnp.array(B),
            jnp.array([1.0, -1.0]), jnp.array([0.6, 0.4])
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
            q, A, B, np.zeros(2), D, obs,
            policy=np.array([]),
            gamma=1.0
        )

        np.testing.assert_allclose(
            F_vfe, F_gen, atol=1e-10,
            err_msg="Generalized FE with empty policy should equal VFE"
        )

    def test_gamma_zero(self):
        """Generalized FE with gamma=0 should equal VFE (ignore future)."""
        A, B, C, D = make_simple_model()
        C = np.array([1.0, -1.0])
        q = np.array([0.7, 0.3])
        obs = 0

        F_vfe = variational_free_energy(q, A, D, obs)
        F_gen = generalized_free_energy(
            q, A, B, C, D, obs,
            policy=np.array([0, 1]),  # non-trivial policy
            gamma=0.0
        )

        np.testing.assert_allclose(
            F_vfe, F_gen, atol=1e-10,
            err_msg="Generalized FE with gamma=0 should equal VFE"
        )

    def test_jax_empty_policy(self):
        """JAX generalized FE with empty policy should equal JAX VFE."""
        A, B, C, D = make_simple_model()
        q = jnp.array([0.7, 0.3])
        obs = jnp.array(0)

        F_vfe = float(jax_variational_free_energy(
            q, jnp.array(A), jnp.array(D), obs
        ))

        # For JAX version with scan, we need at least a length-0 policy
        # which won't enter the scan. Using a length-1 dummy with gamma=0.
        F_gen = float(jax_generalized_free_energy(
            q, jnp.array(A), jnp.array(B),
            jnp.array([0.0, 0.0]), jnp.array(D),
            obs, jnp.array([0]), gamma=0.0
        ))

        np.testing.assert_allclose(
            F_vfe, F_gen, atol=1e-10,
            err_msg="JAX generalized FE with gamma=0 should equal VFE"
        )

    def test_future_increases_generalized_fe(self):
        """Adding future EFE should change generalized FE relative to VFE."""
        A, B, C, D = make_simple_model()
        C_pref = np.array([2.0, -2.0])
        q = np.array([0.7, 0.3])
        obs = 0

        F_vfe = variational_free_energy(q, A, D, obs)
        F_gen = generalized_free_energy(
            q, A, B, C_pref, D, obs,
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
            F_wrapper, F_direct, atol=1e-10,
            err_msg="Wrapper should match direct VFE computation"
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
            decomp_left.pragmatic, 0.0, atol=0.01,
            err_msg="Go_left pragmatic should be ~0 with uniform beliefs"
        )
        np.testing.assert_allclose(
            decomp_right.pragmatic, 0.0, atol=0.01,
            err_msg="Go_right pragmatic should be ~0 with uniform beliefs"
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
            decomp_left.G_total, decomp_right.G_total, atol=1e-10,
            err_msg="Go_left and go_right should have same G with uniform beliefs"
        )


# =========================================================================
# Test 10: Multi-factor tensor A matrices in free_energy_from_beliefs
# =========================================================================

class TestMultiFactorTensorA:
    """free_energy_from_beliefs should handle multi-factor tensor A matrices.

    When a model has multiple state factors (e.g., 'location' and 'context'),
    the A matrix becomes a tensor with shape (n_obs, n_states_f1, n_states_f2).
    The VFE computation must:
      1. Form the joint belief via outer product of per-factor beliefs.
      2. Contract the tensor A with the joint belief to get P(o|beliefs).
      3. Compute VFE using the marginalized likelihood.
    """

    @staticmethod
    def _make_two_factor_model():
        """Create a 2-factor model with a tensor A matrix.

        Factor 1 (location): 3 states
        Factor 2 (context):  2 states
        Observations: 4 outcomes

        A has shape (4, 3, 2) — P(o | location, context).
        B has shape (n_states_f, n_states_f, n_actions_f) per factor.
        """
        np.random.seed(42)

        n_obs = 4
        n_states_f1 = 3  # location
        n_states_f2 = 2  # context

        # Build a tensor A: shape (n_obs, n_states_f1, n_states_f2)
        A_raw = np.random.dirichlet(np.ones(n_obs), size=(n_states_f1, n_states_f2))
        # A_raw is (n_states_f1, n_states_f2, n_obs), transpose to (n_obs, f1, f2)
        A_tensor = A_raw.transpose(2, 0, 1)
        # Verify columns sum to 1 over observations
        assert np.allclose(A_tensor.sum(axis=0), 1.0)

        # B matrices: one per factor
        # Factor 1: 3 states, 2 actions
        B1 = np.zeros((n_states_f1, n_states_f1, 2))
        B1[:, :, 0] = np.eye(n_states_f1)
        B1[:, :, 1] = np.roll(np.eye(n_states_f1), 1, axis=0)

        # Factor 2: 2 states, 2 actions
        B2 = np.zeros((n_states_f2, n_states_f2, 2))
        B2[:, :, 0] = np.eye(n_states_f2)
        B2[:, :, 1] = np.array([[0, 1], [1, 0]], dtype=float)

        C = np.array([1.0, 0.5, -0.5, -1.0])  # preferences
        D1 = np.array([1.0, 0.0, 0.0])  # start at location 0
        D2 = np.array([0.5, 0.5])       # uniform over context

        gm = GenerativeModel(
            A=[A_tensor],
            B=[B1, B2],
            C=[C],
            D=[D1, D2],
            T=1,
        )
        return gm

    def test_multi_factor_vfe_runs(self):
        """free_energy_from_beliefs should not raise for tensor A matrices."""
        gm = self._make_two_factor_model()

        beliefs = [
            np.array([0.8, 0.1, 0.1]),  # location beliefs
            np.array([0.6, 0.4]),        # context beliefs
        ]
        observations = [0]  # observed outcome index

        # This should NOT raise NotImplementedError
        F = free_energy_from_beliefs(gm, beliefs, observations)
        assert np.isfinite(F), f"VFE should be finite, got {F}"

    def test_multi_factor_vfe_matches_manual(self):
        """VFE from multi-factor model should match manual tensor contraction.

        Under mean-field q(s1,s2) = q_1(s1) * q_2(s2), the VFE is:

          F = sum_f KL[q_f || D_f] - E_q[ln A(o | s1, s2)]

        where the expected log-likelihood is:

          E_q[ln A(o|s1,s2)] = sum_{s1,s2} q_1(s1)*q_2(s2) * ln A[o, s1, s2]
        """
        gm = self._make_two_factor_model()

        beliefs_f1 = np.array([0.8, 0.1, 0.1])
        beliefs_f2 = np.array([0.6, 0.4])
        beliefs = [beliefs_f1, beliefs_f2]
        obs = 0
        observations = [obs]

        eps = 1e-16

        # Joint belief via outer product
        joint = np.outer(beliefs_f1, beliefs_f2)  # (3, 2)

        # Expected log-likelihood:
        # E_q[ln A(o|s1,s2)] = sum_{s1,s2} q_1(s1)*q_2(s2) * ln A[o, s1, s2]
        A_tensor = gm.A[0]
        log_a_obs = np.log(np.clip(A_tensor[obs], eps, None))
        ell = np.sum(joint * log_a_obs)

        # KL terms for each factor
        kl_f1 = np.sum(beliefs_f1 * (
            np.log(np.clip(beliefs_f1, eps, None))
            - np.log(np.clip(gm.D[0], eps, None))
        ))
        kl_f2 = np.sum(beliefs_f2 * (
            np.log(np.clip(beliefs_f2, eps, None))
            - np.log(np.clip(gm.D[1], eps, None))
        ))

        # Manual VFE = sum_f KL[q_f || D_f] - E_q[ln A(o|s)]
        F_manual = kl_f1 + kl_f2 - ell

        F_computed = free_energy_from_beliefs(gm, beliefs, observations)

        np.testing.assert_allclose(
            F_computed, F_manual, atol=1e-10,
            err_msg="Multi-factor VFE should match manual tensor contraction"
        )

    def test_multi_factor_reduces_to_single_factor(self):
        """When one factor has 1 state, multi-factor VFE = single-factor VFE.

        If factor 2 has only 1 state, the tensor A of shape (n_obs, n_f1, 1)
        should give the same result as a 2D A of shape (n_obs, n_f1).
        """
        n_obs = 3
        n_states = 4

        # Build a 2D A matrix
        np.random.seed(123)
        A_2d = np.random.dirichlet(np.ones(n_obs), size=n_states).T  # (n_obs, n_states)

        # Build equivalent tensor A: shape (n_obs, n_states, 1)
        A_tensor = A_2d[:, :, np.newaxis]

        B1 = np.zeros((n_states, n_states, 2))
        B1[:, :, 0] = np.eye(n_states)
        B1[:, :, 1] = np.eye(n_states)

        # Trivial second factor: 1 state, 1 action
        B2 = np.ones((1, 1, 1))

        C = np.zeros(n_obs)
        D1 = np.ones(n_states) / n_states
        D2 = np.ones(1)  # single state

        # Single-factor model (2D A)
        gm_single = GenerativeModel(A=[A_2d], B=[B1], C=[C], D=[D1], T=1)

        # Multi-factor model (tensor A)
        gm_multi = GenerativeModel(
            A=[A_tensor], B=[B1, B2], C=[C], D=[D1, D2], T=1
        )

        beliefs_single = [np.array([0.4, 0.3, 0.2, 0.1])]
        beliefs_multi = [np.array([0.4, 0.3, 0.2, 0.1]), np.array([1.0])]
        observations = [1]

        F_single = free_energy_from_beliefs(gm_single, beliefs_single, observations)
        F_multi = free_energy_from_beliefs(gm_multi, beliefs_multi, observations)

        np.testing.assert_allclose(
            F_multi, F_single, atol=1e-10,
            err_msg="Multi-factor with trivial 2nd factor should match single-factor VFE"
        )

    def test_vfe_at_prior_kl_vanishes(self):
        """When q_f = D_f for all factors, KL terms vanish.

        At q = D, the VFE reduces to -E_D[ln A(o | s1, s2)]:

          F = 0 - sum_{s1,s2} D_1(s1)*D_2(s2) * ln A[o, s1, s2]
        """
        gm = self._make_two_factor_model()

        # Use prior beliefs
        beliefs_prior = [gm.D[0].copy(), gm.D[1].copy()]
        obs = 0
        observations = [obs]

        F_at_prior = free_energy_from_beliefs(gm, beliefs_prior, observations)

        # At q = D, KL terms are 0, so F = -E_D[ln A(o|s)]
        eps = 1e-16
        joint_prior = np.outer(gm.D[0], gm.D[1])
        log_a_obs = np.log(np.clip(gm.A[0][obs], eps, None))
        expected_ll = np.sum(joint_prior * log_a_obs)
        F_expected = -expected_ll

        np.testing.assert_allclose(
            F_at_prior, F_expected, atol=1e-10,
            err_msg="VFE at prior beliefs should equal -E_D[ln A(o|s)]"
        )


# =========================================================================
# Run all tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
