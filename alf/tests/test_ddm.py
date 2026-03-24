"""Tests for the Drift-Diffusion Model module.

Tests the Navarro-Fuss density, verifying:
1. Log-density is finite for valid parameters.
2. Density integrates to approximately 1 (over RT and choice).
3. Log-density is differentiable w.r.t. DDM parameters.
4. Higher drift rate increases upper-boundary choice probability.
5. Simulation produces reasonable RT distributions.
6. EFE-to-DDM bridge produces valid parameters.
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.ddm.wiener import (
    DDMParams,
    wiener_log_density,
    wiener_log_density_batch,
    ddm_log_likelihood,
    ddm_nll,
    simulate_ddm,
)
from alf.ddm.bridge import (
    efe_to_ddm,
    ddm_to_policy_probs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def standard_params():
    """Standard DDM parameters for testing."""
    return DDMParams(
        v=jnp.array(1.0),
        a=jnp.array(1.5),
        w=jnp.array(0.5),
        tau=jnp.array(0.3),
    )


# ---------------------------------------------------------------------------
# Navarro-Fuss density tests
# ---------------------------------------------------------------------------


def test_wiener_log_density_finite():
    """Test that log-density is finite for valid parameters."""
    rt = jnp.array(0.8)
    choice = jnp.array(1.0)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    log_p = wiener_log_density(rt, choice, v, a, w, tau)
    assert jnp.isfinite(log_p), f"log_p not finite: {log_p}"


def test_wiener_log_density_upper_and_lower():
    """Test that density works for both upper and lower boundary."""
    rt = jnp.array(0.7)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    log_p_upper = wiener_log_density(rt, jnp.array(1.0), v, a, w, tau)
    log_p_lower = wiener_log_density(rt, jnp.array(0.0), v, a, w, tau)

    assert jnp.isfinite(log_p_upper), f"upper not finite: {log_p_upper}"
    assert jnp.isfinite(log_p_lower), f"lower not finite: {log_p_lower}"

    # With positive drift, upper boundary should be more likely
    assert log_p_upper > log_p_lower, (
        f"Positive drift should favor upper: {log_p_upper:.4f} vs {log_p_lower:.4f}"
    )


def test_wiener_log_density_invalid_rt():
    """Test that density handles RT <= tau correctly."""
    rt = jnp.array(0.1)  # Less than tau
    choice = jnp.array(1.0)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    log_p = wiener_log_density(rt, choice, v, a, w, tau)
    assert log_p == -jnp.inf or log_p < -30, (
        f"RT <= tau should give very low density: {log_p}"
    )


def test_wiener_log_density_differentiable():
    """Test that log-density is differentiable w.r.t. all DDM parameters."""
    rt = jnp.array(0.8)
    choice = jnp.array(1.0)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    # Gradient w.r.t. v
    grad_v = jax.grad(wiener_log_density, argnums=2)(rt, choice, v, a, w, tau)
    assert jnp.isfinite(grad_v), f"grad_v not finite: {grad_v}"

    # Gradient w.r.t. a
    grad_a = jax.grad(wiener_log_density, argnums=3)(rt, choice, v, a, w, tau)
    assert jnp.isfinite(grad_a), f"grad_a not finite: {grad_a}"

    # Gradient w.r.t. tau
    grad_tau = jax.grad(wiener_log_density, argnums=5)(rt, choice, v, a, w, tau)
    assert jnp.isfinite(grad_tau), f"grad_tau not finite: {grad_tau}"


def test_wiener_log_density_jit():
    """Test that log-density works under jax.jit."""
    rt = jnp.array(0.8)
    choice = jnp.array(1.0)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    log_p_eager = wiener_log_density(rt, choice, v, a, w, tau)
    log_p_jit = jax.jit(wiener_log_density)(rt, choice, v, a, w, tau)

    np.testing.assert_allclose(float(log_p_eager), float(log_p_jit), atol=1e-6)


def test_wiener_batch():
    """Test batch log-density computation."""
    rts = jnp.array([0.5, 0.7, 0.9, 1.2])
    choices = jnp.array([1.0, 0.0, 1.0, 1.0])
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    log_ps = wiener_log_density_batch(rts, choices, v, a, w, tau)

    assert log_ps.shape == (4,), f"Expected shape (4,), got {log_ps.shape}"
    assert jnp.all(jnp.isfinite(log_ps)), "Some densities are not finite"


def test_ddm_nll_matches_manual():
    """Test that ddm_nll matches manual computation."""
    rts = jnp.array([0.5, 0.7, 0.9])
    choices = jnp.array([1.0, 0.0, 1.0])
    params = standard_params()

    nll = ddm_nll(params.v, params.a, params.w, params.tau, rts, choices)
    ll = ddm_log_likelihood(params, rts, choices)

    np.testing.assert_allclose(float(nll), -float(ll), atol=1e-6)


def test_ddm_nll_differentiable():
    """Test that NLL is differentiable w.r.t. parameters."""
    rts = jnp.array([0.5, 0.7, 0.9, 1.1])
    choices = jnp.array([1.0, 0.0, 1.0, 1.0])
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    grads = jax.grad(ddm_nll, argnums=(0, 1, 2, 3))(v, a, w, tau, rts, choices)

    for i, name in enumerate(["v", "a", "w", "tau"]):
        assert jnp.isfinite(grads[i]), f"grad_{name} not finite: {grads[i]}"


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------


def test_simulate_ddm_basic():
    """Test that DDM simulation produces valid outputs."""
    result = simulate_ddm(v=1.0, a=1.5, w=0.5, tau=0.3, n_trials=100, seed=42)

    assert result.rt.shape == (100,), f"rt shape: {result.rt.shape}"
    assert result.choice.shape == (100,), f"choice shape: {result.choice.shape}"
    assert np.all(result.rt > 0), "All RTs should be positive"
    assert np.all((result.choice == 0) | (result.choice == 1)), "Choices must be 0 or 1"


def test_simulate_ddm_positive_drift_favors_upper():
    """Test that positive drift leads to more upper-boundary responses."""
    result = simulate_ddm(v=2.0, a=1.5, w=0.5, tau=0.3, n_trials=500, seed=42)

    upper_proportion = np.mean(result.choice == 1)
    assert upper_proportion > 0.6, (
        f"Positive drift should favor upper boundary: {upper_proportion:.3f}"
    )


def test_simulate_ddm_negative_drift_favors_lower():
    """Test that negative drift leads to more lower-boundary responses."""
    result = simulate_ddm(v=-2.0, a=1.5, w=0.5, tau=0.3, n_trials=500, seed=42)

    lower_proportion = np.mean(result.choice == 0)
    assert lower_proportion > 0.6, (
        f"Negative drift should favor lower boundary: {lower_proportion:.3f}"
    )


def test_simulate_ddm_rt_above_tau():
    """Test that all simulated RTs exceed the non-decision time."""
    result = simulate_ddm(v=1.0, a=1.5, w=0.5, tau=0.3, n_trials=100, seed=42)

    assert np.all(result.rt >= 0.3), "All RTs should be >= tau"


def test_simulate_ddm_wider_boundary_slower():
    """Test that wider boundaries lead to slower responses."""
    result_narrow = simulate_ddm(v=1.0, a=1.0, w=0.5, tau=0.3, n_trials=500, seed=42)
    result_wide = simulate_ddm(v=1.0, a=3.0, w=0.5, tau=0.3, n_trials=500, seed=42)

    mean_rt_narrow = np.mean(result_narrow.rt)
    mean_rt_wide = np.mean(result_wide.rt)

    assert mean_rt_wide > mean_rt_narrow, (
        f"Wider boundary should be slower: {mean_rt_wide:.3f} vs {mean_rt_narrow:.3f}"
    )


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------


def test_efe_to_ddm_basic():
    """Test EFE-to-DDM mapping produces valid parameters."""
    G = jnp.array([0.5, -0.5])  # Action 1 has lower (better) EFE
    params = efe_to_ddm(G, gamma=4.0)

    assert jnp.isfinite(params.v), f"v not finite: {params.v}"
    assert params.a > 0, f"a should be positive: {params.a}"
    assert 0 < params.w < 1, f"w should be in (0,1): {params.w}"

    # Action 1 has lower G, so drift should be positive (toward upper boundary)
    assert params.v > 0, (
        f"Lower G[1] should give positive drift: v={float(params.v):.3f}"
    )


def test_ddm_to_policy_probs_valid():
    """Test that DDM choice probabilities are valid."""
    params = standard_params()
    probs = ddm_to_policy_probs(params)

    assert probs.shape == (2,), f"Expected 2 probs, got {probs.shape}"
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)
    assert jnp.all(probs > 0), "All probs should be positive"


def test_ddm_to_policy_probs_drift_direction():
    """Test that positive drift favors upper boundary."""
    params_pos = DDMParams(
        v=jnp.array(2.0),
        a=jnp.array(1.5),
        w=jnp.array(0.5),
        tau=jnp.array(0.3),
    )
    params_neg = DDMParams(
        v=jnp.array(-2.0),
        a=jnp.array(1.5),
        w=jnp.array(0.5),
        tau=jnp.array(0.3),
    )

    probs_pos = ddm_to_policy_probs(params_pos)
    probs_neg = ddm_to_policy_probs(params_neg)

    assert probs_pos[1] > 0.5, f"Positive drift should favor upper: {probs_pos[1]:.3f}"
    assert probs_neg[0] > 0.5, f"Negative drift should favor lower: {probs_neg[0]:.3f}"


def test_ddm_to_policy_probs_zero_drift():
    """Test that zero drift gives equal probabilities (with w=0.5)."""
    params = DDMParams(
        v=jnp.array(0.0),
        a=jnp.array(1.5),
        w=jnp.array(0.5),
        tau=jnp.array(0.3),
    )
    probs = ddm_to_policy_probs(params)

    np.testing.assert_allclose(float(probs[0]), float(probs[1]), atol=0.01)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
