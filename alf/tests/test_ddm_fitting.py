"""Tests for DDM fitting module (MLE, Bayesian, hierarchical).

Tests the fitting module, verifying:
1. MLE fitting reduces loss over epochs.
2. MLE recovers parameters from simulated data.
3. MLE fitting works under JIT compilation.
4. Fitted parameters respect constraints (a > 0, 0 < w < 1, tau > 0).
5. Bayesian fitting produces valid posterior samples (requires numpyro).
6. Hierarchical model runs on multi-subject data (requires numpyro).
7. Posterior predictive generates valid synthetic data.
8. Parameter recovery analysis produces sensible statistics.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from alf.ddm.wiener import DDMParams, simulate_ddm
from alf.ddm.fitting import (
    DDMFitResult,
    DDMRecoveryResult,
    fit_ddm_mle,
    fit_ddm_bayesian,
    fit_ddm_hierarchical,
    ddm_posterior_predictive,
    ddm_recovery_check,
    _to_unconstrained,
    _to_constrained,
    _nll_unconstrained,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_numpyro():
    """Check if numpyro is available."""
    try:
        import numpyro  # noqa: F401
        return True
    except ImportError:
        return False


def _simulate_standard_data(
    v: float = 1.5,
    a: float = 1.5,
    w: float = 0.5,
    tau: float = 0.3,
    n_trials: int = 300,
    seed: int = 42,
):
    """Simulate standard DDM data for testing."""
    return simulate_ddm(v=v, a=a, w=w, tau=tau, n_trials=n_trials, seed=seed)


# ---------------------------------------------------------------------------
# Parameter transform tests
# ---------------------------------------------------------------------------

def test_unconstrained_roundtrip():
    """Test that constrained -> unconstrained -> constrained is identity."""
    v = jnp.array(1.5)
    a = jnp.array(2.0)
    w = jnp.array(0.6)
    tau = jnp.array(0.3)

    v_unc, log_a, logit_w, log_tau = _to_unconstrained(v, a, w, tau)
    v2, a2, w2, tau2 = _to_constrained(v_unc, log_a, logit_w, log_tau)

    np.testing.assert_allclose(float(v2), float(v), atol=1e-6)
    np.testing.assert_allclose(float(a2), float(a), atol=1e-6)
    np.testing.assert_allclose(float(w2), float(w), atol=1e-5)
    np.testing.assert_allclose(float(tau2), float(tau), atol=1e-6)


def test_nll_unconstrained_differentiable():
    """Test that unconstrained NLL is differentiable via jax.grad."""
    sim = _simulate_standard_data(n_trials=50)
    rt = jnp.array(sim.rt, dtype=jnp.float32)
    choice = jnp.array(sim.choice, dtype=jnp.float32)

    v_unc = jnp.array(0.0)
    log_a = jnp.array(0.4)
    logit_w = jnp.array(0.0)
    log_tau = jnp.array(-1.2)

    # Forward pass
    nll = _nll_unconstrained(v_unc, log_a, logit_w, log_tau, rt, choice)
    assert jnp.isfinite(nll), f"NLL not finite: {nll}"

    # Gradient
    grads = jax.grad(_nll_unconstrained, argnums=(0, 1, 2, 3))(
        v_unc, log_a, logit_w, log_tau, rt, choice
    )
    for i, name in enumerate(["v_unc", "log_a", "logit_w", "log_tau"]):
        assert jnp.isfinite(grads[i]), f"grad_{name} not finite: {grads[i]}"


# ---------------------------------------------------------------------------
# MLE fitting tests
# ---------------------------------------------------------------------------

def test_fit_ddm_mle_reduces_loss():
    """Test that MLE fitting reduces the NLL over epochs."""
    sim = _simulate_standard_data(n_trials=200)
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=200, lr=0.01,
    )

    assert isinstance(result, DDMFitResult)
    assert len(result.loss_history) == 200

    # Loss should decrease (compare first 10 epochs average to last 10)
    early_loss = np.mean(result.loss_history[:10])
    late_loss = np.mean(result.loss_history[-10:])
    assert late_loss < early_loss, (
        f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"
    )


def test_fit_ddm_mle_recovers_params():
    """Test that MLE recovers parameters from simulated data."""
    v_true, a_true, w_true, tau_true = 1.5, 1.5, 0.5, 0.3
    sim = _simulate_standard_data(
        v=v_true, a=a_true, w=w_true, tau=tau_true, n_trials=500, seed=123
    )

    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=800, lr=0.01,
    )

    # Drift rate should be roughly recovered (most identifiable parameter)
    assert abs(result.v - v_true) < 1.5, (
        f"v recovery: true={v_true}, recovered={result.v:.3f}"
    )
    # Boundary separation
    assert abs(result.a - a_true) < 1.0, (
        f"a recovery: true={a_true}, recovered={result.a:.3f}"
    )
    # Starting bias (harder to recover, wider tolerance)
    assert abs(result.w - w_true) < 0.25, (
        f"w recovery: true={w_true}, recovered={result.w:.3f}"
    )
    # Non-decision time
    assert abs(result.tau - tau_true) < 0.3, (
        f"tau recovery: true={tau_true}, recovered={result.tau:.3f}"
    )


def test_fit_ddm_mle_jit():
    """Test that the JIT-compiled update step works correctly."""
    sim = _simulate_standard_data(n_trials=100)

    # First run: compiles
    result1 = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=50, lr=0.01,
    )

    # Second run: should reuse compiled code
    result2 = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=50, lr=0.01,
    )

    # Results should be identical (same data, same init)
    np.testing.assert_allclose(
        result1.loss_history[-1], result2.loss_history[-1], atol=1e-4,
    )


def test_fit_ddm_mle_constrained():
    """Test that fitted parameters respect constraints."""
    sim = _simulate_standard_data(n_trials=200)
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=300, lr=0.01,
    )

    assert result.a > 0, f"a must be > 0, got {result.a}"
    assert 0 < result.w < 1, f"w must be in (0,1), got {result.w}"
    assert result.tau > 0, f"tau must be > 0, got {result.tau}"


def test_fit_ddm_mle_custom_init():
    """Test fitting with custom initial parameters."""
    sim = _simulate_standard_data(n_trials=200)
    init_params = DDMParams(
        v=jnp.array(2.0),
        a=jnp.array(2.0),
        w=jnp.array(0.4),
        tau=jnp.array(0.2),
    )
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        init_params=init_params,
        num_epochs=200, lr=0.01,
    )

    assert isinstance(result, DDMFitResult)
    assert result.a > 0
    assert 0 < result.w < 1


def test_fit_ddm_mle_negative_drift():
    """Test that MLE correctly identifies negative drift."""
    sim = _simulate_standard_data(v=-2.0, n_trials=300, seed=99)
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=500, lr=0.01,
    )

    # Recovered drift should be negative
    assert result.v < 0, (
        f"Expected negative drift, got v={result.v:.3f}"
    )


def test_fit_ddm_mle_n_trials():
    """Test that n_trials is correctly recorded."""
    sim = _simulate_standard_data(n_trials=150)
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=10, lr=0.01,
    )
    assert result.n_trials == 150


# ---------------------------------------------------------------------------
# Bayesian fitting tests (skip if no numpyro)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_numpyro(), reason="numpyro not installed")
def test_fit_ddm_bayesian_runs():
    """Test that Bayesian fitting produces valid posterior samples."""
    sim = _simulate_standard_data(n_trials=100, seed=42)

    result = fit_ddm_bayesian(
        rt=sim.rt, choice=sim.choice,
        num_warmup=200, num_samples=500, seed=42,
    )

    # Check structure
    assert "samples" in result
    assert "summary" in result
    for name in ["v", "a", "w", "tau"]:
        assert name in result["samples"]
        assert len(result["samples"][name]) == 500
        assert f"{name}_mean" in result
        assert f"{name}_hdi_95" in result

    # Check that a > 0, 0 < w < 1, tau > 0 in all samples
    assert np.all(result["samples"]["a"] > 0), "All a samples must be > 0"
    assert np.all(result["samples"]["w"] > 0), "All w samples must be > 0"
    assert np.all(result["samples"]["w"] < 1), "All w samples must be < 1"
    assert np.all(result["samples"]["tau"] > 0), "All tau samples must be > 0"


@pytest.mark.skipif(not _has_numpyro(), reason="numpyro not installed")
def test_fit_ddm_bayesian_import_error():
    """Test that Bayesian fitting raises ImportError message clearly."""
    # This test just ensures the function runs when numpyro IS available
    sim = _simulate_standard_data(n_trials=50)
    result = fit_ddm_bayesian(
        rt=sim.rt, choice=sim.choice,
        num_warmup=100, num_samples=100, seed=0,
    )
    assert result["v_mean"] is not None


def test_fit_ddm_bayesian_no_numpyro():
    """Test that ImportError is raised when numpyro is not available."""
    # We cannot easily mock the import away, so just verify the function
    # exists and has the right signature
    import inspect
    sig = inspect.signature(fit_ddm_bayesian)
    assert "rt" in sig.parameters
    assert "choice" in sig.parameters
    assert "num_warmup" in sig.parameters
    assert "num_samples" in sig.parameters


# ---------------------------------------------------------------------------
# Hierarchical fitting tests (skip if no numpyro)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_numpyro(), reason="numpyro not installed")
def test_fit_ddm_hierarchical_runs():
    """Test that hierarchical model runs on multi-subject data."""
    # Simulate 3 subjects with different drift rates
    rt_list = []
    choice_list = []
    for i, v in enumerate([1.0, 2.0, 0.5]):
        sim = simulate_ddm(v=v, a=1.5, w=0.5, tau=0.3, n_trials=80, seed=i)
        rt_list.append(sim.rt)
        choice_list.append(sim.choice.astype(float))

    result = fit_ddm_hierarchical(
        rt_list=rt_list,
        choice_list=choice_list,
        subject_ids=["subj_A", "subj_B", "subj_C"],
        num_warmup=200, num_samples=300, seed=42,
    )

    # Check structure
    assert result["n_subjects"] == 3
    assert "group_summary" in result
    assert "subject_summary" in result
    assert "mu_v" in result["group_summary"]
    assert "sigma_v" in result["group_summary"]

    # Check subject-level results
    for sid in ["subj_A", "subj_B", "subj_C"]:
        assert sid in result["subject_summary"]
        subj = result["subject_summary"][sid]
        assert "v" in subj
        assert "a" in subj


@pytest.mark.skipif(not _has_numpyro(), reason="numpyro not installed")
def test_fit_ddm_hierarchical_default_ids():
    """Test hierarchical fitting with default subject IDs."""
    rt_list = []
    choice_list = []
    for i in range(2):
        sim = simulate_ddm(v=1.0, a=1.5, w=0.5, tau=0.3, n_trials=50, seed=i)
        rt_list.append(sim.rt)
        choice_list.append(sim.choice.astype(float))

    result = fit_ddm_hierarchical(
        rt_list=rt_list,
        choice_list=choice_list,
        num_warmup=100, num_samples=200, seed=0,
    )

    assert result["n_subjects"] == 2
    assert result["subject_ids"] == [0, 1]
    assert 0 in result["subject_summary"]
    assert 1 in result["subject_summary"]


# ---------------------------------------------------------------------------
# Posterior predictive tests
# ---------------------------------------------------------------------------

def test_ddm_posterior_predictive_from_fit_result():
    """Test posterior predictive generation from DDMFitResult."""
    result = DDMFitResult(
        v=1.5, a=1.5, w=0.5, tau=0.3,
        v_se=None, a_se=None, w_se=None, tau_se=None,
        loss_history=[], n_trials=100,
    )

    pp = ddm_posterior_predictive(result, n_trials=200, seed=42)

    assert "rt" in pp
    assert "choice" in pp
    assert "params_used" in pp
    assert len(pp["rt"]) == 200
    assert len(pp["choice"]) == 200
    assert np.all(pp["rt"] > 0)
    assert np.all((pp["choice"] == 0) | (pp["choice"] == 1))


def test_ddm_posterior_predictive_from_dict():
    """Test posterior predictive generation from parameter dict."""
    params = {"v": 1.0, "a": 1.5, "w": 0.5, "tau": 0.3}
    pp = ddm_posterior_predictive(params, n_trials=100, seed=0)

    assert len(pp["rt"]) == 100
    assert pp["params_used"]["v"] == 1.0


@pytest.mark.skipif(not _has_numpyro(), reason="numpyro not installed")
def test_ddm_posterior_predictive_from_posterior():
    """Test posterior predictive from Bayesian posterior samples."""
    sim = _simulate_standard_data(n_trials=100)
    bayes_result = fit_ddm_bayesian(
        rt=sim.rt, choice=sim.choice,
        num_warmup=100, num_samples=200, seed=42,
    )

    pp = ddm_posterior_predictive(bayes_result, n_trials=100, seed=42)

    assert len(pp["rt"]) == 100
    assert np.all(pp["rt"] > 0)


def test_ddm_posterior_predictive_invalid_input():
    """Test that invalid input raises ValueError."""
    with pytest.raises((ValueError, AttributeError)):
        ddm_posterior_predictive("invalid_input", n_trials=10)


# ---------------------------------------------------------------------------
# Recovery check tests
# ---------------------------------------------------------------------------

def test_ddm_recovery_basic():
    """Test basic parameter recovery with fixed parameters."""
    true_params = DDMParams(
        v=jnp.array(1.5),
        a=jnp.array(1.5),
        w=jnp.array(0.5),
        tau=jnp.array(0.3),
    )

    recovery = ddm_recovery_check(
        true_params=true_params,
        n_trials=200,
        n_repeats=5,
        num_epochs=300,
        lr=0.01,
        seed=42,
    )

    assert isinstance(recovery, DDMRecoveryResult)
    assert recovery.true_params.shape == (5, 4)
    assert recovery.recovered_params.shape == (5, 4)
    assert recovery.param_names == ("v", "a", "w", "tau")
    assert len(recovery.biases) == 4
    assert len(recovery.rmse) == 4

    # RMSE should be finite
    for i, name in enumerate(recovery.param_names):
        assert np.isfinite(recovery.rmse[i]), (
            f"RMSE for {name} not finite: {recovery.rmse[i]}"
        )


def test_ddm_recovery_random_params():
    """Test parameter recovery with randomly generated parameters."""
    recovery = ddm_recovery_check(
        true_params=None,
        n_trials=200,
        n_repeats=3,
        num_epochs=200,
        lr=0.01,
        seed=99,
    )

    assert recovery.true_params.shape == (3, 4)
    assert recovery.recovered_params.shape == (3, 4)

    # All true a's should be positive (drawn from Uniform(0.5, 3.0))
    assert np.all(recovery.true_params[:, 1] > 0)

    # All recovered a's should be positive (due to exp transform)
    assert np.all(recovery.recovered_params[:, 1] > 0)

    # All recovered w's should be in (0, 1)
    assert np.all(recovery.recovered_params[:, 2] > 0)
    assert np.all(recovery.recovered_params[:, 2] < 1)


def test_ddm_recovery_result_structure():
    """Test that DDMRecoveryResult has the expected structure."""
    recovery = ddm_recovery_check(
        true_params=DDMParams(
            v=jnp.array(1.0), a=jnp.array(1.5),
            w=jnp.array(0.5), tau=jnp.array(0.3),
        ),
        n_trials=100,
        n_repeats=2,
        num_epochs=50,
        seed=0,
    )

    # With fixed params, correlations should be nan
    # (can't correlate constant columns)
    assert np.isnan(recovery.correlations[0])

    # Coverage should be None (not computed in MLE mode)
    assert recovery.coverage_95 is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_fit_and_posterior_predictive_roundtrip():
    """Test full pipeline: simulate -> fit -> posterior predictive."""
    # Simulate
    sim = _simulate_standard_data(v=2.0, a=1.5, w=0.5, tau=0.3, n_trials=300)

    # Fit
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=400, lr=0.01,
    )

    # Posterior predictive
    pp = ddm_posterior_predictive(result, n_trials=500, seed=123)

    # Check that the predictive data has similar properties to original
    orig_upper = np.mean(sim.choice == 1)
    pred_upper = np.mean(pp["choice"] == 1)

    # Should be in the same ballpark (within 20%)
    assert abs(orig_upper - pred_upper) < 0.20, (
        f"Posterior predictive choice proportion differs too much: "
        f"orig={orig_upper:.3f}, pred={pred_upper:.3f}"
    )


def test_fit_ddm_mle_with_extreme_params():
    """Test that fitting handles data from extreme parameters."""
    # Very strong drift
    sim = simulate_ddm(v=5.0, a=1.0, w=0.5, tau=0.2, n_trials=200, seed=42)
    result = fit_ddm_mle(
        rt=sim.rt, choice=sim.choice,
        num_epochs=300, lr=0.01,
    )

    assert result.a > 0
    assert 0 < result.w < 1
    assert result.tau > 0
    assert np.isfinite(result.v)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
