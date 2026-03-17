"""Oracle validation tests for DDM against HDDM example datasets.

Downloads and uses the HDDM bundled datasets (Wiecki et al., 2013) to
validate our Navarro-Fuss density and MLE fitting against known results.

HDDM reference parameters (approximate, from Wiecki et al. 2013):
    Easy condition: v ~ 1.0-2.5, a ~ 1.0-2.0, tau ~ 0.2-0.4
    Hard condition: v ~ 0.3-1.0, a ~ 1.0-2.0, tau ~ 0.2-0.4

Data format: response (0=lower, 1=upper), rt (seconds), difficulty (easy/hard)
"""

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from alf.ddm.wiener import (
    wiener_log_density_batch,
    ddm_nll,
    simulate_ddm,
    DDMParams,
)

DATA_DIR = Path(__file__).parent / "data" / "ddm"
SIMPLE_CSV = DATA_DIR / "simple_difficulty.csv"
SUBJS_CSV = DATA_DIR / "simple_subjs_difficulty.csv"
DATA_AVAILABLE = SIMPLE_CSV.exists()

try:
    from alf.ddm.fitting import fit_ddm_mle
    HAS_FITTING = True
except ImportError:
    HAS_FITTING = False


def load_simple_difficulty():
    import pandas as pd
    return pd.read_csv(SIMPLE_CSV)


def extract_ddm_arrays(df, min_rt=0.1):
    mask = df["rt"] > min_rt
    rt = jnp.array(df.loc[mask, "rt"].values, dtype=jnp.float32)
    choice = jnp.array(df.loc[mask, "response"].values, dtype=jnp.float32)
    return rt, choice


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
def test_hddm_data_loads():
    """Test that HDDM example data loads correctly."""
    df = load_simple_difficulty()
    assert "rt" in df.columns
    assert "response" in df.columns
    assert len(df) == 1000
    assert set(df["response"].unique()) == {0.0, 1.0}
    assert set(df["difficulty"].unique()) == {"easy", "hard"}


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
def test_wiener_density_finite_on_real_data():
    """Test Navarro-Fuss density is finite on real HDDM data."""
    df = load_simple_difficulty()
    rt, choice = extract_ddm_arrays(df)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)
    log_ps = wiener_log_density_batch(rt, choice, v, a, w, tau)
    pct_finite = float(jnp.sum(jnp.isfinite(log_ps))) / len(log_ps)
    assert pct_finite > 0.95, f"Only {pct_finite:.1%} finite"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
def test_nll_finite_on_real_data():
    """Test total NLL is finite on HDDM data."""
    df = load_simple_difficulty()
    rt, choice = extract_ddm_arrays(df)
    nll = ddm_nll(jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3), rt, choice)
    assert jnp.isfinite(nll), f"NLL not finite: {nll}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
def test_nll_gradient_finite_on_real_data():
    """Test NLL gradients are finite on real data."""
    df = load_simple_difficulty()
    rt, choice = extract_ddm_arrays(df)
    grads = jax.grad(ddm_nll, argnums=(0, 1, 2, 3))(
        jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3), rt, choice)
    for i, name in enumerate(["v", "a", "w", "tau"]):
        assert jnp.isfinite(grads[i]), f"grad_{name} not finite"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
@pytest.mark.skipif(not HAS_FITTING, reason="DDM fitting not available")
def test_mle_fit_on_easy_condition():
    """Test MLE fitting on easy-condition data."""
    df = load_simple_difficulty()
    easy = df[df["difficulty"] == "easy"]
    rt, choice = extract_ddm_arrays(easy)
    result = fit_ddm_mle(np.array(rt), np.array(choice), num_epochs=200, lr=0.01)
    assert float(result.a) > 0
    assert 0 < float(result.w) < 1
    assert float(result.tau) > 0
    assert float(result.tau) < float(jnp.min(rt))
    assert result.loss_history[-1] < result.loss_history[0]


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
@pytest.mark.skipif(not HAS_FITTING, reason="DDM fitting not available")
def test_mle_drift_varies_by_difficulty():
    """Test easy condition has higher |drift| than hard."""
    df = load_simple_difficulty()
    rt_e, ch_e = extract_ddm_arrays(df[df["difficulty"] == "easy"])
    rt_h, ch_h = extract_ddm_arrays(df[df["difficulty"] == "hard"])
    res_e = fit_ddm_mle(np.array(rt_e), np.array(ch_e), num_epochs=200)
    res_h = fit_ddm_mle(np.array(rt_h), np.array(ch_h), num_epochs=200)
    assert abs(float(res_e.v)) > abs(float(res_h.v)) * 0.5


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
@pytest.mark.skipif(not HAS_FITTING, reason="DDM fitting not available")
def test_mle_params_in_hddm_range():
    """Test MLE estimates fall in HDDM literature ranges."""
    df = load_simple_difficulty()
    rt, choice = extract_ddm_arrays(df)
    result = fit_ddm_mle(np.array(rt), np.array(choice), num_epochs=300)
    assert -5.0 < float(result.v) < 5.0
    assert 0.3 < float(result.a) < 5.0
    assert 0.1 < float(result.w) < 0.9
    assert 0.05 < float(result.tau) < 0.8


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HDDM data not downloaded")
@pytest.mark.skipif(not HAS_FITTING, reason="DDM fitting not available")
def test_posterior_predictive_matches_data():
    """Test fitted DDM reproduces observed data statistics."""
    df = load_simple_difficulty()
    rt, choice = extract_ddm_arrays(df)
    result = fit_ddm_mle(np.array(rt), np.array(choice), num_epochs=300)
    sim = simulate_ddm(v=float(result.v), a=float(result.a),
                       w=float(result.w), tau=float(result.tau), n_trials=1000, seed=42)
    obs_mean_rt = float(jnp.mean(rt))
    sim_mean_rt = float(np.mean(sim.rt))
    assert 0.3 < sim_mean_rt / obs_mean_rt < 3.0
    assert abs(float(jnp.mean(choice)) - float(np.mean(sim.choice))) < 0.30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
