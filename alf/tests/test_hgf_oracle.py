"""Oracle validation tests for HGF against reference data and pyhgf.

Tests that our HGF implementation produces stable, reproducible results
on canonical datasets. Uses saved reference trajectories for regression
testing, and optionally compares against pyhgf when installed.

The reference data was generated with known parameters:
    Binary: omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0
    Continuous: omega_1=-3.0, omega_2=-3.0, kappa_1=1.0, kappa_2=1.0,
                theta=0.01, pi_u=100.0
"""

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf,
    continuous_hgf,
    binary_hgf_surprise,
    continuous_hgf_surprise,
)

DATA_DIR = Path(__file__).parent / "data"
REF_FILE = DATA_DIR / "hgf_reference.npz"
HAS_REFERENCE = REF_FILE.exists()

try:
    import pyhgf
    HAS_PYHGF = True
except ImportError:
    HAS_PYHGF = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reference():
    """Load saved reference trajectories."""
    return np.load(REF_FILE)


def make_binary_params():
    return BinaryHGFParams(
        omega_2=jnp.array(-2.0),
        mu_2_0=jnp.array(0.0),
        sigma_2_0=jnp.array(1.0),
    )


def make_continuous_params():
    return ContinuousHGFParams(
        omega_1=jnp.array(-3.0), omega_2=jnp.array(-3.0),
        kappa_1=jnp.array(1.0), kappa_2=jnp.array(1.0),
        theta=jnp.array(0.01), pi_u=jnp.array(100.0),
        mu_1_0=jnp.array(0.0), sigma_1_0=jnp.array(1.0),
        mu_2_0=jnp.array(0.0), sigma_2_0=jnp.array(1.0),
        mu_3_0=jnp.array(0.0), sigma_3_0=jnp.array(1.0),
    )


# ---------------------------------------------------------------------------
# Regression tests against saved reference
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_binary_hgf_mu_matches_reference():
    """Test that binary HGF mu trajectory matches saved reference."""
    ref = load_reference()
    result = binary_hgf(jnp.array(ref["binary_obs"]), make_binary_params())

    np.testing.assert_allclose(
        np.array(result.mu), ref["binary_mu"], atol=1e-5,
        err_msg="Binary HGF mu trajectory drifted from reference"
    )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_binary_hgf_pi_matches_reference():
    """Test that binary HGF precision trajectory matches saved reference."""
    ref = load_reference()
    result = binary_hgf(jnp.array(ref["binary_obs"]), make_binary_params())

    np.testing.assert_allclose(
        np.array(result.pi), ref["binary_pi"], atol=1e-5,
        err_msg="Binary HGF pi trajectory drifted from reference"
    )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_binary_hgf_surprise_matches_reference():
    """Test that binary HGF surprise matches saved reference."""
    ref = load_reference()
    result = binary_hgf(jnp.array(ref["binary_obs"]), make_binary_params())

    np.testing.assert_allclose(
        np.array(result.surprise), ref["binary_surprise"], atol=1e-5,
        err_msg="Binary HGF surprise drifted from reference"
    )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_binary_hgf_total_surprise_stable():
    """Test total binary surprise is stable."""
    ref = load_reference()
    total = float(binary_hgf_surprise(
        jnp.array(ref["binary_obs"]), make_binary_params()
    ))
    expected = float(np.sum(ref["binary_surprise"]))

    np.testing.assert_allclose(total, expected, atol=1e-4,
        err_msg=f"Total surprise changed: {total:.4f} vs {expected:.4f}"
    )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_continuous_hgf_mu_matches_reference():
    """Test that continuous HGF mu trajectory matches saved reference."""
    ref = load_reference()
    result = continuous_hgf(jnp.array(ref["continuous_obs"]), make_continuous_params())

    np.testing.assert_allclose(
        np.array(result.mu), ref["continuous_mu"], atol=1e-4,
        err_msg="Continuous HGF mu trajectory drifted from reference"
    )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference data not generated")
def test_continuous_hgf_surprise_matches_reference():
    """Test that continuous HGF surprise matches saved reference."""
    ref = load_reference()
    result = continuous_hgf(jnp.array(ref["continuous_obs"]), make_continuous_params())

    np.testing.assert_allclose(
        np.array(result.surprise), ref["continuous_surprise"], atol=1e-4,
        err_msg="Continuous HGF surprise drifted from reference"
    )


# ---------------------------------------------------------------------------
# Hand-computed validation (first few trials)
# ---------------------------------------------------------------------------

def test_binary_hgf_first_trial_hand_computed():
    """Verify first trial of binary HGF against hand computation.

    With mu_2=0, sigma_2=1, omega_2=-2:
    - hat_mu_2 = 0.0
    - hat_sigma_2 = 1.0 + exp(-2) = 1.1353
    - hat_mu_1 = sigmoid(0) = 0.5
    - For u=1: delta_1 = 1 - 0.5 = 0.5
    - info_gain = 0.5 * 0.5 = 0.25
    - new_pi_2 = 1/1.1353 + 0.25 = 1.1308
    - new_mu_2 = 0.0 + 0.5 / 1.1308 = 0.4422
    - surprise = -log(0.5) = 0.6931
    """
    obs = jnp.array([1.0])
    params = make_binary_params()
    result = binary_hgf(obs, params)

    expected_surprise = -np.log(0.5)
    np.testing.assert_allclose(
        float(result.surprise[0]), expected_surprise, atol=1e-3,
        err_msg="First trial surprise should be -log(0.5)"
    )

    # mu_2 should move toward positive (since we observed 1)
    assert float(result.mu[0, 0]) > 0.0, "mu_2 should increase after observing 1"

    # Verify exact mu_2 value
    hat_sigma_2 = 1.0 + np.exp(-2.0)
    hat_mu_1 = 1.0 / (1.0 + np.exp(0.0))  # sigmoid(0) = 0.5
    delta_1 = 1.0 - hat_mu_1
    info_gain = hat_mu_1 * (1.0 - hat_mu_1)
    new_pi_2 = 1.0 / hat_sigma_2 + info_gain
    expected_mu_2 = 0.0 + delta_1 / new_pi_2

    np.testing.assert_allclose(
        float(result.mu[0, 0]), expected_mu_2, atol=1e-4,
        err_msg="First trial mu_2 doesn't match hand computation"
    )


def test_continuous_hgf_constant_input_converges():
    """Test that continuous HGF converges to constant input."""
    obs = jnp.ones(100) * 3.0
    params = make_continuous_params()
    result = continuous_hgf(obs, params)

    # Level 1 should converge near 3.0
    final_mu_1 = float(result.mu[-1, 0])
    np.testing.assert_allclose(final_mu_1, 3.0, atol=0.3,
        err_msg=f"Level 1 should converge to 3.0, got {final_mu_1:.3f}"
    )

    # Surprise should decrease as the model learns
    early_surprise = float(jnp.mean(result.surprise[:10]))
    late_surprise = float(jnp.mean(result.surprise[-10:]))
    assert late_surprise < early_surprise, (
        f"Surprise should decrease: early={early_surprise:.3f} late={late_surprise:.3f}"
    )


# ---------------------------------------------------------------------------
# pyhgf oracle tests (skip if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYHGF, reason="pyhgf not installed")
def test_binary_hgf_matches_pyhgf():
    """Compare binary HGF output against pyhgf reference implementation.

    Runs the same binary observation sequence through both alf's binary_hgf
    and pyhgf's Network API with identical parameters, then asserts that
    mu_2, pi_2, and per-trial surprise trajectories match within tolerance.
    """
    from pyhgf import load_data
    from pyhgf.model import Network
    from pyhgf.response import first_level_binary_surprise

    # Load pyhgf's canonical binary dataset
    u, _y = load_data("binary")
    u = np.asarray(u, dtype=float).ravel()

    # Shared parameters
    omega_2 = -2.0
    mu_2_0 = 0.0
    sigma_2_0 = 1.0

    # --- Run pyhgf ---
    network = Network()
    network.add_nodes(kind="binary-state")
    network.add_nodes(
        kind="continuous-state",
        node_parameters={
            "tonic_volatility": omega_2,
            "mean": mu_2_0,
            "precision": 1.0 / sigma_2_0,
        },
        value_children=0,
    )
    network.input_data(input_data=u.reshape(-1, 1))

    pyhgf_mu = np.array(network.node_trajectories[1]["mean"])
    pyhgf_pi = np.array(network.node_trajectories[1]["precision"])
    pyhgf_surprise = np.array(
        network.surprise(response_function=first_level_binary_surprise)
    )

    # --- Run alf ---
    params = BinaryHGFParams(
        omega_2=jnp.array(omega_2),
        mu_2_0=jnp.array(mu_2_0),
        sigma_2_0=jnp.array(sigma_2_0),
    )
    alf_result = binary_hgf(jnp.array(u), params)
    alf_mu = np.array(alf_result.mu[:, 0])
    alf_pi = np.array(alf_result.pi[:, 0])
    alf_surprise = np.array(alf_result.surprise)

    # --- Compare trajectories ---
    np.testing.assert_allclose(
        alf_mu, pyhgf_mu, atol=1e-5,
        err_msg="Binary HGF mu_2 trajectory diverges from pyhgf",
    )
    np.testing.assert_allclose(
        alf_pi, pyhgf_pi, atol=1e-5,
        err_msg="Binary HGF pi_2 trajectory diverges from pyhgf",
    )
    np.testing.assert_allclose(
        alf_surprise, pyhgf_surprise, atol=1e-5,
        err_msg="Binary HGF per-trial surprise diverges from pyhgf",
    )

    # Sanity: total surprise should also match
    np.testing.assert_allclose(
        float(np.sum(alf_surprise)),
        float(np.sum(pyhgf_surprise)),
        atol=1e-3,
        err_msg="Binary HGF total surprise diverges from pyhgf",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
