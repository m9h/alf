"""Tests for the Hierarchical Gaussian Filter module.

Tests the HGF update equations, verifying:
1. Binary HGF tracks a switching probability source correctly.
2. Continuous HGF tracks a volatile signal.
3. Surprise is differentiable w.r.t. parameters.
4. HGF-to-discrete bridge produces valid categorical distributions.
5. Parameter learning reduces NLL on synthetic data.
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf_update,
    binary_hgf,
    continuous_hgf_update,
    continuous_hgf,
    binary_hgf_surprise,
    continuous_hgf_surprise,
)
from alf.hgf.bridge import (
    discretize_belief,
    hgf_to_categorical,
)
from alf.hgf.learning import (
    binary_hgf_nll,
    learn_binary_hgf,
    learn_continuous_hgf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_binary_params(omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0):
    """Create standard binary HGF parameters."""
    return BinaryHGFParams(
        omega_2=jnp.array(omega_2),
        mu_2_0=jnp.array(mu_2_0),
        sigma_2_0=jnp.array(sigma_2_0),
    )


def make_continuous_params():
    """Create standard 3-level continuous HGF parameters."""
    return ContinuousHGFParams(
        omega_1=jnp.array(-3.0),
        omega_2=jnp.array(-3.0),
        kappa_1=jnp.array(1.0),
        kappa_2=jnp.array(1.0),
        theta=jnp.array(0.01),
        pi_u=jnp.array(100.0),
        mu_1_0=jnp.array(0.0),
        sigma_1_0=jnp.array(1.0),
        mu_2_0=jnp.array(0.0),
        sigma_2_0=jnp.array(1.0),
        mu_3_0=jnp.array(0.0),
        sigma_3_0=jnp.array(1.0),
    )


def generate_switching_binary(n=200, switch_prob=0.05, seed=42):
    """Generate binary observations from a switching source."""
    rng = np.random.RandomState(seed)
    p = 0.8
    obs = np.zeros(n)
    for t in range(n):
        obs[t] = rng.binomial(1, p)
        if rng.random() < switch_prob:
            p = 1.0 - p
    return obs


def generate_volatile_continuous(n=200, seed=42):
    """Generate continuous observations from a volatile source."""
    rng = np.random.RandomState(seed)
    x = 0.0
    obs = np.zeros(n)
    volatility = 0.1
    for t in range(n):
        volatility_noise = rng.normal() * 0.01
        volatility = max(0.01, volatility + volatility_noise)
        x += rng.normal() * np.sqrt(volatility)
        obs[t] = x + rng.normal() * 0.1  # observation noise
    return obs


# ---------------------------------------------------------------------------
# Binary HGF tests
# ---------------------------------------------------------------------------

def test_binary_hgf_update_returns_valid():
    """Test that a single binary HGF update returns finite values."""
    mu_2 = jnp.array(0.0)
    pi_2 = jnp.array(1.0)
    u = jnp.array(1.0)
    omega_2 = jnp.array(-2.0)

    new_mu, new_pi, surprise = binary_hgf_update(mu_2, pi_2, u, omega_2)

    assert jnp.isfinite(new_mu), f"mu not finite: {new_mu}"
    assert jnp.isfinite(new_pi), f"pi not finite: {new_pi}"
    assert new_pi > 0, f"pi should be positive: {new_pi}"
    assert jnp.isfinite(surprise), f"surprise not finite: {surprise}"
    assert surprise >= 0, f"surprise should be non-negative: {surprise}"


def test_binary_hgf_update_moves_toward_observation():
    """Test that observing 1s increases predicted probability."""
    mu_2 = jnp.array(0.0)
    pi_2 = jnp.array(1.0)
    omega_2 = jnp.array(-2.0)

    # After seeing several 1s, mu_2 should increase (sigmoid(mu_2) -> higher)
    for _ in range(10):
        mu_2, pi_2, _ = binary_hgf_update(mu_2, pi_2, jnp.array(1.0), omega_2)

    predicted_p = float(jax.nn.sigmoid(mu_2))
    assert predicted_p > 0.6, (
        f"After 10 observations of 1, predicted P(1) should be > 0.6, "
        f"got {predicted_p:.3f}"
    )


def test_binary_hgf_sequence():
    """Test binary HGF on a full sequence."""
    obs = generate_switching_binary(n=100)
    params = make_binary_params()
    obs_jnp = jnp.array(obs)

    result = binary_hgf(obs_jnp, params)

    assert result.mu.shape == (100, 1), f"mu shape: {result.mu.shape}"
    assert result.pi.shape == (100, 1), f"pi shape: {result.pi.shape}"
    assert result.surprise.shape == (100,), f"surprise shape: {result.surprise.shape}"
    assert jnp.all(jnp.isfinite(result.mu)), "mu contains non-finite values"
    assert jnp.all(jnp.isfinite(result.pi)), "pi contains non-finite values"
    assert jnp.all(result.pi > 0), "pi should be positive"
    assert jnp.all(jnp.isfinite(result.surprise)), "surprise not finite"


def test_binary_hgf_surprise_differentiable():
    """Test that binary HGF surprise is differentiable w.r.t. omega_2."""
    obs = jnp.array(generate_switching_binary(n=50))
    params = make_binary_params()

    # Forward pass
    total_surprise = binary_hgf_surprise(obs, params)
    assert jnp.isfinite(total_surprise), f"surprise not finite: {total_surprise}"

    # Gradient w.r.t. omega_2
    def loss_fn(omega_2):
        p = BinaryHGFParams(
            omega_2=omega_2,
            mu_2_0=params.mu_2_0,
            sigma_2_0=params.sigma_2_0,
        )
        return binary_hgf_surprise(obs, p)

    grad = jax.grad(loss_fn)(params.omega_2)
    assert jnp.isfinite(grad), f"gradient not finite: {grad}"


def test_binary_hgf_jit():
    """Test that binary HGF works under jax.jit."""
    obs = jnp.array(generate_switching_binary(n=50))
    params = make_binary_params()

    result_eager = binary_hgf(obs, params)
    result_jit = jax.jit(binary_hgf)(obs, params)

    np.testing.assert_allclose(
        np.array(result_eager.surprise),
        np.array(result_jit.surprise),
        atol=1e-5,
    )


def test_binary_hgf_high_volatility_tracks_faster():
    """Test that higher volatility (less negative omega) tracks changes faster."""
    obs = np.zeros(40)
    obs[:20] = 1.0  # Switch from 1 to 0 at trial 20
    obs_jnp = jnp.array(obs)

    # Low volatility (slow tracking)
    params_low = make_binary_params(omega_2=-4.0)
    result_low = binary_hgf(obs_jnp, params_low)

    # High volatility (fast tracking)
    params_high = make_binary_params(omega_2=-1.0)
    result_high = binary_hgf(obs_jnp, params_high)

    # After the switch (trial 30), high-vol should have lower mu_2
    # (closer to 0 prediction = sigmoid(negative))
    mu_low_30 = float(result_low.mu[30, 0])
    mu_high_30 = float(result_high.mu[30, 0])

    assert mu_high_30 < mu_low_30, (
        f"High-vol agent should track the switch faster: "
        f"mu_high={mu_high_30:.3f} vs mu_low={mu_low_30:.3f}"
    )


# ---------------------------------------------------------------------------
# Continuous HGF tests
# ---------------------------------------------------------------------------

def test_continuous_hgf_update_returns_valid():
    """Test that a single continuous HGF update returns finite values."""
    state = (
        jnp.array(0.0), jnp.array(1.0),  # mu_1, pi_1
        jnp.array(0.0), jnp.array(1.0),  # mu_2, pi_2
        jnp.array(0.0), jnp.array(1.0),  # mu_3, pi_3
    )
    u = jnp.array(0.5)

    result = continuous_hgf_update(
        *state, u,
        omega_1=jnp.array(-3.0), omega_2=jnp.array(-3.0),
        kappa_1=jnp.array(1.0), kappa_2=jnp.array(1.0),
        theta=jnp.array(0.01), pi_u=jnp.array(100.0),
    )

    for i, name in enumerate(["mu_1", "pi_1", "mu_2", "pi_2", "mu_3", "pi_3", "surprise"]):
        assert jnp.isfinite(result[i]), f"{name} not finite: {result[i]}"

    # Precisions should be positive
    assert result[1] > 0, f"pi_1 should be positive: {result[1]}"
    assert result[3] > 0, f"pi_2 should be positive: {result[3]}"
    assert result[5] > 0, f"pi_3 should be positive: {result[5]}"


def test_continuous_hgf_sequence():
    """Test continuous HGF on a full sequence."""
    obs = generate_volatile_continuous(n=100)
    params = make_continuous_params()
    obs_jnp = jnp.array(obs)

    result = continuous_hgf(obs_jnp, params)

    assert result.mu.shape == (100, 3), f"mu shape: {result.mu.shape}"
    assert result.pi.shape == (100, 3), f"pi shape: {result.pi.shape}"
    assert result.surprise.shape == (100,), f"surprise shape: {result.surprise.shape}"
    assert jnp.all(jnp.isfinite(result.mu)), "mu contains non-finite values"
    assert jnp.all(jnp.isfinite(result.surprise)), "surprise not finite"


def test_continuous_hgf_tracks_input():
    """Test that level 1 mean tracks the input signal."""
    # Simple signal: constant at 2.0
    obs = jnp.ones(50) * 2.0
    params = make_continuous_params()

    result = continuous_hgf(obs, params)

    # Level 1 mean should converge toward 2.0
    final_mu_1 = float(result.mu[-1, 0])
    assert abs(final_mu_1 - 2.0) < 0.5, (
        f"Level 1 should track constant input: mu_1={final_mu_1:.3f}"
    )


def test_continuous_hgf_surprise_differentiable():
    """Test that continuous HGF surprise is differentiable."""
    obs = jnp.array(generate_volatile_continuous(n=50))
    params = make_continuous_params()

    total = continuous_hgf_surprise(obs, params)
    assert jnp.isfinite(total), f"surprise not finite: {total}"

    # Gradient w.r.t. omega_1
    def loss_fn(omega_1):
        p = params._replace(omega_1=omega_1)
        return continuous_hgf_surprise(obs, p)

    grad = jax.grad(loss_fn)(params.omega_1)
    assert jnp.isfinite(grad), f"gradient not finite: {grad}"


def test_continuous_hgf_jit():
    """Test continuous HGF under jax.jit."""
    obs = jnp.array(generate_volatile_continuous(n=50))
    params = make_continuous_params()

    result_eager = continuous_hgf(obs, params)
    result_jit = jax.jit(continuous_hgf)(obs, params)

    np.testing.assert_allclose(
        np.array(result_eager.surprise),
        np.array(result_jit.surprise),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------

def test_discretize_belief_sums_to_one():
    """Test that discretized belief is a valid probability distribution."""
    mu = jnp.array(0.0)
    pi = jnp.array(4.0)
    bin_edges = jnp.array([-jnp.inf, -1.0, 0.0, 1.0, jnp.inf])

    probs = discretize_belief(mu, pi, bin_edges)

    assert probs.shape == (4,), f"Expected 4 bins, got {probs.shape}"
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)
    assert jnp.all(probs > 0), "All probabilities should be positive"


def test_discretize_belief_centered():
    """Test that a centered Gaussian belief puts mass in center bins."""
    mu = jnp.array(0.0)
    pi = jnp.array(10.0)  # High precision = narrow
    bin_edges = jnp.array([-jnp.inf, -1.0, 0.0, 1.0, jnp.inf])

    probs = discretize_belief(mu, pi, bin_edges)

    # Center bins ([-1,0] and [0,1]) should have most mass
    center_mass = float(probs[1] + probs[2])
    assert center_mass > 0.95, f"Center mass should be > 0.95: {center_mass:.3f}"


def test_hgf_to_categorical():
    """Test the convenience discretization function."""
    probs = hgf_to_categorical(
        mu=jnp.array(0.0), pi=jnp.array(1.0),
        num_states=5, state_range=(-3.0, 3.0),
    )

    assert probs.shape == (5,), f"Expected 5 states, got {probs.shape}"
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Learning tests
# ---------------------------------------------------------------------------

def test_binary_hgf_nll_differentiable():
    """Test that binary HGF NLL is differentiable."""
    obs = jnp.array(generate_switching_binary(n=50))
    omega_2 = jnp.array(-2.0)
    mu_2_0 = jnp.array(0.0)
    sigma_2_0 = jnp.array(1.0)

    nll = binary_hgf_nll(omega_2, obs, mu_2_0, sigma_2_0)
    assert jnp.isfinite(nll), f"NLL not finite: {nll}"

    grad = jax.grad(binary_hgf_nll)(omega_2, obs, mu_2_0, sigma_2_0)
    assert jnp.isfinite(grad), f"gradient not finite: {grad}"


def test_learn_binary_hgf():
    """Test that learning reduces NLL for binary HGF."""
    obs = generate_switching_binary(n=200)

    result = learn_binary_hgf(
        observations=obs,
        init_omega_2=-2.0,
        num_epochs=50,
        lr=0.01,
    )

    assert len(result.loss_history) == 50
    assert result.loss_history[-1] <= result.loss_history[0], (
        f"Loss should decrease: {result.loss_history[0]:.4f} -> "
        f"{result.loss_history[-1]:.4f}"
    )


def test_learn_continuous_hgf():
    """Test that learning reduces NLL for continuous HGF."""
    obs = generate_volatile_continuous(n=100)

    result = learn_continuous_hgf(
        observations=obs,
        num_epochs=30,
        lr=0.001,
    )

    assert len(result.loss_history) == 30
    # Loss should at least not increase substantially
    assert result.loss_history[-1] <= result.loss_history[0] * 1.1, (
        f"Loss should not increase: {result.loss_history[0]:.4f} -> "
        f"{result.loss_history[-1]:.4f}"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
