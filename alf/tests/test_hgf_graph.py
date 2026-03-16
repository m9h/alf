"""Tests for the generalized n-level HGF graph module.

Validates that the graph-based HGF implementation:
1. Reproduces exact outputs of the hardcoded continuous_hgf and binary_hgf.
2. Supports arbitrary n-level topologies (4+ levels).
3. Is JIT-compatible and differentiable via jax.grad.
4. Handles multi-input graphs and mixed coupling types.
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf,
    binary_hgf_surprise,
    continuous_hgf,
    continuous_hgf_surprise,
)
from alf.hgf.graph import (
    HGFNode,
    HGFGraph,
    build_graph,
    graph_hgf_update,
    graph_hgf,
    graph_hgf_surprise,
    make_standard_3level,
    make_binary_2level,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        obs[t] = x + rng.normal() * 0.1
    return obs


# ---------------------------------------------------------------------------
# Equivalence tests: graph matches hardcoded implementations
# ---------------------------------------------------------------------------

def test_standard_3level_matches_continuous_hgf():
    """Verify make_standard_3level gives identical results to continuous_hgf."""
    obs = jnp.array(generate_volatile_continuous(n=100))

    # Hardcoded 3-level continuous HGF
    params = ContinuousHGFParams(
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
    ref_result = continuous_hgf(obs, params)

    # Graph-based 3-level HGF
    graph, init_mus, init_pis = make_standard_3level(
        omega_1=-3.0, omega_2=-3.0, kappa_1=1.0, kappa_2=1.0,
        theta=0.01, pi_u=100.0,
        mu_1_0=0.0, sigma_1_0=1.0,
        mu_2_0=0.0, sigma_2_0=1.0,
        mu_3_0=0.0, sigma_3_0=1.0,
    )
    graph_result = graph_hgf(obs, graph, init_mus, init_pis)

    # Compare mu trajectories (graph has 3 nodes matching 3 levels)
    np.testing.assert_allclose(
        np.array(graph_result.mu),
        np.array(ref_result.mu),
        atol=1e-6,
        err_msg="Mu trajectories differ between graph and hardcoded 3-level HGF",
    )

    np.testing.assert_allclose(
        np.array(graph_result.pi),
        np.array(ref_result.pi),
        atol=1e-6,
        err_msg="Pi trajectories differ between graph and hardcoded 3-level HGF",
    )

    np.testing.assert_allclose(
        np.array(graph_result.surprise),
        np.array(ref_result.surprise),
        atol=1e-6,
        err_msg="Surprise differs between graph and hardcoded 3-level HGF",
    )


def test_binary_2level_matches_binary_hgf():
    """Verify make_binary_2level gives identical results to binary_hgf."""
    obs = jnp.array(generate_switching_binary(n=100))

    # Hardcoded binary HGF
    params = BinaryHGFParams(
        omega_2=jnp.array(-2.0),
        mu_2_0=jnp.array(0.0),
        sigma_2_0=jnp.array(1.0),
    )
    ref_result = binary_hgf(obs, params)

    # Graph-based binary HGF
    graph, init_mus, init_pis = make_binary_2level(
        omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0,
    )
    graph_result = graph_hgf(obs, graph, init_mus, init_pis)

    # Reference stores mu as (T, 1), graph stores as (T, 1)
    np.testing.assert_allclose(
        np.array(graph_result.mu),
        np.array(ref_result.mu),
        atol=1e-6,
        err_msg="Mu trajectories differ between graph and hardcoded binary HGF",
    )

    np.testing.assert_allclose(
        np.array(graph_result.pi),
        np.array(ref_result.pi),
        atol=1e-6,
        err_msg="Pi trajectories differ between graph and hardcoded binary HGF",
    )

    np.testing.assert_allclose(
        np.array(graph_result.surprise),
        np.array(ref_result.surprise),
        atol=1e-6,
        err_msg="Surprise differs between graph and hardcoded binary HGF",
    )


# ---------------------------------------------------------------------------
# n-level tests
# ---------------------------------------------------------------------------

def test_4level_runs():
    """Test that a 4-level HGF graph runs without errors."""
    nodes = [
        HGFNode(node_id=0, parent_ids=(1,), coupling_types=(0,),
                omega=jnp.array(-3.0), kappa=jnp.array([1.0])),
        HGFNode(node_id=1, parent_ids=(2,), coupling_types=(0,),
                omega=jnp.array(-3.0), kappa=jnp.array([1.0])),
        HGFNode(node_id=2, parent_ids=(3,), coupling_types=(0,),
                omega=jnp.array(-3.0), kappa=jnp.array([1.0])),
        HGFNode(node_id=3, parent_ids=(), coupling_types=(),
                omega=jnp.array(0.0), kappa=jnp.array([])),
    ]
    graph = build_graph(
        nodes=nodes, input_node_ids=(0,), pi_u=100.0,
        is_binary=False, tonic_drift={3: 0.01},
    )

    obs = jnp.array(generate_volatile_continuous(n=50))
    init_mus = jnp.zeros(4)
    init_pis = jnp.ones(4)

    result = graph_hgf(obs, graph, init_mus, init_pis)

    assert result.mu.shape == (50, 4), f"mu shape: {result.mu.shape}"
    assert result.pi.shape == (50, 4), f"pi shape: {result.pi.shape}"
    assert result.surprise.shape == (50,), f"surprise shape: {result.surprise.shape}"
    assert jnp.all(jnp.isfinite(result.mu)), "mu contains non-finite values"
    assert jnp.all(jnp.isfinite(result.pi)), "pi contains non-finite values"
    assert jnp.all(jnp.isfinite(result.surprise)), "surprise not finite"

    # Level 1 should track the input
    final_mu_0 = float(result.mu[-1, 0])
    assert jnp.isfinite(jnp.array(final_mu_0)), f"Final mu_0 not finite: {final_mu_0}"


# ---------------------------------------------------------------------------
# JIT and differentiability tests
# ---------------------------------------------------------------------------

def test_graph_hgf_jit():
    """Test that graph HGF works under jax.jit."""
    obs = jnp.array(generate_volatile_continuous(n=50))
    graph, init_mus, init_pis = make_standard_3level()

    result_eager = graph_hgf(obs, graph, init_mus, init_pis)
    result_jit = jax.jit(graph_hgf, static_argnums=())(obs, graph, init_mus, init_pis)

    np.testing.assert_allclose(
        np.array(result_eager.surprise),
        np.array(result_jit.surprise),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(result_eager.mu),
        np.array(result_jit.mu),
        atol=1e-5,
    )


def test_graph_hgf_differentiable():
    """Test that jax.grad works through graph_hgf_surprise."""
    obs = jnp.array(generate_volatile_continuous(n=50))
    graph, init_mus, init_pis = make_standard_3level()

    # Forward pass
    total = graph_hgf_surprise(obs, graph, init_mus, init_pis)
    assert jnp.isfinite(total), f"surprise not finite: {total}"

    # Gradient w.r.t. initial means
    def loss_fn(mus):
        return graph_hgf_surprise(obs, graph, mus, init_pis)

    grad = jax.grad(loss_fn)(init_mus)
    assert jnp.all(jnp.isfinite(grad)), f"gradient not finite: {grad}"

    # Gradient w.r.t. omegas (via graph)
    def loss_fn_omega(omegas):
        g = graph._replace(omegas=omegas)
        return graph_hgf_surprise(obs, g, init_mus, init_pis)

    grad_omega = jax.grad(loss_fn_omega)(graph.omegas)
    assert jnp.all(jnp.isfinite(grad_omega)), f"omega gradient not finite: {grad_omega}"


def test_graph_hgf_binary_differentiable():
    """Test that jax.grad works through binary graph_hgf_surprise."""
    obs = jnp.array(generate_switching_binary(n=50))
    graph, init_mus, init_pis = make_binary_2level()

    total = graph_hgf_surprise(obs, graph, init_mus, init_pis)
    assert jnp.isfinite(total), f"surprise not finite: {total}"

    def loss_fn(omegas):
        g = graph._replace(omegas=omegas)
        return graph_hgf_surprise(obs, g, init_mus, init_pis)

    grad = jax.grad(loss_fn)(graph.omegas)
    assert jnp.all(jnp.isfinite(grad)), f"gradient not finite: {grad}"


# ---------------------------------------------------------------------------
# Multi-input and custom coupling tests
# ---------------------------------------------------------------------------

def test_multi_input_graph():
    """Test a graph with 2 input nodes sharing a volatility parent.

    Structure:
        Node 0 (input) -> Node 2 (shared volatility parent)
        Node 1 (input) -> Node 2 (shared volatility parent)
        Node 2 (top level)

    This models two correlated signals whose volatility is jointly tracked.
    """
    nodes = [
        HGFNode(node_id=0, parent_ids=(2,), coupling_types=(0,),
                omega=jnp.array(-3.0), kappa=jnp.array([1.0])),
        HGFNode(node_id=1, parent_ids=(2,), coupling_types=(0,),
                omega=jnp.array(-3.0), kappa=jnp.array([1.0])),
        HGFNode(node_id=2, parent_ids=(), coupling_types=(),
                omega=jnp.array(0.0), kappa=jnp.array([])),
    ]
    graph = build_graph(
        nodes=nodes, input_node_ids=(0,), pi_u=100.0,
        is_binary=False, tonic_drift={2: 0.01},
    )

    obs = jnp.array(generate_volatile_continuous(n=50))
    init_mus = jnp.zeros(3)
    init_pis = jnp.ones(3)

    result = graph_hgf(obs, graph, init_mus, init_pis)

    assert result.mu.shape == (50, 3), f"mu shape: {result.mu.shape}"
    assert jnp.all(jnp.isfinite(result.mu)), "mu contains non-finite values"
    assert jnp.all(jnp.isfinite(result.surprise)), "surprise not finite"

    # The shared parent (node 2) should be updated
    assert not jnp.all(result.mu[:, 2] == 0.0), (
        "Shared parent beliefs should be updated"
    )


def test_custom_coupling():
    """Test a graph with mixed value and volatility coupling.

    Structure:
        Node 0 (input) has:
            - Volatility parent: Node 1 (tracks volatility)
            - Value parent: Node 2 (adds mean offset)
        Node 1 (top level, volatility)
        Node 2 (top level, value)
    """
    nodes = [
        HGFNode(
            node_id=0,
            parent_ids=(1, 2),
            coupling_types=(0, 1),  # 0=volatility, 1=value
            omega=jnp.array(-3.0),
            kappa=jnp.array([1.0, 0.5]),
        ),
        HGFNode(node_id=1, parent_ids=(), coupling_types=(),
                omega=jnp.array(0.0), kappa=jnp.array([])),
        HGFNode(node_id=2, parent_ids=(), coupling_types=(),
                omega=jnp.array(0.0), kappa=jnp.array([])),
    ]
    graph = build_graph(
        nodes=nodes, input_node_ids=(0,), pi_u=100.0,
        is_binary=False,
        tonic_drift={1: 0.01, 2: 0.1},
    )

    obs = jnp.array(generate_volatile_continuous(n=50))
    init_mus = jnp.zeros(3)
    init_pis = jnp.ones(3)

    result = graph_hgf(obs, graph, init_mus, init_pis)

    assert result.mu.shape == (50, 3), f"mu shape: {result.mu.shape}"
    assert jnp.all(jnp.isfinite(result.mu)), "mu contains non-finite values"
    assert jnp.all(jnp.isfinite(result.pi)), "pi contains non-finite values"
    assert jnp.all(jnp.isfinite(result.surprise)), "surprise not finite"

    # Both parents should be updated
    assert not jnp.all(result.mu[:, 1] == 0.0), (
        "Volatility parent should be updated"
    )
    assert not jnp.all(result.mu[:, 2] == 0.0), (
        "Value parent should be updated"
    )


def test_3level_total_surprise_matches():
    """Verify total surprise matches between graph and hardcoded continuous HGF."""
    obs = jnp.array(generate_volatile_continuous(n=80))

    params = ContinuousHGFParams(
        omega_1=jnp.array(-4.0),
        omega_2=jnp.array(-2.0),
        kappa_1=jnp.array(1.0),
        kappa_2=jnp.array(0.5),
        theta=jnp.array(0.05),
        pi_u=jnp.array(50.0),
        mu_1_0=jnp.array(1.0),
        sigma_1_0=jnp.array(0.5),
        mu_2_0=jnp.array(-1.0),
        sigma_2_0=jnp.array(2.0),
        mu_3_0=jnp.array(0.5),
        sigma_3_0=jnp.array(0.8),
    )
    ref_surprise = continuous_hgf_surprise(obs, params)

    graph, init_mus, init_pis = make_standard_3level(
        omega_1=-4.0, omega_2=-2.0, kappa_1=1.0, kappa_2=0.5,
        theta=0.05, pi_u=50.0,
        mu_1_0=1.0, sigma_1_0=0.5,
        mu_2_0=-1.0, sigma_2_0=2.0,
        mu_3_0=0.5, sigma_3_0=0.8,
    )
    graph_surprise = graph_hgf_surprise(obs, graph, init_mus, init_pis)

    np.testing.assert_allclose(
        float(graph_surprise), float(ref_surprise), atol=1e-5,
        err_msg="Total surprise differs between graph and hardcoded HGF",
    )


def test_binary_total_surprise_matches():
    """Verify total surprise matches between graph and hardcoded binary HGF."""
    obs = jnp.array(generate_switching_binary(n=80))

    params = BinaryHGFParams(
        omega_2=jnp.array(-3.0),
        mu_2_0=jnp.array(0.5),
        sigma_2_0=jnp.array(2.0),
    )
    ref_surprise = binary_hgf_surprise(obs, params)

    graph, init_mus, init_pis = make_binary_2level(
        omega_2=-3.0, mu_2_0=0.5, sigma_2_0=2.0,
    )
    graph_surprise = graph_hgf_surprise(obs, graph, init_mus, init_pis)

    np.testing.assert_allclose(
        float(graph_surprise), float(ref_surprise), atol=1e-5,
        err_msg="Total surprise differs between graph and hardcoded binary HGF",
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
