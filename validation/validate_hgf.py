#!/usr/bin/env python3
"""Validate alf.hgf against pyhgf reference implementation.

Runs identical data through both implementations and compares
posterior trajectories (mu, pi) and surprise at each trial.

Usage:
    python validate_hgf.py

Requires: pyhgf (`pip install pyhgf`)
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------

try:
    import jax.numpy as jnp
    from alf.hgf.updates import (
        BinaryHGFParams, binary_hgf,
        ContinuousHGFParams, continuous_hgf,
    )
except ImportError:
    print("FAIL: alf not importable. Install with: pip install -e /path/to/alf")
    sys.exit(1)

try:
    from pyhgf import load_data
    from pyhgf.model import HGF
    HAS_PYHGF = True
except ImportError:
    HAS_PYHGF = False
    print("WARN: pyhgf not installed. Install with: pip install pyhgf")
    print("      Will run self-consistency checks only.\n")


def compare(name, alf_val, ref_val, atol=1e-2):
    """Compare two arrays and report result."""
    alf_val = np.asarray(alf_val).ravel()
    ref_val = np.asarray(ref_val).ravel()

    if len(alf_val) != len(ref_val):
        print(f"  FAIL {name}: shape mismatch alf={alf_val.shape} ref={ref_val.shape}")
        return False

    max_diff = float(np.max(np.abs(alf_val - ref_val)))
    mean_diff = float(np.mean(np.abs(alf_val - ref_val)))
    corr = float(np.corrcoef(alf_val, ref_val)[0, 1]) if len(alf_val) > 1 else 1.0

    if max_diff < atol:
        print(f"  PASS {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} r={corr:.6f}")
        return True
    elif max_diff < atol * 10:
        print(f"  WARN {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} r={corr:.6f}")
        return True
    else:
        print(f"  FAIL {name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} r={corr:.6f}")
        return False


# ---------------------------------------------------------------------------
# Validation 1: Binary HGF with pyhgf data
# ---------------------------------------------------------------------------

def validate_binary_hgf_with_pyhgf():
    """Compare binary HGF against pyhgf on its built-in dataset."""
    print("=" * 60)
    print("Binary HGF: alf vs pyhgf")
    print("=" * 60)

    if not HAS_PYHGF:
        print("  SKIP: pyhgf not installed\n")
        return None

    # Load pyhgf's built-in binary data
    u, y = load_data("binary")
    u = np.asarray(u, dtype=float).ravel()

    # Run pyhgf
    omega_2 = -2.0
    mu_2_0 = 0.0
    sigma_2_0 = 1.0

    try:
        pyhgf_model = HGF(
            n_levels=2,
            model_type="binary",
            initial_mean={"1": mu_2_0},
            initial_precision={"1": 1.0 / sigma_2_0},
            tonic_volatility={"1": omega_2},
        )
        pyhgf_model.input_data(input_data=u)

        # Extract pyhgf trajectories
        pyhgf_mu = pyhgf_model.node_trajectories[1]["mean"]
        pyhgf_pi = pyhgf_model.node_trajectories[1]["precision"]
        pyhgf_surprise = pyhgf_model.surprise()
    except Exception as e:
        print(f"  WARN: pyhgf API may have changed: {e}")
        print("  Trying alternative API...")
        try:
            # pyhgf >= 0.2 API
            from pyhgf.model import Network
            network = Network()
            network.add_nodes(kind="binary-input")
            network.add_nodes(
                kind="continuous-state",
                tonic_volatility=omega_2,
            )
            network.add_edges(child_idxs=0, parent_idxs=1)
            network.input_data(input_data=u)
            pyhgf_mu = np.array(network.node_trajectories[1]["mean"])
            pyhgf_pi = np.array(network.node_trajectories[1]["precision"])
            pyhgf_surprise = np.array(network.surprise())
        except Exception as e2:
            print(f"  FAIL: Cannot run pyhgf: {e2}\n")
            return False

    # Run alf
    params = BinaryHGFParams(
        omega_2=jnp.array(omega_2),
        mu_2_0=jnp.array(mu_2_0),
        sigma_2_0=jnp.array(sigma_2_0),
    )
    alf_result = binary_hgf(jnp.array(u), params)
    alf_mu = np.array(alf_result.mu[:, 0])
    alf_pi = np.array(alf_result.pi[:, 0])
    alf_surprise = np.array(alf_result.surprise)

    # Trim to same length
    T = min(len(alf_mu), len(pyhgf_mu))
    alf_mu, pyhgf_mu = alf_mu[:T], pyhgf_mu[:T]
    alf_pi, pyhgf_pi = alf_pi[:T], pyhgf_pi[:T]
    alf_surprise, pyhgf_surprise = alf_surprise[:T], pyhgf_surprise[:T]

    ok = True
    ok &= compare("mu_2 trajectory", alf_mu, pyhgf_mu, atol=0.1)
    ok &= compare("pi_2 trajectory", alf_pi, pyhgf_pi, atol=0.5)
    ok &= compare("per-trial surprise", alf_surprise, pyhgf_surprise, atol=0.1)

    total_alf = float(np.sum(alf_surprise))
    total_pyhgf = float(np.sum(pyhgf_surprise))
    print(f"  Total surprise: alf={total_alf:.4f} pyhgf={total_pyhgf:.4f} "
          f"diff={abs(total_alf - total_pyhgf):.4f}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Validation 2: Self-consistency checks (no pyhgf needed)
# ---------------------------------------------------------------------------

def validate_hgf_self_consistency():
    """Run self-consistency checks that don't need pyhgf."""
    print("=" * 60)
    print("HGF Self-Consistency Checks")
    print("=" * 60)

    ok = True

    # Check 1: Binary HGF surprise is non-negative
    np.random.seed(42)
    obs = np.random.binomial(1, 0.7, 100).astype(float)
    params = BinaryHGFParams(
        omega_2=jnp.array(-2.0), mu_2_0=jnp.array(0.0), sigma_2_0=jnp.array(1.0)
    )
    result = binary_hgf(jnp.array(obs), params)
    all_nonneg = bool(np.all(np.array(result.surprise) >= -1e-10))
    if all_nonneg:
        print("  PASS binary surprise non-negative")
    else:
        print("  FAIL binary surprise has negative values")
        ok = False

    # Check 2: Binary HGF precisions are positive
    all_pos = bool(np.all(np.array(result.pi) > 0))
    if all_pos:
        print("  PASS binary precisions positive")
    else:
        print("  FAIL binary precisions have non-positive values")
        ok = False

    # Check 3: First trial surprise = -log(0.5) when starting from mu_2=0
    obs_one = jnp.array([1.0])
    result_one = binary_hgf(obs_one, params)
    expected = -np.log(0.5)
    actual = float(result_one.surprise[0])
    if abs(actual - expected) < 1e-3:
        print(f"  PASS first trial surprise = {actual:.4f} (expected {expected:.4f})")
    else:
        print(f"  FAIL first trial surprise = {actual:.4f} (expected {expected:.4f})")
        ok = False

    # Check 4: Continuous HGF tracks constant input
    obs_const = jnp.ones(100) * 2.0
    params_c = ContinuousHGFParams(
        omega_1=jnp.array(-3.0), omega_2=jnp.array(-3.0),
        kappa_1=jnp.array(1.0), kappa_2=jnp.array(1.0),
        theta=jnp.array(0.01), pi_u=jnp.array(100.0),
        mu_1_0=jnp.array(0.0), sigma_1_0=jnp.array(1.0),
        mu_2_0=jnp.array(0.0), sigma_2_0=jnp.array(1.0),
        mu_3_0=jnp.array(0.0), sigma_3_0=jnp.array(1.0),
    )
    result_c = continuous_hgf(obs_const, params_c)
    final_mu = float(result_c.mu[-1, 0])
    if abs(final_mu - 2.0) < 0.5:
        print(f"  PASS continuous HGF tracks constant input: mu_1={final_mu:.3f}")
    else:
        print(f"  FAIL continuous HGF doesn't track: mu_1={final_mu:.3f} (expected ~2.0)")
        ok = False

    # Check 5: JIT produces same results
    import jax
    result_jit = jax.jit(binary_hgf)(jnp.array(obs), params)
    jit_diff = float(np.max(np.abs(np.array(result.surprise) - np.array(result_jit.surprise))))
    if jit_diff < 1e-6:
        print(f"  PASS JIT consistency: max_diff={jit_diff:.8f}")
    else:
        print(f"  FAIL JIT inconsistency: max_diff={jit_diff:.8f}")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nALF HGF Validation")
    print("=" * 60)
    print()

    results = {}
    results["pyhgf_comparison"] = validate_binary_hgf_with_pyhgf()
    results["self_consistency"] = validate_hgf_self_consistency()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status}: {name}")

    n_fail = sum(1 for v in results.values() if v is False)
    sys.exit(n_fail)
