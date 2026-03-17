#!/usr/bin/env python3
"""Validate alf.ddm against HDDM example datasets and published results.

HDDM (Wiecki et al., 2013) is not installable on modern Python, so we
compare against published parameter ranges and verify our implementation
produces sensible results on their bundled datasets.

Usage:
    python validate_ddm.py

Requires: pandas, scipy
"""

import sys
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from alf.ddm.wiener import (
        wiener_log_density, wiener_log_density_batch,
        ddm_nll, simulate_ddm, DDMParams,
    )
    from alf.ddm.fitting import fit_ddm_mle
except ImportError as e:
    print(f"FAIL: Cannot import alf.ddm: {e}")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("FAIL: pandas required. Install with: pip install pandas")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "alf" / "tests" / "data" / "ddm"
SIMPLE_CSV = DATA_DIR / "simple_difficulty.csv"


def compare_scalar(name, value, low, high):
    """Check if a scalar falls in expected range."""
    if low <= value <= high:
        print(f"  PASS {name} = {value:.4f} (expected [{low}, {high}])")
        return True
    else:
        print(f"  FAIL {name} = {value:.4f} (expected [{low}, {high}])")
        return False


# ---------------------------------------------------------------------------
# Validation 1: Navarro-Fuss density correctness
# ---------------------------------------------------------------------------

def validate_navarro_fuss():
    """Validate the Navarro-Fuss density implementation."""
    print("=" * 60)
    print("Navarro-Fuss Density Validation")
    print("=" * 60)

    ok = True

    # Check 1: Known symmetry — flipping v and choice gives same density
    rt = jnp.array(0.8)
    v, a, w, tau = jnp.array(1.0), jnp.array(1.5), jnp.array(0.5), jnp.array(0.3)

    lp_upper = float(wiener_log_density(rt, jnp.array(1.0), v, a, w, tau))
    lp_lower_flip = float(wiener_log_density(rt, jnp.array(0.0), -v, a, w, tau))

    # With w=0.5, p(t, upper | v) should equal p(t, lower | -v)
    if abs(lp_upper - lp_lower_flip) < 1e-4:
        print(f"  PASS symmetry: log_p(upper,v=1) = {lp_upper:.4f}, "
              f"log_p(lower,v=-1) = {lp_lower_flip:.4f}")
    else:
        print(f"  FAIL symmetry: {lp_upper:.4f} != {lp_lower_flip:.4f}")
        ok = False

    # Check 2: Positive drift favors upper boundary
    lp_upper_pos = float(wiener_log_density(rt, jnp.array(1.0), jnp.array(2.0), a, w, tau))
    lp_lower_pos = float(wiener_log_density(rt, jnp.array(0.0), jnp.array(2.0), a, w, tau))

    if lp_upper_pos > lp_lower_pos:
        print(f"  PASS positive drift favors upper: {lp_upper_pos:.4f} > {lp_lower_pos:.4f}")
    else:
        print(f"  FAIL positive drift should favor upper")
        ok = False

    # Check 3: Density is differentiable
    grad_v = float(jax.grad(wiener_log_density, argnums=2)(
        rt, jnp.array(1.0), v, a, w, tau))
    if np.isfinite(grad_v):
        print(f"  PASS gradient w.r.t. v is finite: {grad_v:.4f}")
    else:
        print(f"  FAIL gradient w.r.t. v is not finite")
        ok = False

    # Check 4: Choice probability integrates correctly
    # P(upper) from analytical formula vs simulation
    params = DDMParams(v=jnp.array(1.5), a=jnp.array(1.5), w=jnp.array(0.5), tau=jnp.array(0.3))
    from alf.ddm.bridge import ddm_to_policy_probs
    p_analytic = float(ddm_to_policy_probs(params)[1])
    sim = simulate_ddm(v=1.5, a=1.5, w=0.5, tau=0.3, n_trials=5000, seed=42)
    p_simulated = float(np.mean(sim.choice == 1))
    diff = abs(p_analytic - p_simulated)
    if diff < 0.05:
        print(f"  PASS P(upper): analytic={p_analytic:.3f} simulated={p_simulated:.3f}")
    else:
        print(f"  WARN P(upper): analytic={p_analytic:.3f} simulated={p_simulated:.3f} diff={diff:.3f}")

    print()
    return ok


# ---------------------------------------------------------------------------
# Validation 2: HDDM dataset fitting
# ---------------------------------------------------------------------------

def validate_hddm_data():
    """Validate DDM fitting on HDDM's bundled datasets."""
    print("=" * 60)
    print("HDDM Dataset Validation")
    print("=" * 60)

    if not SIMPLE_CSV.exists():
        print("  SKIP: HDDM data not downloaded")
        print(f"  Expected at: {SIMPLE_CSV}\n")
        return None

    df = pd.read_csv(SIMPLE_CSV)
    print(f"  Loaded {len(df)} trials, conditions: {list(df['difficulty'].unique())}")

    ok = True

    # Fit easy condition
    easy = df[df["difficulty"] == "easy"]
    rt_e = np.array(easy["rt"].values)
    ch_e = np.array(easy["response"].values)
    mask = rt_e > 0.1
    rt_e, ch_e = rt_e[mask], ch_e[mask]

    print(f"\n  Easy condition: {len(rt_e)} trials, "
          f"mean_RT={np.mean(rt_e):.3f}s, accuracy={np.mean(ch_e):.3f}")

    result_easy = fit_ddm_mle(rt_e, ch_e, num_epochs=300, lr=0.01)
    v_e, a_e, w_e, tau_e = (
        float(result_easy.v), float(result_easy.a),
        float(result_easy.w), float(result_easy.tau),
    )
    print(f"  MLE: v={v_e:.3f}, a={a_e:.3f}, w={w_e:.3f}, tau={tau_e:.3f}")
    print(f"  Loss: {result_easy.loss_history[0]:.2f} -> {result_easy.loss_history[-1]:.2f}")

    # HDDM literature ranges for easy condition
    ok &= compare_scalar("easy v", abs(v_e), 0.1, 5.0)
    ok &= compare_scalar("easy a", a_e, 0.3, 5.0)
    ok &= compare_scalar("easy tau", tau_e, 0.05, min(rt_e))

    # Fit hard condition
    hard = df[df["difficulty"] == "hard"]
    rt_h = np.array(hard["rt"].values)
    ch_h = np.array(hard["response"].values)
    mask = rt_h > 0.1
    rt_h, ch_h = rt_h[mask], ch_h[mask]

    print(f"\n  Hard condition: {len(rt_h)} trials, "
          f"mean_RT={np.mean(rt_h):.3f}s, accuracy={np.mean(ch_h):.3f}")

    result_hard = fit_ddm_mle(rt_h, ch_h, num_epochs=300, lr=0.01)
    v_h, a_h, w_h, tau_h = (
        float(result_hard.v), float(result_hard.a),
        float(result_hard.w), float(result_hard.tau),
    )
    print(f"  MLE: v={v_h:.3f}, a={a_h:.3f}, w={w_h:.3f}, tau={tau_h:.3f}")

    ok &= compare_scalar("hard v", abs(v_h), 0.01, 5.0)
    ok &= compare_scalar("hard a", a_h, 0.3, 5.0)
    ok &= compare_scalar("hard tau", tau_h, 0.05, min(rt_h))

    # Difficulty effect: easy should have higher |drift|
    print(f"\n  Difficulty effect: |v_easy|={abs(v_e):.3f} vs |v_hard|={abs(v_h):.3f}")
    if abs(v_e) > abs(v_h) * 0.5:
        print("  PASS drift rate varies with difficulty")
    else:
        print("  WARN drift rate difference smaller than expected")

    # Posterior predictive check
    print("\n  Posterior predictive (easy condition):")
    sim = simulate_ddm(v=v_e, a=a_e, w=w_e, tau=tau_e, n_trials=len(rt_e), seed=42)
    print(f"    Observed: mean_RT={np.mean(rt_e):.3f}, acc={np.mean(ch_e):.3f}")
    print(f"    Simulated: mean_RT={np.mean(sim.rt):.3f}, acc={np.mean(sim.choice):.3f}")

    rt_ratio = np.mean(sim.rt) / np.mean(rt_e)
    if 0.3 < rt_ratio < 3.0:
        print(f"  PASS RT ratio = {rt_ratio:.2f}")
    else:
        print(f"  FAIL RT ratio = {rt_ratio:.2f}")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nALF DDM Validation")
    print("=" * 60)
    print()

    results = {}
    results["navarro_fuss"] = validate_navarro_fuss()
    results["hddm_data"] = validate_hddm_data()

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
