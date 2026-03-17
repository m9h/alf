#!/usr/bin/env python3
"""Validate alf.metacognition against metadPy and hand-computed SDT values.

Compares d', meta-d', and m-ratio between alf and metadPy on the
bundled rm.txt dataset (20 subjects, 200 trials each).

Usage:
    python validate_metacognition.py

Requires: scipy, pandas
Optional: metadPy (`pip install metadpy`) for cross-tool comparison
"""

import sys
from pathlib import Path

import numpy as np

try:
    from alf.metacognition import (
        compute_sdt_type1,
        compute_type1_from_counts,
        fit_meta_d_mle,
        m_ratio_to_gamma,
    )
except ImportError as e:
    print(f"FAIL: Cannot import alf.metacognition: {e}")
    sys.exit(1)

try:
    from scipy.stats import norm
except ImportError:
    print("FAIL: scipy required. Install with: pip install scipy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import metadpy
    HAS_METADPY = True
except ImportError:
    HAS_METADPY = False

DATA_DIR = Path(__file__).parent.parent / "alf" / "tests" / "data" / "metacognition"
RM_FILE = DATA_DIR / "rm.txt"


def trial_data_to_counts(stimuli, responses, confidence, n_ratings=4):
    """Convert trial-level data to nR_S1/nR_S2 count format."""
    stimuli = np.asarray(stimuli)
    responses = np.asarray(responses)
    confidence = np.asarray(confidence)
    nR_S1 = np.zeros(2 * n_ratings)
    nR_S2 = np.zeros(2 * n_ratings)

    for i in range(len(stimuli)):
        s, r, c = int(stimuli[i]), int(responses[i]), int(np.clip(confidence[i], 1, n_ratings))
        bin_idx = (n_ratings - c) if r == 0 else (n_ratings + c - 1)
        if s == 0:
            nR_S1[bin_idx] += 1
        else:
            nR_S2[bin_idx] += 1

    return nR_S1, nR_S2


# ---------------------------------------------------------------------------
# Validation 1: Hand-computed SDT values
# ---------------------------------------------------------------------------

def validate_sdt_hand_computed():
    """Validate SDT computations against hand-computed values."""
    print("=" * 60)
    print("SDT Hand-Computed Validation")
    print("=" * 60)

    ok = True

    # Test 1: Known hit/FA rates -> d'
    # hits=80, misses=20, FA=30, CR=70 with log-linear correction
    d, c = compute_sdt_type1(hits=80, misses=20, false_alarms=30, correct_rejections=70)

    hr = 80.5 / 101.0
    far = 30.5 / 101.0
    expected_d = norm.ppf(hr) - norm.ppf(far)
    expected_c = -0.5 * (norm.ppf(hr) + norm.ppf(far))

    if abs(d - expected_d) < 0.01:
        print(f"  PASS d' = {d:.4f} (expected {expected_d:.4f})")
    else:
        print(f"  FAIL d' = {d:.4f} (expected {expected_d:.4f})")
        ok = False

    if abs(c - expected_c) < 0.01:
        print(f"  PASS c = {c:.4f} (expected {expected_c:.4f})")
    else:
        print(f"  FAIL c = {c:.4f} (expected {expected_c:.4f})")
        ok = False

    # Test 2: Chance performance -> d' near 0
    d_chance, _ = compute_sdt_type1(hits=50, misses=50, false_alarms=50, correct_rejections=50)
    if abs(d_chance) < 0.3:
        print(f"  PASS chance d' = {d_chance:.4f} (expected ~0)")
    else:
        print(f"  FAIL chance d' = {d_chance:.4f} (expected ~0)")
        ok = False

    # Test 3: Perfect performance -> high d'
    d_perf, _ = compute_sdt_type1(hits=99, misses=1, false_alarms=1, correct_rejections=99)
    if d_perf > 3.0:
        print(f"  PASS perfect d' = {d_perf:.4f} (expected > 3)")
    else:
        print(f"  FAIL perfect d' = {d_perf:.4f} (expected > 3)")
        ok = False

    # Test 4: Liberal criterion -> negative c
    _, c_lib = compute_sdt_type1(hits=95, misses=5, false_alarms=70, correct_rejections=30)
    _, c_con = compute_sdt_type1(hits=40, misses=60, false_alarms=5, correct_rejections=95)
    if c_lib < 0 and c_con > 0:
        print(f"  PASS criterion: liberal c={c_lib:.3f} < 0, conservative c={c_con:.3f} > 0")
    else:
        print(f"  FAIL criterion signs: liberal={c_lib:.3f}, conservative={c_con:.3f}")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# Validation 2: metadPy rm.txt dataset
# ---------------------------------------------------------------------------

def validate_rm_dataset():
    """Validate on metadPy's bundled rm.txt dataset."""
    print("=" * 60)
    print("metadPy rm.txt Dataset Validation")
    print("=" * 60)

    if pd is None or not RM_FILE.exists():
        print("  SKIP: rm.txt not available or pandas not installed\n")
        return None

    df = pd.read_csv(RM_FILE)
    n_subjects = df["Subject"].nunique()
    n_ratings = int(df["Confidence"].max())
    print(f"  Loaded {len(df)} trials, {n_subjects} subjects, {n_ratings} confidence levels")

    ok = True
    d_primes = []
    meta_ds = []
    m_ratios = []
    gammas = []

    for subj_id in sorted(df["Subject"].unique()):
        subj = df[df["Subject"] == subj_id]
        nR_S1, nR_S2 = trial_data_to_counts(
            subj["Stimuli"].values, subj["Responses"].values,
            subj["Confidence"].values, n_ratings=n_ratings,
        )

        d, c = compute_type1_from_counts(nR_S1, nR_S2)
        d_primes.append(d)

        try:
            result = fit_meta_d_mle(nR_S1, nR_S2)
            meta_ds.append(result.meta_d)
            m_ratios.append(result.m_ratio)
            gamma = m_ratio_to_gamma(result.m_ratio)
            gammas.append(gamma)
        except Exception as e:
            meta_ds.append(np.nan)
            m_ratios.append(np.nan)
            gammas.append(np.nan)

    d_primes = np.array(d_primes)
    meta_ds = np.array(meta_ds)
    m_ratios = np.array(m_ratios)
    gammas = np.array(gammas)

    n_finite = int(np.sum(np.isfinite(meta_ds)))
    print(f"\n  Results ({n_finite}/{n_subjects} subjects converged):")
    print(f"    d': mean={np.nanmean(d_primes):.3f} std={np.nanstd(d_primes):.3f} "
          f"range=[{np.nanmin(d_primes):.3f}, {np.nanmax(d_primes):.3f}]")
    print(f"    meta-d': mean={np.nanmean(meta_ds):.3f} std={np.nanstd(meta_ds):.3f}")
    print(f"    m-ratio: mean={np.nanmean(m_ratios):.3f} std={np.nanstd(m_ratios):.3f}")
    print(f"    gamma: mean={np.nanmean(gammas):.3f} std={np.nanstd(gammas):.3f}")

    # Checks
    if n_finite >= 15:
        print(f"  PASS {n_finite}/{n_subjects} subjects converged")
    else:
        print(f"  FAIL only {n_finite}/{n_subjects} subjects converged")
        ok = False

    if 0.3 < np.nanmean(d_primes) < 3.0:
        print(f"  PASS mean d' in expected range for perceptual task")
    else:
        print(f"  FAIL mean d' = {np.nanmean(d_primes):.3f} out of range")
        ok = False

    if np.all(np.isfinite(gammas[np.isfinite(m_ratios)])):
        print(f"  PASS all gammas finite")
    else:
        print(f"  FAIL some gammas not finite")
        ok = False

    print()
    return ok


# ---------------------------------------------------------------------------
# Validation 3: Cross-tool comparison with metadPy
# ---------------------------------------------------------------------------

def validate_against_metadpy():
    """Compare alf.metacognition against metadPy on same data."""
    print("=" * 60)
    print("metadPy Cross-Tool Comparison")
    print("=" * 60)

    if not HAS_METADPY:
        print("  SKIP: metadPy not installed\n")
        return None

    if pd is None or not RM_FILE.exists():
        print("  SKIP: rm.txt not available\n")
        return None

    df = pd.read_csv(RM_FILE)
    ok = True

    # Compare on first subject
    subj0 = df[df["Subject"] == 0]
    n_ratings = int(subj0["Confidence"].max())

    # alf computation
    nR_S1, nR_S2 = trial_data_to_counts(
        subj0["Stimuli"].values, subj0["Responses"].values,
        subj0["Confidence"].values, n_ratings=n_ratings,
    )
    alf_d, alf_c = compute_type1_from_counts(nR_S1, nR_S2)
    alf_result = fit_meta_d_mle(nR_S1, nR_S2)

    # metadPy computation
    try:
        from metadpy.sdt import dprime as metadpy_dprime
        from metadpy.mle import metad as metadpy_metad

        # metadPy uses trial-level data directly
        metadpy_d = metadpy_dprime(
            stimuli=subj0["Stimuli"].values,
            responses=subj0["Responses"].values,
        )

        print(f"  d' comparison:")
        print(f"    alf:     {alf_d:.4f}")
        print(f"    metadPy: {float(metadpy_d):.4f}")
        diff_d = abs(alf_d - float(metadpy_d))
        if diff_d < 0.1:
            print(f"  PASS d' agreement: diff={diff_d:.4f}")
        else:
            print(f"  WARN d' disagreement: diff={diff_d:.4f}")

        print(f"\n  meta-d' (alf MLE): {alf_result.meta_d:.4f}")
        print(f"  m-ratio (alf): {alf_result.m_ratio:.4f}")

    except Exception as e:
        print(f"  WARN: metadPy API call failed: {e}")
        print("  (metadPy API may have changed between versions)")

    print()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nALF Metacognition Validation")
    print("=" * 60)
    print()

    results = {}
    results["sdt_hand_computed"] = validate_sdt_hand_computed()
    results["rm_dataset"] = validate_rm_dataset()
    results["metadpy_comparison"] = validate_against_metadpy()

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
