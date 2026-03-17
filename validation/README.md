# ALF Validation: Cross-Tool Numerical Comparison

## Purpose

ALF reimplements four computational psychiatry tools in JAX:

| ALF Module | Replaces | Original Language | Key Paper |
|------------|----------|-------------------|-----------|
| `alf.hgf` | pyhgf / TAPAS HGF | Python (JAX) / MATLAB | Mathys et al. (2011, 2014) |
| `alf.ddm` | HDDM | Python (Cython/PyMC) | Wiecki, Sofer & Frank (2013) |
| `alf.metacognition` | HMeta-d / metadPy | MATLAB (JAGS) / Python (NumPyro) | Fleming (2017) |
| `alf.normative` | PCNtoolkit | Python (PyMC) | Marquand et al. (2016) |

The unit tests in `alf/tests/` verify internal correctness (shapes, signs,
differentiability, JIT compatibility). The oracle tests verify against real
datasets and hand-computed values. But neither of these proves that ALF
produces the **same numerical output** as the original tools on identical input.

This directory contains standalone scripts that do exactly that: run the same
data through both the original tool and ALF, then compare numbers.

## Requirements

Run on a machine with:
- Python 3.10+
- ARM macOS or Linux (x86_64 wheels available for all deps)
- ~2GB RAM

```bash
# Create isolated environment
uv venv validation-env
source validation-env/bin/activate

# Install ALF
uv pip install -e /path/to/alf

# Install reference tools
uv pip install pyhgf          # HGF oracle
uv pip install metadpy         # Metacognition oracle (includes NumPyro)
uv pip install pandas scipy    # Data loading
# HDDM is not pip-installable on modern Python — we compare against
# published parameter estimates instead
```

## Scripts

### `validate_hgf.py`
Runs pyhgf and alf.hgf on identical input, compares:
- Per-trial posterior mean (mu) at each level
- Per-trial posterior precision (pi) at each level
- Per-trial surprise (negative log-likelihood)
- Total surprise (sum)

Expected tolerance: atol=1e-2 (implementations may differ in edge-case
numerical handling, e.g., epsilon clipping values).

### `validate_ddm.py`
Runs alf.ddm on HDDM's bundled dataset, compares:
- NLL at published HDDM parameter estimates
- MLE parameter recovery against HDDM literature values
- Posterior predictive statistics vs observed data

No direct HDDM comparison (not installable on modern Python). Instead
validates against published ranges from Wiecki et al. (2013).

### `validate_metacognition.py`
Runs metadPy and alf.metacognition on identical input, compares:
- Type 1 d' (should match exactly — both use the same SDT formula)
- meta-d' MLE (should be close — different optimizers may find slightly
  different optima, but within atol=0.5)
- m-ratio derived from each

### `validate_all.py`
Runs all three validations and prints a summary report.

## Interpreting Results

Each comparison reports:
- **PASS**: Values agree within tolerance
- **WARN**: Values differ more than expected but are in the right ballpark
- **FAIL**: Significant numerical disagreement — likely a bug

Common sources of small disagreement:
- Different epsilon clipping (1e-16 vs 1e-10)
- Different softmax/sigmoid overflow handling
- Different optimizer convergence for MLE fits
- pyhgf uses a mutable-dictionary design vs ALF's functional jax.lax.scan

## Data Sources

| Dataset | Source | Trials | Subjects | Used By |
|---------|--------|--------|----------|---------|
| pyhgf `load_data("binary")` | Iglesias et al. (2013) | ~320 | 1 | HGF validation |
| HDDM `simple_difficulty.csv` | Wiecki et al. (2013) | 1000 | 1 | DDM validation |
| HDDM `simple_subjs_difficulty.csv` | Wiecki et al. (2013) | 1200 | ~12 | DDM hierarchical |
| metadPy `rm.txt` | Bundled dataset | 4000 | 20 | Metacognition validation |

## References

- Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*, 5, 39.
- Mathys, C. D., Lomakina, E. I., Daunizeau, J., et al. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. *Frontiers in Human Neuroscience*, 8, 825.
- Weber, L. A., Imbach, L. L., Legrand, N., et al. (2024). The generalized Hierarchical Gaussian Filter. *arXiv:2305.10937*.
- Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python. *Frontiers in Neuroinformatics*, 7, 14.
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology*, 53(4), 222-230.
- Fleming, S. M. (2017). HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings. *Neuroscience of Consciousness*, 3(1), nix007.
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. *Consciousness and Cognition*, 21(1), 422-430.
- Marquand, A. F., Rezek, I., Buitelaar, J., & Beckmann, C. F. (2016). Understanding heterogeneity in clinical cohorts using normative models. *Biological Psychiatry*, 80(7), 552-561.
- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.
