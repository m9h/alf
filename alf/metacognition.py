"""Metacognitive modeling: meta-d' and precision calibration.

Implements the meta-d' framework (Maniscalco & Lau, 2012; Fleming, 2017) for
measuring metacognitive efficiency, and bridges it to active inference by
mapping the m-ratio (meta-d'/d') to agent precision gamma.

Signal Detection Theory (SDT) provides the foundation:
    - Type 1 performance: d' measures perceptual sensitivity.
    - Type 2 performance: meta-d' measures metacognitive sensitivity,
      defined as the d' that would produce the observed confidence ratings
      under optimal metacognition.
    - m-ratio = meta-d'/d': metacognitive efficiency. An m-ratio of 1 means
      perfectly calibrated confidence; < 1 means underconfident/miscalibrated.

Active Inference connection: m-ratio IS precision calibration. An agent with
m-ratio < 1 systematically under- or over-estimates its belief precision gamma.
This module provides both standalone metacognitive analysis and a bridge to
parameterize alf agents' precision from empirical metacognitive data.

Both NumPy (analytical) and NumPyro (hierarchical Bayesian) implementations
are provided. The NumPyro model follows Fleming (2017) and metadPy.

References:
    Maniscalco & Lau (2012). A signal detection theoretic approach for
        estimating metacognitive sensitivity from confidence ratings.
        Consciousness and Cognition.
    Fleming (2017). HMeta-d: hierarchical Bayesian estimation of
        metacognitive efficiency from confidence ratings. Neuroscience
        of Consciousness.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SDTCounts(NamedTuple):
    """Response counts for metacognitive SDT analysis.

    Attributes:
        nR_S1: Response counts for stimulus=1 trials, shape (2*nRatings,).
            Ordered from highest confidence "S1" to highest confidence "S2".
        nR_S2: Response counts for stimulus=2 trials, shape (2*nRatings,).
            Same ordering as nR_S1.
    """
    nR_S1: np.ndarray
    nR_S2: np.ndarray


class MetaDResult(NamedTuple):
    """Result of meta-d' estimation.

    Attributes:
        d_prime: Type 1 d' (perceptual sensitivity).
        meta_d: Meta-d' (metacognitive sensitivity in d' units).
        m_ratio: Metacognitive efficiency (meta-d'/d').
        c: Type 1 criterion.
    """
    d_prime: float
    meta_d: float
    m_ratio: float
    c: float


# ---------------------------------------------------------------------------
# Type 1 SDT
# ---------------------------------------------------------------------------

def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (Phi function)."""
    from scipy.stats import norm
    return norm.cdf(x)


def _phi_inv(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (probit function)."""
    from scipy.stats import norm
    return norm.ppf(p)


def compute_sdt_type1(
    hits: int,
    misses: int,
    false_alarms: int,
    correct_rejections: int,
    correction: float = 0.5,
) -> tuple[float, float]:
    """Compute Type 1 d' and criterion c from hit/FA rates.

    Uses the standard SDT formulas:
        d' = z(hit_rate) - z(false_alarm_rate)
        c  = -0.5 * (z(hit_rate) + z(false_alarm_rate))

    Applies the log-linear correction (Hautus, 1995) to avoid infinite d'.

    Args:
        hits: Number of hits (stimulus present, response "yes").
        misses: Number of misses.
        false_alarms: Number of false alarms.
        correct_rejections: Number of correct rejections.
        correction: Log-linear correction factor. Default 0.5.

    Returns:
        Tuple of (d_prime, criterion).
    """
    # Log-linear correction (Hautus, 1995)
    n_s2 = hits + misses
    n_s1 = false_alarms + correct_rejections

    hit_rate = (hits + correction) / (n_s2 + 2 * correction)
    fa_rate = (false_alarms + correction) / (n_s1 + 2 * correction)

    d_prime = float(_phi_inv(hit_rate) - _phi_inv(fa_rate))
    c = float(-0.5 * (_phi_inv(hit_rate) + _phi_inv(fa_rate)))

    return d_prime, c


def compute_type1_from_counts(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
) -> tuple[float, float]:
    """Compute Type 1 d' and c from confidence rating counts.

    Collapses confidence ratings into binary response counts.

    Args:
        nR_S1: Counts for S1 trials, shape (2*nRatings,).
        nR_S2: Counts for S2 trials, shape (2*nRatings,).

    Returns:
        Tuple of (d_prime, criterion).
    """
    n_ratings = len(nR_S1) // 2

    # S1 trials: first half = "respond S1", second half = "respond S2"
    cr = int(np.sum(nR_S1[:n_ratings]))  # correct rejections
    fa = int(np.sum(nR_S2[:n_ratings]))  # false alarms
    miss = int(np.sum(nR_S1[n_ratings:]))  # misses (said S2 when S1)
    hit = int(np.sum(nR_S2[n_ratings:]))  # hits (said S2 when S2)

    return compute_sdt_type1(hit, miss, fa, cr)


# ---------------------------------------------------------------------------
# Meta-d' estimation (maximum likelihood, single subject)
# ---------------------------------------------------------------------------

def _sdt_expected_counts(
    d: float,
    c: float,
    criteria: np.ndarray,
    n_s1: int,
    n_s2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected SDT response counts given d' and criteria.

    Args:
        d: Sensitivity (d' or meta-d').
        c: Type 1 criterion.
        criteria: Confidence criteria positions, shape (2*nRatings - 1,).
        n_s1: Total S1 trials.
        n_s2: Total S2 trials.

    Returns:
        Tuple of (expected_nR_S1, expected_nR_S2).
    """
    n_criteria = len(criteria)
    n_ratings = (n_criteria + 1) // 2

    # Cumulative probabilities at each criterion
    # For S1 (noise) distribution: N(0, 1) shifted by -d/2 relative to criterion
    # For S2 (signal) distribution: N(0, 1) shifted by +d/2
    cum_S1 = _phi(criteria + d / 2.0)  # P(response <= criterion | S1)
    cum_S2 = _phi(criteria - d / 2.0)  # P(response <= criterion | S2)

    # Bin probabilities (between consecutive criteria)
    p_S1 = np.diff(np.concatenate([[0.0], cum_S1, [1.0]]))
    p_S2 = np.diff(np.concatenate([[0.0], cum_S2, [1.0]]))

    p_S1 = np.clip(p_S1, 1e-10, None)
    p_S2 = np.clip(p_S2, 1e-10, None)

    return p_S1 * n_s1, p_S2 * n_s2


def fit_meta_d_mle(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
) -> MetaDResult:
    """Estimate meta-d' via maximum likelihood.

    Fits the meta-d' model by finding the d' value that maximizes the
    likelihood of the observed confidence ratings, given the observed
    Type 1 criterion.

    Args:
        nR_S1: Response counts for S1 trials, shape (2*nRatings,).
        nR_S2: Response counts for S2 trials, shape (2*nRatings,).

    Returns:
        MetaDResult with d', meta-d', m-ratio, and criterion.
    """
    from scipy.optimize import minimize_scalar

    nR_S1 = np.asarray(nR_S1, dtype=np.float64)
    nR_S2 = np.asarray(nR_S2, dtype=np.float64)
    n_ratings = len(nR_S1) // 2

    # Type 1 performance
    d_prime, c = compute_type1_from_counts(nR_S1, nR_S2)

    # Type 1 criterion in SDT coordinates
    n_s1 = float(np.sum(nR_S1))
    n_s2 = float(np.sum(nR_S2))

    # Observed confidence criteria from cumulative rates
    cum_S1 = np.cumsum(nR_S1) / n_s1
    cum_S2 = np.cumsum(nR_S2) / n_s2

    # z-transform to get criteria positions
    # Exclude last bin (cumulative = 1.0)
    criteria_S1 = _phi_inv(np.clip(cum_S1[:-1], 1e-5, 1.0 - 1e-5))
    criteria_S2 = _phi_inv(np.clip(cum_S2[:-1], 1e-5, 1.0 - 1e-5))

    # Use average criteria (simplified approach)
    criteria = 0.5 * (criteria_S1 + criteria_S2)

    # Fit meta-d': find the d' that best explains the confidence data
    observed = np.concatenate([nR_S1, nR_S2])
    observed = np.clip(observed, 0.5, None)  # Correction for empty bins

    def neg_log_likelihood(meta_d):
        expected_S1, expected_S2 = _sdt_expected_counts(
            meta_d, c, criteria, int(n_s1), int(n_s2)
        )
        expected = np.concatenate([expected_S1, expected_S2])
        expected = np.clip(expected, 1e-10, None)
        # Multinomial log-likelihood (proportional)
        return -float(np.sum(observed * np.log(expected / expected.sum())))

    result = minimize_scalar(
        neg_log_likelihood, bounds=(-5.0, 5.0), method='bounded'
    )
    meta_d = result.x

    # Compute m-ratio
    eps = 1e-16
    m_ratio = meta_d / max(abs(d_prime), eps)

    return MetaDResult(
        d_prime=d_prime,
        meta_d=meta_d,
        m_ratio=m_ratio,
        c=c,
    )


# ---------------------------------------------------------------------------
# Hierarchical Bayesian meta-d' (NumPyro)
# ---------------------------------------------------------------------------

def fit_meta_d_bayesian(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Fit hierarchical Bayesian meta-d' model using NumPyro NUTS.

    Implements Fleming (2017) single-subject model. Requires numpyro
    and jax as dependencies.

    Args:
        nR_S1: Response counts for S1 trials, shape (2*nRatings,).
        nR_S2: Response counts for S2 trials, shape (2*nRatings,).
        num_warmup: NUTS warmup steps.
        num_samples: Posterior samples.
        seed: Random seed.

    Returns:
        Dict with 'meta_d_samples', 'd_prime', 'm_ratio_samples',
        and 'summary' keys.

    Raises:
        ImportError: If numpyro is not installed.
    """
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError:
        raise ImportError(
            "numpyro is required for Bayesian meta-d' estimation. "
            "Install with: pip install numpyro"
        )

    nR_S1 = np.asarray(nR_S1, dtype=np.float64)
    nR_S2 = np.asarray(nR_S2, dtype=np.float64)
    n_ratings = len(nR_S1) // 2

    # Compute Type 1 d' and c
    d_prime, c = compute_type1_from_counts(nR_S1, nR_S2)

    # Observed data
    counts_S1 = jnp.array(nR_S1)
    counts_S2 = jnp.array(nR_S2)
    n_total_S1 = jnp.sum(counts_S1)
    n_total_S2 = jnp.sum(counts_S2)

    def model():
        # Prior on meta-d'
        meta_d = numpyro.sample("meta_d", dist.Normal(d_prime, 2.0))

        # Confidence criteria (ordered)
        c_raw = numpyro.sample(
            "c_raw",
            dist.Normal(0.0, 1.5).expand([2 * n_ratings - 1]),
        )
        # Sort to ensure ordering
        criteria = jnp.sort(c_raw)

        # Expected cumulative probabilities under meta-d'
        cum_S1 = jax.scipy.stats.norm.cdf(criteria + meta_d / 2.0)
        cum_S2 = jax.scipy.stats.norm.cdf(criteria - meta_d / 2.0)

        # Bin probabilities
        p_S1 = jnp.diff(jnp.concatenate([jnp.zeros(1), cum_S1, jnp.ones(1)]))
        p_S2 = jnp.diff(jnp.concatenate([jnp.zeros(1), cum_S2, jnp.ones(1)]))

        # Clip for numerical stability
        p_S1 = jnp.clip(p_S1, 1e-8, 1.0)
        p_S2 = jnp.clip(p_S2, 1e-8, 1.0)
        p_S1 = p_S1 / p_S1.sum()
        p_S2 = p_S2 / p_S2.sum()

        # Likelihood
        numpyro.sample(
            "obs_S1",
            dist.Multinomial(total_count=n_total_S1, probs=p_S1),
            obs=counts_S1,
        )
        numpyro.sample(
            "obs_S2",
            dist.Multinomial(total_count=n_total_S2, probs=p_S2),
            obs=counts_S2,
        )

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(seed))

    samples = mcmc.get_samples()
    meta_d_samples = np.array(samples["meta_d"])

    eps = 1e-16
    m_ratio_samples = meta_d_samples / max(abs(d_prime), eps)

    return {
        "meta_d_samples": meta_d_samples,
        "d_prime": d_prime,
        "c": c,
        "m_ratio_samples": m_ratio_samples,
        "meta_d_mean": float(np.mean(meta_d_samples)),
        "m_ratio_mean": float(np.mean(m_ratio_samples)),
        "meta_d_hdi_95": (
            float(np.percentile(meta_d_samples, 2.5)),
            float(np.percentile(meta_d_samples, 97.5)),
        ),
    }


# ---------------------------------------------------------------------------
# Precision bridge: m-ratio -> agent gamma
# ---------------------------------------------------------------------------

def m_ratio_to_gamma(
    m_ratio: float,
    base_gamma: float = 4.0,
    min_gamma: float = 0.1,
    max_gamma: float = 16.0,
) -> float:
    """Convert metacognitive efficiency to agent precision.

    Maps m-ratio to policy precision gamma for active inference agents.
    An m-ratio of 1.0 (perfect metacognition) maps to base_gamma.
    Lower m-ratio (worse metacognition) reduces gamma (less precise policies).
    Higher m-ratio (overconfident) increases gamma.

    Args:
        m_ratio: Metacognitive efficiency (meta-d'/d').
        base_gamma: Precision for perfect metacognition. Default 4.0.
        min_gamma: Minimum precision. Default 0.1.
        max_gamma: Maximum precision. Default 16.0.

    Returns:
        Agent precision gamma.
    """
    gamma = base_gamma * m_ratio
    return float(np.clip(gamma, min_gamma, max_gamma))


def update_gamma_from_confidence(
    gamma: float,
    predicted_confidence: float,
    actual_accuracy: float,
    learning_rate: float = 0.1,
    min_gamma: float = 0.1,
    max_gamma: float = 16.0,
) -> float:
    """Online precision update from confidence-accuracy calibration.

    Adjusts gamma based on how well the agent's confidence (derived from
    policy precision) predicts its actual accuracy. This implements online
    metacognitive monitoring — the agent learns its own precision.

    Args:
        gamma: Current policy precision.
        predicted_confidence: Agent's predicted probability of being correct
            (e.g., max policy probability).
        actual_accuracy: Whether the agent was actually correct (0 or 1).
        learning_rate: Update rate. Default 0.1.
        min_gamma: Minimum precision.
        max_gamma: Maximum precision.

    Returns:
        Updated gamma.
    """
    # Calibration error: if confidence > accuracy, reduce gamma
    calibration_error = predicted_confidence - actual_accuracy

    # Negative calibration error (underconfident) -> increase gamma
    # Positive calibration error (overconfident) -> decrease gamma
    gamma_delta = -learning_rate * calibration_error * gamma
    new_gamma = gamma + gamma_delta

    return float(np.clip(new_gamma, min_gamma, max_gamma))
