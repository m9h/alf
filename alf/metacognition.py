"""Metacognitive modeling: meta-d', precision calibration, and online monitoring.

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

Phase 2-3 extensions:
    - EFEMonitor: Online tracking of EFE prediction accuracy and calibration.
    - MetacognitiveAgent: Wraps AnalyticAgent with real-time self-monitoring,
      adjusting gamma based on confidence-accuracy calibration.
    - PopulationMetacognition: Aggregate metacognitive statistics across a
      population of MetacognitiveAgents (for multi-agent simulations).

Both NumPy (analytical) and NumPyro (hierarchical Bayesian) implementations
are provided. The NumPyro model follows Fleming (2017) and metadPy.

References:
    Maniscalco & Lau (2012). A signal detection theoretic approach for
        estimating metacognitive sensitivity from confidence ratings.
        Consciousness and Cognition.
    Fleming (2017). HMeta-d: hierarchical Bayesian estimation of
        metacognitive efficiency from confidence ratings. Neuroscience
        of Consciousness.
    Hesp et al. (2021). Deeply felt affect: The emergence of valence in
        deep active inference. Neural Computation.
    Smith et al. (2022). A Step-by-Step Tutorial on Active Inference.
        Journal of Mathematical Psychology.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from alf.agent import AnalyticAgent
from alf.generative_model import GenerativeModel


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
    (n_criteria + 1) // 2

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
    len(nR_S1) // 2

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


# ---------------------------------------------------------------------------
# EFE Monitor: online tracking of prediction accuracy and calibration
# ---------------------------------------------------------------------------

class EFERecord(NamedTuple):
    """A single record of an EFE prediction paired with its outcome.

    Attributes:
        predicted_efe: The EFE value predicted by the agent for the chosen
            policy (lower = agent expected a better outcome).
        confidence: Agent's confidence in its chosen policy (max policy
            probability, range [0, 1]).
        outcome_valence: Actual outcome quality (positive = good, negative
            = bad). For binary accuracy, use 1.0 (correct) or 0.0 (incorrect).
    """
    predicted_efe: float
    confidence: float
    outcome_valence: float


class EFEMonitor:
    """Online monitor for EFE prediction accuracy and calibration.

    Tracks a running history of the agent's EFE predictions alongside
    actual outcomes, computing calibration error and an online estimate
    of metacognitive efficiency (m-ratio).

    The monitor uses exponential moving averages (EMA) for smooth online
    estimation and maintains a sliding window for recent-trial statistics.

    Args:
        decay: Exponential moving average decay factor. Values closer to
            1.0 give more weight to recent trials. Default 0.95.
        window_size: Number of recent trials to keep for windowed
            statistics. Default 50.
    """

    def __init__(
        self,
        decay: float = 0.95,
        window_size: int = 50,
    ):
        self.decay = decay
        self.window_size = window_size

        # Full history
        self.records: list[EFERecord] = []

        # Exponential moving averages
        self._ema_confidence: float = 0.5
        self._ema_accuracy: float = 0.5
        self._ema_squared_error: float = 0.0
        self._n_records: int = 0

    def record(
        self,
        predicted_efe: float,
        confidence: float,
        outcome_valence: float,
    ) -> None:
        """Record one trial of EFE prediction versus actual outcome.

        Args:
            predicted_efe: EFE value for the chosen policy.
            confidence: Agent's confidence (max policy probability).
            outcome_valence: Actual outcome (1.0 = correct/good,
                0.0 = incorrect/bad, or continuous).
        """
        rec = EFERecord(
            predicted_efe=float(predicted_efe),
            confidence=float(confidence),
            outcome_valence=float(outcome_valence),
        )
        self.records.append(rec)
        self._n_records += 1

        # Update exponential moving averages
        alpha = 1.0 - self.decay
        self._ema_confidence = (
            self.decay * self._ema_confidence + alpha * rec.confidence
        )
        accuracy = 1.0 if rec.outcome_valence > 0.0 else 0.0
        self._ema_accuracy = (
            self.decay * self._ema_accuracy + alpha * accuracy
        )
        sq_err = (rec.confidence - accuracy) ** 2
        self._ema_squared_error = (
            self.decay * self._ema_squared_error + alpha * sq_err
        )

    def get_calibration(self) -> float:
        """Compute calibration error from recent trials.

        Calibration error is the root-mean-square difference between
        predicted confidence and actual accuracy over the recent window.
        A perfectly calibrated agent has calibration error of 0.

        Returns:
            Calibration error (RMSE of confidence - accuracy).
            Returns 0.0 if no records exist.
        """
        if not self.records:
            return 0.0

        window = self.records[-self.window_size:]
        eps = 1e-16

        confidences = np.array([r.confidence for r in window])
        accuracies = np.array([
            1.0 if r.outcome_valence > 0.0 else 0.0 for r in window
        ])

        mse = np.mean((confidences - accuracies) ** 2)
        return float(np.sqrt(mse + eps))

    def get_online_m_ratio(self) -> float:
        """Estimate metacognitive efficiency from online statistics.

        Computes an online approximation of the m-ratio using the
        relationship between confidence discrimination and accuracy.
        Uses the EMA statistics for a smooth estimate.

        The approximation maps the ratio of confidence-accuracy
        covariance to confidence variance, analogous to how m-ratio
        measures how well Type 2 (confidence) tracks Type 1 (accuracy).

        Returns:
            Estimated m-ratio (1.0 = perfect metacognition).
            Returns 1.0 if insufficient data.
        """
        if self._n_records < 2:
            return 1.0

        window = self.records[-self.window_size:]
        eps = 1e-16

        confidences = np.array([r.confidence for r in window])
        accuracies = np.array([
            1.0 if r.outcome_valence > 0.0 else 0.0 for r in window
        ])

        # Variance of confidence
        conf_var = np.var(confidences)
        if conf_var < eps:
            return 1.0

        # Covariance between confidence and accuracy
        cov = np.mean(
            (confidences - confidences.mean())
            * (accuracies - accuracies.mean())
        )

        # m-ratio approximation: how well confidence tracks accuracy
        # Positive covariance = good metacognition (m-ratio ~ 1)
        # Zero/negative covariance = poor metacognition (m-ratio ~ 0)
        m_ratio = float(np.clip(cov / conf_var, 0.0, 2.0))

        return m_ratio

    def get_summary(self) -> dict[str, float]:
        """Return a summary dictionary of monitor statistics.

        Returns:
            Dict with keys: n_records, ema_confidence, ema_accuracy,
            calibration_error, online_m_ratio.
        """
        return {
            "n_records": self._n_records,
            "ema_confidence": self._ema_confidence,
            "ema_accuracy": self._ema_accuracy,
            "calibration_error": self.get_calibration(),
            "online_m_ratio": self.get_online_m_ratio(),
        }


# ---------------------------------------------------------------------------
# MetacognitiveAgent: wraps AnalyticAgent with self-monitoring
# ---------------------------------------------------------------------------

class MetacognitiveAgent:
    """Active Inference agent with metacognitive self-monitoring.

    Wraps an AnalyticAgent (has-a composition) and augments each decision
    step with metacognitive monitoring. On each step, the agent records
    its confidence (max policy probability) and on receiving outcome
    feedback, updates an EFEMonitor and periodically adjusts its policy
    precision (gamma) based on calibration.

    This is a drop-in replacement for AnalyticAgent: the step() method
    returns the same (action, info) tuple.

    Args:
        inner_agent: The AnalyticAgent to wrap. Can also accept a
            GenerativeModel (creates an AnalyticAgent internally) or
            a pymdp Agent (converted to ALF GM, then wrapped).
        gamma: Initial policy precision. Only used if inner_agent is a
            GenerativeModel or pymdp Agent. Default 4.0.
        monitor_decay: EMA decay for the EFEMonitor. Default 0.95.
        monitor_window: Window size for the EFEMonitor. Default 50.
        gamma_learning_rate: Learning rate for gamma adjustment based
            on metacognitive feedback. Default 0.1.
        gamma_update_interval: How often (in trials) to adjust gamma
            from metacognitive statistics. Default 5.
        min_gamma: Minimum allowed gamma. Default 0.1.
        max_gamma: Maximum allowed gamma. Default 16.0.
        seed: Random seed (only used when inner_agent is a
            GenerativeModel). Default 42.
    """

    def __init__(
        self,
        inner_agent,
        gamma: float = 4.0,
        monitor_decay: float = 0.95,
        monitor_window: int = 50,
        gamma_learning_rate: float = 0.1,
        gamma_update_interval: int = 5,
        min_gamma: float = 0.1,
        max_gamma: float = 16.0,
        seed: int = 42,
    ):
        if isinstance(inner_agent, GenerativeModel):
            self.agent = AnalyticAgent(inner_agent, gamma=gamma, seed=seed)
        elif isinstance(inner_agent, AnalyticAgent):
            self.agent = inner_agent
        else:
            # Assume pymdp Agent — convert to ALF
            from alf.compat import pymdp_to_alf
            gm = pymdp_to_alf(inner_agent)
            self.agent = AnalyticAgent(gm, gamma=gamma, seed=seed)

        self.monitor = EFEMonitor(
            decay=monitor_decay,
            window_size=monitor_window,
        )
        self.gamma_learning_rate = gamma_learning_rate
        self.gamma_update_interval = gamma_update_interval
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

        # History tracking
        self.confidence_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.gamma_history: list[float] = [self.agent.gamma]
        self._trial_count: int = 0
        self._last_confidence: float = 0.5
        self._last_efe: float = 0.0

    # -- Properties that proxy to the inner agent --------------------------

    @property
    def gm(self) -> GenerativeModel:
        """The inner agent's generative model."""
        return self.agent.gm

    @property
    def gamma(self) -> float:
        """Current policy precision."""
        return self.agent.gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self.agent.gamma = value

    @property
    def beliefs(self) -> list[np.ndarray]:
        """Current beliefs over hidden states."""
        return self.agent.beliefs

    @property
    def action_history(self) -> list[int]:
        """History of actions taken by the inner agent."""
        return self.agent.action_history

    # -- Core interface (drop-in for AnalyticAgent) ------------------------

    def step(
        self,
        observation: list[int],
    ) -> tuple[int, dict[str, Any]]:
        """Perform one step: delegate to inner agent + metacognitive monitoring.

        1. Delegates to inner agent's step() for belief update, EFE
           evaluation, and action selection.
        2. Extracts confidence (max policy probability) and EFE of the
           selected policy.
        3. Stores metacognitive statistics for later learn() call.

        Args:
            observation: List of observation indices, one per modality.

        Returns:
            Tuple of (action_index, info_dict). The info_dict includes all
            keys from the inner agent plus 'metacognitive_confidence'.
        """
        action, info = self.agent.step(observation)

        # Extract metacognitive signals
        policy_probs = info["policy_probs"]
        self._last_confidence = float(np.max(policy_probs))
        selected_idx = info["selected_policy"]
        self._last_efe = float(info["G"][selected_idx])

        self.confidence_history.append(self._last_confidence)

        info["metacognitive_confidence"] = self._last_confidence

        return action, info

    def learn(self, outcome_valence: float) -> None:
        """Update habits, metacognitive monitor, and optionally gamma.

        1. Delegates to inner agent's learn() for habit updating.
        2. Records the trial in the EFEMonitor.
        3. Periodically adjusts gamma based on calibration.

        Args:
            outcome_valence: How good the outcome was. Positive values
                indicate correct/rewarding outcomes, negative values
                indicate incorrect/punishing outcomes.
        """
        # 1. Inner agent habit learning
        self.agent.learn(outcome_valence)

        # 2. Record in monitor
        accuracy = 1.0 if outcome_valence > 0.0 else 0.0
        self.monitor.record(
            predicted_efe=self._last_efe,
            confidence=self._last_confidence,
            outcome_valence=outcome_valence,
        )
        self.accuracy_history.append(accuracy)
        self._trial_count += 1

        # 3. Periodic gamma adjustment from metacognitive feedback
        if self._trial_count % self.gamma_update_interval == 0:
            self._adjust_gamma()

    def _adjust_gamma(self) -> None:
        """Adjust gamma based on metacognitive calibration.

        Uses update_gamma_from_confidence with EMA-smoothed confidence
        and accuracy to avoid noisy single-trial updates.
        """
        new_gamma = update_gamma_from_confidence(
            gamma=self.agent.gamma,
            predicted_confidence=self.monitor._ema_confidence,
            actual_accuracy=self.monitor._ema_accuracy,
            learning_rate=self.gamma_learning_rate,
            min_gamma=self.min_gamma,
            max_gamma=self.max_gamma,
        )
        self.agent.gamma = new_gamma
        self.gamma_history.append(new_gamma)

    def reset(self) -> None:
        """Reset the inner agent's beliefs (keeps learned habits and history)."""
        self.agent.reset()

    def get_metacognitive_summary(self) -> dict[str, Any]:
        """Return a comprehensive summary of metacognitive state.

        Returns:
            Dict with keys: m_ratio, calibration_error, current_gamma,
            gamma_history, mean_confidence, mean_accuracy,
            confidence_accuracy_gap, n_trials, monitor_summary.
        """
        mean_conf = (
            float(np.mean(self.confidence_history))
            if self.confidence_history else 0.5
        )
        mean_acc = (
            float(np.mean(self.accuracy_history))
            if self.accuracy_history else 0.5
        )

        return {
            "m_ratio": self.monitor.get_online_m_ratio(),
            "calibration_error": self.monitor.get_calibration(),
            "current_gamma": self.agent.gamma,
            "gamma_history": list(self.gamma_history),
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "confidence_accuracy_gap": mean_conf - mean_acc,
            "n_trials": self._trial_count,
            "monitor_summary": self.monitor.get_summary(),
        }

    def get_state_summary(self) -> dict[str, Any]:
        """Return combined inner-agent and metacognitive summary."""
        summary = self.agent.get_state_summary()
        summary["metacognition"] = self.get_metacognitive_summary()
        return summary


# ---------------------------------------------------------------------------
# PopulationMetacognition: aggregate stats for multi-agent simulations
# ---------------------------------------------------------------------------

class PopulationMetacognition:
    """Aggregate metacognitive statistics across a population of agents.

    Useful for Concordia/SustainHub multi-agent simulations where you
    want to track the distribution of metacognitive efficiency across
    heterogeneous agents.

    Args:
        agents: List of MetacognitiveAgent instances.
    """

    def __init__(self, agents: list[MetacognitiveAgent]):
        self.agents = agents

    def get_population_m_ratios(self) -> np.ndarray:
        """Get m-ratio for each agent in the population.

        Returns:
            Array of m-ratios, shape (n_agents,).
        """
        return np.array([
            a.monitor.get_online_m_ratio() for a in self.agents
        ])

    def get_heterogeneity_stats(self) -> dict[str, float]:
        """Compute summary statistics of metacognitive heterogeneity.

        Returns:
            Dict with keys: mean_m_ratio, std_m_ratio, min_m_ratio,
            max_m_ratio, range_m_ratio, mean_gamma, std_gamma,
            mean_calibration_error.
        """
        m_ratios = self.get_population_m_ratios()
        gammas = np.array([a.gamma for a in self.agents])
        calibrations = np.array([
            a.monitor.get_calibration() for a in self.agents
        ])

        return {
            "mean_m_ratio": float(np.mean(m_ratios)),
            "std_m_ratio": float(np.std(m_ratios)),
            "min_m_ratio": float(np.min(m_ratios)),
            "max_m_ratio": float(np.max(m_ratios)),
            "range_m_ratio": float(np.ptp(m_ratios)),
            "mean_gamma": float(np.mean(gammas)),
            "std_gamma": float(np.std(gammas)),
            "mean_calibration_error": float(np.mean(calibrations)),
        }

    def get_population_summary(self) -> dict[str, Any]:
        """Get full per-agent metacognitive summaries plus population stats.

        Returns:
            Dict with keys: agent_summaries (list of dicts),
            heterogeneity (dict of population-level stats).
        """
        return {
            "agent_summaries": [
                a.get_metacognitive_summary() for a in self.agents
            ],
            "heterogeneity": self.get_heterogeneity_stats(),
        }
