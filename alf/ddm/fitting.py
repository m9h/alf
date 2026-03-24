"""Hierarchical Bayesian DDM fitting via gradient-based MLE and NumPyro NUTS.

Provides three levels of DDM parameter estimation:

1. **Maximum Likelihood (MLE)**: Fast point estimates via jax.grad + optax.
   Works with just JAX, no external sampling libraries needed.

2. **Single-subject Bayesian**: Full posterior via NumPyro NUTS sampler.
   Provides uncertainty estimates and enables posterior predictive checks.

3. **Hierarchical Bayesian**: Multi-subject model with group-level priors.
   The key contribution — enables computational phenotyping by estimating
   both individual and population-level DDM parameters simultaneously.

The hierarchical model places group-level priors on each DDM parameter:
    v_i ~ Normal(mu_v, sigma_v)       per-subject drift rate
    a_i ~ LogNormal(mu_a, sigma_a)    per-subject boundary separation
    w_i ~ Beta(alpha_w, beta_w)       per-subject starting bias
    tau_i ~ LogNormal(mu_tau, sigma_tau)  per-subject non-decision time

All functions use the Navarro-Fuss likelihood from alf.ddm.wiener.

References:
    Navarro & Fuss (2009). Fast and accurate calculations for first-passage
        times in Wiener diffusion models. Journal of Mathematical Psychology.
    Wiecki, Sofer & Frank (2013). HDDM: Hierarchical Bayesian estimation of
        the Drift-Diffusion Model in Python. Frontiers in Neuroinformatics.
    Ratcliff & McKoon (2008). The diffusion decision model: Theory and data
        for two-choice decision tasks. Neural Computation.
    Schwartenbeck & Friston (2016). Computational Phenotyping in Psychiatry.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from alf.ddm.wiener import (
    DDMParams,
    wiener_log_density_batch,
    ddm_nll,
    simulate_ddm,
)


# ---------------------------------------------------------------------------
# Numerical constant
# ---------------------------------------------------------------------------

eps = 1e-16


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

def _try_import_optax():
    """Try to import optax, return None if unavailable."""
    try:
        import optax
        return optax
    except ImportError:
        return None


def _try_import_numpyro():
    """Try to import numpyro, return None if unavailable."""
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        return numpyro, dist, MCMC, NUTS
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DDMFitResult(NamedTuple):
    """Result of DDM parameter fitting (MLE).

    Attributes:
        v: Drift rate point estimate.
        a: Boundary separation point estimate.
        w: Starting point bias point estimate.
        tau: Non-decision time point estimate.
        v_se: Standard error for v (None if not computed).
        a_se: Standard error for a (None if not computed).
        w_se: Standard error for w (None if not computed).
        tau_se: Standard error for tau (None if not computed).
        loss_history: NLL at each epoch.
        n_trials: Number of trials used for fitting.
    """
    v: float
    a: float
    w: float
    tau: float
    v_se: Optional[float]
    a_se: Optional[float]
    w_se: Optional[float]
    tau_se: Optional[float]
    loss_history: list
    n_trials: int


class DDMRecoveryResult(NamedTuple):
    """Result of parameter recovery analysis.

    Attributes:
        true_params: Array of true parameter sets, shape (n_repeats, 4).
        recovered_params: Array of recovered parameter sets, shape (n_repeats, 4).
        param_names: Names of the four parameters.
        correlations: Correlation between true and recovered for each param.
        biases: Mean(recovered - true) for each parameter.
        rmse: Root mean squared error for each parameter.
        coverage_95: Proportion of true values within 95% CI (if available).
    """
    true_params: np.ndarray
    recovered_params: np.ndarray
    param_names: tuple
    correlations: np.ndarray
    biases: np.ndarray
    rmse: np.ndarray
    coverage_95: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Parameter transforms (unconstrained <-> constrained)
# ---------------------------------------------------------------------------

def _to_unconstrained(
    v: jnp.ndarray,
    a: jnp.ndarray,
    w: jnp.ndarray,
    tau: jnp.ndarray,
) -> tuple:
    """Map DDM parameters to unconstrained space.

    Args:
        v: Drift rate (unconstrained).
        a: Boundary separation (> 0).
        w: Starting bias (0, 1).
        tau: Non-decision time (> 0).

    Returns:
        Tuple of (v_unc, log_a, logit_w, log_tau) in unconstrained space.
    """
    v_unc = v
    log_a = jnp.log(jnp.clip(a, eps))
    logit_w = jnp.log(jnp.clip(w, eps) / jnp.clip(1.0 - w, eps))
    log_tau = jnp.log(jnp.clip(tau, eps))
    return v_unc, log_a, logit_w, log_tau


def _to_constrained(
    v_unc: jnp.ndarray,
    log_a: jnp.ndarray,
    logit_w: jnp.ndarray,
    log_tau: jnp.ndarray,
) -> tuple:
    """Map unconstrained parameters back to DDM parameter space.

    Args:
        v_unc: Unconstrained drift rate.
        log_a: Log boundary separation.
        logit_w: Logit starting bias.
        log_tau: Log non-decision time.

    Returns:
        Tuple of (v, a, w, tau) in constrained space.
    """
    v = v_unc
    a = jnp.exp(log_a)
    w = jax.nn.sigmoid(logit_w)
    tau = jnp.exp(log_tau)
    return v, a, w, tau


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def _nll_unconstrained(
    v_unc: jnp.ndarray,
    log_a: jnp.ndarray,
    logit_w: jnp.ndarray,
    log_tau: jnp.ndarray,
    rt: jnp.ndarray,
    choice: jnp.ndarray,
) -> jnp.ndarray:
    """Negative log-likelihood in unconstrained parameterization.

    This wrapper applies the inverse transforms to enforce constraints
    (a > 0, 0 < w < 1, tau > 0) while keeping all parameters unconstrained
    for gradient-based optimization.

    Args:
        v_unc: Unconstrained drift rate.
        log_a: Log boundary separation.
        logit_w: Logit starting bias.
        log_tau: Log non-decision time.
        rt: Reaction times, shape (N,).
        choice: Responses, shape (N,).

    Returns:
        Negative log-likelihood (scalar).
    """
    v, a, w, tau = _to_constrained(v_unc, log_a, logit_w, log_tau)
    return ddm_nll(v, a, w, tau, rt, choice)


def fit_ddm_mle(
    rt: np.ndarray,
    choice: np.ndarray,
    init_params: Optional[DDMParams] = None,
    num_epochs: int = 500,
    lr: float = 0.01,
    verbose: bool = False,
) -> DDMFitResult:
    """Fit DDM parameters via maximum likelihood estimation.

    Uses gradient descent through the Navarro-Fuss log-likelihood.
    Parameters are optimized in unconstrained space (exp/sigmoid transforms)
    to ensure valid DDM parameters at every step.

    Prefers optax.adam when available; falls back to manual SGD.

    Args:
        rt: Reaction times, shape (N,).
        choice: Responses (1 = upper, 0 = lower), shape (N,).
        init_params: Initial DDM parameters. If None, uses sensible defaults
            (v=0, a=1.5, w=0.5, tau=min(rt)*0.5).
        num_epochs: Number of optimization steps. Default 500.
        lr: Learning rate. Default 0.01.
        verbose: If True, print loss every 50 epochs.

    Returns:
        DDMFitResult with point estimates and loss history.
    """
    rt_jnp = jnp.array(rt, dtype=jnp.float32)
    choice_jnp = jnp.array(choice, dtype=jnp.float32)
    n_trials = len(rt)

    # Initialize parameters
    if init_params is None:
        v_init = jnp.array(0.0)
        a_init = jnp.array(1.5)
        w_init = jnp.array(0.5)
        tau_init = jnp.array(float(np.min(rt)) * 0.5)
        tau_init = jnp.clip(tau_init, 0.01, None)
    else:
        v_init = jnp.array(float(init_params.v))
        a_init = jnp.array(float(init_params.a))
        w_init = jnp.array(float(init_params.w))
        tau_init = jnp.array(float(init_params.tau))

    # Transform to unconstrained space
    v_unc, log_a, logit_w, log_tau = _to_unconstrained(
        v_init, a_init, w_init, tau_init
    )

    grad_fn = jax.grad(_nll_unconstrained, argnums=(0, 1, 2, 3))

    optax = _try_import_optax()

    loss_history = []

    if optax is not None:
        optimizer = optax.adam(lr)
        opt_state = optimizer.init((v_unc, log_a, logit_w, log_tau))

        @jax.jit
        def update_step(v_unc, log_a, logit_w, log_tau, opt_state):
            loss = _nll_unconstrained(
                v_unc, log_a, logit_w, log_tau, rt_jnp, choice_jnp
            )
            grads = grad_fn(v_unc, log_a, logit_w, log_tau, rt_jnp, choice_jnp)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, (v_unc, log_a, logit_w, log_tau)
            )
            new_v_unc = v_unc + updates[0]
            new_log_a = log_a + updates[1]
            new_logit_w = logit_w + updates[2]
            new_log_tau = log_tau + updates[3]
            return new_v_unc, new_log_a, new_logit_w, new_log_tau, new_opt_state, loss

        for epoch in range(num_epochs):
            v_unc, log_a, logit_w, log_tau, opt_state, loss = update_step(
                v_unc, log_a, logit_w, log_tau, opt_state
            )
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    else:
        # Fallback: manual SGD
        @jax.jit
        def sgd_step(v_unc, log_a, logit_w, log_tau):
            loss = _nll_unconstrained(
                v_unc, log_a, logit_w, log_tau, rt_jnp, choice_jnp
            )
            g_v, g_a, g_w, g_tau = grad_fn(
                v_unc, log_a, logit_w, log_tau, rt_jnp, choice_jnp
            )
            new_v_unc = v_unc - lr * g_v
            new_log_a = log_a - lr * g_a
            new_logit_w = logit_w - lr * g_w
            new_log_tau = log_tau - lr * g_tau
            return new_v_unc, new_log_a, new_logit_w, new_log_tau, loss

        for epoch in range(num_epochs):
            v_unc, log_a, logit_w, log_tau, loss = sgd_step(
                v_unc, log_a, logit_w, log_tau
            )
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    # Transform back to constrained space
    v_fit, a_fit, w_fit, tau_fit = _to_constrained(v_unc, log_a, logit_w, log_tau)

    return DDMFitResult(
        v=float(v_fit),
        a=float(a_fit),
        w=float(w_fit),
        tau=float(tau_fit),
        v_se=None,
        a_se=None,
        w_se=None,
        tau_se=None,
        loss_history=loss_history,
        n_trials=n_trials,
    )


# ---------------------------------------------------------------------------
# Bayesian fitting (single subject)
# ---------------------------------------------------------------------------

def fit_ddm_bayesian(
    rt: np.ndarray,
    choice: np.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Fit DDM parameters via Bayesian inference using NumPyro NUTS.

    Places informative priors on DDM parameters and samples from the
    posterior using the No-U-Turn Sampler (NUTS). The likelihood is the
    Navarro-Fuss first-passage time density.

    Priors:
        v ~ Normal(0, 3)         — drift rate
        a ~ LogNormal(0.5, 0.5)  — boundary separation
        w ~ Beta(5, 5)           — starting bias (centered near 0.5)
        tau ~ LogNormal(-1, 0.5) — non-decision time

    Args:
        rt: Reaction times, shape (N,).
        choice: Responses (1 = upper, 0 = lower), shape (N,).
        num_warmup: Number of NUTS warmup (adaptation) steps.
        num_samples: Number of posterior samples to draw.
        seed: Random seed for the MCMC sampler.

    Returns:
        Dict with keys:
            'samples': dict of posterior samples for v, a, w, tau.
            'v_mean', 'a_mean', 'w_mean', 'tau_mean': Posterior means.
            'v_hdi_95', 'a_hdi_95', 'w_hdi_95', 'tau_hdi_95': 95% HDIs.
            'summary': dict of summary statistics per parameter.

    Raises:
        ImportError: If numpyro is not installed.
    """
    imports = _try_import_numpyro()
    if imports is None:
        raise ImportError(
            "numpyro is required for Bayesian DDM fitting. "
            "Install with: pip install numpyro"
        )
    numpyro, dist, MCMC, NUTS = imports

    rt_jnp = jnp.array(rt, dtype=jnp.float32)
    choice_jnp = jnp.array(choice, dtype=jnp.float32)

    def model(rt_obs, choice_obs):
        # Priors
        v = numpyro.sample("v", dist.Normal(0.0, 3.0))
        a = numpyro.sample("a", dist.LogNormal(0.5, 0.5))
        w = numpyro.sample("w", dist.Beta(5.0, 5.0))
        tau = numpyro.sample("tau", dist.LogNormal(-1.0, 0.5))

        # Likelihood via Navarro-Fuss density
        log_densities = wiener_log_density_batch(
            rt_obs, choice_obs, v, a, w, tau
        )
        numpyro.factor("obs", jnp.sum(log_densities))

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), rt_jnp, choice_jnp)

    samples = mcmc.get_samples()

    # Build summary
    param_names = ["v", "a", "w", "tau"]
    summary = {}
    result = {"samples": {}}

    for name in param_names:
        s = np.array(samples[name])
        result["samples"][name] = s
        result[f"{name}_mean"] = float(np.mean(s))
        result[f"{name}_std"] = float(np.std(s))
        result[f"{name}_hdi_95"] = (
            float(np.percentile(s, 2.5)),
            float(np.percentile(s, 97.5)),
        )
        summary[name] = {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "hdi_2.5%": float(np.percentile(s, 2.5)),
            "hdi_97.5%": float(np.percentile(s, 97.5)),
            "median": float(np.median(s)),
        }

    result["summary"] = summary
    return result


# ---------------------------------------------------------------------------
# Hierarchical Bayesian fitting (multi-subject)
# ---------------------------------------------------------------------------

def fit_ddm_hierarchical(
    rt_list: list,
    choice_list: list,
    subject_ids: Optional[list] = None,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Fit hierarchical Bayesian DDM with group and subject-level parameters.

    The hierarchical model estimates group-level distributions for each DDM
    parameter and shrinks individual-subject estimates toward the group mean.
    This enables more reliable parameter estimation for subjects with few
    trials, and directly estimates population-level effects.

    Model structure:
        Group level:
            mu_v ~ Normal(0, 3)
            sigma_v ~ HalfNormal(2)
            mu_log_a ~ Normal(0.5, 1)
            sigma_log_a ~ HalfNormal(0.5)
            mu_logit_w ~ Normal(0, 1)
            sigma_logit_w ~ HalfNormal(0.5)
            mu_log_tau ~ Normal(-1, 1)
            sigma_log_tau ~ HalfNormal(0.5)

        Subject level:
            v_i ~ Normal(mu_v, sigma_v)
            log(a_i) ~ Normal(mu_log_a, sigma_log_a)
            logit(w_i) ~ Normal(mu_logit_w, sigma_logit_w)
            log(tau_i) ~ Normal(mu_log_tau, sigma_log_tau)

    Args:
        rt_list: List of RT arrays, one per subject. Each shape (N_i,).
        choice_list: List of choice arrays, one per subject. Each shape (N_i,).
        subject_ids: Optional subject identifiers. If None, uses 0..N-1.
        num_warmup: Number of NUTS warmup steps.
        num_samples: Number of posterior samples.
        seed: Random seed for the MCMC sampler.

    Returns:
        Dict with keys:
            'samples': dict of posterior samples (group + subject level).
            'group_summary': summary statistics for group-level parameters.
            'subject_summary': per-subject summary statistics.
            'n_subjects': number of subjects.

    Raises:
        ImportError: If numpyro is not installed.
    """
    imports = _try_import_numpyro()
    if imports is None:
        raise ImportError(
            "numpyro is required for hierarchical DDM fitting. "
            "Install with: pip install numpyro"
        )
    numpyro, dist, MCMC, NUTS = imports

    n_subjects = len(rt_list)
    if subject_ids is None:
        subject_ids = list(range(n_subjects))

    # Pad and concatenate data for vectorized likelihood computation
    # Build subject index arrays
    rt_all = []
    choice_all = []
    subj_idx_all = []
    for i in range(n_subjects):
        rt_i = np.asarray(rt_list[i], dtype=np.float32)
        ch_i = np.asarray(choice_list[i], dtype=np.float32)
        rt_all.append(rt_i)
        choice_all.append(ch_i)
        subj_idx_all.append(np.full(len(rt_i), i, dtype=np.int32))

    rt_concat = jnp.array(np.concatenate(rt_all))
    choice_concat = jnp.array(np.concatenate(choice_all))
    subj_idx_concat = jnp.array(np.concatenate(subj_idx_all))

    def model(rt_obs, choice_obs, subj_idx, n_subj):
        # Group-level hyperpriors
        mu_v = numpyro.sample("mu_v", dist.Normal(0.0, 3.0))
        sigma_v = numpyro.sample("sigma_v", dist.HalfNormal(2.0))

        mu_log_a = numpyro.sample("mu_log_a", dist.Normal(0.5, 1.0))
        sigma_log_a = numpyro.sample("sigma_log_a", dist.HalfNormal(0.5))

        mu_logit_w = numpyro.sample("mu_logit_w", dist.Normal(0.0, 1.0))
        sigma_logit_w = numpyro.sample("sigma_logit_w", dist.HalfNormal(0.5))

        mu_log_tau = numpyro.sample("mu_log_tau", dist.Normal(-1.0, 1.0))
        sigma_log_tau = numpyro.sample("sigma_log_tau", dist.HalfNormal(0.5))

        # Subject-level parameters (non-centered parameterization)
        with numpyro.plate("subjects", n_subj):
            v_offset = numpyro.sample("v_offset", dist.Normal(0.0, 1.0))
            v = numpyro.deterministic("v", mu_v + sigma_v * v_offset)

            log_a_offset = numpyro.sample("log_a_offset", dist.Normal(0.0, 1.0))
            log_a = mu_log_a + sigma_log_a * log_a_offset
            a = numpyro.deterministic("a", jnp.exp(log_a))

            logit_w_offset = numpyro.sample(
                "logit_w_offset", dist.Normal(0.0, 1.0)
            )
            logit_w = mu_logit_w + sigma_logit_w * logit_w_offset
            w = numpyro.deterministic("w", jax.nn.sigmoid(logit_w))

            log_tau_offset = numpyro.sample(
                "log_tau_offset", dist.Normal(0.0, 1.0)
            )
            log_tau = mu_log_tau + sigma_log_tau * log_tau_offset
            tau = numpyro.deterministic("tau", jnp.exp(log_tau))

        # Likelihood: index into subject-level params for each trial
        v_trial = v[subj_idx]
        a_trial = a[subj_idx]
        w_trial = w[subj_idx]
        tau_trial = tau[subj_idx]

        # Compute per-trial log-densities using vmap over trials
        # (each trial may have different subject parameters)
        def single_trial_ll(rt_i, choice_i, v_i, a_i, w_i, tau_i):
            from alf.ddm.wiener import wiener_log_density
            return wiener_log_density(rt_i, choice_i, v_i, a_i, w_i, tau_i)

        log_densities = jax.vmap(single_trial_ll)(
            rt_obs, choice_obs, v_trial, a_trial, w_trial, tau_trial
        )
        numpyro.factor("obs", jnp.sum(log_densities))

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed),
        rt_concat, choice_concat, subj_idx_concat, n_subjects
    )

    samples = mcmc.get_samples()

    # Build group-level summary
    group_params = [
        "mu_v", "sigma_v", "mu_log_a", "sigma_log_a",
        "mu_logit_w", "sigma_logit_w", "mu_log_tau", "sigma_log_tau",
    ]
    group_summary = {}
    for name in group_params:
        if name in samples:
            s = np.array(samples[name])
            group_summary[name] = {
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
                "hdi_2.5%": float(np.percentile(s, 2.5)),
                "hdi_97.5%": float(np.percentile(s, 97.5)),
            }

    # Build subject-level summary
    subject_summary = {}
    for i, sid in enumerate(subject_ids):
        subj_dict = {}
        for pname in ["v", "a", "w", "tau"]:
            if pname in samples:
                s = np.array(samples[pname][:, i])
                subj_dict[pname] = {
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s)),
                    "hdi_2.5%": float(np.percentile(s, 2.5)),
                    "hdi_97.5%": float(np.percentile(s, 97.5)),
                }
        subject_summary[sid] = subj_dict

    return {
        "samples": {k: np.array(v) for k, v in samples.items()},
        "group_summary": group_summary,
        "subject_summary": subject_summary,
        "n_subjects": n_subjects,
        "subject_ids": subject_ids,
    }


# ---------------------------------------------------------------------------
# Posterior predictive simulation
# ---------------------------------------------------------------------------

def ddm_posterior_predictive(
    params_or_samples: dict,
    n_trials: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate synthetic data from fitted DDM parameters.

    Supports both point estimates (from MLE) and posterior samples
    (from Bayesian fitting). For posterior samples, generates data from
    randomly selected posterior draws for posterior predictive checks.

    Args:
        params_or_samples: Either:
            - DDMFitResult (uses point estimates), or
            - dict with 'samples' key containing posterior samples, or
            - dict with 'v', 'a', 'w', 'tau' keys (point estimates).
        n_trials: Number of synthetic trials to generate.
        seed: Random seed.

    Returns:
        Dict with keys:
            'rt': Simulated reaction times, shape (n_trials,).
            'choice': Simulated choices, shape (n_trials,).
            'params_used': dict of parameter values used for simulation.
    """
    rng = np.random.RandomState(seed)

    # Extract parameters
    if isinstance(params_or_samples, DDMFitResult):
        v = params_or_samples.v
        a = params_or_samples.a
        w = params_or_samples.w
        tau = params_or_samples.tau
    elif isinstance(params_or_samples, dict) and "samples" in params_or_samples:
        # Draw from a random posterior sample
        posterior = params_or_samples["samples"]
        n_post = len(posterior["v"])
        idx = rng.randint(0, n_post)
        v = float(posterior["v"][idx])
        a = float(posterior["a"][idx])
        w = float(posterior["w"][idx])
        tau = float(posterior["tau"][idx])
    elif isinstance(params_or_samples, dict):
        v = float(params_or_samples["v"])
        a = float(params_or_samples["a"])
        w = float(params_or_samples["w"])
        tau = float(params_or_samples["tau"])
    else:
        raise ValueError(
            "params_or_samples must be a DDMFitResult, or a dict with "
            "'samples' or individual parameter keys."
        )

    # Simulate
    result = simulate_ddm(
        v=v, a=a, w=w, tau=tau,
        n_trials=n_trials, seed=seed,
    )

    return {
        "rt": result.rt,
        "choice": result.choice,
        "params_used": {"v": v, "a": a, "w": w, "tau": tau},
    }


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------

def ddm_recovery_check(
    true_params: Optional[DDMParams] = None,
    n_trials: int = 200,
    n_repeats: int = 50,
    num_epochs: int = 500,
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = False,
) -> DDMRecoveryResult:
    """Parameter recovery analysis for DDM MLE fitting.

    Simulates data with known parameters, fits the model, and compares
    recovered parameters to the ground truth. Repeats multiple times with
    different random seeds to assess recovery reliability.

    If true_params is None, randomly generates parameter sets for each
    repeat from reasonable ranges:
        v ~ Uniform(-3, 3)
        a ~ Uniform(0.5, 3.0)
        w ~ Uniform(0.3, 0.7)
        tau ~ Uniform(0.1, 0.5)

    Args:
        true_params: Fixed true parameters for all repeats. If None,
            generates random parameters per repeat.
        n_trials: Number of trials to simulate per dataset.
        n_repeats: Number of recovery repetitions.
        num_epochs: Optimization epochs for each fit.
        lr: Learning rate for MLE.
        seed: Base random seed.
        verbose: If True, print progress.

    Returns:
        DDMRecoveryResult with recovery statistics.
    """
    rng = np.random.RandomState(seed)
    param_names = ("v", "a", "w", "tau")

    all_true = np.zeros((n_repeats, 4))
    all_recovered = np.zeros((n_repeats, 4))

    for rep in range(n_repeats):
        rep_seed = rng.randint(0, 2**31)

        # Generate or use fixed true parameters
        if true_params is not None:
            v_true = float(true_params.v)
            a_true = float(true_params.a)
            w_true = float(true_params.w)
            tau_true = float(true_params.tau)
        else:
            v_true = rng.uniform(-3.0, 3.0)
            a_true = rng.uniform(0.5, 3.0)
            w_true = rng.uniform(0.3, 0.7)
            tau_true = rng.uniform(0.1, 0.5)

        all_true[rep] = [v_true, a_true, w_true, tau_true]

        # Simulate data
        sim = simulate_ddm(
            v=v_true, a=a_true, w=w_true, tau=tau_true,
            n_trials=n_trials, seed=rep_seed,
        )

        # Fit
        result = fit_ddm_mle(
            rt=sim.rt, choice=sim.choice,
            num_epochs=num_epochs, lr=lr, verbose=False,
        )

        all_recovered[rep] = [result.v, result.a, result.w, result.tau]

        if verbose and (rep % 10 == 0 or rep == n_repeats - 1):
            print(
                f"  Recovery {rep+1}/{n_repeats}: "
                f"v={v_true:.2f}->{result.v:.2f}, "
                f"a={a_true:.2f}->{result.a:.2f}, "
                f"w={w_true:.2f}->{result.w:.2f}, "
                f"tau={tau_true:.2f}->{result.tau:.2f}"
            )

    # Compute statistics
    correlations = np.zeros(4)
    biases = np.zeros(4)
    rmse = np.zeros(4)

    for i in range(4):
        true_col = all_true[:, i]
        rec_col = all_recovered[:, i]

        # Handle constant columns (when true_params is fixed)
        if np.std(true_col) < eps:
            correlations[i] = np.nan
        else:
            correlations[i] = np.corrcoef(true_col, rec_col)[0, 1]

        biases[i] = np.mean(rec_col - true_col)
        rmse[i] = np.sqrt(np.mean((rec_col - true_col) ** 2))

    return DDMRecoveryResult(
        true_params=all_true,
        recovered_params=all_recovered,
        param_names=param_names,
        correlations=correlations,
        biases=biases,
        rmse=rmse,
        coverage_95=None,
    )
