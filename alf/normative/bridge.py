"""Bridge between normative modeling Z-scores and active inference free energy.

The key mathematical insight: a normative model's Z-score IS a form of
surprisal (negative log-probability) under the population generative model.
Specifically, for a Gaussian normative model:

    VFE = 0.5 * z^2 + 0.5 * log(2*pi)

This is exactly the surprise of observing a brain measure z standard
deviations from the population norm. In active inference terms, the
normative model provides the generative model p(y | x), and the Z-score
quantifies the free energy of an individual's data under that model.

This module provides functions to:
    1. Convert Z-scores to variational free energy (surprise).
    2. Compute individual-level VFE from raw predictions.
    3. Convert deviation profiles to categorical beliefs for action selection.
    4. Run a full normative-AIF pipeline (ComBat -> BLR -> Z-scores -> VFE).

References:
    Marquand, Rezek, Buitelaar & Beckmann (2016). Understanding heterogeneity
        in clinical cohorts using normative models. Biological Psychiatry.
    Friston, FitzGerald, Rigoli, Schwartenbeck, Daunizeau & Pezzulo (2016).
        Active inference and learning. Neuroscience & Biobehavioral Reviews.
    Rutherford, Fraza, Dinga, et al. (2022). Charting brain growth and aging
        at high spatial precision. eLife.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Numerical stability constant
# ---------------------------------------------------------------------------

eps = 1e-16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NormativeAIFResult(NamedTuple):
    """Combined result of normative modeling and active inference bridge.

    Attributes:
        z_scores: Deviation Z-scores, shape (n_test, n_features).
        vfe: Variational free energy per subject per feature,
            shape (n_test, n_features).
        vfe_total: Total VFE per subject (summed across features),
            shape (n_test,).
        deviation_mask: Boolean mask of deviant features,
            shape (n_test, n_features).
        belief_vectors: Categorical belief vectors (normal vs deviant),
            shape (n_test, n_features, 2).
        y_pred: Predicted means, shape (n_test, n_features).
        y_std: Predicted standard deviations, shape (n_test, n_features).
    """
    z_scores: jnp.ndarray
    vfe: jnp.ndarray
    vfe_total: jnp.ndarray
    deviation_mask: jnp.ndarray
    belief_vectors: jnp.ndarray
    y_pred: jnp.ndarray
    y_std: jnp.ndarray


# ---------------------------------------------------------------------------
# Core conversions
# ---------------------------------------------------------------------------

def zscore_to_vfe(z_scores: jnp.ndarray) -> jnp.ndarray:
    """Convert normative Z-scores to variational free energy (surprise).

    Under a Gaussian population model, the surprise (negative log-probability)
    of observing a value z standard deviations from the mean is:

        VFE = -log p(z) = 0.5 * z^2 + 0.5 * log(2*pi)

    This is the direct mathematical connection between normative modeling
    and the free energy principle: each Z-score IS a free energy.

    Args:
        z_scores: Deviation Z-scores, shape (*,). Can be any shape;
            the computation is element-wise.

    Returns:
        Variational free energy, same shape as z_scores. Units are nats.
    """
    z_scores = jnp.asarray(z_scores)
    return 0.5 * z_scores ** 2 + 0.5 * jnp.log(2.0 * jnp.pi)


def individual_free_energy(
    y_obs: jnp.ndarray,
    y_pred: jnp.ndarray,
    y_var: jnp.ndarray,
) -> jnp.ndarray:
    """Compute individual-level variational free energy from BLR predictions.

    The VFE under a Gaussian generative model p(y | mu, sigma^2) is:

        F = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2

    This is the negative log-likelihood (surprise) of the individual's
    observation under the population normative model. It decomposes into:
        - Complexity: 0.5 * log(2*pi*sigma^2) (uncertainty of the model)
        - Accuracy: 0.5 * (y - mu)^2 / sigma^2 (squared normalized error)

    This function is designed to be vmappable over brain regions.

    Args:
        y_obs: Observed values, shape (n_subjects,) or scalar.
        y_pred: Predicted means from BLR, shape (n_subjects,) or scalar.
        y_var: Predicted variances from BLR, shape (n_subjects,) or scalar.

    Returns:
        Free energy per observation, same shape as inputs. Units are nats.
    """
    y_obs = jnp.asarray(y_obs)
    y_pred = jnp.asarray(y_pred)
    y_var = jnp.asarray(y_var)

    y_var_safe = jnp.clip(y_var, eps)
    complexity = 0.5 * jnp.log(2.0 * jnp.pi * y_var_safe)
    accuracy = 0.5 * (y_obs - y_pred) ** 2 / y_var_safe

    return complexity + accuracy


# Vmappable version over brain regions (columns)
individual_free_energy_vmap = jax.vmap(
    individual_free_energy,
    in_axes=(1, 1, 1),
    out_axes=1,
)


# ---------------------------------------------------------------------------
# Deviation profiles -> beliefs
# ---------------------------------------------------------------------------

def deviation_profile_to_beliefs(
    z_scores: jnp.ndarray,
    threshold: float = 2.0,
) -> jnp.ndarray:
    """Convert a profile of Z-scores into categorical beliefs.

    For each brain region, computes a soft categorical distribution over
    {"Normal", "Deviant"} states, where the confidence in "Deviant"
    increases with |z| relative to the threshold.

    The mapping uses a sigmoid function centered at the threshold:
        P(Deviant | z) = sigmoid(|z| - threshold)
        P(Normal | z) = 1 - P(Deviant | z)

    This produces a belief vector suitable for alf's action selection
    machinery (e.g., as input to policy evaluation over interventions).

    Args:
        z_scores: Deviation Z-scores, shape (n_features,) for a single
            subject or (n_subjects, n_features) for a batch.
        threshold: Z-score threshold for deviation. Default 2.0
            (approximately p < 0.05 two-tailed under normality).

    Returns:
        Belief vectors, shape (*z_scores.shape, 2). The last axis is
        [P(Normal), P(Deviant)] for each region.
    """
    z_scores = jnp.asarray(z_scores)

    # Soft classification via sigmoid
    p_deviant = jax.nn.sigmoid(jnp.abs(z_scores) - threshold)
    p_normal = 1.0 - p_deviant

    # Stack into belief vectors [..., 2]
    beliefs = jnp.stack([p_normal, p_deviant], axis=-1)

    return beliefs


def deviation_mask(
    z_scores: jnp.ndarray,
    threshold: float = 2.0,
) -> jnp.ndarray:
    """Hard threshold Z-scores to identify deviant brain regions.

    Args:
        z_scores: Deviation Z-scores, shape (*,).
        threshold: Absolute Z-score threshold. Default 2.0.

    Returns:
        Boolean mask where True indicates a deviant region, same shape
        as z_scores.
    """
    return jnp.abs(jnp.asarray(z_scores)) > threshold


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def normative_aif_pipeline(
    x_train: np.ndarray,
    Y_train: np.ndarray,
    x_test: np.ndarray,
    Y_test: np.ndarray,
    site_train: Optional[np.ndarray] = None,
    site_test: Optional[np.ndarray] = None,
    n_basis: int = 10,
    degree: int = 3,
    z_threshold: float = 2.0,
) -> NormativeAIFResult:
    """Full normative modeling to active inference pipeline.

    Runs the complete workflow:
        1. (Optional) ComBat harmonization to remove site effects.
        2. BLR normative modeling across brain regions (vmapped).
        3. Z-score computation.
        4. Z-score -> VFE conversion (the AIF bridge).
        5. Deviation profiling for belief formation.

    Args:
        x_train: Training covariates (e.g., age), shape (n_train,).
        Y_train: Training brain measures, shape (n_train, n_features).
        x_test: Test covariates, shape (n_test,).
        Y_test: Test brain measures, shape (n_test, n_features).
        site_train: Optional training site labels, shape (n_train,).
            If provided along with site_test, ComBat is applied.
        site_test: Optional test site labels, shape (n_test,).
        n_basis: Number of B-spline basis functions for BLR.
        degree: B-spline degree.
        z_threshold: Z-score threshold for deviation detection.

    Returns:
        NormativeAIFResult combining normative and AIF quantities.
    """
    from alf.normative.blr import normative_model_vmap

    Y_train_use = np.asarray(Y_train)
    Y_test_use = np.asarray(Y_test)

    # Step 1: ComBat harmonization (if site labels provided)
    if site_train is not None and site_test is not None:
        from alf.normative.combat import combat_harmonize
        Y_train_harm, Y_test_harm = combat_harmonize(
            jnp.array(Y_train_use),
            jnp.array(site_train),
            jnp.array(Y_test_use),
            jnp.array(site_test),
        )
        Y_train_use = np.asarray(Y_train_harm)
        Y_test_use = np.asarray(Y_test_harm)

    # Step 2-3: BLR + Z-scores (vmapped over brain regions)
    z_scores, y_pred, y_std = normative_model_vmap(
        x_train, Y_train_use, x_test, Y_test_use,
        n_basis=n_basis, degree=degree,
    )

    # Step 4: Z-scores -> VFE (the bridge)
    vfe = zscore_to_vfe(z_scores)
    vfe_total = jnp.sum(vfe, axis=1)  # Sum across features per subject

    # Step 5: Deviation profiling
    dev_mask = deviation_mask(z_scores, threshold=z_threshold)
    belief_vectors = deviation_profile_to_beliefs(
        z_scores, threshold=z_threshold
    )

    return NormativeAIFResult(
        z_scores=z_scores,
        vfe=vfe,
        vfe_total=vfe_total,
        deviation_mask=dev_mask,
        belief_vectors=belief_vectors,
        y_pred=y_pred,
        y_std=y_std,
    )
