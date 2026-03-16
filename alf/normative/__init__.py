"""Normative modeling for individual deviation scoring.

JAX-native implementation of normative models (Marquand et al., 2016; Wolfers
et al., 2018) that fit population distributions and score individual deviations
as Z-scores. The key computation pattern — independent regressions over 1000s
of brain regions — is ideally suited to jax.vmap for massive parallelism.

Active Inference connection: Z-score = surprisal under a population prior.
A normative deviation is the individual's variational free energy relative
to the population generative model. This connects normative modeling to the
precision estimation and individual differences framework in AIF.

Modules:
    blr: Bayesian Linear Regression with B-spline basis (the workhorse).
    warping: SHASH (sinh-arcsinh) likelihood warping for non-Gaussian data.

References:
    Marquand, Rezek, Buitelaar & Beckmann (2016). Understanding heterogeneity
        in clinical cohorts using normative models. Biological Psychiatry.
    Wolfers, Buitelaar, Beckmann, Franke & Marquand (2015). From estimating
        activation locality to predicting disorder. NeuroImage.
    Fraza, Dinga, Beckmann & Marquand (2021). Warped Bayesian linear regression
        for normative modelling of big data. NeuroImage.
"""

from alf.normative.blr import (
    BLRParams,
    BLRResult,
    NormativeResult,
    bspline_basis,
    fit_blr,
    predict_blr,
    compute_zscore,
    normative_model,
    normative_model_vmap,
)
from alf.normative.warping import (
    shash_log_prob,
    shash_transform,
    shash_inverse_transform,
    fit_blr_shash,
)

__all__ = [
    # BLR
    "BLRParams",
    "BLRResult",
    "NormativeResult",
    "bspline_basis",
    "fit_blr",
    "predict_blr",
    "compute_zscore",
    "normative_model",
    "normative_model_vmap",
    # Warping
    "shash_log_prob",
    "shash_transform",
    "shash_inverse_transform",
    "fit_blr_shash",
]
