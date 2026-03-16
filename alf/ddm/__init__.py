"""Drift-Diffusion Models (DDM) for Active Inference.

JAX-native implementation of drift-diffusion models (Ratcliff, 1978) with
the Navarro-Fuss (2009) first-passage time density for the Wiener process.
No production JAX DDM exists elsewhere — this is a novel contribution.

The DDM describes binary decisions as noisy evidence accumulation:
    dx = v*dt + s*dW    (Wiener diffusion with drift v)
with absorbing boundaries at 0 and a (threshold separation).

Active Inference connection: DDM parameters map directly onto AIF quantities:
    - Drift rate v = precision-weighted prediction error (gamma * delta_G)
    - Boundary a = policy precision gamma
    - Bias z = prior policy preference E(pi)

Modules:
    wiener: Navarro-Fuss first-passage time density in pure JAX.
    bridge: Map DDM parameters to/from Active Inference EFE.

References:
    Ratcliff (1978). A theory of memory retrieval. Psychological Review.
    Navarro & Fuss (2009). Fast and accurate calculations for first-passage
        times in Wiener diffusion models. Journal of Mathematical Psychology.
    Ratcliff, Smith, Brown & McKoon (2016). Diffusion Decision Model: Current
        Issues and History. Trends in Cognitive Sciences.
"""

from alf.ddm.wiener import (
    DDMParams,
    DDMResult,
    wiener_log_density,
    wiener_log_density_batch,
    ddm_log_likelihood,
    ddm_nll,
    simulate_ddm,
)
from alf.ddm.bridge import (
    efe_to_ddm,
    ddm_to_policy_probs,
)

__all__ = [
    # Wiener density
    "DDMParams",
    "DDMResult",
    "wiener_log_density",
    "wiener_log_density_batch",
    "ddm_log_likelihood",
    "ddm_nll",
    "simulate_ddm",
    # Bridge
    "efe_to_ddm",
    "ddm_to_policy_probs",
]
