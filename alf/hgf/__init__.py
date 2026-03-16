"""Hierarchical Gaussian Filter (HGF) for Active Inference.

JAX-native reimplementation of the Hierarchical Gaussian Filter (Mathys et al.
2011, 2014). The HGF is a Bayesian perception model that tracks hidden states
at multiple timescales via precision-weighted prediction errors — the
continuous-valued analog of alf's discrete HMM forward algorithm.

Architecturally, HGF is a generative model where each level provides the
volatility (step-size) for the level below. Inference uses Gaussian sufficient
statistics (mu, pi) updated via the same jax.lax.scan pattern as alf.learning.

Modules:
    updates: Core HGF update equations (2-level binary, 3-level continuous).
    bridge: Connect HGF perception to alf's discrete action selection.
    learning: Differentiable parameter learning via jax.grad through surprise.

References:
    Mathys, Daunizeau, Friston & Stephan (2011). A Bayesian foundation for
        individual learning under uncertainty. Frontiers in Human Neuroscience.
    Mathys, Lomakina, Daunizeau et al. (2014). Uncertainty in perception and
        the Hierarchical Gaussian Filter. Frontiers in Human Neuroscience.
    Weber, Imbach, Legrand et al. (2024). The generalized Hierarchical Gaussian
        Filter. arXiv:2305.10937.
"""

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    HGFBelief,
    HGFResult,
    binary_hgf_update,
    binary_hgf,
    continuous_hgf_update,
    continuous_hgf,
    binary_hgf_surprise,
    continuous_hgf_surprise,
)
from alf.hgf.bridge import (
    discretize_belief,
    hgf_to_categorical,
    HGFPerceptualAgent,
)
from alf.hgf.learning import (
    binary_hgf_nll,
    continuous_hgf_nll,
    learn_binary_hgf,
    learn_continuous_hgf,
    BinaryHGFLearningResult,
    ContinuousHGFLearningResult,
)

__all__ = [
    # Updates
    "BinaryHGFParams",
    "ContinuousHGFParams",
    "HGFBelief",
    "HGFResult",
    "binary_hgf_update",
    "binary_hgf",
    "continuous_hgf_update",
    "continuous_hgf",
    "binary_hgf_surprise",
    "continuous_hgf_surprise",
    # Bridge
    "discretize_belief",
    "hgf_to_categorical",
    "HGFPerceptualAgent",
    # Learning
    "binary_hgf_nll",
    "continuous_hgf_nll",
    "learn_binary_hgf",
    "learn_continuous_hgf",
    "BinaryHGFLearningResult",
    "ContinuousHGFLearningResult",
]
