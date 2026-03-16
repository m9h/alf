"""ALF — Active inference/Learning Framework.

A standalone JAX-native library for Active Inference on discrete POMDPs.
No PGMax dependency. Fully differentiable via jax.grad, vectorizable
via jax.vmap, and GPU-acceleratable via jax.jit.

Modules:
    generative_model: Data-only POMDP representation (A, B, C, D, E matrices).
    free_energy: VFE, EFE decomposition, generalized free energy.
    sequential_efe: Forward-rollout EFE with value-of-information.
    jax_native: JAX-native policy functions and BatchAgent.
    deep_aif: Neural network generative models.
    learning: Differentiable HMM learning via forward algorithm.
    policy: Action selection, habit learning, precision updating.
    agent: AnalyticAgent wrapping the full AIF loop.
    multitask: Hierarchical/compositional generative models for task batteries.
    multitask_agent: MultitaskAgent for switching between cognitive tasks.
    hgf: Hierarchical Gaussian Filter for continuous belief updating.
    metacognition: Meta-d' estimation and precision calibration.
    ddm: Drift-Diffusion Models with Navarro-Fuss likelihood.
    normative: Normative modeling with Bayesian Linear Regression.

Example:
    >>> import alf
    >>> gm = alf.GenerativeModel(
    ...     A=likelihood_matrix,
    ...     B=transition_matrices,
    ...     C=preference_vector,
    ...     D=prior_beliefs,
    ... )
    >>> agent = alf.AnalyticAgent(gm)
    >>> action, info = agent.step(observation)

Positioning:
    | Library | Backend | Strength | Best for |
    |---------|---------|----------|----------|
    | pymdp   | NumPy   | Reference impl    | Standard AIF experiments |
    | ALF     | JAX     | Differentiable, GPU | Learning, scaling, deep AIF |
    | pgmax/aif | PGMax BP | Message passing | Large state spaces |
"""

from alf.generative_model import GenerativeModel
from alf.policy import select_action
from alf.policy import update_habits
from alf.policy import update_precision
from alf.agent import AnalyticAgent
from alf.jax_native import BatchAgent
from alf.sequential_efe import sequential_efe
from alf.sequential_efe import evaluate_all_policies_sequential
from alf.sequential_efe import select_action_sequential
from alf.learning import LearnableGenerativeModel
from alf.learning import learn_model
from alf.learning import learn_from_agent_data
from alf.learning import analytic_nll
from alf.deep_aif import DeepGenerativeModel
from alf.deep_aif import deep_analytic_nll
from alf.deep_aif import learn_deep_model
from alf.deep_aif import init_encoder
from alf.deep_aif import init_transition
from alf.deep_aif import encode
from alf.deep_aif import predict_transition
from alf.deep_aif import extract_A_matrix
from alf.deep_aif import DecoderGenerativeModel
from alf.deep_aif import DecoderLearningResult
from alf.deep_aif import init_decoder
from alf.deep_aif import decode
from alf.deep_aif import decoder_log_likelihood
from alf.deep_aif import decoder_analytic_nll
from alf.deep_aif import learn_decoder_model
from alf.deep_aif import init_gaussian_decoder
from alf.deep_aif import gaussian_decode
from alf.deep_aif import gaussian_log_likelihood
from alf.hierarchical import HierarchicalLevel
from alf.hierarchical import HierarchicalGenerativeModel
from alf.hierarchical import hierarchical_infer
from alf.hierarchical import hierarchical_efe
from alf.hierarchical import evaluate_all_policies_hierarchical
from alf.hierarchical import HierarchicalAgent
from alf.multitask import MultitaskGenerativeModel
from alf.multitask import CompositionalModel
from alf.multitask import build_compositional_battery
from alf.multitask import build_simple_task_pair
from alf.multitask import multifactor_sequential_efe
from alf.multitask import evaluate_all_policies_multifactor
from alf.multitask import YANG_TASK_DEFINITIONS
from alf.multitask_agent import MultitaskAgent

# Subpackages (imported as namespaces)
from alf import hgf
from alf import ddm
from alf import normative
from alf import metacognition as metacognition_module

# Key exports from new modules
from alf.hgf import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf,
    continuous_hgf,
)
from alf.ddm import (
    DDMParams,
    wiener_log_density,
    simulate_ddm,
)
from alf.metacognition import (
    MetaDResult,
    fit_meta_d_mle,
    m_ratio_to_gamma,
)
from alf.normative import (
    normative_model,
    normative_model_vmap,
)

__version__ = "0.1.0"

__all__ = [
    "GenerativeModel",
    "select_action",
    "update_habits",
    "update_precision",
    "AnalyticAgent",
    "BatchAgent",
    "sequential_efe",
    "evaluate_all_policies_sequential",
    "select_action_sequential",
    "LearnableGenerativeModel",
    "learn_model",
    "learn_from_agent_data",
    "analytic_nll",
    "DeepGenerativeModel",
    "deep_analytic_nll",
    "learn_deep_model",
    "init_encoder",
    "init_transition",
    "encode",
    "predict_transition",
    "extract_A_matrix",
    "DecoderGenerativeModel",
    "DecoderLearningResult",
    "init_decoder",
    "decode",
    "decoder_log_likelihood",
    "decoder_analytic_nll",
    "learn_decoder_model",
    "init_gaussian_decoder",
    "gaussian_decode",
    "gaussian_log_likelihood",
    "HierarchicalLevel",
    "HierarchicalGenerativeModel",
    "hierarchical_infer",
    "hierarchical_efe",
    "evaluate_all_policies_hierarchical",
    "HierarchicalAgent",
    "MultitaskGenerativeModel",
    "CompositionalModel",
    "build_compositional_battery",
    "build_simple_task_pair",
    "multifactor_sequential_efe",
    "evaluate_all_policies_multifactor",
    "YANG_TASK_DEFINITIONS",
    "MultitaskAgent",
    # HGF
    "hgf",
    "BinaryHGFParams",
    "ContinuousHGFParams",
    "binary_hgf",
    "continuous_hgf",
    # DDM
    "ddm",
    "DDMParams",
    "wiener_log_density",
    "simulate_ddm",
    # Metacognition
    "metacognition_module",
    "MetaDResult",
    "fit_meta_d_mle",
    "m_ratio_to_gamma",
    # Normative
    "normative",
    "normative_model",
    "normative_model_vmap",
]
