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
]
