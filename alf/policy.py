"""Policy selection and action selection for Active Inference.

Implements the policy selection mechanism described in Smith et al. (2022):

    P(pi) = sigma( -gamma * G(pi) + ln E(pi) )

Where:
    G(pi) = Expected Free Energy of policy pi
    E(pi) = Prior probability of policy pi (habits)
    gamma = Precision (inverse temperature) over policies
    sigma = Softmax function

This module provides only the analytic (non-BP) policy functions:
    - select_action: softmax policy selection
    - update_habits: Dirichlet-like habit learning
    - update_precision: adaptive precision

The BP-based evaluate_policies() stays in pgmax/aif where it belongs.
For analytic EFE evaluation, use sequential_efe.evaluate_all_policies_sequential
or jax_native.jax_evaluate_all_actions.

References:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
"""

from typing import Optional

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def select_action(
    G: np.ndarray,
    E: np.ndarray,
    gamma: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[int, np.ndarray]:
    """Select an action by sampling from the posterior over policies.

    Implements: P(pi) = sigma( -gamma * G(pi) + ln E(pi) )

    Args:
        G: Expected free energy for each policy, shape (num_policies,).
        E: Policy prior (habits), shape (num_policies,).
        gamma: Policy precision (inverse temperature). Default 1.0.
        rng: Random state for reproducible sampling.

    Returns:
        Tuple of (selected_policy_index, policy_probabilities).
    """
    log_E = np.log(np.clip(E, 1e-16, None))
    log_posterior = -gamma * G + log_E
    policy_probs = _softmax(log_posterior)

    if rng is not None:
        selected = rng.choice(len(policy_probs), p=policy_probs)
    else:
        selected = np.random.choice(len(policy_probs), p=policy_probs)

    return int(selected), policy_probs


def update_habits(
    E: np.ndarray,
    selected_policy: int,
    outcome_valence: float,
    learning_rate: float = 0.1,
) -> np.ndarray:
    """Update the policy prior (habits) based on outcome.

    Implements Dirichlet-like habit learning:
        E(pi) <- E(pi) + eta * valence   (for the selected policy)

    Args:
        E: Current policy prior, shape (num_policies,).
        selected_policy: Index of the policy that was executed.
        outcome_valence: Scalar indicating how good the outcome was.
        learning_rate: How quickly habits form. Default 0.1.

    Returns:
        Updated E vector (normalized to sum to 1).
    """
    E = E.copy()
    E[selected_policy] += learning_rate * outcome_valence
    E = np.clip(E, 1e-8, None)
    E = E / E.sum()
    return E


def update_precision(
    gamma: float,
    prediction_error: float,
    learning_rate: float = 0.3,
    min_gamma: float = 0.1,
    max_gamma: float = 10.0,
) -> float:
    """Adapt policy precision based on prediction error.

    Args:
        gamma: Current precision.
        prediction_error: Magnitude of recent prediction error.
        learning_rate: Update rate. Default 0.3.
        min_gamma: Minimum precision. Default 0.1.
        max_gamma: Maximum precision. Default 10.0.

    Returns:
        Updated gamma value.
    """
    target = 1.0 / (1.0 + prediction_error)
    gamma = (1.0 - learning_rate) * gamma + learning_rate * target
    return float(np.clip(gamma, min_gamma, max_gamma))
