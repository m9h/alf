"""Analytic Active Inference agent using JAX-native EFE computation.

Unlike pgmax/aif's ActiveInferenceAgent which uses belief propagation on
factor graphs, this agent uses analytic EFE computation (sequential rollout
or single-step vmap). No PGMax dependency.

The agent loop follows Smith et al. (2022) Figure 1:

    1. Observe -> 2. Infer states -> 3. Evaluate policies ->
    4. Select action -> 5. Act -> 6. Learn -> repeat

Example usage:
    >>> from alf import agent, generative_model
    >>> import numpy as np
    >>>
    >>> gm = generative_model.GenerativeModel(A=A, B=B, C=C, D=D)
    >>> aia = agent.AnalyticAgent(gm, gamma=4.0)
    >>>
    >>> obs = [0]
    >>> for t in range(10):
    ...     action, info = aia.step(obs)
    ...     obs = environment.step(action)
    ...     aia.learn(reward)

References:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
"""

from typing import Any, Optional

import numpy as np

from alf.generative_model import GenerativeModel
from alf import policy as alf_policy
from alf.sequential_efe import evaluate_all_policies_sequential


class AnalyticAgent:
    """An Active Inference agent using analytic (non-BP) EFE computation.

    Maintains beliefs about hidden states and selects actions by
    minimizing expected free energy over candidate policies using
    sequential forward rollout.

    Args:
        gm: The generative model (POMDP).
        gamma: Policy precision (inverse temperature). Default 4.0.
        learning_rate: Rate of habit learning. Default 0.1.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        gm: GenerativeModel,
        gamma: float = 4.0,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.gm = gm
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)

        # Initialize beliefs from priors
        self.beliefs = [d.copy() for d in gm.D]

        # Policy prior (habits)
        self.E = gm.E.copy()

        # History for analysis
        self.belief_history: list[list[np.ndarray]] = []
        self.action_history: list[int] = []
        self.efe_history: list[np.ndarray] = []
        self.policy_prob_history: list[np.ndarray] = []

    def step(
        self,
        observation: list[int],
    ) -> tuple[int, dict[str, Any]]:
        """Perform one step of the Active Inference loop.

        1. Update beliefs given observation (analytic Bayesian update)
        2. Evaluate expected free energy for all policies (sequential rollout)
        3. Select action from posterior over policies

        Args:
            observation: List of observation indices, one per modality.

        Returns:
            Tuple of (action_index, info_dict).
        """
        # 1. Belief updating (analytic Bayesian update)
        for f in range(self.gm.num_factors):
            for m in range(self.gm.num_modalities):
                a_matrix = self.gm.A[m]
                if a_matrix.ndim == 2:
                    likelihood = a_matrix[observation[m], :]
                    posterior = self.beliefs[f] * likelihood
                    posterior = np.clip(posterior, 1e-16, None)
                    self.beliefs[f] = posterior / posterior.sum()

        self.belief_history.append([b.copy() for b in self.beliefs])

        # 2. Policy evaluation (sequential EFE)
        G = evaluate_all_policies_sequential(self.gm, self.beliefs)
        self.efe_history.append(G.copy())

        # 3. Action selection
        policy_idx, policy_probs = alf_policy.select_action(
            G, self.E, self.gamma, rng=self.rng,
        )
        self.policy_prob_history.append(policy_probs.copy())

        # Extract the first action from the selected policy
        selected_policy = self.gm.policies[policy_idx]
        action = int(selected_policy[0, 0])
        self.action_history.append(action)

        info = {
            "beliefs": [b.copy() for b in self.beliefs],
            "G": G,
            "policy_probs": policy_probs,
            "selected_policy": policy_idx,
        }
        return action, info

    def learn(self, outcome_valence: float) -> None:
        """Update habits and precision based on outcome.

        Args:
            outcome_valence: How good the outcome was.
        """
        if self.action_history:
            last_policy_idx = (
                self.policy_prob_history[-1].argmax()
                if self.policy_prob_history
                else 0
            )
            self.E = alf_policy.update_habits(
                self.E, last_policy_idx, outcome_valence,
                learning_rate=self.learning_rate,
            )

    def update_precision(self, prediction_error: float) -> None:
        """Adapt policy precision based on prediction error."""
        self.gamma = alf_policy.update_precision(
            self.gamma, prediction_error,
        )

    def reset(self) -> None:
        """Reset beliefs to priors (keep learned habits)."""
        self.beliefs = [d.copy() for d in self.gm.D]
        self.belief_history.clear()
        self.action_history.clear()
        self.efe_history.clear()
        self.policy_prob_history.clear()

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the agent's internal state."""
        return {
            "beliefs": {
                f"factor_{f}": self.beliefs[f].tolist()
                for f in range(self.gm.num_factors)
            },
            "gamma": self.gamma,
            "E": self.E.tolist(),
            "num_actions_taken": len(self.action_history),
        }
