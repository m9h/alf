"""Multitask active inference agent for the Yang et al. (2019) task battery.

Extends AnalyticAgent to handle hierarchical task switching, rule-dependent
preference modulation, and trial phase tracking across 20 cognitive tasks.

The agent maintains:
    - A MultitaskGenerativeModel with the full task battery
    - A current-task pointer determining which A/C/D matrices are active
    - Per-task habit memories (E vectors) that persist across task switches
    - A posterior distribution over task identity (for task inference)

Usage:
    >>> from alf.multitask import build_compositional_battery
    >>> from alf.multitask_agent import MultitaskAgent
    >>>
    >>> mtm = build_compositional_battery(["Go", "Anti", "DelayGo"])
    >>> agent = MultitaskAgent(mtm, gamma=4.0)
    >>>
    >>> agent.set_task("Go")
    >>> for trial in range(50):
    ...     result = agent.run_trial(env, "Go")

References:
    Yang, Joiner, Bhatt, et al. (2019). Task representations in neural
        networks trained to perform many cognitive tasks. Nature Neuroscience.
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active Inference.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from alf.generative_model import GenerativeModel
from alf.multitask import (
    MultitaskGenerativeModel,
    evaluate_all_policies_multifactor,
)
from alf import policy as alf_policy


class MultitaskAgent:
    """Active inference agent for the cognitive task battery.

    Extends the AnalyticAgent pattern to handle:
        - Task switching (reset beliefs but keep learned parameters)
        - Rule-dependent preference modulation
        - Trial phase tracking
        - Per-task habit memories
        - Task inference from observations

    The agent wraps a MultitaskGenerativeModel and delegates policy evaluation
    to the multi-factor EFE computation when models have multiple factors
    and modalities, or to the standard single-factor EFE for simpler models.

    Args:
        multitask_model: MultitaskGenerativeModel containing all tasks.
        gamma: Policy precision (inverse temperature). Default 4.0.
        learning_rate: Rate of habit learning. Default 0.1.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        multitask_model: MultitaskGenerativeModel,
        gamma: float = 4.0,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.multitask_model = multitask_model
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)

        # Current task state
        self._current_task: Optional[str] = None
        self._current_gm: Optional[GenerativeModel] = None

        # Per-task habits (E vectors), initialized lazily on first use
        self._task_habits: dict[str, np.ndarray] = {}

        # Current beliefs (per factor)
        self.beliefs: list[np.ndarray] = []

        # History for the current task/trial
        self.belief_history: list[list[np.ndarray]] = []
        self.action_history: list[int] = []
        self.efe_history: list[np.ndarray] = []
        self.policy_prob_history: list[np.ndarray] = []

        # Cross-task performance log
        self.performance_log: list[dict[str, Any]] = []

        # Task inference state
        self._task_posterior: np.ndarray = multitask_model.task_prior.copy()
        self._observation_buffer: list[list[int]] = []

    @property
    def current_task(self) -> Optional[str]:
        """Name of the currently active task."""
        return self._current_task

    @property
    def task_posterior(self) -> np.ndarray:
        """Current posterior distribution over task identity."""
        return self._task_posterior.copy()

    def set_task(self, task_name: str) -> None:
        """Switch to a new task.

        Resets beliefs to the new task's priors but preserves learned habits
        for the previous task (and loads any previously learned habits for
        the new task).

        Args:
            task_name: Name of the task to switch to.

        Raises:
            KeyError: If task_name is not in the multitask model.
        """
        # Save habits for the current task before switching
        if self._current_task is not None and self._current_gm is not None:
            self._task_habits[self._current_task] = self._get_current_E()

        # Load new task model
        self._current_gm = self.multitask_model.get_model_for_task(task_name)
        self._current_task = task_name

        # Reset beliefs to new task's priors
        self.beliefs = [d.copy() for d in self._current_gm.D]

        # Load or initialize habits for the new task
        if task_name not in self._task_habits:
            self._task_habits[task_name] = self._current_gm.E.copy()

        # Clear per-trial histories
        self.belief_history.clear()
        self.action_history.clear()
        self.efe_history.clear()
        self.policy_prob_history.clear()

        # Reset observation buffer for task inference
        self._observation_buffer.clear()

    def _get_current_E(self) -> np.ndarray:
        """Get the current habit vector for the active task."""
        if self._current_task in self._task_habits:
            return self._task_habits[self._current_task].copy()
        elif self._current_gm is not None:
            return self._current_gm.E.copy()
        else:
            raise RuntimeError("No task is currently set.")

    def step(
        self,
        observation: list[int],
        rule_input: Optional[int] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Perform one step of the active inference loop.

        1. Update beliefs given observation (Bayesian update)
        2. Evaluate expected free energy for all policies
        3. Select action from posterior over policies

        For context-dependent tasks, an optional rule_input can modulate
        which modality/context is attended. This updates the context factor
        beliefs.

        Args:
            observation: List of observation indices, one per modality.
            rule_input: Optional integer specifying context/rule state.
                For context-dependent tasks, 0 = attend modality 1,
                1 = attend modality 2.

        Returns:
            Tuple of (action_index, info_dict).

        Raises:
            RuntimeError: If no task has been set.
        """
        if self._current_gm is None or self._current_task is None:
            raise RuntimeError("No task set. Call set_task() before stepping.")

        gm = self._current_gm
        eps = 1e-16

        # Optional: update context beliefs from rule input
        if rule_input is not None and gm.num_factors >= 3:
            # Context factor is typically factor index 2
            ctx_beliefs = np.ones(gm.num_states[2]) * eps
            ctx_idx = min(rule_input, gm.num_states[2] - 1)
            ctx_beliefs[ctx_idx] = 1.0
            ctx_beliefs /= ctx_beliefs.sum()
            self.beliefs[2] = ctx_beliefs

        # Buffer observation for task inference
        self._observation_buffer.append(observation)

        # 1. Belief updating (Bayesian update)
        for f in range(gm.num_factors):
            for m in range(gm.num_modalities):
                A_m = gm.A[m]
                o_m = observation[m]

                if A_m.ndim == 2 and gm.num_factors == 1:
                    # Single-factor, 2D A matrix
                    likelihood = A_m[o_m, :]
                    posterior = self.beliefs[f] * likelihood
                    posterior = np.clip(posterior, eps, None)
                    self.beliefs[f] = posterior / posterior.sum()
                elif A_m.ndim > 2:
                    # Multi-factor tensor A matrix: marginalize over other
                    # factors to get P(o_m | s_f) for Bayesian update on f.
                    # A_slice shape: (ns_f0, ns_f1, ..., ns_{N-1})
                    A_slice = A_m[o_m]

                    # Use einsum to contract all factors except f.
                    # E.g. for 3 factors and f=1: "abc,a,c->b"
                    ndim = gm.num_factors
                    letters = [chr(ord("a") + i) for i in range(ndim)]
                    a_sub = "".join(letters)
                    operands = [A_slice]
                    subs = [a_sub]
                    for other_f in range(ndim):
                        if other_f != f:
                            subs.append(letters[other_f])
                            operands.append(self.beliefs[other_f])
                    out_sub = letters[f]
                    einsum_str = ",".join(subs) + "->" + out_sub
                    likelihood_f = np.einsum(einsum_str, *operands)

                    if likelihood_f.shape[0] == gm.num_states[f]:
                        posterior = self.beliefs[f] * likelihood_f
                        posterior = np.clip(posterior, eps, None)
                        self.beliefs[f] = posterior / posterior.sum()

        self.belief_history.append([b.copy() for b in self.beliefs])

        # 2. Policy evaluation
        if gm.num_factors == 1 and gm.num_modalities == 1:
            # Use fast single-factor path
            from alf.sequential_efe import evaluate_all_policies_sequential

            G = evaluate_all_policies_sequential(gm, self.beliefs)
        else:
            # Use multi-factor path
            G = evaluate_all_policies_multifactor(gm, self.beliefs)

        self.efe_history.append(G.copy())

        # 3. Action selection
        E = self._get_current_E()
        policy_idx, policy_probs = alf_policy.select_action(
            G,
            E,
            self.gamma,
            rng=self.rng,
        )
        self.policy_prob_history.append(policy_probs.copy())

        # Extract the first action from the selected policy
        selected_policy = gm.policies[policy_idx]
        # For multi-factor: return action for the first controllable factor
        # Typically the phase factor (factor 1) for compositional models
        if gm.num_factors > 1:
            # Find the factor with the most actions (the controllable one)
            action_factor = max(
                range(gm.num_factors),
                key=lambda f: gm.num_actions[f],
            )
            action = int(selected_policy[0, action_factor])
        else:
            action = int(selected_policy[0, 0])

        self.action_history.append(action)

        info = {
            "beliefs": [b.copy() for b in self.beliefs],
            "G": G,
            "policy_probs": policy_probs,
            "selected_policy": policy_idx,
            "task": self._current_task,
        }
        return action, info

    def learn(self, outcome_valence: float) -> None:
        """Update habits based on outcome for the current task.

        Args:
            outcome_valence: How good the outcome was (positive = good).
        """
        if self._current_task is None:
            return

        E = self._get_current_E()
        if self.policy_prob_history:
            last_policy_idx = self.policy_prob_history[-1].argmax()
            E = alf_policy.update_habits(
                E,
                last_policy_idx,
                outcome_valence,
                learning_rate=self.learning_rate,
            )
            self._task_habits[self._current_task] = E

    def update_precision(self, prediction_error: float) -> None:
        """Adapt policy precision based on prediction error."""
        self.gamma = alf_policy.update_precision(
            self.gamma,
            prediction_error,
        )

    def reset(self) -> None:
        """Reset beliefs to the current task's priors (keep learned habits)."""
        if self._current_gm is not None:
            self.beliefs = [d.copy() for d in self._current_gm.D]
        self.belief_history.clear()
        self.action_history.clear()
        self.efe_history.clear()
        self.policy_prob_history.clear()
        self._observation_buffer.clear()

    def infer_task(self) -> np.ndarray:
        """Infer which task is active from buffered observations.

        Uses the observation buffer accumulated since the last set_task()
        or reset() call.

        Returns:
            Posterior distribution over tasks.
        """
        if not self._observation_buffer:
            return self.multitask_model.task_prior.copy()

        self._task_posterior = self.multitask_model.infer_task(self._observation_buffer)
        return self._task_posterior.copy()

    def run_trial(
        self,
        env: Any,
        task_name: str,
        max_steps: int = 10,
        rule_input: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run a complete trial on a given task.

        Sets the task, resets the environment and agent beliefs, then runs
        the perception-action loop until the environment signals done or
        max_steps is reached.

        The environment must implement:
            - reset() -> initial_observation (list of int)
            - step(action) -> (observation, reward, done)

        Args:
            env: Environment object with reset() and step() methods.
            task_name: Name of the task to run.
            max_steps: Maximum number of steps per trial.
            rule_input: Optional rule/context input for context-dependent tasks.

        Returns:
            Dictionary with trial results including actions, rewards,
            beliefs, and performance.
        """
        self.set_task(task_name)
        obs = env.reset()

        # Ensure observation is a list
        if isinstance(obs, (int, np.integer)):
            obs = [int(obs)]

        total_reward = 0.0
        actions_taken = []
        observations = [obs]
        rewards = []

        for step_idx in range(max_steps):
            action, info = self.step(obs, rule_input=rule_input)
            result = env.step(action)

            if len(result) == 3:
                obs, reward, done = result
            elif len(result) == 2:
                obs, reward = result
                done = False
            else:
                raise ValueError(
                    f"Environment step() returned {len(result)} values, "
                    f"expected 2 or 3."
                )

            if isinstance(obs, (int, np.integer)):
                obs = [int(obs)]

            total_reward += reward
            actions_taken.append(action)
            observations.append(obs)
            rewards.append(reward)

            if done:
                break

        # Learn from outcome
        self.learn(total_reward)

        trial_result = {
            "task": task_name,
            "actions": actions_taken,
            "observations": observations,
            "rewards": rewards,
            "total_reward": total_reward,
            "num_steps": len(actions_taken),
            "final_beliefs": [b.copy() for b in self.beliefs],
        }

        self.performance_log.append(trial_result)
        return trial_result

    def run_battery(
        self,
        envs: dict[str, Any],
        n_trials_per_task: int = 50,
        task_order: Optional[list[str]] = None,
        interleaved: bool = False,
    ) -> dict[str, Any]:
        """Run the full Yang et al. task battery.

        Supports two scheduling modes:
            - blocked: run all trials for each task before moving to the next
            - interleaved: alternate between tasks each trial

        Args:
            envs: Dictionary mapping task names to environment objects.
            n_trials_per_task: Number of trials per task. Default 50.
            task_order: Order in which to run tasks. If None, uses the
                order from the multitask model.
            interleaved: If True, interleave tasks rather than blocking.

        Returns:
            Dictionary with per-task performance metrics and overall summary.
        """
        if task_order is None:
            task_order = [
                name for name in self.multitask_model.task_names if name in envs
            ]

        if not task_order:
            raise ValueError("No tasks to run. Ensure envs keys match task names.")

        all_results: dict[str, list[dict]] = {name: [] for name in task_order}

        if interleaved:
            # Interleaved: cycle through tasks
            for trial_idx in range(n_trials_per_task):
                for task_name in task_order:
                    env = envs[task_name]
                    result = self.run_trial(env, task_name)
                    result["global_trial_idx"] = trial_idx * len(
                        task_order
                    ) + task_order.index(task_name)
                    all_results[task_name].append(result)
        else:
            # Blocked: all trials for each task sequentially
            global_idx = 0
            for task_name in task_order:
                env = envs[task_name]
                for trial_idx in range(n_trials_per_task):
                    result = self.run_trial(env, task_name)
                    result["global_trial_idx"] = global_idx
                    all_results[task_name].append(result)
                    global_idx += 1

        # Compute per-task metrics
        task_metrics = {}
        for task_name in task_order:
            results = all_results[task_name]
            rewards = [r["total_reward"] for r in results]
            task_metrics[task_name] = {
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "reward_rate": (
                    sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
                ),
                "n_trials": len(results),
                "mean_steps": (
                    float(np.mean([r["num_steps"] for r in results]))
                    if results
                    else 0.0
                ),
            }

        # Overall summary
        all_rewards = []
        for results in all_results.values():
            all_rewards.extend(r["total_reward"] for r in results)

        summary = {
            "task_metrics": task_metrics,
            "overall_mean_reward": (
                float(np.mean(all_rewards)) if all_rewards else 0.0
            ),
            "overall_reward_rate": (
                sum(1 for r in all_rewards if r > 0) / len(all_rewards)
                if all_rewards
                else 0.0
            ),
            "total_trials": len(all_rewards),
            "n_tasks": len(task_order),
            "task_order": task_order,
            "mode": self.multitask_model.mode,
            "trial_log": all_results,
        }

        return summary

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the agent's internal state."""
        summary: dict[str, Any] = {
            "current_task": self._current_task,
            "gamma": self.gamma,
            "num_actions_taken": len(self.action_history),
            "task_posterior": self._task_posterior.tolist(),
        }

        if self.beliefs:
            summary["beliefs"] = {
                f"factor_{f}": self.beliefs[f].tolist()
                for f in range(len(self.beliefs))
            }

        if self._task_habits:
            summary["tasks_with_learned_habits"] = list(self._task_habits.keys())

        return summary
