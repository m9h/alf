"""Unified cognitive task battery (Yang et al. 2019).

Provides a single entry point to run multiple cognitive neuroscience tasks
from the Yang et al. (2019) battery, all specified as discrete POMDPs
for Active Inference agents.

Current tasks:
    - context_dm: Context-Dependent Decision Making (Mante et al. 2013)
    - delayed_match: Delayed Match-to-Sample (working memory)
    - go_nogo: Go/NoGo with Anti variant (inhibitory control)

References:
    Yang, G.R., Joglekar, M.R., Song, H.F., Newsome, W.T. & Wang, X.-J.
        (2019). Task representations in neural networks trained to perform
        many cognitive tasks. Nature Neuroscience, 22, 297-306.
"""

from typing import Any, Optional

from alf.benchmarks.context_dm import (
    build_context_dm_model,
    run_context_dm,
)
from alf.benchmarks.delayed_match import (
    build_delayed_match_model,
    run_delayed_match,
)
from alf.benchmarks.go_nogo import (
    build_go_nogo_model,
    run_go_nogo,
)


# Registry of all available tasks
BATTERY = {
    "context_dm": build_context_dm_model,
    "delayed_match": build_delayed_match_model,
    "go_nogo": build_go_nogo_model,
}

# Registry of run functions
_RUN_FUNCTIONS = {
    "context_dm": run_context_dm,
    "delayed_match": run_delayed_match,
    "go_nogo": run_go_nogo,
}

# Human-readable task descriptions
TASK_DESCRIPTIONS = {
    "context_dm": "Context-Dependent Decision Making (Mante et al. 2013)",
    "delayed_match": "Delayed Match-to-Sample (working memory)",
    "go_nogo": "Go/NoGo with Anti variant (inhibitory control)",
}


def run_battery(
    n_trials: int = 50,
    gamma: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
    tasks: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Run the full cognitive task battery and report performance.

    Runs each task independently with the specified parameters and
    collects performance metrics.

    Args:
        n_trials: Number of trials per task.
        gamma: Agent policy precision (inverse temperature).
        seed: Random seed for reproducibility.
        verbose: Print trial-by-trial details.
        tasks: List of task names to run. If None, runs all tasks.

    Returns:
        Dict mapping task name -> results dict. Each results dict contains
        task-specific metrics (accuracy, etc.) plus a 'trial_log' key.
    """
    if tasks is None:
        tasks = list(_RUN_FUNCTIONS.keys())

    for task_name in tasks:
        if task_name not in _RUN_FUNCTIONS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(_RUN_FUNCTIONS.keys())}"
            )

    battery_results = {}

    for task_name in tasks:
        if verbose:
            desc = TASK_DESCRIPTIONS.get(task_name, task_name)
            print(f"\n{'=' * 60}")
            print(f"Running: {desc}")
            print(f"{'=' * 60}")

        run_fn = _RUN_FUNCTIONS[task_name]
        results = run_fn(
            num_trials=n_trials,
            gamma=gamma,
            seed=seed,
            verbose=verbose,
        )
        battery_results[task_name] = results

    if verbose:
        print(f"\n{'=' * 60}")
        print("BATTERY SUMMARY")
        print(f"{'=' * 60}")
        for task_name, results in battery_results.items():
            acc = results.get("accuracy", results.get("reward_rate", 0.0))
            mr = results.get("mean_reward", 0.0)
            print(f"  {task_name:20s}  accuracy={acc:.2%}  mean_reward={mr:+.2f}")

    return battery_results


def get_task_list() -> list[str]:
    """Return list of available task names."""
    return list(BATTERY.keys())


def get_task_model(task_name: str, **kwargs):
    """Build a generative model for a specific task.

    Args:
        task_name: Name of the task.
        **kwargs: Task-specific parameters.

    Returns:
        GenerativeModel for the specified task.
    """
    if task_name not in BATTERY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(BATTERY.keys())}"
        )
    return BATTERY[task_name](**kwargs)
