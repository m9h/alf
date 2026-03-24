"""Convenience constructors for Yang et al. (2019) cognitive task battery.

Provides a thin layer over NeurogymAdapter to create ready-to-use
discretized environments for the 20 cognitive neuroscience tasks studied
in:

    Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., &
        Wang, X.-J. (2019). Task representations in neural networks
        trained to perform many cognitive tasks. Nature Neuroscience,
        22(2), 297-306.

Usage:
    >>> from alf.envs.cognitive_tasks import make_task, make_all_tasks
    >>> adapter = make_task("dm1", n_bins=4)
    >>> obs = adapter.reset()
    >>> obs, reward, done = adapter.step(0)
"""

from __future__ import annotations

from typing import Any

from alf.envs.neurogym_bridge import NeurogymAdapter, _require_neurogym

# ---------------------------------------------------------------------------
# Mapping from short names to NeuroGym registered environment IDs.
#
# NeuroGym uses the naming convention "TaskName-v0".  The short names
# follow Yang et al. (2019) Table 1 / Figure 1.
# ---------------------------------------------------------------------------

YANG19_TASKS: dict[str, str] = {
    # Go / Anti family
    "go": "PerceptualDecisionMaking-v0",
    "rtgo": "GoNogo-v0",
    "dlygo": "DelayedMatchToSample-v0",
    "anti": "AntiReach-v0",
    "rtanti": "AntiReach-v0",  # variant; kwargs can adjust
    "dlyanti": "AntiReach-v0",  # variant; kwargs can adjust
    # Decision-making family
    "dm1": "PerceptualDecisionMaking-v0",
    "dm2": "PerceptualDecisionMaking-v0",
    "ctxdm1": "PerceptualDecisionMaking-v0",
    "ctxdm2": "PerceptualDecisionMaking-v0",
    "multidm": "PerceptualDecisionMaking-v0",
    "dlydm1": "DelayedMatchToSample-v0",
    "dlydm2": "DelayedMatchToSample-v0",
    "ctxdlydm1": "DelayedMatchToSample-v0",
    "ctxdlydm2": "DelayedMatchToSample-v0",
    # Match-to-sample family
    "dms": "DelayedMatchToSample-v0",
    "dnms": "DelayedMatchToSample-v0",
    "dmc": "DelayedMatchCategory-v0",
    "dnmc": "DelayedMatchCategory-v0",
}


# Some tasks benefit from specific timing or configuration overrides.
_TASK_KWARGS: dict[str, dict[str, Any]] = {
    # These are intentionally empty; users can override via **kwargs.
}


def make_task(
    task_name: str,
    discretization: str = "bin",
    n_bins: int = 5,
    n_clusters: int = 20,
    seed: int = 42,
    **kwargs: Any,
) -> NeurogymAdapter:
    """Create a discretized NeuroGym task ready for ALF.

    Args:
        task_name: One of the Yang et al. (2019) short task names.
            Valid names: 'go', 'rtgo', 'dlygo', 'anti', 'rtanti',
            'dlyanti', 'dm1', 'dm2', 'ctxdm1', 'ctxdm2', 'multidm',
            'dlydm1', 'dlydm2', 'ctxdlydm1', 'ctxdlydm2', 'dms',
            'dnms', 'dmc', 'dnmc'.
        discretization: Observation discretization strategy
            (``"bin"`` or ``"kmeans"``).
        n_bins: Bins per dimension for the ``"bin"`` strategy.
        n_clusters: Number of clusters for the ``"kmeans"`` strategy.
        seed: Random seed.
        **kwargs: Extra keyword arguments forwarded to ``neurogym.make()``.

    Returns:
        A ``NeurogymAdapter`` wrapping the requested task.

    Raises:
        ValueError: If *task_name* is not recognised.
        ImportError: If neurogym is not installed.
    """
    _require_neurogym()
    import neurogym as ngym

    task_key = task_name.lower().replace("-", "").replace("_", "")
    if task_key not in YANG19_TASKS:
        valid = ", ".join(sorted(YANG19_TASKS.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Valid names: {valid}")

    env_id = YANG19_TASKS[task_key]

    # Merge default task kwargs with user overrides.
    env_kwargs: dict[str, Any] = {}
    if task_key in _TASK_KWARGS:
        env_kwargs.update(_TASK_KWARGS[task_key])
    env_kwargs.update(kwargs)

    env = ngym.make(env_id, **env_kwargs)

    adapter = NeurogymAdapter(
        env,
        discretization=discretization,
        n_bins=n_bins,
        n_clusters=n_clusters,
        seed=seed,
    )

    # If using kmeans, auto-fit so the adapter is immediately usable.
    if discretization == "kmeans":
        adapter.fit()

    return adapter


def make_all_tasks(
    discretization: str = "bin",
    n_bins: int = 5,
    n_clusters: int = 20,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, NeurogymAdapter]:
    """Create all 20 Yang et al. (2019) tasks as discretized adapters.

    Args:
        discretization: Observation discretization strategy.
        n_bins: Bins per dimension for ``"bin"`` strategy.
        n_clusters: Clusters for ``"kmeans"`` strategy.
        seed: Random seed (incremented per task for reproducibility).
        **kwargs: Extra keyword arguments forwarded to ``neurogym.make()``.

    Returns:
        Dict mapping short task name to ``NeurogymAdapter``.
    """
    tasks: dict[str, NeurogymAdapter] = {}
    for i, name in enumerate(sorted(YANG19_TASKS.keys())):
        tasks[name] = make_task(
            name,
            discretization=discretization,
            n_bins=n_bins,
            n_clusters=n_clusters,
            seed=seed + i,
            **kwargs,
        )
    return tasks
