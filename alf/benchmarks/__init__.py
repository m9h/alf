"""Benchmarks for ALF Active Inference."""

from alf.benchmarks.t_maze import build_t_maze_model
from alf.benchmarks.t_maze import TMazeEnv
from alf.benchmarks.t_maze import run_t_maze

from alf.benchmarks.context_dm import build_context_dm_model
from alf.benchmarks.context_dm import ContextDMEnv
from alf.benchmarks.context_dm import run_context_dm

from alf.benchmarks.delayed_match import build_delayed_match_model
from alf.benchmarks.delayed_match import DelayedMatchEnv
from alf.benchmarks.delayed_match import run_delayed_match

from alf.benchmarks.go_nogo import build_go_nogo_model
from alf.benchmarks.go_nogo import GoNoGoEnv
from alf.benchmarks.go_nogo import run_go_nogo

from alf.benchmarks.cognitive_battery import run_battery
from alf.benchmarks.cognitive_battery import BATTERY
from alf.benchmarks.cognitive_battery import get_task_list
from alf.benchmarks.cognitive_battery import get_task_model

__all__ = [
    # T-maze (Smith et al. 2022)
    "build_t_maze_model",
    "TMazeEnv",
    "run_t_maze",
    # Context-dependent DM (Mante et al. 2013)
    "build_context_dm_model",
    "ContextDMEnv",
    "run_context_dm",
    # Delayed match-to-sample
    "build_delayed_match_model",
    "DelayedMatchEnv",
    "run_delayed_match",
    # Go/NoGo/Anti
    "build_go_nogo_model",
    "GoNoGoEnv",
    "run_go_nogo",
    # Cognitive battery
    "run_battery",
    "BATTERY",
    "get_task_list",
    "get_task_model",
]
