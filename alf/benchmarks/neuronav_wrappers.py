"""ALF wrappers for neuro-nav environments.

Bridges the neuro-nav RL environment suite (Juliani, 2022) with ALF's
Active Inference framework. Provides converters that extract transition
and reward structure from neuro-nav GraphEnv and GridEnv instances and
build ALF GenerativeModel objects suitable for AIF planning.

References:
    Juliani, A. (2022). neuro-nav: A library for neurally-plausible
        reinforcement learning research. github.com/awjuliani/neuro-nav
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


def GraphEnvToGenerativeModel(
    graph_env,
    obs_type: str = "index",
    reward_magnitude: float = 3.0,
    T: int = 1,
) -> GenerativeModel:
    """Convert a neuro-nav GraphEnv into an ALF GenerativeModel.

    Args:
        graph_env: An instantiated neuronav GraphEnv. Must have been reset().
        obs_type: Observation mapping style ("index" or "onehot").
        reward_magnitude: Scaling for preference vector.
        T: Planning horizon.

    Returns:
        GenerativeModel for the graph environment.
    """
    num_states = graph_env.state_size
    num_actions = graph_env.action_space.n

    A = np.eye(num_states)

    B = np.zeros((num_states, num_states, num_actions))
    edges = graph_env.edges

    for s, neighbours in enumerate(edges):
        if len(neighbours) == 0:
            B[s, s, :] = 1.0
        else:
            for a in range(num_actions):
                if a < len(neighbours):
                    target = neighbours[a]
                    if isinstance(target, tuple):
                        dests, probs = target
                        for d_idx, dest in enumerate(dests):
                            B[dest, s, a] = probs[d_idx]
                    else:
                        B[target, s, a] = 1.0
                else:
                    B[s, s, a] = 1.0

    C = np.zeros(num_states)
    reward_dict = _get_reward_dict(graph_env)
    for state_idx, reward_val in reward_dict.items():
        C[state_idx] = reward_magnitude * np.sign(reward_val) * abs(reward_val)

    D = np.zeros(num_states)
    D[graph_env.agent_start_pos] = 1.0

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


def GridEnvToGenerativeModel(
    grid_env,
    obs_type: str = "index",
    reward_magnitude: float = 3.0,
    T: int = 1,
) -> GenerativeModel:
    """Convert a neuro-nav GridEnv into an ALF GenerativeModel.

    Args:
        grid_env: An instantiated neuronav GridEnv with fixed orientation.
        obs_type: Observation style.
        reward_magnitude: Scaling for the C vector.
        T: Planning horizon.

    Returns:
        GenerativeModel for the linearised grid.
    """
    gs = grid_env.grid_size
    num_states = gs * gs
    num_actions = 4

    direction_map = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    blocks_set = set(tuple(b) for b in grid_env.blocks)

    def _rc_to_idx(r: int, c: int) -> int:
        return r * gs + c

    A = np.eye(num_states)

    B = np.zeros((num_states, num_states, num_actions))
    for r in range(gs):
        for c in range(gs):
            s = _rc_to_idx(r, c)
            if (r, c) in blocks_set:
                B[s, s, :] = 1.0
                continue
            for a, (dr, dc) in enumerate(direction_map):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs and (nr, nc) not in blocks_set:
                    B[_rc_to_idx(nr, nc), s, a] = 1.0
                else:
                    B[s, s, a] = 1.0

    C = np.zeros(num_states)
    reward_dict = _get_reward_dict(grid_env)
    for pos, reward_val in reward_dict.items():
        if isinstance(pos, tuple) and len(pos) == 2:
            idx = _rc_to_idx(pos[0], pos[1])
            if isinstance(reward_val, list):
                reward_val = reward_val[0]
            C[idx] = reward_magnitude * reward_val

    D = np.zeros(num_states)
    start = grid_env.agent_start_pos
    if isinstance(start, (list, tuple)):
        D[_rc_to_idx(start[0], start[1])] = 1.0
    else:
        D[start] = 1.0

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


def _get_reward_dict(env) -> dict:
    """Return the reward dictionary from a neuro-nav env."""
    if hasattr(env, "objects") and env.objects is not None:
        obj = env.objects
    elif hasattr(env, "template_objects"):
        obj = env.template_objects
    else:
        return {}
    return obj.get("rewards", {})


def build_neuronav_t_maze(
    reward_magnitude: float = 3.0,
    T: int = 2,
) -> tuple[GenerativeModel, Any]:
    """Build a T-maze from neuro-nav graph templates.

    Returns:
        Tuple of (GenerativeModel, GraphEnv instance).
    """
    from neuronav.envs.graph_env import GraphEnv
    from neuronav.envs.graph_templates import GraphTemplate

    env = GraphEnv(template=GraphTemplate.t_graph, obs_type="index")
    env.reset()
    gm = GraphEnvToGenerativeModel(
        env,
        reward_magnitude=reward_magnitude,
        T=T,
    )
    return gm, env


def build_neuronav_two_step(
    reward_magnitude: float = 3.0,
    T: int = 2,
) -> tuple[GenerativeModel, Any]:
    """Build the two-step decision task from neuro-nav.

    Returns:
        Tuple of (GenerativeModel, GraphEnv instance).
    """
    from neuronav.envs.graph_env import GraphEnv
    from neuronav.envs.graph_templates import GraphTemplate

    env = GraphEnv(template=GraphTemplate.two_step, obs_type="index")
    env.reset()
    gm = GraphEnvToGenerativeModel(
        env,
        reward_magnitude=reward_magnitude,
        T=T,
    )
    return gm, env


def build_neuronav_three_arm_bandit(
    reward_magnitude: float = 3.0,
    T: int = 1,
) -> tuple[GenerativeModel, Any]:
    """Build a three-arm bandit from neuro-nav.

    Returns:
        Tuple of (GenerativeModel, GraphEnv instance).
    """
    from neuronav.envs.graph_env import GraphEnv
    from neuronav.envs.graph_templates import GraphTemplate

    env = GraphEnv(template=GraphTemplate.three_arm_bandit, obs_type="index")
    env.reset()
    gm = GraphEnvToGenerativeModel(
        env,
        reward_magnitude=reward_magnitude,
        T=T,
    )
    return gm, env


class NeuroNavRunner:
    """Run comparative experiments between ALF AIF and neuro-nav RL agents.

    Args:
        gm: ALF GenerativeModel for the task.
        env: A neuro-nav environment (GraphEnv or GridEnv).
        aif_gamma: Policy precision for the AIF agent.
        rl_agent_cls: A neuro-nav agent class (e.g. TDQ).
        rl_agent_kwargs: Keyword arguments for the RL agent.
        max_steps: Maximum steps per episode.
        seed: Random seed.
    """

    def __init__(
        self,
        gm: GenerativeModel,
        env,
        aif_gamma: float = 4.0,
        rl_agent_cls=None,
        rl_agent_kwargs: Optional[dict] = None,
        max_steps: int = 50,
        seed: int = 42,
    ):
        self.gm = gm
        self.env = env
        self.aif_gamma = aif_gamma
        self.max_steps = max_steps
        self.seed = seed

        self.aif_agent = AnalyticAgent(gm, gamma=aif_gamma, seed=seed)

        self.rl_agent = None
        if rl_agent_cls is not None:
            kwargs = dict(rl_agent_kwargs or {})
            kwargs.setdefault("state_size", env.state_size)
            kwargs.setdefault("action_size", env.action_space.n)
            self.rl_agent = rl_agent_cls(**kwargs)

    def _run_aif_episode(self, env_reset_kwargs: dict | None = None) -> dict:
        reset_kw = env_reset_kwargs or {}
        obs = self.env.reset(**reset_kw)
        self.aif_agent.reset()

        total_reward = 0.0
        steps = 0
        visited_states: list[int] = []
        actions_taken: list[int] = []

        obs_idx = self._obs_to_index(obs)
        visited_states.append(obs_idx)

        for _ in range(self.max_steps):
            action, _info = self.aif_agent.step([obs_idx])
            obs, reward, done, _ = self.env.step(action)
            obs_idx = self._obs_to_index(obs)

            total_reward += reward
            steps += 1
            visited_states.append(obs_idx)
            actions_taken.append(action)

            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "visited_states": visited_states,
            "actions": actions_taken,
            "done": done,
        }

    def _run_rl_episode(self, env_reset_kwargs: dict | None = None) -> dict:
        if self.rl_agent is None:
            raise RuntimeError("No RL agent configured.")

        reset_kw = env_reset_kwargs or {}
        obs = self.env.reset(**reset_kw)
        if hasattr(self.rl_agent, "reset"):
            self.rl_agent.reset()

        total_reward = 0.0
        steps = 0
        visited_states: list[int] = []
        actions_taken: list[int] = []

        obs_idx = self._obs_to_index(obs)
        visited_states.append(obs_idx)

        for _ in range(self.max_steps):
            action = self.rl_agent.sample_action(obs_idx)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs_idx = self._obs_to_index(next_obs)

            exp = (obs_idx, action, next_obs_idx, reward, done)
            self.rl_agent.update(exp)

            total_reward += reward
            steps += 1
            visited_states.append(next_obs_idx)
            actions_taken.append(action)
            obs_idx = next_obs_idx

            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "visited_states": visited_states,
            "actions": actions_taken,
            "done": done,
        }

    def run_comparison(
        self,
        num_episodes: int = 20,
        env_reset_kwargs: dict | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        aif_episodes = []
        rl_episodes = []

        for ep in range(num_episodes):
            aif_result = self._run_aif_episode(env_reset_kwargs)
            aif_episodes.append(aif_result)

            if self.rl_agent is not None:
                rl_result = self._run_rl_episode(env_reset_kwargs)
                rl_episodes.append(rl_result)

            if verbose:
                aif_r = aif_result["total_reward"]
                msg = (
                    f"  Ep {ep:3d} | AIF reward={aif_r:.2f} steps={aif_result['steps']}"
                )
                if rl_episodes:
                    rl_r = rl_result["total_reward"]
                    msg += f" | RL reward={rl_r:.2f} steps={rl_result['steps']}"
                print(msg)

        result: dict[str, Any] = {
            "aif": self._summarise(aif_episodes),
        }
        if rl_episodes:
            result["rl"] = self._summarise(rl_episodes)

        return result

    @staticmethod
    def _obs_to_index(obs) -> int:
        if isinstance(obs, (int, np.integer)):
            return int(obs)
        if isinstance(obs, np.ndarray):
            if obs.ndim == 0:
                return int(obs)
            if obs.ndim == 1 and obs.sum() == 1:
                return int(np.argmax(obs))
            return int(obs.flat[0])
        return int(obs)

    @staticmethod
    def _summarise(episodes: list[dict]) -> dict:
        rewards = [e["total_reward"] for e in episodes]
        steps = [e["steps"] for e in episodes]
        return {
            "reward_rate": sum(1 for r in rewards if r > 0) / max(len(rewards), 1),
            "mean_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
            "episodes": episodes,
        }
