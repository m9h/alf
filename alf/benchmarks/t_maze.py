"""T-maze benchmark for Active Inference (Smith et al. 2022, Section 5).

The T-maze is the canonical Active Inference benchmark. An agent starts at
the center of a T-shaped maze and must reach the rewarded arm. A cue at one
location reveals which arm is rewarded.

State space (single factor, 8 states):
    States encode (location, reward_condition):
    0: center, reward-left    1: center, reward-right
    2: cue,    reward-left    3: cue,    reward-right
    4: left,   reward-left    5: left,   reward-right
    6: right,  reward-left    7: right,  reward-right

Observations (single modality, 5 outcomes):
    0: null, 1: cue_left, 2: cue_right, 3: reward, 4: punishment

Actions (4):
    0: stay, 1: go_cue, 2: go_left, 3: go_right

Reference:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
    Inference. Journal of Mathematical Psychology, 107, 102632.
"""

import numpy as np
from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


# State indices
CENTER_LEFT = 0
CENTER_RIGHT = 1
CUE_LEFT = 2
CUE_RIGHT = 3
LEFT_LEFT = 4
LEFT_RIGHT = 5
RIGHT_LEFT = 6
RIGHT_RIGHT = 7

NUM_STATES = 8

# Observation indices
OBS_NULL = 0
OBS_CUE_LEFT = 1
OBS_CUE_RIGHT = 2
OBS_REWARD = 3
OBS_PUNISHMENT = 4

NUM_OBS = 5

# Action indices
ACT_STAY = 0
ACT_CUE = 1
ACT_LEFT = 2
ACT_RIGHT = 3

NUM_ACTIONS = 4

ACTION_NAMES = ["stay", "go_cue", "go_left", "go_right"]
STATE_NAMES = [
    "center/L", "center/R", "cue/L", "cue/R",
    "left/L", "left/R", "right/L", "right/R",
]
OBS_NAMES = ["null", "cue_left", "cue_right", "reward", "punishment"]


def build_t_maze_model(
    cue_reliability: float = 0.9,
    reward_magnitude: float = 3.0,
    T: int = 2,
) -> GenerativeModel:
    """Build the T-maze generative model.

    Args:
        cue_reliability: Probability that the cue correctly indicates
            the reward location. Default 0.9.
        reward_magnitude: Strength of reward/punishment preference.
        T: Planning horizon. Default 2.

    Returns:
        GenerativeModel configured for the T-maze.
    """
    A = _build_A(cue_reliability)
    B = _build_B()
    C = _build_C(reward_magnitude)
    D = _build_D()

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


def _build_A(cue_reliability: float) -> np.ndarray:
    """Likelihood matrix P(o|s), shape (5, 8)."""
    A = np.zeros((NUM_OBS, NUM_STATES))

    A[OBS_NULL, CENTER_LEFT] = 1.0
    A[OBS_NULL, CENTER_RIGHT] = 1.0

    A[OBS_CUE_LEFT, CUE_LEFT] = cue_reliability
    A[OBS_CUE_RIGHT, CUE_LEFT] = 1.0 - cue_reliability
    A[OBS_CUE_RIGHT, CUE_RIGHT] = cue_reliability
    A[OBS_CUE_LEFT, CUE_RIGHT] = 1.0 - cue_reliability

    A[OBS_REWARD, LEFT_LEFT] = 1.0
    A[OBS_PUNISHMENT, LEFT_RIGHT] = 1.0
    A[OBS_PUNISHMENT, RIGHT_LEFT] = 1.0
    A[OBS_REWARD, RIGHT_RIGHT] = 1.0

    return A


def _build_B() -> np.ndarray:
    """Transition matrix P(s'|s, a), shape (8, 8, 4)."""
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for reward_side in range(2):
        center = reward_side
        cue = 2 + reward_side
        left = 4 + reward_side
        right = 6 + reward_side

        for s in [center, cue, left, right]:
            B[s, s, ACT_STAY] = 1.0

        B[cue, center, ACT_CUE] = 1.0
        B[cue, cue, ACT_CUE] = 1.0
        B[left, left, ACT_CUE] = 1.0
        B[right, right, ACT_CUE] = 1.0

        B[left, center, ACT_LEFT] = 1.0
        B[left, cue, ACT_LEFT] = 1.0
        B[left, left, ACT_LEFT] = 1.0
        B[right, right, ACT_LEFT] = 1.0

        B[right, center, ACT_RIGHT] = 1.0
        B[right, cue, ACT_RIGHT] = 1.0
        B[right, right, ACT_RIGHT] = 1.0
        B[left, left, ACT_RIGHT] = 1.0

    return B


def _build_C(reward_magnitude: float) -> np.ndarray:
    """Preference vector (log preferences over observations)."""
    C = np.zeros(NUM_OBS)
    C[OBS_REWARD] = reward_magnitude
    C[OBS_PUNISHMENT] = -reward_magnitude
    return C


def _build_D() -> np.ndarray:
    """Prior beliefs: agent starts at center, doesn't know reward side."""
    D = np.zeros(NUM_STATES)
    D[CENTER_LEFT] = 0.5
    D[CENTER_RIGHT] = 0.5
    return D


class TMazeEnv:
    """Simple T-maze environment.

    Args:
        reward_side: Which arm has the reward ("left" or "right").
        cue_reliability: How reliably the cue indicates reward side.
        seed: Random seed.
    """

    def __init__(
        self,
        reward_side: str = "left",
        cue_reliability: float = 0.9,
        seed: int = 42,
    ):
        self.reward_offset = 0 if reward_side == "left" else 1
        self.cue_reliability = cue_reliability
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> int:
        """Reset to center. Returns initial observation."""
        self.location = "center"
        self.true_state = self.reward_offset
        return OBS_NULL

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action. Returns (observation, reward, done)."""
        if action == ACT_CUE and self.location == "center":
            self.location = "cue"
            self.true_state = 2 + self.reward_offset
        elif action == ACT_LEFT and self.location in ("center", "cue"):
            self.location = "left"
            self.true_state = 4 + self.reward_offset
        elif action == ACT_RIGHT and self.location in ("center", "cue"):
            self.location = "right"
            self.true_state = 6 + self.reward_offset

        obs = self._observe()

        reward = 0.0
        done = False
        if self.location in ("left", "right"):
            done = True
            if obs == OBS_REWARD:
                reward = 1.0
            elif obs == OBS_PUNISHMENT:
                reward = -1.0

        return obs, reward, done

    def _observe(self) -> int:
        """Generate observation from current state."""
        if self.location == "center":
            return OBS_NULL
        elif self.location == "cue":
            if self.reward_offset == 0:
                if self.rng.random() < self.cue_reliability:
                    return OBS_CUE_LEFT
                else:
                    return OBS_CUE_RIGHT
            else:
                if self.rng.random() < self.cue_reliability:
                    return OBS_CUE_RIGHT
                else:
                    return OBS_CUE_LEFT
        elif self.location == "left":
            return OBS_REWARD if self.reward_offset == 0 else OBS_PUNISHMENT
        else:
            return OBS_REWARD if self.reward_offset == 1 else OBS_PUNISHMENT


def run_t_maze(
    num_trials: int = 20,
    cue_reliability: float = 0.9,
    gamma: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run the T-maze benchmark over multiple trials.

    Args:
        num_trials: Number of T-maze trials.
        cue_reliability: Cue accuracy.
        gamma: Agent policy precision.
        seed: Random seed.
        verbose: Print trial-by-trial details.

    Returns:
        Dict with reward_rate, cue_visit_rate, mean_reward, trial_log.
    """
    gm = build_t_maze_model(cue_reliability=cue_reliability, T=2)
    agent = AnalyticAgent(gm, gamma=gamma, seed=seed)

    rng = np.random.RandomState(seed)
    results = []

    for trial in range(num_trials):
        reward_side = "left" if rng.random() < 0.5 else "right"
        env = TMazeEnv(reward_side=reward_side, cue_reliability=cue_reliability,
                       seed=seed + trial)

        agent.reset()
        obs = env.reset()
        visited_cue = False
        total_reward = 0.0
        actions_taken = []

        for step in range(2):
            action, info = agent.step([obs])
            obs, reward, done = env.step(action)
            total_reward += reward
            actions_taken.append(ACTION_NAMES[action])

            if action == ACT_CUE:
                visited_cue = True

            if done:
                break

        trial_info = {
            "trial": trial,
            "reward_side": reward_side,
            "actions": actions_taken,
            "visited_cue": visited_cue,
            "reward": total_reward,
            "final_location": env.location,
        }
        results.append(trial_info)

        if verbose:
            status = "REWARD" if total_reward > 0 else "PUNISH" if total_reward < 0 else "NONE"
            print(f"  Trial {trial:2d}: {reward_side:5s} | "
                  f"{' -> '.join(actions_taken):20s} | "
                  f"cue={'Y' if visited_cue else 'N'} | {status}")

    rewards = [r["reward"] for r in results]
    cue_visits = [r["visited_cue"] for r in results]

    return {
        "reward_rate": sum(1 for r in rewards if r > 0) / num_trials,
        "cue_visit_rate": sum(cue_visits) / num_trials,
        "mean_reward": np.mean(rewards),
        "num_trials": num_trials,
        "trial_log": results,
    }
