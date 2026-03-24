"""Delayed Match-to-Sample (DMS) benchmark (Yang et al. 2019).

A sample stimulus is presented, followed by a delay period, then a test
stimulus. The agent must report whether the test matches the sample.
This is a canonical working memory task: success requires maintaining
the sample representation across the delay.

State space (single factor, 20 states):
    States encode (sample, phase, test_matches):
    sample: left(0) / right(1)
    phase: fixation(0) / sample(1) / delay(2) / test(3) / response(4)
    test_matches: match(0) / nonmatch(1)
    Total: 2 x 5 x 2 = 20 states

    Index = sample * 10 + phase * 2 + test_matches

Observations (single modality, 7 outcomes):
    0: fixation_on
    1: sample_left
    2: sample_right
    3: test_left
    4: test_right
    5: reward
    6: punishment

Actions (3):
    0: fixate
    1: respond_match
    2: respond_nonmatch

References:
    Yang, G.R., Joglekar, M.R., Song, H.F., Newsome, W.T. & Wang, X.-J.
        (2019). Task representations in neural networks trained to perform
        many cognitive tasks. Nature Neuroscience, 22, 297-306.
"""

import numpy as np
from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


# --- State encoding ---
# Index = sample * 10 + phase * 2 + test_matches

NUM_STATES = 20  # 2 * 5 * 2

# Phase constants
PHASE_FIX = 0
PHASE_SAMPLE = 1
PHASE_DELAY = 2
PHASE_TEST = 3
PHASE_RESP = 4
NUM_PHASES = 5

# Sample direction
DIR_LEFT = 0
DIR_RIGHT = 1

# Match condition
MATCH = 0
NONMATCH = 1

# Observation indices
OBS_FIXATION = 0
OBS_SAMPLE_LEFT = 1
OBS_SAMPLE_RIGHT = 2
OBS_TEST_LEFT = 3
OBS_TEST_RIGHT = 4
OBS_REWARD = 5
OBS_PUNISHMENT = 6

NUM_OBS = 7

# Action indices
ACT_FIXATE = 0
ACT_RESPOND_MATCH = 1
ACT_RESPOND_NONMATCH = 2

NUM_ACTIONS = 3

ACTION_NAMES = ["fixate", "respond_match", "respond_nonmatch"]
OBS_NAMES = [
    "fixation",
    "sample_left",
    "sample_right",
    "test_left",
    "test_right",
    "reward",
    "punishment",
]


def _state_index(sample: int, phase: int, test_matches: int) -> int:
    """Convert (sample, phase, test_matches) to flat state index."""
    return sample * 10 + phase * 2 + test_matches


def _decode_state(s: int) -> tuple[int, int, int]:
    """Decode flat state index to (sample, phase, test_matches)."""
    sample = s // 10
    remainder = s % 10
    phase = remainder // 2
    test_matches = remainder % 2
    return sample, phase, test_matches


def _test_direction(sample: int, test_matches: int) -> int:
    """Return the direction of the test stimulus."""
    if test_matches == MATCH:
        return sample  # Same direction as sample
    else:
        return 1 - sample  # Opposite direction


def _correct_response(test_matches: int) -> int:
    """Return the correct action for the given match condition."""
    return ACT_RESPOND_MATCH if test_matches == MATCH else ACT_RESPOND_NONMATCH


def _build_A() -> np.ndarray:
    """Likelihood matrix P(o|s), shape (7, 20).

    - Fixation phase: observe fixation (1.0)
    - Sample phase: observe sample stimulus (1.0 for correct direction)
    - Delay phase: observe fixation (1.0) — no stimulus present
    - Test phase: observe test stimulus (1.0 for correct direction)
    - Response phase: observe reward/punishment (50/50 as placeholder;
      actual outcome determined by environment)
    """
    A = np.zeros((NUM_OBS, NUM_STATES))

    for s in range(NUM_STATES):
        sample, phase, test_matches = _decode_state(s)

        if phase == PHASE_FIX:
            A[OBS_FIXATION, s] = 1.0

        elif phase == PHASE_SAMPLE:
            if sample == DIR_LEFT:
                A[OBS_SAMPLE_LEFT, s] = 1.0
            else:
                A[OBS_SAMPLE_RIGHT, s] = 1.0

        elif phase == PHASE_DELAY:
            # During delay, only fixation is visible
            A[OBS_FIXATION, s] = 1.0

        elif phase == PHASE_TEST:
            test_dir = _test_direction(sample, test_matches)
            if test_dir == DIR_LEFT:
                A[OBS_TEST_LEFT, s] = 1.0
            else:
                A[OBS_TEST_RIGHT, s] = 1.0

        elif phase == PHASE_RESP:
            # Response phase: 50/50 reward/punishment (env resolves)
            A[OBS_REWARD, s] = 0.5
            A[OBS_PUNISHMENT, s] = 0.5

    return A


def _build_B() -> np.ndarray:
    """Transition matrix P(s'|s, a), shape (20, 20, 3).

    Phase transitions (fixate action advances through phases):
    - fixate during fixation -> sample phase
    - fixate during sample -> delay phase
    - fixate during delay -> test phase
    - fixate during test -> response phase
    - respond_match/respond_nonmatch during any non-response phase -> response
    - Response phase is absorbing.

    Sample identity and match condition are immutable.
    """
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for s in range(NUM_STATES):
        sample, phase, test_matches = _decode_state(s)

        for action in range(NUM_ACTIONS):
            if phase == PHASE_RESP:
                # Absorbing
                s_next = s
            elif action == ACT_FIXATE:
                # Advance to next phase
                next_phase = min(phase + 1, PHASE_RESP)
                s_next = _state_index(sample, next_phase, test_matches)
            else:
                # Any response action -> response phase
                s_next = _state_index(sample, PHASE_RESP, test_matches)

            B[s_next, s, action] = 1.0

    return B


def _build_C(reward_magnitude: float) -> np.ndarray:
    """Preference vector (log preferences over observations)."""
    C = np.zeros(NUM_OBS)
    C[OBS_REWARD] = reward_magnitude
    C[OBS_PUNISHMENT] = -reward_magnitude
    return C


def _build_D() -> np.ndarray:
    """Prior beliefs: uniform over fixation-phase states.

    Agent knows it starts in fixation but doesn't know sample direction
    or match condition.
    """
    D = np.zeros(NUM_STATES)
    for sample in range(2):
        for test_matches in range(2):
            s = _state_index(sample, PHASE_FIX, test_matches)
            D[s] = 1.0
    D = D / D.sum()  # 1/4 for each fixation state
    return D


def build_delayed_match_model(
    reward_magnitude: float = 3.0,
    T: int = 5,
) -> GenerativeModel:
    """Build the delayed match-to-sample generative model.

    Args:
        reward_magnitude: Strength of reward/punishment preference.
        T: Planning horizon (fix -> sample -> delay -> test -> response = 5).

    Returns:
        GenerativeModel configured for delayed match-to-sample.
    """
    A = _build_A()
    B = _build_B()
    C = _build_C(reward_magnitude)
    D = _build_D()

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


class DelayedMatchEnv:
    """Delayed match-to-sample environment.

    Presents a sample stimulus, waits through a delay, then presents a
    test stimulus. Agent must report match/nonmatch.

    Args:
        sample_dir: Direction of sample stimulus ("left" or "right").
        test_matches: Whether the test matches the sample (True/False).
        seed: Random seed.
    """

    def __init__(
        self,
        sample_dir: str = "left",
        test_matches: bool = True,
        seed: int = 42,
    ):
        self.sample = DIR_LEFT if sample_dir == "left" else DIR_RIGHT
        self.match_cond = MATCH if test_matches else NONMATCH
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> int:
        """Reset to fixation phase. Returns initial observation."""
        self.phase = PHASE_FIX
        self.true_state = _state_index(self.sample, PHASE_FIX, self.match_cond)
        self.agent_response = None
        return OBS_FIXATION

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action. Returns (observation, reward, done)."""
        if self.phase == PHASE_RESP:
            # Absorbing state
            return self._observe(), 0.0, True

        if action == ACT_FIXATE:
            # Advance to next phase
            self.phase = min(self.phase + 1, PHASE_RESP)
            if self.phase == PHASE_RESP:
                # Fixated through to response without responding
                self.agent_response = None
        else:
            # Agent made a response
            self.phase = PHASE_RESP
            self.agent_response = action

        self.true_state = _state_index(self.sample, self.phase, self.match_cond)
        obs = self._observe()

        reward = 0.0
        done = False
        if self.phase == PHASE_RESP:
            done = True
            if self.agent_response is not None:
                correct = _correct_response(self.match_cond)
                if self.agent_response == correct:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                reward = -1.0  # No response is incorrect

        return obs, reward, done

    def _observe(self) -> int:
        """Generate observation from current phase."""
        if self.phase == PHASE_FIX:
            return OBS_FIXATION
        elif self.phase == PHASE_SAMPLE:
            return OBS_SAMPLE_LEFT if self.sample == DIR_LEFT else OBS_SAMPLE_RIGHT
        elif self.phase == PHASE_DELAY:
            return OBS_FIXATION
        elif self.phase == PHASE_TEST:
            test_dir = _test_direction(self.sample, self.match_cond)
            return OBS_TEST_LEFT if test_dir == DIR_LEFT else OBS_TEST_RIGHT
        elif self.phase == PHASE_RESP:
            if self.agent_response is not None:
                correct = _correct_response(self.match_cond)
                if self.agent_response == correct:
                    return OBS_REWARD
                else:
                    return OBS_PUNISHMENT
            return OBS_PUNISHMENT  # No response = punishment

        return OBS_FIXATION  # fallback


def run_delayed_match(
    num_trials: int = 50,
    reward_magnitude: float = 3.0,
    gamma: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run the delayed match-to-sample benchmark.

    Args:
        num_trials: Number of trials to run.
        reward_magnitude: Strength of reward/punishment preference.
        gamma: Agent policy precision.
        seed: Random seed.
        verbose: Print trial-by-trial details.

    Returns:
        Dict with accuracy, match_accuracy, nonmatch_accuracy,
        mean_reward, trial_log.
    """
    gm = build_delayed_match_model(reward_magnitude=reward_magnitude, T=5)
    agent = AnalyticAgent(gm, gamma=gamma, seed=seed)

    rng = np.random.RandomState(seed)
    results = []

    for trial in range(num_trials):
        sample_dir = "left" if rng.random() < 0.5 else "right"
        test_matches = rng.random() < 0.5

        env = DelayedMatchEnv(
            sample_dir=sample_dir,
            test_matches=test_matches,
            seed=seed + trial,
        )

        agent.reset()
        obs = env.reset()
        total_reward = 0.0
        actions_taken = []

        for step in range(5):  # fix -> sample -> delay -> test -> response
            action, info = agent.step([obs])
            obs, reward, done = env.step(action)
            total_reward += reward
            actions_taken.append(ACTION_NAMES[action])

            if done:
                break

        responded_correctly = total_reward > 0

        trial_info = {
            "trial": trial,
            "sample_dir": sample_dir,
            "test_matches": test_matches,
            "actions": actions_taken,
            "reward": total_reward,
            "correct": responded_correctly,
        }
        results.append(trial_info)

        if verbose:
            status = "CORRECT" if responded_correctly else "WRONG"
            match_str = "match" if test_matches else "nonmatch"
            print(
                f"  Trial {trial:2d}: sample={sample_dir:5s} "
                f"({match_str:8s}) | "
                f"{' -> '.join(actions_taken):40s} | {status}"
            )

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / num_trials

    match_trials = [r for r in results if r["test_matches"]]
    nonmatch_trials = [r for r in results if not r["test_matches"]]

    match_acc = (
        sum(1 for r in match_trials if r["correct"]) / len(match_trials)
        if match_trials
        else 0.0
    )
    nonmatch_acc = (
        sum(1 for r in nonmatch_trials if r["correct"]) / len(nonmatch_trials)
        if nonmatch_trials
        else 0.0
    )

    rewards = [r["reward"] for r in results]

    return {
        "accuracy": accuracy,
        "match_accuracy": match_acc,
        "nonmatch_accuracy": nonmatch_acc,
        "mean_reward": float(np.mean(rewards)),
        "num_trials": num_trials,
        "trial_log": results,
    }
