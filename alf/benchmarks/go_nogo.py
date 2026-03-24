"""Go/NoGo with Anti variant benchmark (Yang et al. 2019).

Combines two classic inhibitory control tasks:

- Go/NoGo: A stimulus appears. On "Go" trials the agent must respond in
  the stimulus direction. On "NoGo" trials the agent must withhold its
  response (continue fixating).

- Anti: A stimulus appears. Under the "Pro" rule the agent responds in
  the stimulus direction. Under the "Anti" rule the agent must respond
  in the OPPOSITE direction.

We unify these into a single task with a rule cue (go/nogo/anti) and
a stimulus direction (left/right).

State space (single factor, 18 states):
    States encode (stimulus_dir, rule, trial_phase):
    stimulus_dir: left(0) / right(1)
    rule: go(0) / nogo(1) / anti(2)
    trial_phase: fixation(0) / stimulus(1) / response(2)
    Total: 2 x 3 x 3 = 18 states

    Index = stim_dir * 9 + rule * 3 + phase

Observations (single modality, 8 outcomes):
    0: fixation_on
    1: stim_left
    2: stim_right
    3: rule_go
    4: rule_nogo
    5: rule_anti
    6: reward
    7: punishment

Actions (3):
    0: fixate (withhold response)
    1: respond_left
    2: respond_right

References:
    Yang, G.R., Joglekar, M.R., Song, H.F., Newsome, W.T. & Wang, X.-J.
        (2019). Task representations in neural networks trained to perform
        many cognitive tasks. Nature Neuroscience, 22, 297-306.
"""

import numpy as np
from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


# --- State encoding ---
# Index = stim_dir * 9 + rule * 3 + phase

NUM_STATES = 18  # 2 * 3 * 3

# Phase constants
PHASE_FIX = 0
PHASE_STIM = 1
PHASE_RESP = 2
NUM_PHASES = 3

# Stimulus direction
DIR_LEFT = 0
DIR_RIGHT = 1

# Rule constants
RULE_GO = 0
RULE_NOGO = 1
RULE_ANTI = 2
NUM_RULES = 3

# Observation indices
OBS_FIXATION = 0
OBS_STIM_LEFT = 1
OBS_STIM_RIGHT = 2
OBS_RULE_GO = 3
OBS_RULE_NOGO = 4
OBS_RULE_ANTI = 5
OBS_REWARD = 6
OBS_PUNISHMENT = 7

NUM_OBS = 8

# Action indices
ACT_FIXATE = 0
ACT_RESPOND_LEFT = 1
ACT_RESPOND_RIGHT = 2

NUM_ACTIONS = 3

ACTION_NAMES = ["fixate", "respond_left", "respond_right"]
OBS_NAMES = [
    "fixation",
    "stim_left",
    "stim_right",
    "rule_go",
    "rule_nogo",
    "rule_anti",
    "reward",
    "punishment",
]
RULE_NAMES = ["go", "nogo", "anti"]


def _state_index(stim_dir: int, rule: int, phase: int) -> int:
    """Convert (stim_dir, rule, phase) to flat state index."""
    return stim_dir * 9 + rule * 3 + phase


def _decode_state(s: int) -> tuple[int, int, int]:
    """Decode flat state index to (stim_dir, rule, phase)."""
    stim_dir = s // 9
    remainder = s % 9
    rule = remainder // 3
    phase = remainder % 3
    return stim_dir, rule, phase


def _correct_response(stim_dir: int, rule: int) -> int:
    """Return the correct action for the given stimulus and rule.

    - Go: respond in stimulus direction
    - NoGo: fixate (withhold response)
    - Anti: respond in opposite direction
    """
    if rule == RULE_GO:
        return ACT_RESPOND_LEFT if stim_dir == DIR_LEFT else ACT_RESPOND_RIGHT
    elif rule == RULE_NOGO:
        return ACT_FIXATE
    elif rule == RULE_ANTI:
        return ACT_RESPOND_RIGHT if stim_dir == DIR_LEFT else ACT_RESPOND_LEFT
    return ACT_FIXATE  # fallback


def _build_A() -> np.ndarray:
    """Likelihood matrix P(o|s), shape (8, 18).

    - Fixation phase: observe fixation (0.5) + rule cue (0.5)
    - Stimulus phase: observe stimulus direction (1.0)
    - Response phase: reward/punishment (50/50 placeholder)
    """
    A = np.zeros((NUM_OBS, NUM_STATES))

    for s in range(NUM_STATES):
        stim_dir, rule, phase = _decode_state(s)

        if phase == PHASE_FIX:
            # See fixation cross and rule cue
            A[OBS_FIXATION, s] = 0.5
            if rule == RULE_GO:
                A[OBS_RULE_GO, s] = 0.5
            elif rule == RULE_NOGO:
                A[OBS_RULE_NOGO, s] = 0.5
            elif rule == RULE_ANTI:
                A[OBS_RULE_ANTI, s] = 0.5

        elif phase == PHASE_STIM:
            # See stimulus direction unambiguously
            if stim_dir == DIR_LEFT:
                A[OBS_STIM_LEFT, s] = 1.0
            else:
                A[OBS_STIM_RIGHT, s] = 1.0

        elif phase == PHASE_RESP:
            # Response phase: 50/50 reward/punishment (env resolves)
            A[OBS_REWARD, s] = 0.5
            A[OBS_PUNISHMENT, s] = 0.5

    return A


def _build_B() -> np.ndarray:
    """Transition matrix P(s'|s, a), shape (18, 18, 3).

    Phase transitions:
    - fixate during fixation -> stimulus phase
    - fixate during stimulus -> response phase (withholding response)
    - respond during stimulus -> response phase
    - respond during fixation -> response phase (premature)
    - Response phase is absorbing.

    Stimulus direction and rule are immutable.
    """
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for s in range(NUM_STATES):
        stim_dir, rule, phase = _decode_state(s)

        for action in range(NUM_ACTIONS):
            if phase == PHASE_RESP:
                # Absorbing
                s_next = s
            elif phase == PHASE_FIX:
                if action == ACT_FIXATE:
                    # Fixate -> advance to stimulus phase
                    s_next = _state_index(stim_dir, rule, PHASE_STIM)
                else:
                    # Premature response -> response phase
                    s_next = _state_index(stim_dir, rule, PHASE_RESP)
            elif phase == PHASE_STIM:
                # Any action during stimulus -> response phase
                s_next = _state_index(stim_dir, rule, PHASE_RESP)

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

    Agent knows it starts in fixation but doesn't know stimulus direction
    or rule.
    """
    D = np.zeros(NUM_STATES)
    for stim_dir in range(2):
        for rule in range(NUM_RULES):
            s = _state_index(stim_dir, rule, PHASE_FIX)
            D[s] = 1.0
    D = D / D.sum()  # 1/6 for each fixation state
    return D


def build_go_nogo_model(
    reward_magnitude: float = 3.0,
    T: int = 3,
) -> GenerativeModel:
    """Build the Go/NoGo/Anti generative model.

    Args:
        reward_magnitude: Strength of reward/punishment preference.
        T: Planning horizon (fixation -> stimulus -> response = 3 steps).

    Returns:
        GenerativeModel configured for Go/NoGo/Anti.
    """
    A = _build_A()
    B = _build_B()
    C = _build_C(reward_magnitude)
    D = _build_D()

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


class GoNoGoEnv:
    """Go/NoGo/Anti environment.

    A stimulus appears with a rule cue. The agent must:
    - Go: respond in stimulus direction
    - NoGo: withhold response (fixate)
    - Anti: respond in opposite direction

    Args:
        stim_dir: Stimulus direction ("left" or "right").
        rule: Task rule ("go", "nogo", or "anti").
        seed: Random seed.
    """

    def __init__(
        self,
        stim_dir: str = "left",
        rule: str = "go",
        seed: int = 42,
    ):
        self.stim = DIR_LEFT if stim_dir == "left" else DIR_RIGHT
        self.rule = {"go": RULE_GO, "nogo": RULE_NOGO, "anti": RULE_ANTI}[rule]
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> int:
        """Reset to fixation phase. Returns initial observation."""
        self.phase = PHASE_FIX
        self.true_state = _state_index(self.stim, self.rule, PHASE_FIX)
        self.agent_response = None
        return self._observe()

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action. Returns (observation, reward, done)."""
        if self.phase == PHASE_RESP:
            return self._observe(), 0.0, True

        if self.phase == PHASE_FIX:
            if action == ACT_FIXATE:
                self.phase = PHASE_STIM
            else:
                # Premature response
                self.phase = PHASE_RESP
                self.agent_response = action

        elif self.phase == PHASE_STIM:
            # Any action during stimulus -> response phase
            self.phase = PHASE_RESP
            self.agent_response = action

        self.true_state = _state_index(self.stim, self.rule, self.phase)
        obs = self._observe()

        reward = 0.0
        done = False
        if self.phase == PHASE_RESP:
            done = True
            correct = _correct_response(self.stim, self.rule)
            if self.agent_response == correct:
                reward = 1.0
            else:
                reward = -1.0

        return obs, reward, done

    def _observe(self) -> int:
        """Generate observation from current phase."""
        if self.phase == PHASE_FIX:
            # Randomly emit fixation or rule cue
            if self.rng.random() < 0.5:
                return OBS_FIXATION
            else:
                return [OBS_RULE_GO, OBS_RULE_NOGO, OBS_RULE_ANTI][self.rule]

        elif self.phase == PHASE_STIM:
            return OBS_STIM_LEFT if self.stim == DIR_LEFT else OBS_STIM_RIGHT

        elif self.phase == PHASE_RESP:
            correct = _correct_response(self.stim, self.rule)
            if self.agent_response == correct:
                return OBS_REWARD
            else:
                return OBS_PUNISHMENT

        return OBS_FIXATION  # fallback


def run_go_nogo(
    num_trials: int = 50,
    reward_magnitude: float = 3.0,
    gamma: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run the Go/NoGo/Anti benchmark.

    Args:
        num_trials: Number of trials to run.
        reward_magnitude: Strength of reward/punishment preference.
        gamma: Agent policy precision.
        seed: Random seed.
        verbose: Print trial-by-trial details.

    Returns:
        Dict with accuracy, go_accuracy, nogo_accuracy, anti_accuracy,
        false_alarm_rate, mean_reward, trial_log.
    """
    gm = build_go_nogo_model(reward_magnitude=reward_magnitude, T=3)
    agent = AnalyticAgent(gm, gamma=gamma, seed=seed)

    rng = np.random.RandomState(seed)
    results = []

    for trial in range(num_trials):
        stim_dir = "left" if rng.random() < 0.5 else "right"
        rule_idx = rng.choice(3)
        rule = ["go", "nogo", "anti"][rule_idx]

        env = GoNoGoEnv(stim_dir=stim_dir, rule=rule, seed=seed + trial)

        agent.reset()
        obs = env.reset()
        total_reward = 0.0
        actions_taken = []

        for step in range(3):  # fixation -> stimulus -> response
            action, info = agent.step([obs])
            obs, reward, done = env.step(action)
            total_reward += reward
            actions_taken.append(ACTION_NAMES[action])

            if done:
                break

        responded_correctly = total_reward > 0

        # False alarm: responded when should have withheld (NoGo trials)
        false_alarm = (
            rule == "nogo"
            and any(a != "fixate" for a in actions_taken)
            and not responded_correctly
        )

        trial_info = {
            "trial": trial,
            "stim_dir": stim_dir,
            "rule": rule,
            "actions": actions_taken,
            "reward": total_reward,
            "correct": responded_correctly,
            "false_alarm": false_alarm,
        }
        results.append(trial_info)

        if verbose:
            status = "CORRECT" if responded_correctly else "WRONG"
            print(
                f"  Trial {trial:2d}: rule={rule:5s} stim={stim_dir:5s} | "
                f"{' -> '.join(actions_taken):25s} | {status}"
            )

    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / num_trials

    # Per-rule accuracy
    rule_acc = {}
    for rule_name in ["go", "nogo", "anti"]:
        rule_trials = [r for r in results if r["rule"] == rule_name]
        if rule_trials:
            rule_acc[rule_name] = sum(1 for r in rule_trials if r["correct"]) / len(
                rule_trials
            )
        else:
            rule_acc[rule_name] = 0.0

    # False alarm rate (NoGo trials where agent responded)
    nogo_trials = [r for r in results if r["rule"] == "nogo"]
    false_alarm_rate = (
        sum(1 for r in nogo_trials if r["false_alarm"]) / len(nogo_trials)
        if nogo_trials
        else 0.0
    )

    rewards = [r["reward"] for r in results]

    return {
        "accuracy": accuracy,
        "go_accuracy": rule_acc["go"],
        "nogo_accuracy": rule_acc["nogo"],
        "anti_accuracy": rule_acc["anti"],
        "false_alarm_rate": false_alarm_rate,
        "mean_reward": float(np.mean(rewards)),
        "num_trials": num_trials,
        "trial_log": results,
    }
