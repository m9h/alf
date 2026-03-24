"""Context-Dependent Decision Making benchmark (Mante et al. 2013).

Two stimulus modalities are presented simultaneously (e.g., color + motion).
A context cue indicates which modality to base the decision on. The agent
must integrate evidence from the attended modality while ignoring the other.

This is a core task from the Yang et al. (2019) cognitive neuroscience
task battery, and the canonical test of context-dependent computation.

State space (single factor, 24 states):
    States encode (stimulus_mod1, stimulus_mod2, context, trial_phase):
    stimulus_mod1: left(0) / right(1)
    stimulus_mod2: left(0) / right(1)
    context: attend_mod1(0) / attend_mod2(1)
    trial_phase: fixation(0) / stimulus(1) / response(2)
    Total: 2 x 2 x 2 x 3 = 24 states

    Index = mod1 * 12 + mod2 * 6 + ctx * 3 + phase

Observations (single modality, 9 outcomes):
    0: fixation_on
    1: stim_left_mod1
    2: stim_right_mod1
    3: stim_left_mod2
    4: stim_right_mod2
    5: context_attend_mod1
    6: context_attend_mod2
    7: reward
    8: punishment

Actions (3):
    0: fixate
    1: respond_left
    2: respond_right

References:
    Mante, V., Sussillo, D., Shenoy, K.V. & Newsome, W.T. (2013).
        Context-dependent computation by recurrent dynamics in prefrontal
        cortex. Nature, 503, 78-84.
    Yang, G.R., Joglekar, M.R., Song, H.F., Newsome, W.T. & Wang, X.-J.
        (2019). Task representations in neural networks trained to perform
        many cognitive tasks. Nature Neuroscience, 22, 297-306.
"""

import numpy as np
from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent


# --- State encoding ---
# Index = mod1 * 12 + mod2 * 6 + ctx * 3 + phase
# mod1 in {0=left, 1=right}
# mod2 in {0=left, 1=right}
# ctx  in {0=attend_mod1, 1=attend_mod2}
# phase in {0=fixation, 1=stimulus, 2=response}

NUM_STATES = 24  # 2 * 2 * 2 * 3

# Phase constants
PHASE_FIX = 0
PHASE_STIM = 1
PHASE_RESP = 2
NUM_PHASES = 3

# Context constants
CTX_MOD1 = 0
CTX_MOD2 = 1

# Stimulus direction constants
DIR_LEFT = 0
DIR_RIGHT = 1

# Observation indices
OBS_FIXATION = 0
OBS_STIM_LEFT_MOD1 = 1
OBS_STIM_RIGHT_MOD1 = 2
OBS_STIM_LEFT_MOD2 = 3
OBS_STIM_RIGHT_MOD2 = 4
OBS_CTX_ATTEND_MOD1 = 5
OBS_CTX_ATTEND_MOD2 = 6
OBS_REWARD = 7
OBS_PUNISHMENT = 8

NUM_OBS = 9

# Action indices
ACT_FIXATE = 0
ACT_RESPOND_LEFT = 1
ACT_RESPOND_RIGHT = 2

NUM_ACTIONS = 3

ACTION_NAMES = ["fixate", "respond_left", "respond_right"]
OBS_NAMES = [
    "fixation",
    "stim_L_mod1",
    "stim_R_mod1",
    "stim_L_mod2",
    "stim_R_mod2",
    "ctx_mod1",
    "ctx_mod2",
    "reward",
    "punishment",
]


def _state_index(mod1: int, mod2: int, ctx: int, phase: int) -> int:
    """Convert (mod1, mod2, ctx, phase) to flat state index."""
    return mod1 * 12 + mod2 * 6 + ctx * 3 + phase


def _decode_state(s: int) -> tuple[int, int, int, int]:
    """Decode flat state index to (mod1, mod2, ctx, phase)."""
    mod1 = s // 12
    remainder = s % 12
    mod2 = remainder // 6
    remainder = remainder % 6
    ctx = remainder // 3
    phase = remainder % 3
    return mod1, mod2, ctx, phase


def _correct_response(mod1: int, mod2: int, ctx: int) -> int:
    """Return the correct action given stimuli and context.

    If context is attend_mod1, the correct response matches mod1 direction.
    If context is attend_mod2, the correct response matches mod2 direction.
    """
    attended_dir = mod1 if ctx == CTX_MOD1 else mod2
    return ACT_RESPOND_LEFT if attended_dir == DIR_LEFT else ACT_RESPOND_RIGHT


def _build_A() -> np.ndarray:
    """Likelihood matrix P(o|s), shape (9, 24).

    - Fixation phase: observe fixation + context cue (50/50 split to
      represent both being visible, implemented as mixed distribution).
    - Stimulus phase: observe both modality stimuli (mixed distribution
      over the four stimulus observations).
    - Response phase: observe reward or punishment depending on whether
      the state's attended modality matches the eventual response.
      Since the response has already been made by the time we reach the
      response phase, we observe reward if the state is "correct" and
      punishment if "incorrect". However, since the state doesn't encode
      the response, we observe reward with 100% probability in states
      where we assume the agent responded correctly. We handle this by
      making both reward and punishment partially observable (agent doesn't
      know the outcome until it acts).

    Actually, for a POMDP the A matrix is purely P(o|s) -- it maps states
    to observations. The response phase observation should reflect whether
    the trial is a "correct" or "incorrect" state. We split response-phase
    states into those that would yield reward vs punishment based on a
    simple mapping. But since the state doesn't encode the agent's choice,
    we use 50/50 reward/punishment in the response phase to indicate
    uncertainty. The actual reward determination happens in the environment.

    Simpler approach (following T-maze pattern):
    - Fixation: see fixation_on (0.5) + context cue (0.5)
    - Stimulus: see mod1 stimulus (0.5) + mod2 stimulus (0.5)
    - Response: see reward (0.5) + punishment (0.5) as default.
      The environment determines the real outcome.
    """
    A = np.zeros((NUM_OBS, NUM_STATES))

    for s in range(NUM_STATES):
        mod1, mod2, ctx, phase = _decode_state(s)

        if phase == PHASE_FIX:
            # During fixation, agent sees fixation cross and context cue
            A[OBS_FIXATION, s] = 0.5
            if ctx == CTX_MOD1:
                A[OBS_CTX_ATTEND_MOD1, s] = 0.5
            else:
                A[OBS_CTX_ATTEND_MOD2, s] = 0.5

        elif phase == PHASE_STIM:
            # During stimulus, agent sees both modalities
            if mod1 == DIR_LEFT:
                A[OBS_STIM_LEFT_MOD1, s] = 0.5
            else:
                A[OBS_STIM_RIGHT_MOD1, s] = 0.5
            if mod2 == DIR_LEFT:
                A[OBS_STIM_LEFT_MOD2, s] = 0.5
            else:
                A[OBS_STIM_RIGHT_MOD2, s] = 0.5

        elif phase == PHASE_RESP:
            # Response phase: agent sees reward or punishment.
            # We split 50/50 since the state doesn't encode the response.
            # The environment determines the actual outcome.
            A[OBS_REWARD, s] = 0.5
            A[OBS_PUNISHMENT, s] = 0.5

    return A


def _build_B() -> np.ndarray:
    """Transition matrix P(s'|s, a), shape (24, 24, 3).

    Phase transitions:
    - fixate during fixation: advance to stimulus phase
    - fixate during stimulus: stay in stimulus (observe stimuli)
    - respond_left/right from fixation: jump to response (premature)
    - respond_left/right from stimulus: advance to response phase
    - In response phase, all actions keep you in response phase.

    Stimulus directions and context are immutable (set at trial start).
    """
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for s in range(NUM_STATES):
        mod1, mod2, ctx, phase = _decode_state(s)

        for action in range(NUM_ACTIONS):
            if phase == PHASE_FIX:
                if action == ACT_FIXATE:
                    # Fixating during fixation -> advance to stimulus phase
                    s_next = _state_index(mod1, mod2, ctx, PHASE_STIM)
                else:
                    # Responding prematurely during fixation -> response phase
                    s_next = _state_index(mod1, mod2, ctx, PHASE_RESP)

            elif phase == PHASE_STIM:
                if action == ACT_FIXATE:
                    # Fixating during stimulus -> stay in stimulus
                    # (agent needs time to observe)
                    s_next = _state_index(mod1, mod2, ctx, PHASE_STIM)
                else:
                    # Responding during stimulus -> response phase
                    s_next = _state_index(mod1, mod2, ctx, PHASE_RESP)

            elif phase == PHASE_RESP:
                # Response phase is absorbing
                s_next = s

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

    Agent knows it starts in fixation phase but doesn't know stimuli
    or context.
    """
    D = np.zeros(NUM_STATES)
    for mod1 in range(2):
        for mod2 in range(2):
            for ctx in range(2):
                s = _state_index(mod1, mod2, ctx, PHASE_FIX)
                D[s] = 1.0
    D = D / D.sum()  # 1/8 for each fixation state
    return D


def build_context_dm_model(
    reward_magnitude: float = 3.0,
    T: int = 3,
) -> GenerativeModel:
    """Build the context-dependent decision making generative model.

    Args:
        reward_magnitude: Strength of reward/punishment preference.
        T: Planning horizon (fixation -> stimulus -> response = 3 steps).

    Returns:
        GenerativeModel configured for context-dependent DM.
    """
    A = _build_A()
    B = _build_B()
    C = _build_C(reward_magnitude)
    D = _build_D()

    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


class ContextDMEnv:
    """Context-dependent decision making environment.

    Two stimulus modalities (each left or right) are presented simultaneously.
    A context cue indicates which modality is task-relevant. The agent must
    respond with the direction of the attended modality.

    Args:
        mod1_dir: Direction of modality 1 ("left" or "right").
        mod2_dir: Direction of modality 2 ("left" or "right").
        context: Which modality to attend ("mod1" or "mod2").
        seed: Random seed.
    """

    def __init__(
        self,
        mod1_dir: str = "left",
        mod2_dir: str = "right",
        context: str = "mod1",
        seed: int = 42,
    ):
        self.mod1 = DIR_LEFT if mod1_dir == "left" else DIR_RIGHT
        self.mod2 = DIR_LEFT if mod2_dir == "left" else DIR_RIGHT
        self.ctx = CTX_MOD1 if context == "mod1" else CTX_MOD2
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> int:
        """Reset to fixation phase. Returns initial observation."""
        self.phase = PHASE_FIX
        self.true_state = _state_index(self.mod1, self.mod2, self.ctx, PHASE_FIX)
        self.agent_response = None
        return self._observe()

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action. Returns (observation, reward, done)."""
        if self.phase == PHASE_FIX:
            if action == ACT_FIXATE:
                # Correct: fixate during fixation -> advance to stimulus
                self.phase = PHASE_STIM
            else:
                # Premature response
                self.phase = PHASE_RESP
                self.agent_response = action

        elif self.phase == PHASE_STIM:
            if action == ACT_FIXATE:
                # Stay in stimulus (observe stimuli)
                pass
            else:
                # Make a response
                self.phase = PHASE_RESP
                self.agent_response = action

        # else: PHASE_RESP is absorbing

        self.true_state = _state_index(
            self.mod1,
            self.mod2,
            self.ctx,
            self.phase,
        )

        obs = self._observe()

        reward = 0.0
        done = False
        if self.phase == PHASE_RESP:
            done = True
            correct_action = _correct_response(self.mod1, self.mod2, self.ctx)
            if self.agent_response == correct_action:
                reward = 1.0
            else:
                reward = -1.0

        return obs, reward, done

    def _observe(self) -> int:
        """Generate observation from current phase and stimulus config."""
        if self.phase == PHASE_FIX:
            # During fixation, randomly emit fixation or context cue
            if self.rng.random() < 0.5:
                return OBS_FIXATION
            else:
                return (
                    OBS_CTX_ATTEND_MOD1 if self.ctx == CTX_MOD1 else OBS_CTX_ATTEND_MOD2
                )

        elif self.phase == PHASE_STIM:
            # During stimulus, randomly emit one of the two modality obs
            if self.rng.random() < 0.5:
                return (
                    OBS_STIM_LEFT_MOD1 if self.mod1 == DIR_LEFT else OBS_STIM_RIGHT_MOD1
                )
            else:
                return (
                    OBS_STIM_LEFT_MOD2 if self.mod2 == DIR_LEFT else OBS_STIM_RIGHT_MOD2
                )

        elif self.phase == PHASE_RESP:
            correct_action = _correct_response(self.mod1, self.mod2, self.ctx)
            if self.agent_response == correct_action:
                return OBS_REWARD
            else:
                return OBS_PUNISHMENT

        return OBS_FIXATION  # fallback


def run_context_dm(
    num_trials: int = 50,
    reward_magnitude: float = 3.0,
    gamma: float = 4.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run the context-dependent DM benchmark over multiple trials.

    Trials are generated with random combinations of mod1 direction,
    mod2 direction, and context. The key test is whether the agent uses
    the context cue to gate its decision.

    Args:
        num_trials: Number of trials to run.
        reward_magnitude: Strength of reward/punishment preference.
        gamma: Agent policy precision.
        seed: Random seed.
        verbose: Print trial-by-trial details.

    Returns:
        Dict with accuracy, congruent_accuracy, incongruent_accuracy,
        context_use_rate, mean_reward, trial_log.
    """
    gm = build_context_dm_model(reward_magnitude=reward_magnitude, T=3)
    agent = AnalyticAgent(gm, gamma=gamma, seed=seed)

    rng = np.random.RandomState(seed)
    results = []

    for trial in range(num_trials):
        # Random trial configuration
        mod1_dir = "left" if rng.random() < 0.5 else "right"
        mod2_dir = "left" if rng.random() < 0.5 else "right"
        context = "mod1" if rng.random() < 0.5 else "mod2"

        # Congruent = both modalities point same direction
        congruent = mod1_dir == mod2_dir

        env = ContextDMEnv(
            mod1_dir=mod1_dir,
            mod2_dir=mod2_dir,
            context=context,
            seed=seed + trial,
        )

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

        # Check if agent used context correctly
        _correct_response(
            DIR_LEFT if mod1_dir == "left" else DIR_RIGHT,
            DIR_LEFT if mod2_dir == "left" else DIR_RIGHT,
            CTX_MOD1 if context == "mod1" else CTX_MOD2,
        )
        # The agent's final response action (last non-fixate action)
        responded_correctly = total_reward > 0

        trial_info = {
            "trial": trial,
            "mod1_dir": mod1_dir,
            "mod2_dir": mod2_dir,
            "context": context,
            "congruent": congruent,
            "actions": actions_taken,
            "reward": total_reward,
            "correct": responded_correctly,
        }
        results.append(trial_info)

        if verbose:
            status = "CORRECT" if responded_correctly else "WRONG"
            cong = "cong" if congruent else "incong"
            print(
                f"  Trial {trial:2d}: ctx={context:4s} "
                f"m1={mod1_dir:5s} m2={mod2_dir:5s} ({cong:6s}) | "
                f"{' -> '.join(actions_taken):30s} | {status}"
            )

    # Compute summary statistics
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / num_trials

    congruent_trials = [r for r in results if r["congruent"]]
    incongruent_trials = [r for r in results if not r["congruent"]]

    cong_acc = (
        sum(1 for r in congruent_trials if r["correct"]) / len(congruent_trials)
        if congruent_trials
        else 0.0
    )
    incong_acc = (
        sum(1 for r in incongruent_trials if r["correct"]) / len(incongruent_trials)
        if incongruent_trials
        else 0.0
    )

    # Context use: on incongruent trials, above-chance performance indicates
    # the agent is actually using the context cue
    context_use_rate = incong_acc  # 0.5 = chance on incongruent

    rewards = [r["reward"] for r in results]

    return {
        "accuracy": accuracy,
        "congruent_accuracy": cong_acc,
        "incongruent_accuracy": incong_acc,
        "context_use_rate": context_use_rate,
        "mean_reward": float(np.mean(rewards)),
        "num_trials": num_trials,
        "trial_log": results,
    }
