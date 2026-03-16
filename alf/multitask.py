"""Multitask/hierarchical generative model for cognitive task batteries.

Implements a hierarchical active inference generative model that supports
switching between multiple cognitive neuroscience tasks, following the
compositional structure described in Yang et al. (2019).

The key insight: a battery of 20 cognitive tasks (Go/Anti, Delayed/Immediate,
context-dependent matching, etc.) shares common perceptual and motor structure.
An active inference agent benefits from a generative model that captures this
compositionality, yielding:

    1. Transfer between tasks (shared dynamics)
    2. Efficient task switching (only C/rule changes)
    3. Compositional representations (factored state/observation spaces)

Three modes of operation:

    'independent':    Separate A/B/C/D per task. Baseline with no sharing.
    'shared_dynamics': Shared B (transition) matrix across tasks with
                       task-specific A (likelihood) and C (preferences).
    'compositional':  Fully factored model where state space is decomposed
                       into stimulus, phase, and context factors.  Task identity
                       modulates which observation modalities are attended and
                       which response mapping (C matrix) is active.

Hierarchy:
    Top level  : P(task) -- categorical prior over task identity
    Bottom level: P(o, s | task) -- task-conditioned POMDP

References:
    Yang, Jober, Masber, et al. (2019). Task representations in neural
        networks trained to perform many cognitive tasks. Nature Neuroscience.
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active Inference.
    Friston, Rosch, Parr, Price & Bowman (2018). Deep temporal models and
        active inference. Neuroscience & Biobehavioral Reviews.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from alf.generative_model import GenerativeModel


# ============================================================================
# Yang et al. 2019 task definitions
# ============================================================================

# The 20 tasks from Yang et al. (2019), grouped by category.
# Each tuple: (task_name, attended_modality, response_rule, timing)
#   attended_modality: 1 or 2 (which stimulus modality to attend)
#   response_rule: 'go' (respond toward) or 'anti' (respond away)
#   timing: 'immediate' or 'delayed'

YANG_TASK_DEFINITIONS: dict[str, dict] = {
    # --- Go / Anti tasks ---
    "Go":              {"modality": 1, "rule": "go",    "timing": "immediate", "category": "go_anti"},
    "Anti":            {"modality": 1, "rule": "anti",  "timing": "immediate", "category": "go_anti"},
    "DelayGo":         {"modality": 1, "rule": "go",    "timing": "delayed",   "category": "go_anti"},
    "DelayAnti":       {"modality": 1, "rule": "anti",  "timing": "delayed",   "category": "go_anti"},
    # --- Context-dependent Go / Anti (modality 1 vs 2) ---
    "CtxGo1":          {"modality": 1, "rule": "go",    "timing": "immediate", "category": "context"},
    "CtxGo2":          {"modality": 2, "rule": "go",    "timing": "immediate", "category": "context"},
    "CtxAnti1":        {"modality": 1, "rule": "anti",  "timing": "immediate", "category": "context"},
    "CtxAnti2":        {"modality": 2, "rule": "anti",  "timing": "immediate", "category": "context"},
    "CtxDelayGo1":     {"modality": 1, "rule": "go",    "timing": "delayed",   "category": "context"},
    "CtxDelayGo2":     {"modality": 2, "rule": "go",    "timing": "delayed",   "category": "context"},
    "CtxDelayAnti1":   {"modality": 1, "rule": "anti",  "timing": "delayed",   "category": "context"},
    "CtxDelayAnti2":   {"modality": 2, "rule": "anti",  "timing": "delayed",   "category": "context"},
    # --- Matching tasks ---
    "DM1":             {"modality": 1, "rule": "match",     "timing": "delayed", "category": "matching"},
    "DM2":             {"modality": 2, "rule": "match",     "timing": "delayed", "category": "matching"},
    "CtxDM1":          {"modality": 1, "rule": "ctx_match", "timing": "delayed", "category": "matching"},
    "CtxDM2":          {"modality": 2, "rule": "ctx_match", "timing": "delayed", "category": "matching"},
    "MultiDM":         {"modality": 0, "rule": "multi_match", "timing": "delayed", "category": "matching"},
    # --- Anti-matching tasks ---
    "AntiDM1":         {"modality": 1, "rule": "anti_match", "timing": "delayed", "category": "matching"},
    "AntiDM2":         {"modality": 2, "rule": "anti_match", "timing": "delayed", "category": "matching"},
    # --- Duration discrimination ---
    "Dur1vs2":         {"modality": 0, "rule": "duration",   "timing": "delayed", "category": "duration"},
}

# Number of directions for the stimulus ring (Yang et al. use continuous;
# we discretize to 8 directions for a tractable POMDP).
NUM_DIRECTIONS = 8

# Trial phase states
PHASE_FIXATION = 0
PHASE_STIMULUS = 1
PHASE_DELAY = 2
PHASE_RESPONSE = 3
NUM_PHASES = 4
PHASE_NAMES = ["fixation", "stimulus", "delay", "response"]

# Stimulus strength levels
STRENGTH_ABSENT = 0
STRENGTH_WEAK = 1
STRENGTH_STRONG = 2
NUM_STRENGTHS = 3

# Observation modality indices for the compositional model
OBS_MOD_FIXATION = 0   # fixation signal: on/off (2 outcomes)
OBS_MOD_STIM1 = 1      # stimulus modality 1: direction bin or absent (NUM_DIRECTIONS + 1)
OBS_MOD_STIM2 = 2      # stimulus modality 2: direction bin or absent (NUM_DIRECTIONS + 1)
OBS_MOD_FEEDBACK = 3   # feedback: null/correct/incorrect (3 outcomes)

NUM_OBS_FIXATION = 2
NUM_OBS_STIM = NUM_DIRECTIONS + 1  # 8 directions + absent
NUM_OBS_FEEDBACK = 3


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================================
# Multi-factor sequential EFE (extends sequential_efe.py for multi-factor)
# ============================================================================

def multifactor_sequential_efe(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    C_list: list[np.ndarray],
    D_list: list[np.ndarray],
    policy: np.ndarray,
) -> float:
    """Compute sequential EFE for a multi-factor, multi-modality model.

    Extends the single-factor sequential_efe to handle the factored state
    and observation spaces needed by the compositional model.

    For each timestep t in the policy:
        1. Predict next state per factor: Q(s_f') = B_f[:,:,a_f] @ Q(s_f)
        2. For each modality m, predict observations using the outer product
           of factor beliefs contracted through A_m.
        3. Accumulate pragmatic (preference) and epistemic (ambiguity) terms.

    Args:
        A_list: List of likelihood matrices, one per modality. Each A_m has
            shape (num_obs_m, num_states_f1, ..., num_states_fN) for N factors,
            or (num_obs_m, num_states_f) for single-factor.
        B_list: List of transition matrices, one per factor. Each B_f has
            shape (num_states_f, num_states_f, num_actions_f).
        C_list: List of log-preference vectors, one per modality.
            Each C_m has shape (num_obs_m,).
        D_list: List of prior belief vectors, one per factor.
            Each D_f has shape (num_states_f,).
        policy: Action sequence of shape (T, num_factors) with integer
            action indices per factor.

    Returns:
        G: Total expected free energy (scalar, lower is better).
    """
    eps = 1e-16
    num_factors = len(B_list)
    num_modalities = len(A_list)
    policy = np.asarray(policy)
    if policy.ndim == 1:
        policy = policy.reshape(-1, 1)
    T = policy.shape[0]

    # Initialize factor beliefs
    q_s = [np.clip(d.copy(), eps, None) for d in D_list]
    q_s = [d / d.sum() for d in q_s]

    G_total = 0.0

    for t in range(T):
        # 1. Predict next state per factor
        q_s_next = []
        for f in range(num_factors):
            action_f = int(policy[t, f])
            B_a = B_list[f][:, :, action_f]
            q_f = B_a @ q_s[f]
            q_f = np.clip(q_f, eps, None)
            q_f /= q_f.sum()
            q_s_next.append(q_f)

        # 2. For each modality, compute predicted observations and EFE terms
        for m in range(num_modalities):
            A_m = A_list[m]
            C_m = C_list[m]

            if A_m.ndim == 2 and num_factors == 1:
                # Simple single-factor case
                q_o = A_m @ q_s_next[0]
            else:
                # Multi-factor: contract A_m tensor with factor beliefs
                # A_m has shape (num_obs_m, ns_f1, ns_f2, ..., ns_fN)
                # We compute: q_o[o] = sum_{s1,...,sN} A_m[o,s1,...,sN] * q(s1) * ... * q(sN)
                result = A_m.copy()
                for f in range(num_factors - 1, -1, -1):
                    # Contract the last state-factor axis with q_s_next[f]
                    result = np.tensordot(result, q_s_next[f], axes=([-1], [0]))
                q_o = result

            q_o = np.clip(q_o, eps, None)
            q_o /= q_o.sum()

            # Pragmatic value: E[C(o)]
            pragmatic = np.sum(q_o * C_m)

            # Epistemic value: expected ambiguity
            # H[P(o|s)] averaged over predicted states
            log_A_m = np.log(np.clip(A_m, eps, None))
            # Entropy per joint-state configuration
            if A_m.ndim == 2 and num_factors == 1:
                entropy_per_state = -np.sum(A_m * log_A_m, axis=0)
                epistemic = np.sum(q_s_next[0] * entropy_per_state)
            else:
                entropy_per_config = -np.sum(A_m * log_A_m, axis=0)
                # Contract with factor beliefs to get expected entropy
                result = entropy_per_config.copy()
                for f in range(num_factors - 1, -1, -1):
                    result = np.tensordot(result, q_s_next[f], axes=([-1], [0]))
                epistemic = float(result)

            G_total += -pragmatic - epistemic

        q_s = q_s_next

    return float(G_total)


def evaluate_all_policies_multifactor(
    gm: GenerativeModel,
    beliefs: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Evaluate all policies for a multi-factor generative model.

    Drop-in replacement for evaluate_all_policies_sequential that supports
    arbitrary numbers of factors and modalities.

    Args:
        gm: GenerativeModel instance.
        beliefs: Current beliefs per factor. If None, uses gm.D.

    Returns:
        G: Array of shape (num_policies,) with EFE per policy.
    """
    D_list = beliefs if beliefs is not None else gm.D

    G = np.zeros(gm.num_policies)
    for i, policy in enumerate(gm.policies):
        # policy has shape (T, num_factors)
        G[i] = multifactor_sequential_efe(gm.A, gm.B, gm.C, D_list, policy)

    return G


# ============================================================================
# CompositionalModel: factored generative model for the Yang task battery
# ============================================================================

class CompositionalModel:
    """Factored generative model with compositional task structure.

    Mirrors the structure of Yang et al. (2019)'s task battery with
    factored state and observation spaces suitable for active inference.

    State factors:
        - stimulus: encodes direction (NUM_DIRECTIONS states). In matching
          tasks this represents the *remembered* stimulus direction.
        - phase: trial phase (fixation/stimulus/delay/response = 4 states).
        - context: which modality/rule is active. For non-context tasks this
          is a single state. For context-dependent tasks this distinguishes
          modality 1 vs modality 2 (2 states).

    Observation modalities:
        - fixation: on/off (2 outcomes)
        - stimulus_mod1: direction bin or absent (NUM_DIRECTIONS + 1 outcomes)
        - stimulus_mod2: direction bin or absent (NUM_DIRECTIONS + 1 outcomes)
        - feedback: null/correct/incorrect (3 outcomes)

    The key design:
        A matrices for stimulus modalities are SHARED across tasks.
        The C (preference) matrix changes based on the task rule to implement
        go vs anti, match vs non-match response mappings.
        B matrices (phase transitions) are shared; context transitions depend
        on task instruction.

    Args:
        task_name: Name of the task from YANG_TASK_DEFINITIONS.
        stimulus_direction: True stimulus direction (0..NUM_DIRECTIONS-1)
            for building task-specific priors. If None, uniform prior.
        num_context_states: Number of context states (1 for simple tasks,
            2 for context-dependent tasks).
    """

    def __init__(
        self,
        task_name: str,
        stimulus_direction: Optional[int] = None,
        num_context_states: int = 2,
    ):
        if task_name not in YANG_TASK_DEFINITIONS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {sorted(YANG_TASK_DEFINITIONS.keys())}"
            )

        self.task_name = task_name
        self.task_def = YANG_TASK_DEFINITIONS[task_name]
        self.stimulus_direction = stimulus_direction
        self.num_context_states = num_context_states

        # Build the factored generative model
        self._A = self._build_A()
        self._B = self._build_B()
        self._C = self._build_C()
        self._D = self._build_D()

    @property
    def num_stimulus_states(self) -> int:
        return NUM_DIRECTIONS

    @property
    def num_phase_states(self) -> int:
        return NUM_PHASES

    @property
    def state_factor_sizes(self) -> list[int]:
        return [self.num_stimulus_states, self.num_phase_states, self.num_context_states]

    @property
    def obs_modality_sizes(self) -> list[int]:
        return [NUM_OBS_FIXATION, NUM_OBS_STIM, NUM_OBS_STIM, NUM_OBS_FEEDBACK]

    def _build_A(self) -> list[np.ndarray]:
        """Build likelihood matrices for all four observation modalities.

        Returns list of A matrices, one per modality. Each has shape
        (num_obs_m, num_stim_states, num_phase_states, num_context_states).
        """
        ns = self.num_stimulus_states
        np_ = self.num_phase_states
        nc = self.num_context_states
        eps = 1e-4  # small noise for non-zero entries

        # -- Fixation modality: on during fixation/stimulus/delay, off during response
        A_fix = np.zeros((NUM_OBS_FIXATION, ns, np_, nc))
        for s in range(ns):
            for c in range(nc):
                # Fixation ON for phases 0,1,2 (fixation, stimulus, delay)
                A_fix[1, s, PHASE_FIXATION, c] = 1.0  # ON
                A_fix[1, s, PHASE_STIMULUS, c] = 1.0
                A_fix[1, s, PHASE_DELAY, c] = 1.0
                # Fixation OFF for response phase
                A_fix[0, s, PHASE_RESPONSE, c] = 1.0

        # Add eps floor and normalize for numerical stability
        A_fix = np.clip(A_fix, eps, None)
        A_fix = self._normalize_A(A_fix)

        # -- Stimulus modality 1: reveals direction during stimulus phase
        #    when context matches modality 1 (context state 0)
        A_stim1 = np.ones((NUM_OBS_STIM, ns, np_, nc)) * eps
        for s in range(ns):
            for c in range(nc):
                for ph in range(np_):
                    if ph == PHASE_STIMULUS:
                        # During stimulus phase, modality 1 shows direction
                        # but only informatively when context = mod1 (c=0)
                        # or for non-context tasks
                        if c == 0 or self.task_def["category"] != "context":
                            A_stim1[s, s, ph, c] = 1.0  # direction s -> obs s
                        else:
                            # Context says attend mod2; mod1 still visible but
                            # treated as uninformative (uniform-ish)
                            A_stim1[s, s, ph, c] = 1.0
                    else:
                        # Absent observation (index = NUM_DIRECTIONS)
                        A_stim1[NUM_DIRECTIONS, s, ph, c] = 1.0
        # Normalize columns
        A_stim1 = self._normalize_A(A_stim1)

        # -- Stimulus modality 2: analogous to modality 1 but for context state 1
        A_stim2 = np.ones((NUM_OBS_STIM, ns, np_, nc)) * eps
        for s in range(ns):
            for c in range(nc):
                for ph in range(np_):
                    if ph == PHASE_STIMULUS:
                        if c == 1 or self.task_def["category"] != "context":
                            A_stim2[s, s, ph, c] = 1.0
                        else:
                            A_stim2[s, s, ph, c] = 1.0
                    else:
                        A_stim2[NUM_DIRECTIONS, s, ph, c] = 1.0
        A_stim2 = self._normalize_A(A_stim2)

        # -- Feedback modality: correct/incorrect during response phase
        A_fb = np.zeros((NUM_OBS_FEEDBACK, ns, np_, nc))
        for s in range(ns):
            for c in range(nc):
                for ph in range(np_):
                    if ph == PHASE_RESPONSE:
                        # During response phase: feedback depends on task
                        # For simplicity in the generative model, we encode
                        # a neutral (null=0) feedback; the C matrix handles
                        # the preference for correct responses.
                        # Correct feedback (1) more likely, incorrect (2) less
                        A_fb[1, s, ph, c] = 0.7
                        A_fb[2, s, ph, c] = 0.3
                    else:
                        A_fb[0, s, ph, c] = 1.0  # null feedback

        # Add eps and normalize
        A_fb = np.clip(A_fb, eps, None)
        A_fb = self._normalize_A(A_fb)

        return [A_fix, A_stim1, A_stim2, A_fb]

    @staticmethod
    def _normalize_A(A: np.ndarray) -> np.ndarray:
        """Normalize A matrix so that each column (over observations) sums to 1."""
        # A has shape (num_obs, dim1, dim2, ...) -- normalize over axis 0
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.clip(col_sums, 1e-16, None)
        return A / col_sums

    def _build_B(self) -> list[np.ndarray]:
        """Build transition matrices for each state factor.

        Returns list of B matrices, one per factor:
            B_stim: shape (num_stim, num_stim, num_actions_stim)
            B_phase: shape (num_phase, num_phase, num_actions_phase)
            B_ctx: shape (num_ctx, num_ctx, num_actions_ctx)

        Actions:
            - Stimulus factor: 1 action (identity -- stimulus is fixed within trial)
            - Phase factor: 2 actions (advance phase / hold phase)
            - Context factor: 1 action (identity -- context is set by task instruction)
        """
        ns = self.num_stimulus_states
        np_ = self.num_phase_states
        nc = self.num_context_states

        # Stimulus factor: identity transition (stimulus doesn't change)
        B_stim = np.zeros((ns, ns, 1))
        B_stim[:, :, 0] = np.eye(ns)

        # Phase factor: advance or hold
        # Action 0: hold current phase
        # Action 1: advance to next phase
        B_phase = np.zeros((np_, np_, 2))
        B_phase[:, :, 0] = np.eye(np_)  # hold
        for ph in range(np_):
            next_ph = min(ph + 1, np_ - 1)
            B_phase[next_ph, ph, 1] = 1.0  # advance

        # Context factor: identity (context set externally)
        B_ctx = np.zeros((nc, nc, 1))
        B_ctx[:, :, 0] = np.eye(nc)

        return [B_stim, B_phase, B_ctx]

    def _build_C(self) -> list[np.ndarray]:
        """Build preference vectors for each observation modality.

        The C vector is the key task-specific component. It encodes:
            - Go tasks: prefer observations matching stimulus direction
            - Anti tasks: prefer observations opposite to stimulus direction
            - Match tasks: prefer match/non-match feedback
        """
        rule = self.task_def["rule"]

        # Fixation: slight preference for fixation being on (stay focused)
        C_fix = np.zeros(NUM_OBS_FIXATION)
        C_fix[1] = 0.5   # mild preference for fixation-on

        # Stimulus modalities: preferences depend on rule
        C_stim1 = np.zeros(NUM_OBS_STIM)
        C_stim2 = np.zeros(NUM_OBS_STIM)

        if self.stimulus_direction is not None:
            target_dir = self.stimulus_direction
            if rule == "go":
                # Prefer observing the target direction
                C_stim1[target_dir] = 2.0
                C_stim2[target_dir] = 2.0
            elif rule == "anti":
                # Prefer the opposite direction
                anti_dir = (target_dir + NUM_DIRECTIONS // 2) % NUM_DIRECTIONS
                C_stim1[anti_dir] = 2.0
                C_stim2[anti_dir] = 2.0

        # Feedback: strongly prefer correct, penalize incorrect
        C_fb = np.zeros(NUM_OBS_FEEDBACK)
        C_fb[0] = 0.0    # null
        C_fb[1] = 3.0    # correct
        C_fb[2] = -3.0   # incorrect

        return [C_fix, C_stim1, C_stim2, C_fb]

    def _build_D(self) -> list[np.ndarray]:
        """Build prior beliefs over initial states for each factor."""
        ns = self.num_stimulus_states
        np_ = self.num_phase_states
        nc = self.num_context_states

        # Stimulus direction: uniform if unknown, peaked if known
        D_stim = np.ones(ns) / ns
        if self.stimulus_direction is not None:
            D_stim = np.ones(ns) * 0.01
            D_stim[self.stimulus_direction] = 1.0
            D_stim /= D_stim.sum()

        # Phase: start at fixation
        D_phase = np.zeros(np_)
        D_phase[PHASE_FIXATION] = 1.0

        # Context: set by task instruction
        D_ctx = np.zeros(nc)
        attended_mod = self.task_def["modality"]
        if attended_mod == 1:
            D_ctx[0] = 1.0
        elif attended_mod == 2:
            D_ctx[min(1, nc - 1)] = 1.0
        else:
            # Both modalities or not applicable -- uniform
            D_ctx[:] = 1.0 / nc

        return [D_stim, D_phase, D_ctx]

    def to_generative_model(self, T: int = 2) -> GenerativeModel:
        """Convert to a standard GenerativeModel for use with ALF agents.

        Args:
            T: Planning horizon.

        Returns:
            GenerativeModel instance with the factored matrices.
        """
        return GenerativeModel(
            A=self._A,
            B=self._B,
            C=self._C,
            D=self._D,
            T=T,
        )

    @property
    def A(self) -> list[np.ndarray]:
        return self._A

    @property
    def B(self) -> list[np.ndarray]:
        return self._B

    @property
    def C(self) -> list[np.ndarray]:
        return self._C

    @property
    def D(self) -> list[np.ndarray]:
        return self._D


# ============================================================================
# MultitaskGenerativeModel: top-level manager for multiple tasks
# ============================================================================

class MultitaskGenerativeModel:
    """Hierarchical generative model for multiple cognitive tasks.

    Two-level hierarchy:
        - Top level: task identity (which of the N tasks)
        - Bottom level: task-specific POMDP dynamics

    Supports three modes:

    1. 'independent':
        Separate A/B/C/D per task. Each task gets its own GenerativeModel.
        No parameter sharing. This is the baseline.

    2. 'shared_dynamics':
        Shared B matrix across all tasks. Task-specific A and C matrices.
        Captures the insight that transition dynamics (how actions change
        states) are common across tasks.

    3. 'compositional':
        Fully factored model using CompositionalModel. The state space is
        decomposed into (stimulus, phase, context) factors. Task identity
        determines which C (preference) and D (prior) vectors are active.
        A and B matrices are fully shared.

    Args:
        task_models: Dictionary mapping task name to GenerativeModel. For
            'compositional' mode these can be built from CompositionalModel.
        mode: One of 'independent', 'shared_dynamics', 'compositional'.
        task_prior: Prior distribution over tasks. If None, uniform.
    """

    VALID_MODES = ("independent", "shared_dynamics", "compositional")

    def __init__(
        self,
        task_models: dict[str, GenerativeModel],
        mode: str = "independent",
        task_prior: Optional[np.ndarray] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}."
            )
        if not task_models:
            raise ValueError("task_models must contain at least one task.")

        self.mode = mode
        self.task_names = list(task_models.keys())
        self.num_tasks = len(self.task_names)
        self._task_models = dict(task_models)

        # Task prior P(task)
        if task_prior is not None:
            self.task_prior = np.asarray(task_prior, dtype=np.float64)
            self.task_prior = np.clip(self.task_prior, 1e-16, None)
            self.task_prior /= self.task_prior.sum()
        else:
            self.task_prior = np.ones(self.num_tasks) / self.num_tasks

        # Mode-specific setup
        if mode == "shared_dynamics":
            self._setup_shared_dynamics()
        elif mode == "compositional":
            self._setup_compositional()

    def _setup_shared_dynamics(self) -> None:
        """For shared_dynamics mode: compute average B matrix and replace."""
        first_gm = next(iter(self._task_models.values()))
        num_factors = first_gm.num_factors

        # Average B matrices across tasks
        shared_B = []
        for f in range(num_factors):
            B_sum = np.zeros_like(first_gm.B[f])
            for gm in self._task_models.values():
                B_sum += gm.B[f]
            B_avg = B_sum / self.num_tasks
            # Re-normalize columns
            for a in range(B_avg.shape[-1]):
                col_sums = B_avg[:, :, a].sum(axis=0, keepdims=True)
                col_sums = np.clip(col_sums, 1e-16, None)
                B_avg[:, :, a] /= col_sums
            shared_B.append(B_avg)

        # Replace B in all task models
        for name in self._task_models:
            gm = self._task_models[name]
            self._task_models[name] = GenerativeModel(
                A=gm.A, B=shared_B, C=gm.C, D=gm.D, E=gm.E, T=gm.T,
            )

    def _setup_compositional(self) -> None:
        """For compositional mode: verify models share compatible structure."""
        first_gm = next(iter(self._task_models.values()))
        for name, gm in self._task_models.items():
            if gm.num_factors != first_gm.num_factors:
                raise ValueError(
                    f"Compositional mode requires all tasks to have the same "
                    f"number of factors. Task '{name}' has {gm.num_factors}, "
                    f"expected {first_gm.num_factors}."
                )
            if gm.num_modalities != first_gm.num_modalities:
                raise ValueError(
                    f"Compositional mode requires all tasks to have the same "
                    f"number of modalities. Task '{name}' has "
                    f"{gm.num_modalities}, expected {first_gm.num_modalities}."
                )

    def get_model_for_task(self, task_name: str) -> GenerativeModel:
        """Return the effective generative model for a specific task.

        Args:
            task_name: Name of the task.

        Returns:
            GenerativeModel configured for the given task.

        Raises:
            KeyError: If task_name is not in the model.
        """
        if task_name not in self._task_models:
            raise KeyError(
                f"Unknown task '{task_name}'. "
                f"Available: {sorted(self._task_models.keys())}"
            )
        return self._task_models[task_name]

    def infer_task(self, observation_history: list[list[int]]) -> np.ndarray:
        """Infer which task is active from a sequence of observations.

        Uses Bayesian model comparison: computes the marginal likelihood
        of the observation sequence under each task's generative model,
        then applies Bayes' rule with the task prior.

        P(task | o_{1:T}) ∝ P(o_{1:T} | task) * P(task)

        The marginal likelihood is approximated using the predictive
        distribution from the prior:

            P(o_t | task) = sum_s A[o_t, s] * q(s_t | task)

        where q(s_t | task) is propagated forward using the task's B matrix
        under a default (hold) policy.

        Args:
            observation_history: List of observations, each a list of
                observation indices (one per modality).

        Returns:
            Posterior distribution over tasks, shape (num_tasks,).
        """
        eps = 1e-16
        log_likelihoods = np.zeros(self.num_tasks)

        for task_idx, task_name in enumerate(self.task_names):
            gm = self._task_models[task_name]

            # Initialize beliefs from prior
            beliefs = [d.copy() for d in gm.D]
            beliefs = [np.clip(b, eps, None) for b in beliefs]
            beliefs = [b / b.sum() for b in beliefs]

            log_lik = 0.0

            for obs in observation_history:
                for m in range(gm.num_modalities):
                    A_m = gm.A[m]
                    o_m = obs[m]

                    if A_m.ndim == 2:
                        # Single-factor: P(o_m) = A_m[o_m, :] @ q(s)
                        p_o = np.dot(A_m[o_m, :], beliefs[0])
                    else:
                        # Multi-factor: contract A tensor with beliefs
                        # Extract the observation slice
                        A_slice = A_m[o_m]  # shape: (ns_f1, ..., ns_fN)
                        result = A_slice.copy()
                        for f in range(gm.num_factors - 1, -1, -1):
                            result = np.tensordot(
                                result, beliefs[f], axes=([-1], [0])
                            )
                        p_o = float(result)

                    p_o = max(p_o, eps)
                    log_lik += np.log(p_o)

                # Update beliefs using Bayesian update (simplified)
                for f in range(gm.num_factors):
                    for m_idx in range(gm.num_modalities):
                        A_m = gm.A[m_idx]
                        o_m = obs[m_idx]

                        if A_m.ndim == 2 and gm.num_factors == 1:
                            likelihood = A_m[o_m, :]
                            beliefs[f] = beliefs[f] * likelihood
                            beliefs[f] = np.clip(beliefs[f], eps, None)
                            beliefs[f] /= beliefs[f].sum()

                # Propagate forward with default action (hold: action 0)
                for f in range(gm.num_factors):
                    B_f = gm.B[f]
                    beliefs[f] = B_f[:, :, 0] @ beliefs[f]
                    beliefs[f] = np.clip(beliefs[f], eps, None)
                    beliefs[f] /= beliefs[f].sum()

            log_likelihoods[task_idx] = log_lik

        # Bayes' rule
        log_posterior = log_likelihoods + np.log(np.clip(self.task_prior, eps, None))
        posterior = _softmax(log_posterior)

        return posterior

    def get_task_index(self, task_name: str) -> int:
        """Return the index of a task in the task list."""
        return self.task_names.index(task_name)

    @property
    def models(self) -> dict[str, GenerativeModel]:
        """Return dictionary of all task models."""
        return dict(self._task_models)


# ============================================================================
# Factory functions for building task batteries
# ============================================================================

def build_compositional_battery(
    task_names: Optional[list[str]] = None,
    T: int = 2,
    num_context_states: int = 2,
) -> MultitaskGenerativeModel:
    """Build a MultitaskGenerativeModel using compositional factorization.

    Creates CompositionalModel instances for each task and wraps them in
    a MultitaskGenerativeModel with 'compositional' mode.

    Args:
        task_names: Which tasks to include. If None, uses all 20 Yang tasks.
        T: Planning horizon for each task model. Default 2.
        num_context_states: Number of context states. Default 2.

    Returns:
        MultitaskGenerativeModel in compositional mode.
    """
    if task_names is None:
        task_names = list(YANG_TASK_DEFINITIONS.keys())

    task_models = {}
    for name in task_names:
        comp = CompositionalModel(
            task_name=name,
            stimulus_direction=None,  # uniform prior
            num_context_states=num_context_states,
        )
        task_models[name] = comp.to_generative_model(T=T)

    return MultitaskGenerativeModel(task_models, mode="compositional")


def build_simple_task_pair(
    num_states: int = 4,
    num_obs: int = 4,
    num_actions: int = 2,
    T: int = 1,
    seed: int = 42,
) -> MultitaskGenerativeModel:
    """Build a minimal 2-task MultitaskGenerativeModel for testing.

    Creates two simple tasks ('task_A' and 'task_B') with shared state/action
    spaces but different observation mappings and preferences.

    Args:
        num_states: Number of hidden states per task.
        num_obs: Number of observations per task.
        num_actions: Number of actions per task.
        T: Planning horizon.
        seed: Random seed for reproducible matrix generation.

    Returns:
        MultitaskGenerativeModel in 'independent' mode.
    """
    rng = np.random.RandomState(seed)

    def _random_stochastic(rows, cols, rng_):
        """Generate a random column-stochastic matrix."""
        M = rng_.dirichlet(np.ones(rows), size=cols).T
        return M

    def _make_task(name_seed):
        rng_t = np.random.RandomState(seed + hash(name_seed) % 10000)
        A = _random_stochastic(num_obs, num_states, rng_t)
        B = np.zeros((num_states, num_states, num_actions))
        for a in range(num_actions):
            B[:, :, a] = _random_stochastic(num_states, num_states, rng_t)
        C = rng_t.randn(num_obs)
        D = np.ones(num_states) / num_states
        return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)

    return MultitaskGenerativeModel(
        {"task_A": _make_task("task_A"), "task_B": _make_task("task_B")},
        mode="independent",
    )
