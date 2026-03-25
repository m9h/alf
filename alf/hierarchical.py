"""Hierarchical Active Inference: multi-level generative models.

Implements hierarchical generative models where higher levels provide
contextual priors that modulate lower-level inference, enabling temporal
abstraction and cross-level information gain.

Architecture:

    Level N (slowest):  abstract context, strategy, goals
        |  top-down: modulates A-matrix at Level N-1
        v
    Level N-1:  intermediate context
        |
        v
    ...
        |
        v
    Level 0 (fastest):  concrete sensorimotor states, actions

Key concepts:
    - Context-dependent A matrices: A[context, obs, state] at lower levels
    - Bottom-up/top-down belief propagation via alternating sweeps
    - Cross-level epistemic value: mutual information between context and obs
    - Temporal abstraction: each level has a temporal_scale parameter

References:
    Friston, Rosch, Parr, Price & Bowman (2017). Deep Temporal Models
        and Active Inference. Neuroscience & Biobehavioral Reviews.
    Pezzulo, Rigoli & Friston (2018). Hierarchical Active Inference:
        A Theory of Motivated Control. Trends in Cognitive Sciences.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# -- Numerical helpers --------------------------------------------------------

_EPS = 1e-16


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize an array to sum to 1 along the last axis, avoiding div-by-zero."""
    s = x.sum()
    if s < _EPS:
        return np.ones_like(x) / x.size
    return x / s


def _log_stable(x: np.ndarray) -> np.ndarray:
    """Element-wise log, clipping to avoid -inf."""
    return np.log(np.maximum(x, _EPS))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum p log p."""
    p_safe = np.maximum(p, _EPS)
    return -float(np.sum(p_safe * np.log(p_safe)))


# ==========================================================================
# HierarchicalLevel
# ==========================================================================

class HierarchicalLevel:
    """One level of a hierarchical generative model.

    Parameters
    ----------
    A : np.ndarray
        Likelihood mapping.  Shape depends on whether this level is
        context-dependent:
        - 3-D ``(num_parent_states, num_obs, num_states)`` if the level
          receives contextual modulation from its parent.
        - 2-D ``(num_obs, num_states)`` if this is the top level or has
          no context dependence.
    B : np.ndarray
        Transition tensor, shape ``(num_states, num_states, num_actions)``.
        ``B[s', s, a] = P(s' | s, a)``.
    C : np.ndarray
        Preference vector over observations, shape ``(num_obs,)``.
    D : np.ndarray
        Prior over initial hidden states, shape ``(num_states,)``.
    temporal_scale : int
        How many lower-level steps correspond to one step at this level.
    level_name : str
        Human-readable label (e.g. ``'navigation'``, ``'context'``).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        temporal_scale: int = 1,
        level_name: str = "level",
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64)
        self.temporal_scale = int(temporal_scale)
        self.level_name = str(level_name)

    # -- properties ---------------------------------------------------------

    @property
    def context_dependent(self) -> bool:
        """True if the A matrix is 3-D (modulated by a parent level)."""
        return self.A.ndim == 3

    @property
    def num_parent_states(self) -> int:
        """Number of parent-level states that modulate this level's A.

        Returns 0 for a non-context-dependent (top) level.
        """
        if self.context_dependent:
            return self.A.shape[0]
        return 0

    @property
    def num_obs(self) -> int:
        if self.context_dependent:
            return self.A.shape[1]
        return self.A.shape[0]

    @property
    def num_states(self) -> int:
        if self.context_dependent:
            return self.A.shape[2]
        return self.A.shape[1]

    @property
    def num_actions(self) -> int:
        return self.B.shape[2]

    # -- A-matrix helpers ---------------------------------------------------

    def get_A_for_context(self, context_index: int) -> np.ndarray:
        """Return the 2-D likelihood matrix for a specific parent context.

        Parameters
        ----------
        context_index : int
            Index into the first axis of the 3-D A tensor.

        Returns
        -------
        np.ndarray
            Shape ``(num_obs, num_states)``.
        """
        if not self.context_dependent:
            return self.A.copy()
        return self.A[context_index]

    def get_A_marginalized(self, parent_beliefs: np.ndarray) -> np.ndarray:
        """Marginalise the A-matrix over parent beliefs.

        .. math::

            A_{\\text{eff}}(o|s) = \\sum_c Q(c) \\cdot A(o|s,c)

        Parameters
        ----------
        parent_beliefs : np.ndarray
            Probability distribution over parent states, shape
            ``(num_parent_states,)``.

        Returns
        -------
        np.ndarray
            Shape ``(num_obs, num_states)``.
        """
        if not self.context_dependent:
            return self.A.copy()
        # A shape: (num_parent_states, num_obs, num_states)
        # parent_beliefs shape: (num_parent_states,)
        return np.einsum("c,cos->os", parent_beliefs, self.A)


# ==========================================================================
# HierarchicalGenerativeModel
# ==========================================================================

class HierarchicalGenerativeModel:
    """A hierarchy of generative model levels.

    Parameters
    ----------
    levels : list[HierarchicalLevel]
        Ordered from lowest (fastest, index 0) to highest (slowest).
    """

    def __init__(self, levels: list[HierarchicalLevel]):
        self.levels = list(levels)

    @classmethod
    def from_pymdp(cls, agent, higher_levels: list[HierarchicalLevel] = None,
                   level_name: str = "sensorimotor") -> "HierarchicalGenerativeModel":
        """Create a hierarchy with the lowest level from a pymdp Agent.

        Extracts the A, B, C, D matrices from a pymdp Agent (stripping the
        batch dimension) and uses them as the lowest-level (fastest) model.
        Additional higher levels can be appended.

        Args:
            agent: pymdp.agent.Agent instance.
            higher_levels: Optional list of HierarchicalLevel objects for
                context/strategy levels above the base sensorimotor level.
            level_name: Name for the base level.

        Returns:
            HierarchicalGenerativeModel with the pymdp model as level 0.
        """
        from alf.compat import pymdp_to_alf
        gm = pymdp_to_alf(agent)
        base_level = HierarchicalLevel(
            A=gm.A[0], B=gm.B[0], C=gm.C[0], D=gm.D[0],
            temporal_scale=1, level_name=level_name,
        )
        levels = [base_level] + (higher_levels or [])
        return cls(levels)

    @property
    def num_levels(self) -> int:
        return len(self.levels)


# ==========================================================================
# Hierarchical inference
# ==========================================================================

def hierarchical_infer(
    hierarchy: HierarchicalGenerativeModel,
    observations: list[Optional[int]],
    beliefs: Optional[list[np.ndarray]] = None,
    num_sweeps: int = 5,
) -> list[np.ndarray]:
    """Run bottom-up / top-down belief propagation.

    Parameters
    ----------
    hierarchy : HierarchicalGenerativeModel
        The multi-level model.
    observations : list[int | None]
        One observation per level.  ``None`` means no observation available
        at that level on this step.
    beliefs : list[np.ndarray] | None
        Current beliefs.  If ``None``, each level is initialised from its
        prior ``D``.
    num_sweeps : int
        Number of bottom-up / top-down alternations.

    Returns
    -------
    list[np.ndarray]
        Updated beliefs, one array per level.
    """
    n = hierarchy.num_levels

    # Initialise beliefs
    if beliefs is None:
        beliefs = [_normalize(level.D.copy()) for level in hierarchy.levels]
    else:
        beliefs = [b.copy() for b in beliefs]

    for _sweep in range(num_sweeps):
        # ---- Bottom-up pass (level 0 -> top) ----------------------------
        for i in range(n):
            level = hierarchy.levels[i]
            obs = observations[i] if i < len(observations) else None

            # Get effective A depending on whether we have a parent
            if level.context_dependent and i + 1 < n:
                parent_beliefs = beliefs[i + 1]
                A_eff = level.get_A_marginalized(parent_beliefs)
            elif level.context_dependent:
                # Context-dependent but no parent provided: uniform
                parent_beliefs = np.ones(level.num_parent_states) / level.num_parent_states
                A_eff = level.get_A_marginalized(parent_beliefs)
            else:
                A_eff = level.A

            if obs is not None:
                # Likelihood: P(o | s) for the observed o
                likelihood = A_eff[obs, :]  # shape (num_states,)
                beliefs[i] = _normalize(beliefs[i] * likelihood)

        # ---- Top-down pass: parent beliefs update from child evidence ---
        for i in range(n - 1, 0, -1):
            child_level = hierarchy.levels[i - 1]
            if not child_level.context_dependent:
                continue

            obs = observations[i - 1] if (i - 1) < len(observations) else None
            if obs is None:
                continue

            # For each parent context c, compute likelihood of child's
            # observation given the child's current state beliefs.
            num_parent = child_level.num_parent_states
            context_likelihoods = np.zeros(num_parent)
            for c in range(num_parent):
                A_c = child_level.get_A_for_context(c)
                # P(o | c) = sum_s P(o | s, c) * Q(s)
                context_likelihoods[c] = np.dot(A_c[obs, :], beliefs[i - 1])

            # Update parent beliefs with this bottom-up evidence
            beliefs[i] = _normalize(beliefs[i] * np.maximum(context_likelihoods, _EPS))

    return beliefs


# ==========================================================================
# Hierarchical EFE
# ==========================================================================

def hierarchical_efe(
    hierarchy: HierarchicalGenerativeModel,
    beliefs: list[np.ndarray],
    action: int,
    level_weights: Optional[list[float]] = None,
) -> float:
    """Compute the hierarchical Expected Free Energy for a single action.

    The action is applied at level 0 only.  The EFE includes:
    - Pragmatic term at each level (preference satisfaction).
    - Epistemic term at each level (ambiguity reduction).
    - Cross-level epistemic term for context-dependent levels (mutual
      information between parent state and child observations).

    Parameters
    ----------
    hierarchy : HierarchicalGenerativeModel
        The hierarchy.
    beliefs : list[np.ndarray]
        Current beliefs per level.
    action : int
        Level-0 action index.
    level_weights : list[float] | None
        Weight for each level's contribution.  Defaults to all 1.0.

    Returns
    -------
    float
        Total (weighted) expected free energy.  Lower is better.
    """
    n = hierarchy.num_levels
    if level_weights is None:
        level_weights = [1.0] * n

    G_total = 0.0

    for i in range(n):
        level = hierarchy.levels[i]
        w = level_weights[i] if i < len(level_weights) else 1.0
        if abs(w) < _EPS:
            continue

        if i == 0:
            # Predict next state after taking the action
            B_a = level.B[:, :, action]  # (num_states, num_states)
            qs_next = B_a @ beliefs[i]
            qs_next = _normalize(qs_next)
        else:
            # Higher levels: beliefs stay roughly constant within one
            # lower-level step (temporal abstraction).
            qs_next = beliefs[i]

        # Get effective A
        if level.context_dependent and (i + 1) < n:
            parent_beliefs = beliefs[i + 1]
            A_eff = level.get_A_marginalized(parent_beliefs)
        elif level.context_dependent:
            parent_beliefs = np.ones(level.num_parent_states) / level.num_parent_states
            A_eff = level.get_A_marginalized(parent_beliefs)
        else:
            A_eff = level.A

        # Predicted observation: P(o) = sum_s A_eff(o,s) * Q(s')
        qo = A_eff @ qs_next  # shape (num_obs,)
        qo = np.maximum(qo, _EPS)

        # -- Pragmatic term: C is treated as log-preferences.
        #    pragmatic = sum_o Q(o) * C(o)   (higher = more aligned)
        #    Contribution to G: -pragmatic   (lower G = better)
        pragmatic = float(np.sum(qo * level.C))

        # -- Epistemic term (within-level ambiguity):
        #    Expected entropy of observations given state:
        #    sum_s Q(s') * H[P(o|s)]
        #    High ambiguity contributes positively to G (bad).
        log_A_eff = _log_stable(A_eff)
        entropy_per_state = -np.sum(A_eff * log_A_eff, axis=0)  # (num_states,)
        ambiguity = float(np.sum(qs_next * entropy_per_state))

        G_level = -pragmatic + ambiguity

        # -- Cross-level epistemic term (for context-dependent levels):
        #    Mutual information between parent context and observations
        if level.context_dependent and (i + 1) < n:
            parent_beliefs_here = beliefs[i + 1]
            # H(o) - H(o|c)
            # H(o) already in qo
            H_o = _entropy(qo)

            # H(o|c) = sum_c Q(c) * H(o|c)
            H_o_given_c = 0.0
            for c in range(level.num_parent_states):
                A_c = level.get_A_for_context(c)
                qo_c = A_c @ qs_next
                qo_c = np.maximum(qo_c, _EPS)
                H_o_given_c += parent_beliefs_here[c] * _entropy(qo_c)

            mutual_info = H_o - H_o_given_c
            # Mutual information is valuable (reduces G)
            G_level -= mutual_info

        G_total += w * G_level

    return float(G_total)


def evaluate_all_policies_hierarchical(
    hierarchy: HierarchicalGenerativeModel,
    beliefs: list[np.ndarray],
    level_weights: Optional[list[float]] = None,
) -> np.ndarray:
    """Evaluate the hierarchical EFE for every level-0 action.

    Parameters
    ----------
    hierarchy : HierarchicalGenerativeModel
        The hierarchy.
    beliefs : list[np.ndarray]
        Current beliefs per level.
    level_weights : list[float] | None
        Per-level weight.  Defaults to all 1.0.

    Returns
    -------
    np.ndarray
        Shape ``(num_level0_actions,)``.  EFE value for each action.
    """
    level0 = hierarchy.levels[0]
    num_actions = level0.num_actions
    G = np.zeros(num_actions)
    for a in range(num_actions):
        G[a] = hierarchical_efe(hierarchy, beliefs, a, level_weights)
    return G


# ==========================================================================
# HierarchicalAgent
# ==========================================================================

class HierarchicalAgent:
    """Agent wrapper for hierarchical active inference.

    Maintains beliefs across all levels and selects level-0 actions
    by minimising hierarchical EFE.

    Parameters
    ----------
    hierarchy : HierarchicalGenerativeModel
        The multi-level model.
    gamma : float
        Policy precision (inverse temperature).
    level_weights : list[float] | None
        Per-level EFE weight.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        hierarchy: HierarchicalGenerativeModel,
        gamma: float = 4.0,
        level_weights: Optional[list[float]] = None,
        seed: int = 42,
    ):
        self.hierarchy = hierarchy
        self.gamma = gamma
        self.level_weights = level_weights
        self.rng = np.random.RandomState(seed)

        # Initialise beliefs from priors
        self.beliefs: list[np.ndarray] = [
            _normalize(level.D.copy()) for level in hierarchy.levels
        ]

    def reset(self) -> None:
        """Reset beliefs to priors."""
        self.beliefs = [
            _normalize(level.D.copy()) for level in self.hierarchy.levels
        ]

    def step(
        self,
        observations: list[Optional[int]],
        num_sweeps: int = 5,
    ) -> tuple[int, dict[str, Any]]:
        """Perform one agent step.

        1. Update beliefs via hierarchical inference.
        2. Evaluate hierarchical EFE for each level-0 action.
        3. Select action via softmax policy.

        Parameters
        ----------
        observations : list[int | None]
            One observation per level (``None`` if unobserved).
        num_sweeps : int
            Inference sweeps.

        Returns
        -------
        action : int
            Selected level-0 action.
        info : dict
            ``'beliefs'``, ``'G'``, ``'policy_probs'``.
        """
        # 1. Infer
        self.beliefs = hierarchical_infer(
            self.hierarchy,
            observations,
            beliefs=self.beliefs,
            num_sweeps=num_sweeps,
        )

        # 2. Evaluate
        G = evaluate_all_policies_hierarchical(
            self.hierarchy, self.beliefs, self.level_weights
        )

        # 3. Select (softmax with precision gamma)
        logits = -self.gamma * G
        policy_probs = _softmax(logits)

        action = int(self.rng.choice(len(policy_probs), p=policy_probs))

        info = {
            "beliefs": [b.copy() for b in self.beliefs],
            "G": G,
            "policy_probs": policy_probs,
        }

        return action, info
