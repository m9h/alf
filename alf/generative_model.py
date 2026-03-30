"""Generative model for Active Inference on discrete POMDPs.

Data-only representation of a discrete POMDP generative model. Unlike
the PGMax version, this does NOT build factor graphs — it stores only
the A, B, C, D, E matrices and auto-computed attributes (num_modalities,
num_factors, policies, etc.).

Following Smith et al. (2022), a discrete POMDP generative model consists of:

    A: Likelihood mapping P(o|s) — "What observations do I expect given states?"
    B: Transition model P(s'|s,a) — "How do my actions change the world?"
    C: Preferences ln P(o) — "What observations do I prefer?"
    D: Prior beliefs P(s_0) — "What do I believe before observing anything?"
    E: Policy prior P(pi) — "What habits/policies am I predisposed toward?"

This module is backend-agnostic: it provides the data structures that can
be consumed by analytic JAX computations (free_energy.py, sequential_efe.py,
jax_native.py) or by external inference backends (PGMax BP, BlackJAX SMC).

References:
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
"""

from typing import Optional
import dataclasses

import numpy as np


@dataclasses.dataclass
class GenerativeModel:
    """A discrete POMDP generative model (data-only, no factor graph).

    This is the core data structure for Active Inference. It wraps the
    standard A, B, C, D, E matrices and provides automatic validation,
    normalization, and policy enumeration.

    Compatible attribute names with pgmax/aif's GenerativeModel so code
    is portable between backends, but this version has no PGMax dependency.

    Args:
        A: Likelihood matrices. Shape: (num_modalities,) list of arrays,
            each with shape (num_obs_m, num_states_f1, ..., num_states_fN).
            For single-factor models: list of (num_obs, num_states) arrays.
        B: Transition matrices. Shape: (num_factors,) list of arrays,
            each with shape (num_states_f, num_states_f, num_actions_f).
            B[f][:, s, a] = P(s'|s, a) for factor f.
        C: Observation preferences (log-preferences). Shape: (num_modalities,)
            list of arrays, each with shape (num_obs_m,).
        D: Prior beliefs over initial states. Shape: (num_factors,) list of
            arrays, each with shape (num_states_f,).
        E: Policy prior (habits). Shape: (num_policies,) array.
            Default: uniform over policies.
        T: Planning horizon (number of future timesteps to consider).
    """

    A: list[np.ndarray]
    B: list[np.ndarray]
    C: list[np.ndarray]
    D: list[np.ndarray]
    E: Optional[np.ndarray] = None
    T: int = 1

    def __post_init__(self):
        # Validate and normalize inputs
        self.A = [np.asarray(a, dtype=np.float64) for a in self.A]
        self.B = [np.asarray(b, dtype=np.float64) for b in self.B]
        self.C = [np.asarray(c, dtype=np.float64) for c in self.C]
        self.D = [np.asarray(d, dtype=np.float64) for d in self.D]

        # Normalization logic
        self.A = [a / np.maximum(a.sum(axis=0, keepdims=True), 1e-12) for a in self.A]
        self.B = [b / np.maximum(b.sum(axis=0, keepdims=True), 1e-12) for b in self.B]
        self.D = [d / np.maximum(d.sum(), 1e-12) for d in self.D]

        # Validation logic
        for i, a in enumerate(self.A):
            if np.any(a < 0):
                raise ValueError(f"Matrix A[{i}] contains negative probabilities.")
        for i, b in enumerate(self.B):
            if np.any(b < 0):
                raise ValueError(f"Matrix B[{i}] contains negative probabilities.")
        for i, d in enumerate(self.D):
            if np.any(d < 0):
                raise ValueError(f"Matrix D[{i}] contains negative probabilities.")

        self.num_modalities = len(self.A)
        self.num_factors = len(self.B)
        self.num_obs = [a.shape[0] for a in self.A]
        self.num_states = [b.shape[0] for b in self.B]
        self.num_actions = [b.shape[-1] for b in self.B]

        # Enumerate all policies (action sequences of length T)
        # For single-factor: each policy is a sequence of T actions
        if self.num_factors == 1:
            na = self.num_actions[0]
            if self.T == 1:
                self.policies = np.arange(na).reshape(-1, 1, 1)
            else:
                # All T-length action sequences
                import itertools
                seqs = list(itertools.product(range(na), repeat=self.T))
                self.policies = np.array(seqs).reshape(-1, self.T, 1)
        else:
            # Multi-factor: policies are joint action sequences
            import itertools
            action_combos = list(itertools.product(
                *[range(na) for na in self.num_actions]
            ))
            if self.T == 1:
                self.policies = np.array(action_combos).reshape(
                    -1, 1, self.num_factors
                )
            else:
                time_combos = list(itertools.product(
                    action_combos, repeat=self.T
                ))
                self.policies = np.array(time_combos).reshape(
                    -1, self.T, self.num_factors
                )

        self.num_policies = len(self.policies)

        if self.E is None:
            self.E = np.ones(self.num_policies) / self.num_policies
        self.E = np.asarray(self.E, dtype=np.float64)
