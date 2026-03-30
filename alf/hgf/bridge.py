"""Bridge HGF continuous beliefs to alf's discrete action selection.

Connects HGF perception (continuous Gaussian beliefs) to alf's POMDP action
selection (discrete categorical beliefs). This enables hybrid agents that
perceive via HGF and act via active inference.

The key mapping: HGF posterior N(mu, 1/pi) at level 1 is discretized into
a categorical belief over bins, which feeds into alf's EFE computation and
policy selection.

References:
    Mathys, Lomakina, Daunizeau et al. (2014). Uncertainty in perception and
        the Hierarchical Gaussian Filter. Frontiers in Human Neuroscience.
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active
        Inference. Journal of Mathematical Psychology.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from alf.generative_model import GenerativeModel
from alf import policy as alf_policy
from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf_update,
    continuous_hgf_update,
)


# ---------------------------------------------------------------------------
# Discretization utilities
# ---------------------------------------------------------------------------

def discretize_belief(
    mu: jnp.ndarray,
    pi: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    """Convert a Gaussian belief N(mu, 1/pi) to a categorical distribution.

    Computes the probability mass in each bin defined by bin_edges using
    the Gaussian CDF. Compatible with jax.jit.

    Args:
        mu: Mean of the Gaussian belief (scalar).
        pi: Precision (inverse variance) of the belief (scalar).
        bin_edges: Monotonically increasing bin boundaries, shape (num_bins+1,).
            The first edge should be -inf and last +inf for complete coverage.

    Returns:
        Categorical distribution over bins, shape (num_bins,).
    """
    eps = 1e-16
    sigma = 1.0 / jnp.sqrt(jnp.clip(pi, eps))

    # CDF at each edge: Phi((edge - mu) / sigma)
    z = (bin_edges - mu) / jnp.clip(sigma, eps)
    cdf = jax.scipy.stats.norm.cdf(z)

    # Probability mass in each bin
    probs = jnp.diff(cdf)
    probs = jnp.clip(probs, eps)
    probs = probs / probs.sum()

    return probs


def hgf_to_categorical(
    mu: jnp.ndarray,
    pi: jnp.ndarray,
    num_states: int,
    state_range: tuple[float, float] = (-3.0, 3.0),
) -> jnp.ndarray:
    """Convert HGF belief to categorical over evenly-spaced states.

    Convenience wrapper for discretize_belief using uniform bin spacing
    with infinite tails on both ends.

    Args:
        mu: Mean of the Gaussian belief (scalar).
        pi: Precision of the belief (scalar).
        num_states: Number of discrete states.
        state_range: Range for the interior bin edges (min, max).

    Returns:
        Categorical distribution over states, shape (num_states,).
    """
    interior = jnp.linspace(state_range[0], state_range[1], num_states - 1)
    bin_edges = jnp.concatenate([
        jnp.array([-jnp.inf]),
        interior,
        jnp.array([jnp.inf]),
    ])
    return discretize_belief(mu, pi, bin_edges)


# ---------------------------------------------------------------------------
# Hybrid agent: HGF perception + alf action selection
# ---------------------------------------------------------------------------

class HGFPerceptualAgent:
    """Agent using HGF perception and alf policy selection.

    Perceives the environment via HGF (continuous Gaussian beliefs),
    discretizes beliefs into categorical form, and selects actions
    using alf's softmax policy over expected free energy.

    Supports both binary and continuous HGF variants. The HGF provides
    the perception model (replacing alf's A-matrix-based Bayesian update),
    while the B, C, D matrices and policy selection from alf are reused.

    Args:
        gm: Generative model with B, C matrices for action selection.
            Accepts either an ALF GenerativeModel or a pymdp Agent
            (which will be converted to ALF GM internally).
        hgf_params: HGF parameters (BinaryHGFParams or ContinuousHGFParams).
        gamma: Policy precision (inverse temperature). Default 4.0.
        state_range: Range for discretizing HGF beliefs.
        seed: Random seed.
    """

    def __init__(
        self,
        gm,
        hgf_params: BinaryHGFParams | ContinuousHGFParams,
        gamma: float = 4.0,
        state_range: tuple[float, float] = (-3.0, 3.0),
        seed: int = 42,
    ):
        if not isinstance(gm, GenerativeModel):
            from alf.compat import pymdp_to_alf
            gm = pymdp_to_alf(gm)
        self.gm = gm
        self.hgf_params = hgf_params
        self.gamma = gamma
        self.state_range = state_range
        self.rng = np.random.RandomState(seed)

        self.is_binary = isinstance(hgf_params, BinaryHGFParams)
        self.num_states = gm.num_states[0]
        self.E = gm.E.copy()

        # HGF state
        if self.is_binary:
            self.mu_2 = float(hgf_params.mu_2_0)
            self.pi_2 = 1.0 / max(float(hgf_params.sigma_2_0), 1e-16)
        else:
            self.mu_1 = float(hgf_params.mu_1_0)
            self.pi_1 = 1.0 / max(float(hgf_params.sigma_1_0), 1e-16)
            self.mu_2 = float(hgf_params.mu_2_0)
            self.pi_2 = 1.0 / max(float(hgf_params.sigma_2_0), 1e-16)
            self.mu_3 = float(hgf_params.mu_3_0)
            self.pi_3 = 1.0 / max(float(hgf_params.sigma_3_0), 1e-16)

        # History
        self.belief_history: list[np.ndarray] = []
        self.hgf_mu_history: list[list[float]] = []
        self.hgf_pi_history: list[list[float]] = []
        self.action_history: list[int] = []
        self.surprise_history: list[float] = []

    def step(
        self,
        observation: float,
    ) -> tuple[int, dict[str, Any]]:
        """Perform one step: HGF update + alf action selection.

        Args:
            observation: Scalar observation (binary 0/1 or continuous).

        Returns:
            Tuple of (action_index, info_dict).
        """
        u = jnp.array(float(observation))

        # 1. HGF perceptual update
        if self.is_binary:
            new_mu_2, new_pi_2, surprise = binary_hgf_update(
                jnp.array(self.mu_2), jnp.array(self.pi_2),
                u, self.hgf_params.omega_2,
            )
            self.mu_2 = float(new_mu_2)
            self.pi_2 = float(new_pi_2)
            self.hgf_mu_history.append([self.mu_2])
            self.hgf_pi_history.append([self.pi_2])

            # Discretize level 2 belief for action selection
            beliefs = hgf_to_categorical(
                new_mu_2, new_pi_2,
                self.num_states, self.state_range,
            )
        else:
            p = self.hgf_params
            (new_mu_1, new_pi_1, new_mu_2, new_pi_2,
             new_mu_3, new_pi_3, surprise) = continuous_hgf_update(
                jnp.array(self.mu_1), jnp.array(self.pi_1),
                jnp.array(self.mu_2), jnp.array(self.pi_2),
                jnp.array(self.mu_3), jnp.array(self.pi_3),
                u, p.omega_1, p.omega_2,
                p.kappa_1, p.kappa_2, p.theta, p.pi_u,
            )
            self.mu_1 = float(new_mu_1)
            self.pi_1 = float(new_pi_1)
            self.mu_2 = float(new_mu_2)
            self.pi_2 = float(new_pi_2)
            self.mu_3 = float(new_mu_3)
            self.pi_3 = float(new_pi_3)
            self.hgf_mu_history.append([self.mu_1, self.mu_2, self.mu_3])
            self.hgf_pi_history.append([self.pi_1, self.pi_2, self.pi_3])

            # Discretize level 1 belief for action selection
            beliefs = hgf_to_categorical(
                new_mu_1, new_pi_1,
                self.num_states, self.state_range,
            )

        beliefs_np = np.array(beliefs)
        self.belief_history.append(beliefs_np.copy())
        self.surprise_history.append(float(surprise))

        # 2. Compute EFE for each action (single-step)
        from alf.free_energy import expected_free_energy_decomposed

        num_actions = self.gm.num_actions[0]
        G = np.zeros(num_actions)
        for a in range(num_actions):
            decomp = expected_free_energy_decomposed(
                self.gm.A[0], self.gm.B[0], self.gm.C[0],
                beliefs_np, a,
            )
            G[a] = decomp.G_total

        # 3. Action selection
        action, policy_probs = alf_policy.select_action(
            G, self.E[:num_actions], self.gamma, rng=self.rng,
        )
        self.action_history.append(action)

        info = {
            "beliefs": beliefs_np,
            "G": G,
            "policy_probs": policy_probs,
            "surprise": float(surprise),
            "hgf_mu": self.hgf_mu_history[-1],
            "hgf_pi": self.hgf_pi_history[-1],
        }
        return action, info

    def reset(self) -> None:
        """Reset HGF state to initial conditions."""
        if self.is_binary:
            self.mu_2 = float(self.hgf_params.mu_2_0)
            self.pi_2 = 1.0 / max(float(self.hgf_params.sigma_2_0), 1e-16)
        else:
            self.mu_1 = float(self.hgf_params.mu_1_0)
            self.pi_1 = 1.0 / max(float(self.hgf_params.sigma_1_0), 1e-16)
            self.mu_2 = float(self.hgf_params.mu_2_0)
            self.pi_2 = 1.0 / max(float(self.hgf_params.sigma_2_0), 1e-16)
            self.mu_3 = float(self.hgf_params.mu_3_0)
            self.pi_3 = 1.0 / max(float(self.hgf_params.sigma_3_0), 1e-16)
        self.belief_history.clear()
        self.hgf_mu_history.clear()
        self.hgf_pi_history.clear()
        self.action_history.clear()
        self.surprise_history.clear()
