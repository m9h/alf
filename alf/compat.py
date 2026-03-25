"""Compatibility layer between ALF and pymdp v1.0.0.

Provides bidirectional conversion between ALF's data-only GenerativeModel
and pymdp's Equinox-based Agent. This enables ALF's cognitive extensions
(HGF, DDM, metacognition, hierarchical) to work with pymdp's core POMDP
engine while preserving backward compatibility with existing ALF code.

Convention differences:
    - ALF matrices: no batch dim, numpy float64
    - pymdp matrices: leading batch dim, jax float32
    - ALF EFE: G (lower = better, minimize)
    - pymdp EFE: neg_efe (higher = better, maximize), so neg_efe = -G

Example:
    >>> from alf.compat import alf_to_pymdp, pymdp_to_alf
    >>> import alf
    >>>
    >>> # ALF model -> pymdp Agent
    >>> gm = alf.GenerativeModel(A=[A], B=[B], C=[C], D=[D])
    >>> agent = alf_to_pymdp(gm)
    >>> qs = agent.infer_states(obs, empirical_prior=agent.D)
    >>>
    >>> # pymdp Agent -> ALF model
    >>> gm2 = pymdp_to_alf(agent)
    >>> alf_agent = alf.AnalyticAgent(gm2)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from alf.generative_model import GenerativeModel


def alf_to_pymdp(
    gm: GenerativeModel,
    gamma: float = 1.0,
    alpha: float = 1.0,
    **agent_kwargs,
):
    """Convert an ALF GenerativeModel to a pymdp Agent.

    Args:
        gm: ALF GenerativeModel with A, B, C, D matrices.
        gamma: Policy precision (pymdp convention). Note: pymdp's gamma
            defaults to 1.0 (vs ALF's 4.0). Adjust accordingly.
        alpha: Action precision for pymdp Agent.
        **agent_kwargs: Additional keyword arguments passed to pymdp Agent.

    Returns:
        pymdp.agent.Agent instance with batch_size=1.

    Raises:
        ImportError: If pymdp (inferactively-pymdp) is not installed.
    """
    try:
        from pymdp.agent import Agent as PyMDP_Agent
    except ImportError:
        raise ImportError(
            "pymdp v1.0.0 required. Install with: "
            "uv pip install inferactively-pymdp"
        )

    # Add batch dimension (pymdp expects leading batch axis)
    A = [jnp.array(a, dtype=jnp.float32)[None, ...] for a in gm.A]
    B = [jnp.array(b, dtype=jnp.float32)[None, ...] for b in gm.B]
    C = [jnp.array(c, dtype=jnp.float32)[None, ...] for c in gm.C]
    D = [jnp.array(d, dtype=jnp.float32)[None, ...] for d in gm.D]

    return PyMDP_Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        gamma=gamma,
        alpha=alpha,
        policy_len=gm.T,
        **agent_kwargs,
    )


def pymdp_to_alf(
    agent,
    T: Optional[int] = None,
) -> GenerativeModel:
    """Convert a pymdp Agent to an ALF GenerativeModel.

    Extracts the A, B, C, D matrices from a pymdp Agent and wraps them
    in ALF's data-only GenerativeModel. Strips the batch dimension
    (takes first element if batch_size > 1).

    Args:
        agent: pymdp.agent.Agent instance.
        T: Planning horizon. If None, uses agent.policy_len.

    Returns:
        ALF GenerativeModel.
    """
    # Strip batch dimension (take first batch element)
    A = [np.array(a[0]) for a in agent.A]
    B = [np.array(b[0]) for b in agent.B]
    C = [np.array(c[0]) for c in agent.C]
    D = [np.array(d[0]) for d in agent.D]

    if T is None:
        T = agent.policy_len

    # Extract E (policy prior) if available
    E = np.array(agent.E[0]) if agent.E is not None else None

    return GenerativeModel(A=A, B=B, C=C, D=D, E=E, T=T)


def neg_efe_to_G(neg_efe: jnp.ndarray) -> np.ndarray:
    """Convert pymdp's neg_efe (higher=better) to ALF's G (lower=better).

    Args:
        neg_efe: pymdp negative EFE, shape (batch, num_policies) or (num_policies,).

    Returns:
        G as numpy array with batch dim stripped if present.
    """
    G = -np.array(neg_efe)
    if G.ndim == 2 and G.shape[0] == 1:
        G = G[0]
    return G


def G_to_neg_efe(G: np.ndarray) -> jnp.ndarray:
    """Convert ALF's G (lower=better) to pymdp's neg_efe (higher=better).

    Args:
        G: ALF EFE values, shape (num_policies,).

    Returns:
        neg_efe as JAX array with batch dim added.
    """
    return jnp.array(-G)[None, ...]
