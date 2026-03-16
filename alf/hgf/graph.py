"""Generalized n-level HGF with arbitrary graph topology.

Extends the fixed 2-level binary and 3-level continuous HGF implementations in
``alf.hgf.updates`` to arbitrary directed-acyclic graphs. Each node can have
multiple parents with configurable coupling types (value or volatility), enabling
multi-branch and multi-input HGF architectures.

The core update equations follow Mathys et al. (2014) and Weber et al. (2024):

    Prediction step (top-down):
        hat_sigma_i = 1/pi_i + exp(kappa_{ij} * mu_j + omega_i)   [volatility]
        hat_sigma_i = 1/pi_i + exp(omega_i)                       [value, top-level]

    Update step (bottom-up):
        Value coupling:   Gaussian prediction error propagated through precision
                          weighting.
        Volatility coupling: Precision-weighted squared prediction error (VOPE).

All functions are jax.jit and jax.grad compatible. Temporal sequences use
jax.lax.scan.

References:
    Mathys, Daunizeau, Friston & Stephan (2011). A Bayesian foundation for
        individual learning under uncertainty. Frontiers in Human Neuroscience.
    Mathys, Lomakina, Daunizeau et al. (2014). Uncertainty in perception and
        the Hierarchical Gaussian Filter. Frontiers in Human Neuroscience.
    Weber, Imbach, Legrand et al. (2024). The generalized Hierarchical Gaussian
        Filter. arXiv:2305.10937.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from alf.hgf.updates import HGFResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HGFNode(NamedTuple):
    """One node in a generalized HGF graph.

    Attributes:
        node_id: Unique integer identifier for this node.
        parent_ids: Tuple of parent node ids (higher-level nodes).
        coupling_types: Tuple of coupling types for each parent.
            Each element is 0 for volatility coupling or 1 for value coupling.
        omega: Tonic log-volatility for this node (scalar).
        kappa: Coupling strengths to each parent, shape (len(parent_ids),).
    """
    node_id: int
    parent_ids: tuple[int, ...]
    coupling_types: tuple[int, ...]
    omega: jnp.ndarray
    kappa: jnp.ndarray


class HGFGraph:
    """Complete HGF graph topology with differentiable parameters.

    Registered as a JAX pytree with static topology (used for Python control
    flow) and dynamic parameters (traced for JIT and differentiable via grad).

    Static fields (auxiliary data, not traced):
        n_nodes, input_node_ids, node_parent_ids, node_coupling_types,
        has_vol_parent, is_binary.

    Dynamic fields (JAX arrays, differentiable):
        omegas, kappas, tonic_drift, pi_u.

    Attributes:
        n_nodes: Total number of nodes in the graph.
        input_node_ids: Which nodes receive observations (tuple of ints).
        node_parent_ids: Per-node parent indices as nested Python tuple.
            node_parent_ids[i] is a tuple of int parent ids for node i.
        node_coupling_types: Per-node coupling types as nested Python tuple.
            node_coupling_types[i][j] is 0 (volatility) or 1 (value).
        has_vol_parent: Per-node flag (Python tuple of bool). True if node i
            has at least one volatility parent.
        is_binary: Whether input nodes use sigmoid (binary) link.
        omegas: Tonic log-volatility for each node, shape (n_nodes,).
        kappas: Coupling strengths as nested tuple of JAX scalars.
            kappas[i][j] is the coupling strength from node i to parent j.
        tonic_drift: Top-level drift (theta), shape (n_nodes,). Non-zero only
            for top-level nodes that use a simple random walk.
        pi_u: Input precision (observation noise), scalar.
    """

    def __init__(
        self,
        n_nodes: int,
        input_node_ids: tuple[int, ...],
        node_parent_ids: tuple[tuple[int, ...], ...],
        node_coupling_types: tuple[tuple[int, ...], ...],
        has_vol_parent: tuple[bool, ...],
        is_binary: bool,
        omegas: jnp.ndarray,
        kappas: tuple[tuple[jnp.ndarray, ...], ...],
        tonic_drift: jnp.ndarray,
        pi_u: jnp.ndarray,
    ):
        self.n_nodes = n_nodes
        self.input_node_ids = input_node_ids
        self.node_parent_ids = node_parent_ids
        self.node_coupling_types = node_coupling_types
        self.has_vol_parent = has_vol_parent
        self.is_binary = is_binary
        self.omegas = omegas
        self.kappas = kappas
        self.tonic_drift = tonic_drift
        self.pi_u = pi_u

    def _replace(self, **kwargs):
        """Return a copy with specified fields replaced (NamedTuple compat)."""
        fields = dict(
            n_nodes=self.n_nodes,
            input_node_ids=self.input_node_ids,
            node_parent_ids=self.node_parent_ids,
            node_coupling_types=self.node_coupling_types,
            has_vol_parent=self.has_vol_parent,
            is_binary=self.is_binary,
            omegas=self.omegas,
            kappas=self.kappas,
            tonic_drift=self.tonic_drift,
            pi_u=self.pi_u,
        )
        fields.update(kwargs)
        return HGFGraph(**fields)


def _hgf_graph_flatten(graph: HGFGraph):
    """Flatten HGFGraph into dynamic children and static auxiliary data."""
    # Dynamic: JAX arrays that should be traced/differentiated
    # Flatten kappas into a flat tuple of scalars
    flat_kappas = []
    for node_kappas in graph.kappas:
        for k in node_kappas:
            flat_kappas.append(k)

    children = (graph.omegas, graph.tonic_drift, graph.pi_u, *flat_kappas)

    # Static: topology (Python ints, bools, tuples)
    # Also store kappa structure so we can unflatten
    kappa_structure = tuple(len(nk) for nk in graph.kappas)
    aux_data = (
        graph.n_nodes,
        graph.input_node_ids,
        graph.node_parent_ids,
        graph.node_coupling_types,
        graph.has_vol_parent,
        graph.is_binary,
        kappa_structure,
    )
    return children, aux_data


def _hgf_graph_unflatten(aux_data, children):
    """Reconstruct HGFGraph from flattened form."""
    (n_nodes, input_node_ids, node_parent_ids, node_coupling_types,
     has_vol_parent, is_binary, kappa_structure) = aux_data

    omegas = children[0]
    tonic_drift = children[1]
    pi_u = children[2]
    flat_kappas = children[3:]

    # Reconstruct nested kappa tuples
    kappas = []
    idx = 0
    for n_k in kappa_structure:
        node_kappas = tuple(flat_kappas[idx:idx + n_k])
        kappas.append(node_kappas)
        idx += n_k

    return HGFGraph(
        n_nodes=n_nodes,
        input_node_ids=input_node_ids,
        node_parent_ids=node_parent_ids,
        node_coupling_types=node_coupling_types,
        has_vol_parent=has_vol_parent,
        is_binary=is_binary,
        omegas=omegas,
        kappas=tuple(kappas),
        tonic_drift=tonic_drift,
        pi_u=pi_u,
    )


jax.tree_util.register_pytree_node(
    HGFGraph, _hgf_graph_flatten, _hgf_graph_unflatten,
)


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def build_graph(
    nodes: list[HGFNode],
    input_node_ids: tuple[int, ...],
    pi_u: float = 100.0,
    is_binary: bool = False,
    tonic_drift: dict[int, float] | None = None,
) -> HGFGraph:
    """Build an HGFGraph from a list of HGFNode specifications.

    Args:
        nodes: List of HGFNode objects defining the topology.
        input_node_ids: Which nodes receive observations.
        pi_u: Input precision (inverse observation noise variance).
        is_binary: Whether input uses sigmoid link (binary HGF).
        tonic_drift: Optional dict mapping node_id -> theta (top-level drift).

    Returns:
        Compiled HGFGraph ready for use with graph_hgf_update.
    """
    n_nodes = len(nodes)

    # Sort nodes by id for consistent ordering
    nodes_sorted = sorted(nodes, key=lambda nd: nd.node_id)

    omegas_list = []
    drift_list = []
    parent_ids_list = []
    coupling_types_list = []
    has_vol_list = []
    kappas_list = []

    td = tonic_drift if tonic_drift is not None else {}

    for node in nodes_sorted:
        i = node.node_id
        omegas_list.append(float(node.omega))
        drift_list.append(td.get(i, 0.0))
        parent_ids_list.append(tuple(node.parent_ids))
        coupling_types_list.append(tuple(node.coupling_types))
        has_vol_list.append(any(ct == 0 for ct in node.coupling_types))

        # Store kappas as tuple of JAX scalars
        kappas_node = tuple(jnp.array(float(node.kappa[j]))
                            for j in range(len(node.parent_ids)))
        kappas_list.append(kappas_node)

    return HGFGraph(
        n_nodes=n_nodes,
        input_node_ids=input_node_ids,
        node_parent_ids=tuple(parent_ids_list),
        node_coupling_types=tuple(coupling_types_list),
        has_vol_parent=tuple(has_vol_list),
        is_binary=is_binary,
        omegas=jnp.array(omegas_list),
        kappas=tuple(kappas_list),
        tonic_drift=jnp.array(drift_list),
        pi_u=jnp.array(pi_u),
    )


# ---------------------------------------------------------------------------
# Single-trial update
# ---------------------------------------------------------------------------

def graph_hgf_update(
    mus: jnp.ndarray,
    pis: jnp.ndarray,
    observation: jnp.ndarray,
    graph: HGFGraph,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-trial update for a generalized n-level HGF.

    Computes predictions top-down and updates beliefs bottom-up, handling
    both value and volatility coupling at each level.

    Args:
        mus: Current posterior means, shape (n_nodes,).
        pis: Current posterior precisions, shape (n_nodes,).
        observation: Observation value (scalar).
        graph: HGFGraph defining topology and parameters.

    Returns:
        Tuple of (new_mus, new_pis, surprise) where surprise is a scalar.
    """
    eps = 1e-16
    n = len(graph.node_parent_ids)  # static Python int

    # --- Prediction step (compute hat_sigma, hat_pi for each node) ---
    hat_mus = mus  # predicted means equal current means
    hat_sigmas = jnp.zeros_like(mus)
    vs = jnp.zeros_like(mus)  # volatility contributions per node

    for i in range(n):
        sigma_i = 1.0 / jnp.clip(pis[i], eps)
        pids = graph.node_parent_ids[i]
        ctypes = graph.node_coupling_types[i]
        n_par = len(pids)

        if n_par == 0:
            # Top-level node: random walk.
            # If tonic_drift > 0, use it as step size (theta).
            # Otherwise, use exp(omega) as step size.
            drift = graph.tonic_drift[i]
            v_i = jnp.where(drift > 0.0, drift, jnp.exp(graph.omegas[i]))
            hat_sigma_i = sigma_i + v_i
        else:
            # Node with parent(s): volatility from volatility-coupled parents
            v_i = jnp.array(0.0)
            for j in range(n_par):
                pid = pids[j]
                ct = ctypes[j]
                kappa_ij = graph.kappas[i][j]
                if ct == 0:  # volatility coupling
                    vol_contribution = jnp.exp(
                        kappa_ij * mus[pid] + graph.omegas[i]
                    )
                    v_i = v_i + vol_contribution

            # If no volatility parent, use tonic volatility only
            if not graph.has_vol_parent[i]:
                v_i = jnp.exp(graph.omegas[i])

            hat_sigma_i = sigma_i + v_i

        hat_sigmas = hat_sigmas.at[i].set(jnp.clip(hat_sigma_i, eps))
        vs = vs.at[i].set(v_i)

    hat_pis = 1.0 / hat_sigmas

    # --- Update step (bottom-up) ---
    new_mus = hat_mus.copy()
    new_pis = hat_pis.copy()

    if graph.is_binary:
        # Binary input: sigmoid link from input node
        input_id = graph.input_node_ids[0]
        hat_mu_1 = jax.nn.sigmoid(hat_mus[input_id])
        hat_mu_1 = jnp.clip(hat_mu_1, eps, 1.0 - eps)

        delta_1 = observation - hat_mu_1
        info_gain = hat_mu_1 * (1.0 - hat_mu_1)
        new_pi_input = hat_pis[input_id] + info_gain
        new_mu_input = (
            hat_mus[input_id] + delta_1 / jnp.clip(new_pi_input, eps)
        )

        new_mus = new_mus.at[input_id].set(new_mu_input)
        new_pis = new_pis.at[input_id].set(new_pi_input)

        surprise = -(
            observation * jnp.log(hat_mu_1)
            + (1.0 - observation) * jnp.log(1.0 - hat_mu_1)
        )
    else:
        # Continuous input: Gaussian observation model
        input_id = graph.input_node_ids[0]
        delta_1 = observation - hat_mus[input_id]
        new_pi_input = hat_pis[input_id] + graph.pi_u
        new_sigma_input = 1.0 / jnp.clip(new_pi_input, eps)
        new_mu_input = (
            hat_mus[input_id] + new_sigma_input * graph.pi_u * delta_1
        )

        new_mus = new_mus.at[input_id].set(new_mu_input)
        new_pis = new_pis.at[input_id].set(new_pi_input)

        total_var = 1.0 / jnp.clip(graph.pi_u, eps) + hat_sigmas[input_id]
        surprise = 0.5 * (
            jnp.log(2.0 * jnp.pi * total_var) + delta_1 ** 2 / total_var
        )

    # Propagate updates bottom-up through the graph.
    # Process nodes in order from input toward top levels.
    for i in range(n):
        pids = graph.node_parent_ids[i]
        ctypes = graph.node_coupling_types[i]
        n_par = len(pids)

        if n_par == 0:
            continue

        # Prediction errors from node i
        delta_mu_i = new_mus[i] - hat_mus[i]
        new_sigma_i = 1.0 / jnp.clip(new_pis[i], eps)

        for j in range(n_par):
            pid = pids[j]
            ct = ctypes[j]
            kappa_ij = graph.kappas[i][j]

            if ct == 0:
                # --- Volatility coupling update ---
                vope_i = hat_pis[i] * (new_sigma_i + delta_mu_i ** 2) - 1.0
                w_i = kappa_ij * vs[i] / jnp.clip(hat_sigmas[i], eps)

                pi_update = 0.5 * w_i ** 2 * (1.0 + vope_i)
                mu_numer = 0.5 * w_i * vope_i
            else:
                # --- Value coupling update ---
                pi_update = kappa_ij ** 2 * hat_pis[i]
                mu_numer = kappa_ij * hat_pis[i] * delta_mu_i

            updated_pi = new_pis[pid] + pi_update
            updated_pi = jnp.clip(updated_pi, eps)
            updated_mu = new_mus[pid] + mu_numer / updated_pi

            new_pis = new_pis.at[pid].set(updated_pi)
            new_mus = new_mus.at[pid].set(updated_mu)

    return new_mus, new_pis, surprise


# ---------------------------------------------------------------------------
# Sequence processing
# ---------------------------------------------------------------------------

def graph_hgf(
    observations: jnp.ndarray,
    graph: HGFGraph,
    initial_mus: jnp.ndarray,
    initial_pis: jnp.ndarray,
) -> HGFResult:
    """Run a generalized HGF on a sequence of observations via jax.lax.scan.

    Args:
        observations: Observation sequence, shape (T,).
        graph: HGFGraph defining topology and parameters.
        initial_mus: Initial posterior means, shape (n_nodes,).
        initial_pis: Initial posterior precisions, shape (n_nodes,).

    Returns:
        HGFResult with mu shape (T, n_nodes), pi shape (T, n_nodes),
        surprise shape (T,).
    """
    def scan_step(carry, obs):
        mus, pis = carry
        new_mus, new_pis, surprise = graph_hgf_update(mus, pis, obs, graph)
        return (new_mus, new_pis), (new_mus, new_pis, surprise)

    init_carry = (initial_mus, initial_pis)
    _, (mu_traj, pi_traj, surprises) = jax.lax.scan(
        scan_step, init_carry, observations
    )

    return HGFResult(mu=mu_traj, pi=pi_traj, surprise=surprises)


def graph_hgf_surprise(
    observations: jnp.ndarray,
    graph: HGFGraph,
    initial_mus: jnp.ndarray,
    initial_pis: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total surprise (NLL) for a generalized HGF.

    Args:
        observations: Observation sequence, shape (T,).
        graph: HGFGraph defining topology and parameters.
        initial_mus: Initial posterior means, shape (n_nodes,).
        initial_pis: Initial posterior precisions, shape (n_nodes,).

    Returns:
        Total surprise (scalar, negative log-likelihood).
    """
    result = graph_hgf(observations, graph, initial_mus, initial_pis)
    return jnp.sum(result.surprise)


# ---------------------------------------------------------------------------
# Standard graph constructors
# ---------------------------------------------------------------------------

def make_standard_3level(
    omega_1: float = -3.0,
    omega_2: float = -3.0,
    kappa_1: float = 1.0,
    kappa_2: float = 1.0,
    theta: float = 0.01,
    pi_u: float = 100.0,
    mu_1_0: float = 0.0,
    sigma_1_0: float = 1.0,
    mu_2_0: float = 0.0,
    sigma_2_0: float = 1.0,
    mu_3_0: float = 0.0,
    sigma_3_0: float = 1.0,
) -> tuple[HGFGraph, jnp.ndarray, jnp.ndarray]:
    """Create a standard 3-level continuous HGF graph.

    Produces an equivalent model to ``continuous_hgf()`` from alf.hgf.updates.

    Node 0: Level 1 (input, value tracking)
    Node 1: Level 2 (volatility of level 1)
    Node 2: Level 3 (meta-volatility, top level)

    Args:
        omega_1: Tonic log-volatility at level 1.
        omega_2: Tonic log-volatility at level 2.
        kappa_1: Volatility coupling strength between levels 1 and 2.
        kappa_2: Volatility coupling strength between levels 2 and 3.
        theta: Top-level drift (meta-volatility step size).
        pi_u: Input precision.
        mu_1_0: Initial mean at level 1.
        sigma_1_0: Initial variance at level 1.
        mu_2_0: Initial mean at level 2.
        sigma_2_0: Initial variance at level 2.
        mu_3_0: Initial mean at level 3.
        sigma_3_0: Initial variance at level 3.

    Returns:
        Tuple of (graph, initial_mus, initial_pis).
    """
    eps = 1e-16
    nodes = [
        HGFNode(
            node_id=0,
            parent_ids=(1,),
            coupling_types=(0,),  # volatility coupling to node 1
            omega=jnp.array(omega_1),
            kappa=jnp.array([kappa_1]),
        ),
        HGFNode(
            node_id=1,
            parent_ids=(2,),
            coupling_types=(0,),  # volatility coupling to node 2
            omega=jnp.array(omega_2),
            kappa=jnp.array([kappa_2]),
        ),
        HGFNode(
            node_id=2,
            parent_ids=(),
            coupling_types=(),
            omega=jnp.array(0.0),  # not used for top level
            kappa=jnp.array([]),
        ),
    ]

    graph = build_graph(
        nodes=nodes,
        input_node_ids=(0,),
        pi_u=pi_u,
        is_binary=False,
        tonic_drift={2: theta},
    )

    initial_mus = jnp.array([mu_1_0, mu_2_0, mu_3_0])
    initial_pis = jnp.array([
        1.0 / jnp.clip(jnp.array(sigma_1_0), eps),
        1.0 / jnp.clip(jnp.array(sigma_2_0), eps),
        1.0 / jnp.clip(jnp.array(sigma_3_0), eps),
    ])

    return graph, initial_mus, initial_pis


def make_binary_2level(
    omega_2: float = -2.0,
    mu_2_0: float = 0.0,
    sigma_2_0: float = 1.0,
) -> tuple[HGFGraph, jnp.ndarray, jnp.ndarray]:
    """Create a 2-level binary HGF graph.

    Produces an equivalent model to ``binary_hgf()`` from alf.hgf.updates.

    Node 0: Level 2 (hidden state, evolved as Gaussian random walk).
    Level 1 is implicit via the sigmoid link function.

    Args:
        omega_2: Tonic log-volatility of the hidden state.
        mu_2_0: Initial mean.
        sigma_2_0: Initial variance.

    Returns:
        Tuple of (graph, initial_mus, initial_pis).
    """
    eps = 1e-16
    nodes = [
        HGFNode(
            node_id=0,
            parent_ids=(),
            coupling_types=(),
            omega=jnp.array(omega_2),
            kappa=jnp.array([]),
        ),
    ]

    graph = build_graph(
        nodes=nodes,
        input_node_ids=(0,),
        pi_u=1.0,  # not used for binary
        is_binary=True,
        tonic_drift={},
    )

    initial_mus = jnp.array([mu_2_0])
    initial_pis = jnp.array([1.0 / jnp.clip(jnp.array(sigma_2_0), eps)])

    return graph, initial_mus, initial_pis
