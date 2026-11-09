"""EFE checked against enumeration of Bayesian posterior information gain.

The oracle deliberately uses posterior KL, independently of ALF's entropy
decomposition. C contains raw log preferences (no implicit log-softmax).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alf.free_energy import (
    expected_free_energy_decomposed,
    jax_expected_free_energy_decomposed,
)
from alf.hierarchical import (
    HierarchicalGenerativeModel,
    HierarchicalLevel,
    hierarchical_efe,
)
from alf.jax_native import jax_compute_efe_analytic
from alf.multitask import multifactor_sequential_efe
from alf.sequential_efe import jax_sequential_efe, sequential_efe


def posterior_kl_oracle(A, prior, C):
    """Enumerate observations, update by Bayes, and average KL(post || prior)."""
    information = utility = 0.0
    for observation in range(len(C)):
        joint = A[observation] * prior
        probability = sum(joint)
        if probability == 0:
            continue
        posterior = joint / probability
        kl = sum(
            posterior[s] * np.log(posterior[s] / prior[s])
            for s in range(len(prior))
            if posterior[s] > 0
        )
        information += probability * kl
        utility += probability * C[observation]
    return np.array([-utility - information, utility, information])


CHANNELS = [
    (np.full((2, 2), 0.5), np.array([0.5, 0.5])),
    (np.eye(2), np.array([0.5, 0.5])),
    (np.array([[0.9, 0.2], [0.1, 0.8]]), np.array([0.8, 0.2])),
    (np.eye(2), np.array([1.0, 0.0])),
]


@pytest.mark.parametrize(
    "A,prior", CHANNELS, ids=["noise", "perfect", "noisy", "certain"]
)
@pytest.mark.parametrize(
    "backend", ["numpy", "jax", "analytic", "sequential", "jax-sequential"]
)
def test_single_step_matches_posterior_kl(A, prior, backend):
    B = np.eye(2)[:, :, None]
    C = np.array([0.4, -0.6])
    expected = posterior_kl_oracle(A, prior, C)
    if backend == "numpy":
        actual = expected_free_energy_decomposed(A, B, C, prior, 0)
    elif backend == "jax":
        actual = jax.jit(jax_expected_free_energy_decomposed)(A, B, C, prior, 0)
    else:
        if backend == "analytic":
            actual = jax.jit(jax_compute_efe_analytic)(A, B[:, :, 0], C, prior)
        elif backend == "sequential":
            actual = sequential_efe(A, B, C, prior, np.array([0]))
        else:
            actual = jax.jit(jax_sequential_efe)(A, B, C, prior, jnp.array([0]))
        expected = expected[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize("backend", [sequential_efe, jax.jit(jax_sequential_efe)])
def test_open_loop_rollout_matches_posterior_kl(backend):
    A, prior = CHANNELS[2]
    B = np.stack([np.eye(2), np.array([[0.2, 0.7], [0.8, 0.3]])], axis=-1)
    C = np.array([1.1, -0.7])
    policy = np.array([1, 0, 1])
    expected = 0.0
    belief = prior.copy()
    for action in policy:
        belief = B[:, :, action] @ belief
        expected += posterior_kl_oracle(A, belief, C)[0]
    np.testing.assert_allclose(backend(A, B, C, prior, policy), expected, atol=2e-6)


def test_information_beats_noise_with_identical_outcome_preferences():
    # Action 0 reaches informative states; action 1 reaches noisy states.
    # Both actions predict exactly the same marginal observations.
    A = np.array([[1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5]])
    B = np.zeros((4, 4, 2))
    B[:2, :, 0] = 0.5
    B[2:, :, 1] = 0.5
    C, prior = np.zeros(2), np.full(4, 0.25)
    scores = jax.vmap(jax_compute_efe_analytic, in_axes=(None, 2, None, None))(
        A, B, C, prior
    )
    np.testing.assert_allclose(scores, [-np.log(2), 0], atol=2e-6)


def test_multifactor_matches_enumeration_per_modality():
    A = np.array([[[0.9, 0.3], [0.2, 0.6]], [[0.1, 0.7], [0.8, 0.4]]])
    D = [np.array([0.3, 0.7]), np.array([0.8, 0.2])]
    B = [np.eye(2)[:, :, None]] * 2
    C = np.array([0.5, -0.1])
    joint = np.outer(*D).reshape(-1)
    expected = posterior_kl_oracle(A.reshape(2, -1), joint, C)[0]
    actual = multifactor_sequential_efe([A], B, [C], D, np.zeros((2, 2), dtype=int))
    np.testing.assert_allclose(actual, 2 * expected, atol=1e-10)


@pytest.mark.parametrize(
    "A,prior", CHANNELS, ids=["noise", "perfect", "noisy", "certain"]
)
def test_hierarchy_single_level_matches_enumeration(A, prior):
    C = np.array([0.4, -0.6])
    level = HierarchicalLevel(A, np.eye(2)[:, :, None], C, prior)
    actual = hierarchical_efe(HierarchicalGenerativeModel([level]), [prior], 0)
    np.testing.assert_allclose(actual, posterior_kl_oracle(A, prior, C)[0], atol=1e-10)


def test_hierarchy_values_joint_state_context_information():
    # XOR reveals one bit about (child, parent), but none about either alone.
    A = np.stack([np.eye(2), np.eye(2)[::-1]])  # (context, observation, state)
    D = np.full(2, 0.5)
    B = np.eye(2)[:, :, None]
    child = HierarchicalLevel(A, B, np.zeros(2), D)
    parent = HierarchicalLevel(np.full((2, 2), 0.5), B, np.zeros(2), D)
    hierarchy = HierarchicalGenerativeModel([child, parent])
    expected = posterior_kl_oracle(
        A.transpose(1, 0, 2).reshape(2, 4), np.full(4, 0.25), np.zeros(2)
    )[0]
    np.testing.assert_allclose(
        hierarchical_efe(hierarchy, [D, D], 0), expected, atol=1e-10
    )


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_complete_policy_scores_and_posteriors_match_pymdp(horizon):
    pytest.importorskip("pymdp")
    from alf.compat import alf_to_pymdp
    from alf.generative_model import GenerativeModel
    from alf.policy import select_action
    from alf.sequential_efe import evaluate_all_policies_sequential

    A, D = CHANNELS[2]
    B = np.stack([np.eye(2), np.array([[0.2, 0.7], [0.8, 0.3]])], axis=-1)
    C = np.array([0.6, -0.2])
    gm = GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=horizon)
    agent = alf_to_pymdp(gm, gamma=2.0)
    # Same predictive prior, raw C, and no parameter-learning or habit terms.
    posterior, negative_efe = agent.infer_policies([jnp.asarray(D)[None, None, :]])
    scores = evaluate_all_policies_sequential(gm)
    _, probabilities = select_action(
        scores, gm.E, gamma=2.0, rng=np.random.RandomState(0)
    )
    np.testing.assert_allclose(scores, -negative_efe[0], atol=2e-6)
    np.testing.assert_allclose(probabilities, posterior[0], atol=2e-6)


def test_jitted_efe_gradient_matches_independent_finite_difference():
    B, prior, C = jnp.eye(2)[:, :, None], jnp.array([0.7, 0.3]), jnp.zeros(2)

    def likelihood(x):
        return jnp.array([[x, 0.2], [1 - x, 0.8]])

    def score(x):
        return jax_sequential_efe(likelihood(x), B, C, prior, jnp.array([0, 0]))

    delta = 1e-4

    def oracle(x):
        A = np.array([[x, 0.2], [1 - x, 0.8]])
        return 2 * posterior_kl_oracle(A, np.array([0.7, 0.3]), np.zeros(2))[0]

    expected = (oracle(0.85 + delta) - oracle(0.85 - delta)) / (2 * delta)
    np.testing.assert_allclose(jax.jit(jax.grad(score))(0.85), expected, atol=2e-5)
