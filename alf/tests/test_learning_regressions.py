"""Regression cases for initialization and sequence-length sensitive training."""

import jax
import jax.numpy as jnp
import numpy as np

from alf.deep_aif import extract_A_matrix, learn_deep_model
from alf.learning import analytic_nll_single, generate_data, learn_model
from alf.tests.test_learning import build_simple_model


def test_restarts_select_lowest_final_likelihood_and_are_reproducible():
    gm = build_simple_model()
    actions = np.random.RandomState(42).randint(0, 2, 200)
    obs, _ = generate_data(gm.A[0], gm.B[0], gm.D[0], actions, seed=42)
    actions = np.r_[actions, 0]

    def fit(seed, restarts):
        return learn_model(
            obs, actions, 2, 2, 2, num_epochs=300, seed=seed, num_restarts=restarts
        )

    def nll(result):
        return float(
            analytic_nll_single(
                result.log_A_params[0], result.log_B_params[0], gm.D[0], obs, actions
            )
        )

    candidates = [fit(seed, 1) for seed in (0, 2, 4)]
    result = fit(0, 3)
    np.testing.assert_allclose(nll(result), min(map(nll, candidates)), atol=1e-5)
    assert nll(result) < nll(candidates[0]) - 20
    np.testing.assert_array_equal(result.learned_A[0], fit(0, 3).learned_A[0])


def test_deep_update_is_invariant_to_repeating_independent_observations():
    # With one hidden state the observations are independent. Repetition
    # changes total NLL, but must not change the per-observation update.
    results = []
    for count in (1, 10):
        results.append(
            learn_deep_model(
                np.tile([1.0, 0.0], (count, 1)),
                np.zeros(count, dtype=int),
                obs_dim=2,
                num_states=1,
                num_actions=1,
                encoder_hidden=[4],
                transition_hidden=[4],
                num_epochs=3,
                entropy_reg=0.1,
                lr=0.003,
                seed=42,
            )
        )
    for short, long in zip(
        jax.tree_util.tree_leaves(results[0].encoder_params),
        jax.tree_util.tree_leaves(results[1].encoder_params),
    ):
        np.testing.assert_allclose(short, long, atol=1e-6)


def test_extracted_likelihood_matches_training_and_is_invariant_to_state_offsets():
    A = np.array([[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]])
    for bias in (jnp.zeros(2), jnp.array([3.0, -2.0])):
        params = [(jnp.log(A), bias)]
        extracted = extract_A_matrix(params, 3)
        np.testing.assert_allclose(extracted, A, atol=1e-6)
        np.testing.assert_allclose(extracted.sum(axis=0), 1, atol=1e-6)
