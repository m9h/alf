"""Tests for the ALF ↔ pymdp compatibility layer."""

import numpy as np
import jax.numpy as jnp
import pytest

from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent

try:
    from pymdp.agent import Agent as PyMDP_Agent
    from alf.compat import alf_to_pymdp, pymdp_to_alf, neg_efe_to_G, G_to_neg_efe
    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False

pytestmark = pytest.mark.skipif(not HAS_PYMDP, reason="pymdp not installed")


@pytest.fixture
def simple_model():
    """3-state, 3-obs, 2-action model."""
    A = np.array([[0.9, 0.05, 0.05],
                  [0.05, 0.9, 0.05],
                  [0.05, 0.05, 0.9]])
    B = np.zeros((3, 3, 2))
    B[:, :, 0] = np.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
    B[:, :, 1] = np.array([[0.1, 0.1, 0.8],
                            [0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1]])
    C = np.array([2.0, 0.0, -1.0])
    D = np.array([1/3, 1/3, 1/3])
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D])


class TestAlfToPymdp:
    def test_creates_agent(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        assert isinstance(agent, PyMDP_Agent)

    def test_matrix_shapes(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        assert agent.A[0].shape == (1, 3, 3)
        assert agent.B[0].shape == (1, 3, 3, 2)
        assert agent.C[0].shape == (1, 3)
        assert agent.D[0].shape == (1, 3)

    def test_batch_size_one(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        assert agent.batch_size == 1

    def test_model_dimensions(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        assert agent.num_states == [3]
        assert agent.num_obs == [3]
        assert agent.num_controls == [2]

    def test_can_run_inference(self, simple_model):
        from jax import random as jr
        agent = alf_to_pymdp(simple_model)
        obs = [jnp.array([0])]
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        assert qs[0].shape[0] == 1  # batch dim
        q_pi, neg_efe = agent.infer_policies(qs)
        assert q_pi.shape == (1, 2)


class TestPymdpToAlf:
    def test_roundtrip(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        gm2 = pymdp_to_alf(agent)
        assert gm2.num_modalities == simple_model.num_modalities
        assert gm2.num_factors == simple_model.num_factors
        assert gm2.num_obs == simple_model.num_obs
        assert gm2.num_states == simple_model.num_states

    def test_matrix_values_roundtrip(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        gm2 = pymdp_to_alf(agent)
        np.testing.assert_allclose(gm2.A[0], simple_model.A[0], atol=1e-5)
        np.testing.assert_allclose(gm2.B[0], simple_model.B[0], atol=1e-5)
        np.testing.assert_allclose(gm2.C[0], simple_model.C[0], atol=1e-5)
        np.testing.assert_allclose(gm2.D[0], simple_model.D[0], atol=1e-5)

    def test_alf_agent_works_on_roundtrip(self, simple_model):
        agent = alf_to_pymdp(simple_model)
        gm2 = pymdp_to_alf(agent)
        alf_agent = AnalyticAgent(gm2, gamma=4.0)
        action, info = alf_agent.step([0])
        assert isinstance(action, int)
        assert "G" in info


class TestEFEConversion:
    def test_neg_efe_to_G(self):
        neg_efe = jnp.array([[1.5, 0.5]])
        G = neg_efe_to_G(neg_efe)
        np.testing.assert_allclose(G, [-1.5, -0.5])
        assert G.ndim == 1  # batch stripped

    def test_G_to_neg_efe(self):
        G = np.array([-1.5, -0.5])
        neg_efe = G_to_neg_efe(G)
        np.testing.assert_allclose(neg_efe, [[1.5, 0.5]], atol=1e-6)
        assert neg_efe.ndim == 2  # batch added

    def test_roundtrip_efe(self):
        G_orig = np.array([2.1, -0.3, 1.5])
        G_back = neg_efe_to_G(G_to_neg_efe(G_orig))
        np.testing.assert_allclose(G_back, G_orig, atol=1e-6)


class TestCrossValidation:
    """Test that ALF and pymdp produce consistent results on the same model."""

    def test_both_prefer_same_action(self, simple_model):
        """Both frameworks should prefer the same best action."""
        from jax import random as jr

        # ALF
        alf_agent = AnalyticAgent(simple_model, gamma=4.0, seed=0)
        _, alf_info = alf_agent.step([0])
        alf_best_policy = np.argmin(alf_info["G"])

        # pymdp
        agent = alf_to_pymdp(simple_model, gamma=4.0)
        obs = [jnp.array([0])]
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        q_pi, neg_efe = agent.infer_policies(qs)
        pymdp_best_policy = int(jnp.argmax(neg_efe[0]))

        assert alf_best_policy == pymdp_best_policy
