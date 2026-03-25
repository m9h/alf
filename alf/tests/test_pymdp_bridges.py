"""Tests for pymdp bridge integration in HGF, DDM, metacognition, hierarchical."""

import numpy as np
import jax.numpy as jnp
import pytest

try:
    from pymdp.agent import Agent as PyMDP_Agent
    from alf.compat import alf_to_pymdp
    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False

from alf.generative_model import GenerativeModel

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


@pytest.fixture
def pymdp_agent(simple_model):
    return alf_to_pymdp(simple_model)


class TestHGFBridgePymdp:
    def test_accepts_pymdp_agent(self, pymdp_agent):
        from alf.hgf.bridge import HGFPerceptualAgent
        from alf.hgf.updates import BinaryHGFParams

        params = BinaryHGFParams(omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0)
        agent = HGFPerceptualAgent(pymdp_agent, params, gamma=4.0)
        assert agent.gm is not None
        assert isinstance(agent.gm, GenerativeModel)

    def test_step_works_with_pymdp(self, pymdp_agent):
        from alf.hgf.bridge import HGFPerceptualAgent
        from alf.hgf.updates import BinaryHGFParams

        params = BinaryHGFParams(omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0)
        agent = HGFPerceptualAgent(pymdp_agent, params, gamma=4.0)
        action, info = agent.step(1.0)
        assert isinstance(action, int)
        assert "G" in info


class TestDDMBridgePymdp:
    def test_neg_efe_to_ddm(self):
        from alf.ddm.bridge import neg_efe_to_ddm, efe_to_ddm

        G = jnp.array([-1.5, -0.5])
        neg_efe = jnp.array([1.5, 0.5])

        params_from_G = efe_to_ddm(G)
        params_from_neg_efe = neg_efe_to_ddm(neg_efe)

        np.testing.assert_allclose(float(params_from_G.v),
                                   float(params_from_neg_efe.v), atol=1e-5)
        np.testing.assert_allclose(float(params_from_G.a),
                                   float(params_from_neg_efe.a), atol=1e-5)

    def test_neg_efe_to_ddm_with_batch_dim(self):
        from alf.ddm.bridge import neg_efe_to_ddm

        neg_efe_batched = jnp.array([[1.5, 0.5]])
        params = neg_efe_to_ddm(neg_efe_batched)
        assert jnp.isfinite(params.v)


class TestMetacognitionPymdp:
    def test_accepts_pymdp_agent(self, pymdp_agent):
        from alf.metacognition import MetacognitiveAgent

        agent = MetacognitiveAgent(pymdp_agent, gamma=4.0)
        assert agent.gm is not None

    def test_step_works_with_pymdp(self, pymdp_agent):
        from alf.metacognition import MetacognitiveAgent

        agent = MetacognitiveAgent(pymdp_agent, gamma=4.0)
        action, info = agent.step([0])
        assert isinstance(action, int)
        assert "metacognitive_confidence" in info
        assert 0.0 <= info["metacognitive_confidence"] <= 1.0

    def test_learn_works_with_pymdp(self, pymdp_agent):
        from alf.metacognition import MetacognitiveAgent

        agent = MetacognitiveAgent(pymdp_agent, gamma=4.0)
        agent.step([0])
        agent.learn(1.0)
        assert len(agent.accuracy_history) == 1


class TestHierarchicalPymdp:
    def test_from_pymdp(self, pymdp_agent):
        from alf.hierarchical import HierarchicalGenerativeModel

        hierarchy = HierarchicalGenerativeModel.from_pymdp(pymdp_agent)
        assert hierarchy.num_levels == 1
        assert hierarchy.levels[0].num_states == 3
        assert hierarchy.levels[0].num_actions == 2

    def test_from_pymdp_with_higher_levels(self, pymdp_agent):
        from alf.hierarchical import HierarchicalGenerativeModel, HierarchicalLevel

        context_A = np.eye(2)
        context_B = np.stack([np.eye(2)] * 2, axis=-1)
        context_C = np.array([1.0, 0.0])
        context_D = np.array([0.5, 0.5])
        context_level = HierarchicalLevel(
            A=context_A, B=context_B, C=context_C, D=context_D,
            temporal_scale=5, level_name="context",
        )

        hierarchy = HierarchicalGenerativeModel.from_pymdp(
            pymdp_agent, higher_levels=[context_level]
        )
        assert hierarchy.num_levels == 2
        assert hierarchy.levels[0].level_name == "sensorimotor"
        assert hierarchy.levels[1].level_name == "context"
