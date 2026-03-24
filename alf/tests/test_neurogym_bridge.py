"""Tests for the NeuroGym-to-ALF bridge adapter.

Tests cover:
    - NeurogymAdapter wrapping and running a task
    - Observation discretization (bin and kmeans) producing valid indices
    - build_generative_model returning valid A/B/C/D matrices
    - End-to-end AnalyticAgent stepping through a wrapped environment

Requires: neurogym (``pip install neurogym``).
Tests are skipped gracefully if neurogym is not installed.
"""

from alf.envs.neurogym_bridge import (
    NeurogymAdapter,
    BinDiscretizer,
    KMeansDiscretizer,
)
from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent

import numpy as np
import pytest

# Guard against missing neurogym.
try:
    import neurogym as ngym

    _HAS_NEUROGYM = True
except ImportError:
    _HAS_NEUROGYM = False

pytestmark = pytest.mark.skipif(not _HAS_NEUROGYM, reason="neurogym not installed")
# ---------------------------------------------------------------------------


def _make_simple_env():
    """Create a simple NeuroGym environment for testing."""
    return ngym.make("PerceptualDecisionMaking-v0")


# ---------------------------------------------------------------------------
# BinDiscretizer unit tests
# ---------------------------------------------------------------------------


class TestBinDiscretizer:
    def test_index_in_range(self):
        disc = BinDiscretizer(n_bins=4, low=np.zeros(3), high=np.ones(3), obs_dim=3)
        for _ in range(100):
            obs = np.random.rand(3)
            idx = disc.discretize(obs)
            assert 0 <= idx < disc.n_obs

    def test_deterministic(self):
        disc = BinDiscretizer(n_bins=5, low=np.zeros(2), high=np.ones(2), obs_dim=2)
        obs = np.array([0.3, 0.7])
        idx1 = disc.discretize(obs)
        idx2 = disc.discretize(obs)
        assert idx1 == idx2

    def test_boundary_values(self):
        disc = BinDiscretizer(n_bins=3, low=np.zeros(2), high=np.ones(2), obs_dim=2)
        # Minimum
        assert disc.discretize(np.array([0.0, 0.0])) == 0
        # Near maximum -- should map to highest bin
        idx_max = disc.discretize(np.array([0.99, 0.99]))
        assert 0 <= idx_max < disc.n_obs

    def test_n_obs_count(self):
        disc = BinDiscretizer(n_bins=4, low=np.zeros(3), high=np.ones(3), obs_dim=3)
        assert disc.n_obs == 4**3


# ---------------------------------------------------------------------------
# KMeansDiscretizer unit tests
# ---------------------------------------------------------------------------


class TestKMeansDiscretizer:
    def test_fit_and_discretize(self):
        disc = KMeansDiscretizer(n_clusters=5)
        samples = np.random.rand(200, 3)
        disc.fit(samples)
        assert disc.fitted
        for _ in range(50):
            obs = np.random.rand(3)
            idx = disc.discretize(obs)
            assert 0 <= idx < disc.n_obs

    def test_not_fitted_raises(self):
        disc = KMeansDiscretizer(n_clusters=5)
        with pytest.raises(RuntimeError, match="not been fitted"):
            disc.discretize(np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# NeurogymAdapter tests
# ---------------------------------------------------------------------------


class TestNeurogymAdapterBin:
    """Test the adapter with binning discretization."""

    def test_wrap_and_reset(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=4)
        obs = adapter.reset()
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < adapter.num_obs

    def test_step_returns_valid(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=4)
        adapter.reset()
        obs, reward, done = adapter.step(0)
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < adapter.num_obs
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_full_episode(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        obs = adapter.reset()
        total_steps = 0
        for _ in range(1000):
            action = np.random.randint(adapter.num_actions)
            obs, reward, done = adapter.step(action)
            assert 0 <= obs < adapter.num_obs
            total_steps += 1
            if done:
                break
        assert total_steps > 0

    def test_current_phase_accessible(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        adapter.reset()
        # Phase should be a string.
        assert isinstance(adapter.current_phase, str)

    def test_num_actions_matches_env(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        assert adapter.num_actions == env.action_space.n


class TestNeurogymAdapterKMeans:
    """Test the adapter with KMeans discretization."""

    def test_wrap_fit_and_run(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="kmeans", n_clusters=10)
        adapter.fit(n_samples=500)
        obs = adapter.reset()
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < adapter.num_obs

        obs, reward, done = adapter.step(0)
        assert 0 <= obs < adapter.num_obs


# ---------------------------------------------------------------------------
# build_generative_model tests
# ---------------------------------------------------------------------------


class TestBuildGenerativeModel:
    def test_returns_generative_model(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        assert isinstance(gm, GenerativeModel)

    def test_A_matrix_valid(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        A = gm.A[0]
        # Columns should sum to 1.
        col_sums = A.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)

    def test_B_matrix_valid(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        B = gm.B[0]
        n_act = adapter.num_actions
        for a in range(n_act):
            col_sums = B[:, :, a].sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)

    def test_D_sums_to_one(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        np.testing.assert_allclose(gm.D[0].sum(), 1.0, atol=1e-8)

    def test_C_shape(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        assert gm.C[0].shape == (adapter.num_obs,)

    def test_dimensions_consistent(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        n_obs = gm.num_obs[0]
        n_states = gm.num_states[0]
        assert gm.A[0].shape == (n_obs, n_states)
        assert gm.B[0].shape[0] == n_states
        assert gm.B[0].shape[1] == n_states
        assert gm.B[0].shape[2] == adapter.num_actions

    def test_kmeans_auto_fits(self):
        """build_generative_model with kmeans should auto-fit if needed."""
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="kmeans", n_clusters=8)
        gm = adapter.build_generative_model(n_episodes=10, T=1)
        assert isinstance(gm, GenerativeModel)

    def test_custom_num_states(self):
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=10, num_states=5, T=1)
        assert gm.num_states[0] == 5


# ---------------------------------------------------------------------------
# End-to-end: AnalyticAgent + NeurogymAdapter
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_agent_steps_through_env(self):
        """An AnalyticAgent should be able to step through a wrapped env."""
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=20, T=1)
        agent = AnalyticAgent(gm, gamma=4.0, seed=0)

        obs = adapter.reset()
        agent.reset()

        total_reward = 0.0
        for _ in range(50):
            action, info = agent.step([obs])
            assert 0 <= action < adapter.num_actions
            assert "beliefs" in info
            assert "G" in info

            obs, reward, done = adapter.step(action)
            total_reward += reward
            if done:
                break

        # Just verify it ran without error; reward may be anything.
        assert isinstance(total_reward, float)

    def test_multiple_episodes(self):
        """Run multiple episodes with agent to check stability."""
        env = _make_simple_env()
        adapter = NeurogymAdapter(env, discretization="bin", n_bins=3)
        gm = adapter.build_generative_model(n_episodes=15, T=1)
        agent = AnalyticAgent(gm, gamma=4.0, seed=42)

        for _ep in range(5):
            obs = adapter.reset()
            agent.reset()
            for _ in range(30):
                action, _ = agent.step([obs])
                obs, reward, done = adapter.step(action)
                if done:
                    break


# ---------------------------------------------------------------------------
# Cognitive tasks convenience layer
# ---------------------------------------------------------------------------


class TestCognitiveTasks:
    def test_make_task_known_name(self):
        from alf.envs.cognitive_tasks import make_task

        adapter = make_task("dm1", n_bins=3)
        assert isinstance(adapter, NeurogymAdapter)
        obs = adapter.reset()
        assert 0 <= obs < adapter.num_obs

    def test_make_task_unknown_raises(self):
        from alf.envs.cognitive_tasks import make_task

        with pytest.raises(ValueError, match="Unknown task"):
            make_task("nonexistent_task")

    def test_yang19_tasks_dict(self):
        from alf.envs.cognitive_tasks import YANG19_TASKS

        # Should contain all 19 entries (some env IDs may repeat).
        assert len(YANG19_TASKS) >= 19
        for name, env_id in YANG19_TASKS.items():
            assert isinstance(name, str)
            assert isinstance(env_id, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
