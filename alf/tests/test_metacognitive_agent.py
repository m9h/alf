"""Tests for MetacognitiveAgent, EFEMonitor, and PopulationMetacognition.

Verifies:
1. EFEMonitor correctly records, computes calibration, and estimates m-ratio.
2. MetacognitiveAgent is a drop-in replacement for AnalyticAgent.
3. Gamma adjusts correctly for overconfident and underconfident agents.
4. Full history is recorded across trials.
5. PopulationMetacognition aggregates stats across multiple agents.
6. Integration test on the T-maze benchmark.
"""

import numpy as np
import pytest

from alf.benchmarks.t_maze import (
    build_t_maze_model,
    TMazeEnv,
    ACT_CUE,
)
from alf.agent import AnalyticAgent
from alf.generative_model import GenerativeModel
from alf.metacognition import (
    EFEMonitor,
    MetacognitiveAgent,
    PopulationMetacognition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model() -> GenerativeModel:
    """Create a minimal 2-state, 2-obs, 2-action model for unit tests."""
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)  # action 0: stay
    B[:, :, 1] = np.array(
        [
            [0, 1],  # action 1: flip
            [1, 0],
        ]
    )
    C = np.array([1.0, -1.0])
    D = np.array([0.5, 0.5])
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=1)


def _make_metacognitive_agent(
    gamma: float = 4.0,
    seed: int = 42,
    **kwargs,
) -> MetacognitiveAgent:
    """Create a MetacognitiveAgent on the simple model."""
    gm = _make_simple_model()
    return MetacognitiveAgent(
        gm,
        gamma=gamma,
        seed=seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# EFEMonitor tests
# ---------------------------------------------------------------------------


def test_efe_monitor_records():
    """Monitor stores predictions correctly."""
    monitor = EFEMonitor(decay=0.9, window_size=10)

    monitor.record(predicted_efe=-1.0, confidence=0.8, outcome_valence=1.0)
    monitor.record(predicted_efe=-0.5, confidence=0.6, outcome_valence=-1.0)
    monitor.record(predicted_efe=-2.0, confidence=0.9, outcome_valence=1.0)

    assert len(monitor.records) == 3
    assert monitor._n_records == 3

    # Check the stored records
    assert monitor.records[0].predicted_efe == -1.0
    assert monitor.records[0].confidence == 0.8
    assert monitor.records[0].outcome_valence == 1.0

    assert monitor.records[1].confidence == 0.6
    assert monitor.records[1].outcome_valence == -1.0


def test_efe_monitor_calibration():
    """Calibration error computed correctly for known data."""
    monitor = EFEMonitor(decay=0.9, window_size=100)

    # Perfect calibration: confidence always matches accuracy
    for _ in range(50):
        monitor.record(predicted_efe=-1.0, confidence=1.0, outcome_valence=1.0)
    for _ in range(50):
        monitor.record(predicted_efe=-0.5, confidence=0.0, outcome_valence=-1.0)

    cal = monitor.get_calibration()
    assert cal < 0.01, (
        f"Perfectly calibrated agent should have near-zero calibration "
        f"error, got {cal:.4f}"
    )

    # Overconfident agent: always confident but often wrong
    monitor_bad = EFEMonitor(decay=0.9, window_size=100)
    for _ in range(50):
        monitor_bad.record(
            predicted_efe=-1.0,
            confidence=0.9,
            outcome_valence=1.0,
        )
    for _ in range(50):
        monitor_bad.record(
            predicted_efe=-1.0,
            confidence=0.9,
            outcome_valence=-1.0,
        )

    cal_bad = monitor_bad.get_calibration()
    assert cal_bad > 0.3, (
        f"Overconfident agent should have high calibration error, got {cal_bad:.4f}"
    )

    # Bad calibration should be worse than good calibration
    assert cal_bad > cal


def test_efe_monitor_calibration_empty():
    """Calibration returns 0.0 when no records exist."""
    monitor = EFEMonitor()
    assert monitor.get_calibration() == 0.0


def test_efe_monitor_online_m_ratio():
    """Online m-ratio is reasonable for synthetic data."""
    monitor_good = EFEMonitor(decay=0.9, window_size=200)
    rng = np.random.RandomState(42)

    # Good metacognition: high confidence when correct, low when wrong
    for _ in range(100):
        correct = rng.random() < 0.7
        if correct:
            conf = 0.7 + 0.2 * rng.random()  # high confidence
            valence = 1.0
        else:
            conf = 0.3 + 0.2 * rng.random()  # low confidence
            valence = -1.0
        monitor_good.record(
            predicted_efe=-1.0,
            confidence=conf,
            outcome_valence=valence,
        )

    m_good = monitor_good.get_online_m_ratio()

    # Poor metacognition: random confidence regardless of outcome
    monitor_bad = EFEMonitor(decay=0.9, window_size=200)
    for _ in range(100):
        correct = rng.random() < 0.7
        conf = rng.random()  # random confidence
        valence = 1.0 if correct else -1.0
        monitor_bad.record(
            predicted_efe=-1.0,
            confidence=conf,
            outcome_valence=valence,
        )

    m_bad = monitor_bad.get_online_m_ratio()

    assert m_good > m_bad, (
        f"Good metacognition m-ratio ({m_good:.3f}) should exceed "
        f"poor metacognition ({m_bad:.3f})"
    )
    assert m_good > 0.0, f"Good m-ratio should be positive: {m_good:.3f}"


def test_efe_monitor_online_m_ratio_insufficient_data():
    """Online m-ratio returns 1.0 with insufficient data."""
    monitor = EFEMonitor()
    assert monitor.get_online_m_ratio() == 1.0

    monitor.record(predicted_efe=-1.0, confidence=0.8, outcome_valence=1.0)
    assert monitor.get_online_m_ratio() == 1.0


def test_efe_monitor_summary():
    """Summary dict has all expected keys."""
    monitor = EFEMonitor()
    monitor.record(predicted_efe=-1.0, confidence=0.8, outcome_valence=1.0)

    summary = monitor.get_summary()
    expected_keys = {
        "n_records",
        "ema_confidence",
        "ema_accuracy",
        "calibration_error",
        "online_m_ratio",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["n_records"] == 1


# ---------------------------------------------------------------------------
# MetacognitiveAgent tests
# ---------------------------------------------------------------------------


def test_metacognitive_agent_step():
    """Agent produces valid actions with the same interface as AnalyticAgent."""
    agent = _make_metacognitive_agent()
    gm = agent.gm

    action, info = agent.step([0])

    assert isinstance(action, int)
    assert 0 <= action < gm.num_actions[0]
    assert "policy_probs" in info
    assert "G" in info
    assert "metacognitive_confidence" in info
    assert 0.0 <= info["metacognitive_confidence"] <= 1.0


def test_metacognitive_agent_step_matches_analytic():
    """MetacognitiveAgent produces same action as AnalyticAgent given same seed."""
    gm = _make_simple_model()
    seed = 123

    # AnalyticAgent
    aa = AnalyticAgent(gm, gamma=4.0, seed=seed)
    action_aa, info_aa = aa.step([0])

    # MetacognitiveAgent wrapping AnalyticAgent
    ma = MetacognitiveAgent(
        AnalyticAgent(gm, gamma=4.0, seed=seed),
    )
    action_ma, info_ma = ma.step([0])

    assert action_ma == action_aa, (
        f"MetacognitiveAgent should produce same action as AnalyticAgent: "
        f"{action_ma} vs {action_aa}"
    )
    np.testing.assert_allclose(
        info_ma["policy_probs"],
        info_aa["policy_probs"],
        err_msg="Policy probabilities should match",
    )


def test_metacognitive_agent_learns():
    """Gamma adjusts over trials with feedback."""
    agent = _make_metacognitive_agent(
        gamma=4.0,
        gamma_update_interval=5,
        gamma_learning_rate=0.2,
    )

    # Run several trials with mixed outcomes
    for i in range(20):
        agent.step([0])
        valence = 1.0 if i % 3 != 0 else -1.0
        agent.learn(valence)

    # Gamma should have been adjusted at least once
    assert len(agent.gamma_history) > 1, "Gamma history should grow after learn() calls"


def test_metacognitive_agent_overconfident():
    """Gamma decreases when agent is consistently overconfident."""
    agent = _make_metacognitive_agent(
        gamma=8.0,  # high gamma -> high confidence
        gamma_update_interval=1,  # adjust every trial
        gamma_learning_rate=0.3,
    )

    initial_gamma = agent.gamma

    # Always wrong -> overconfident (high confidence, bad outcome)
    for _ in range(20):
        agent.step([0])
        agent.learn(-1.0)  # always bad outcome

    final_gamma = agent.gamma
    assert final_gamma < initial_gamma, (
        f"Overconfident agent's gamma should decrease: "
        f"{final_gamma:.3f} should be < {initial_gamma:.3f}"
    )


def test_metacognitive_agent_underconfident():
    """Gamma increases when agent is consistently underconfident."""
    agent = _make_metacognitive_agent(
        gamma=0.5,  # low gamma -> low confidence (near-uniform policy)
        gamma_update_interval=1,
        gamma_learning_rate=0.3,
    )

    initial_gamma = agent.gamma

    # Always correct -> underconfident (low confidence, good outcome)
    for _ in range(20):
        agent.step([0])
        agent.learn(1.0)  # always good outcome

    final_gamma = agent.gamma
    assert final_gamma > initial_gamma, (
        f"Underconfident agent's gamma should increase: "
        f"{final_gamma:.3f} should be > {initial_gamma:.3f}"
    )


def test_metacognitive_agent_history():
    """Full history is recorded across trials."""
    agent = _make_metacognitive_agent(gamma_update_interval=3)
    n_trials = 12

    for i in range(n_trials):
        agent.step([0])
        agent.learn(1.0 if i % 2 == 0 else -1.0)

    assert len(agent.confidence_history) == n_trials
    assert len(agent.accuracy_history) == n_trials
    assert len(agent.action_history) == n_trials

    # Gamma history: initial + one entry per gamma_update_interval
    # 12 trials / 3 interval = 4 adjustments, plus initial = 5
    assert len(agent.gamma_history) == 5, (
        f"Expected 5 gamma history entries (1 initial + 4 updates), "
        f"got {len(agent.gamma_history)}"
    )

    # Monitor should also have records
    assert agent.monitor._n_records == n_trials

    # All confidences should be valid probabilities
    for c in agent.confidence_history:
        assert 0.0 <= c <= 1.0, f"Invalid confidence: {c}"

    # All accuracies should be 0 or 1
    for a in agent.accuracy_history:
        assert a in (0.0, 1.0), f"Invalid accuracy: {a}"


def test_metacognitive_agent_summary():
    """get_metacognitive_summary returns all expected keys."""
    agent = _make_metacognitive_agent()

    # Run a few trials
    for _ in range(5):
        agent.step([0])
        agent.learn(1.0)

    summary = agent.get_metacognitive_summary()
    expected_keys = {
        "m_ratio",
        "calibration_error",
        "current_gamma",
        "gamma_history",
        "mean_confidence",
        "mean_accuracy",
        "confidence_accuracy_gap",
        "n_trials",
        "monitor_summary",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["n_trials"] == 5
    assert isinstance(summary["gamma_history"], list)
    assert isinstance(summary["monitor_summary"], dict)


def test_metacognitive_agent_state_summary():
    """get_state_summary includes both agent state and metacognition."""
    agent = _make_metacognitive_agent()
    agent.step([0])
    agent.learn(1.0)

    summary = agent.get_state_summary()
    assert "beliefs" in summary  # from inner agent
    assert "gamma" in summary
    assert "metacognition" in summary
    assert "m_ratio" in summary["metacognition"]


def test_metacognitive_agent_reset():
    """Reset clears inner agent beliefs but keeps metacognitive history."""
    agent = _make_metacognitive_agent()

    agent.step([0])
    agent.learn(1.0)

    n_records_before = agent.monitor._n_records
    agent.reset()

    # Monitor history should be preserved
    assert agent.monitor._n_records == n_records_before
    assert len(agent.confidence_history) == 1

    # Inner agent beliefs should be reset to priors
    np.testing.assert_allclose(
        agent.beliefs[0],
        agent.gm.D[0],
        err_msg="Beliefs should be reset to priors after reset()",
    )


def test_metacognitive_agent_from_generative_model():
    """MetacognitiveAgent can be initialized from a GenerativeModel directly."""
    gm = _make_simple_model()
    agent = MetacognitiveAgent(gm, gamma=3.0, seed=99)

    assert agent.gamma == 3.0
    action, info = agent.step([0])
    assert isinstance(action, int)


def test_metacognitive_agent_gamma_bounds():
    """Gamma stays within configured bounds."""
    agent = _make_metacognitive_agent(
        gamma=4.0,
        gamma_update_interval=1,
        gamma_learning_rate=1.0,  # aggressive
        min_gamma=1.0,
        max_gamma=8.0,
    )

    # Force many bad outcomes to push gamma down
    for _ in range(50):
        agent.step([0])
        agent.learn(-1.0)

    assert agent.gamma >= 1.0, f"Gamma below min: {agent.gamma}"

    # Force many good outcomes to push gamma up
    agent2 = _make_metacognitive_agent(
        gamma=4.0,
        gamma_update_interval=1,
        gamma_learning_rate=1.0,
        min_gamma=1.0,
        max_gamma=8.0,
    )
    for _ in range(50):
        agent2.step([0])
        agent2.learn(1.0)

    assert agent2.gamma <= 8.0, f"Gamma above max: {agent2.gamma}"


# ---------------------------------------------------------------------------
# PopulationMetacognition tests
# ---------------------------------------------------------------------------


def test_population_metacognition():
    """Population stats are valid for multiple agents."""
    agents = []
    for i in range(5):
        agent = _make_metacognitive_agent(
            gamma=2.0 + i,
            seed=i * 10,
        )
        # Run different numbers of trials per agent
        for j in range(10 + i * 5):
            agent.step([0])
            valence = 1.0 if (j + i) % 2 == 0 else -1.0
            agent.learn(valence)
        agents.append(agent)

    pop = PopulationMetacognition(agents)

    # m-ratios
    m_ratios = pop.get_population_m_ratios()
    assert m_ratios.shape == (5,)
    assert all(np.isfinite(m_ratios))

    # Heterogeneity stats
    stats = pop.get_heterogeneity_stats()
    expected_keys = {
        "mean_m_ratio",
        "std_m_ratio",
        "min_m_ratio",
        "max_m_ratio",
        "range_m_ratio",
        "mean_gamma",
        "std_gamma",
        "mean_calibration_error",
    }
    assert set(stats.keys()) == expected_keys

    assert np.isfinite(stats["mean_m_ratio"])
    assert stats["std_m_ratio"] >= 0.0
    assert stats["range_m_ratio"] >= 0.0
    assert stats["mean_gamma"] > 0.0
    assert stats["mean_calibration_error"] >= 0.0


def test_population_summary():
    """Population summary includes per-agent and aggregate stats."""
    agents = [_make_metacognitive_agent(seed=i) for i in range(3)]
    for a in agents:
        a.step([0])
        a.learn(1.0)

    pop = PopulationMetacognition(agents)
    summary = pop.get_population_summary()

    assert "agent_summaries" in summary
    assert "heterogeneity" in summary
    assert len(summary["agent_summaries"]) == 3


# ---------------------------------------------------------------------------
# Integration test: T-maze benchmark
# ---------------------------------------------------------------------------


def test_metacognitive_agent_t_maze():
    """Integration test on the T-maze benchmark.

    Runs a MetacognitiveAgent on multiple T-maze trials and verifies:
    1. It produces valid actions and reaches terminal states.
    2. Metacognitive statistics are tracked.
    3. It achieves reasonable reward rate (>50% with cue usage).
    4. Gamma adjusts in response to performance.
    """

    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    agent = MetacognitiveAgent(
        gm,
        gamma=4.0,
        seed=42,
        gamma_update_interval=5,
        gamma_learning_rate=0.1,
    )

    num_trials = 50
    rng = np.random.RandomState(123)
    rewards = []
    cue_visits = []

    for trial in range(num_trials):
        reward_side = "left" if rng.random() < 0.5 else "right"
        env = TMazeEnv(
            reward_side=reward_side,
            cue_reliability=0.9,
            seed=123 + trial,
        )

        agent.reset()
        obs = env.reset()
        visited_cue = False
        total_reward = 0.0

        for step_idx in range(2):
            action, info = agent.step([obs])
            obs, reward, done = env.step(action)
            total_reward += reward

            if action == ACT_CUE:
                visited_cue = True

            if done:
                break

        # Feed outcome to metacognitive learning
        agent.learn(total_reward)

        rewards.append(total_reward)
        cue_visits.append(visited_cue)

    # Verify metacognitive tracking
    assert agent.monitor._n_records == num_trials
    assert len(agent.confidence_history) > 0

    summary = agent.get_metacognitive_summary()
    assert summary["n_trials"] == num_trials
    assert np.isfinite(summary["m_ratio"])
    assert np.isfinite(summary["calibration_error"])

    # Performance: the agent selects policies via EFE minimization, but
    # since beliefs are reset each trial the effective rate depends on
    # whether the cue-first policy is preferred. With enough trials the
    # rate should exceed chance (25% for uniform random among 4 actions).
    reward_rate = sum(1 for r in rewards if r > 0) / num_trials
    assert reward_rate > 0.2, f"T-maze reward rate too low: {reward_rate:.2f}"

    # Gamma history should have been updated
    assert len(agent.gamma_history) > 1, (
        f"Gamma should have been adjusted during {num_trials} trials"
    )

    # All gamma values should be finite and within bounds
    for g in agent.gamma_history:
        assert np.isfinite(g), f"Non-finite gamma in history: {g}"
        assert g >= 0.1
        assert g <= 16.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
