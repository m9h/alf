"""Tests for the multitask generative model and agent.

Verifies:
    1. MultitaskGenerativeModel construction in all three modes
    2. get_model_for_task returns valid GenerativeModel instances
    3. CompositionalModel factored state space correctness
    4. MultitaskAgent task switching preserves/resets state correctly
    5. MultitaskAgent.run_trial on hand-built tasks
    6. Task inference from observation history
    7. Multi-factor sequential EFE computation
    8. Full battery execution
"""

import numpy as np
import pytest

from alf.generative_model import GenerativeModel
from alf.multitask import (
    MultitaskGenerativeModel,
    CompositionalModel,
    YANG_TASK_DEFINITIONS,
    NUM_DIRECTIONS,
    NUM_PHASES,
    NUM_OBS_FIXATION,
    NUM_OBS_STIM,
    NUM_OBS_FEEDBACK,
    PHASE_FIXATION,
    PHASE_STIMULUS,
    build_compositional_battery,
    build_simple_task_pair,
    multifactor_sequential_efe,
    evaluate_all_policies_multifactor,
)
from alf.multitask_agent import MultitaskAgent


# ---------------------------------------------------------------------------
# Fixtures: reusable task models
# ---------------------------------------------------------------------------


def _make_simple_gm(
    num_states=4,
    num_obs=4,
    num_actions=2,
    seed=42,
    T=1,
) -> GenerativeModel:
    """Build a simple single-factor GenerativeModel for testing."""
    rng = np.random.RandomState(seed)
    A = rng.dirichlet(np.ones(num_obs), size=num_states).T
    B = np.zeros((num_states, num_states, num_actions))
    for a in range(num_actions):
        B[:, :, a] = rng.dirichlet(np.ones(num_states), size=num_states).T
    C = rng.randn(num_obs)
    D = np.ones(num_states) / num_states
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)


class SimpleEnv:
    """Minimal environment for testing run_trial."""

    def __init__(self, num_obs=4, num_steps=3, seed=42):
        self.num_obs = num_obs
        self.num_steps = num_steps
        self.rng = np.random.RandomState(seed)
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return [self.rng.randint(self.num_obs)]

    def step(self, action):
        self._step_count += 1
        obs = [self.rng.randint(self.num_obs)]
        reward = self.rng.choice([-1.0, 0.0, 1.0])
        done = self._step_count >= self.num_steps
        return obs, reward, done


class SimpleMultiModalEnv:
    """Minimal multi-modality environment for compositional model testing."""

    def __init__(self, obs_sizes, num_steps=3, seed=42):
        self.obs_sizes = obs_sizes
        self.num_steps = num_steps
        self.rng = np.random.RandomState(seed)
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return [self.rng.randint(s) for s in self.obs_sizes]

    def step(self, action):
        self._step_count += 1
        obs = [self.rng.randint(s) for s in self.obs_sizes]
        reward = self.rng.choice([-1.0, 0.0, 1.0])
        done = self._step_count >= self.num_steps
        return obs, reward, done


# ===================================================================
# 1. MultitaskGenerativeModel construction
# ===================================================================


class TestMultitaskGenerativeModelConstruction:
    """Test MultitaskGenerativeModel in all three modes."""

    def test_independent_mode(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="independent"
        )
        assert mtm.num_tasks == 2
        assert mtm.mode == "independent"
        assert set(mtm.task_names) == {"task_A", "task_B"}

    def test_shared_dynamics_mode(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="shared_dynamics"
        )
        assert mtm.mode == "shared_dynamics"

        # B matrices should be identical across tasks (shared)
        model_a = mtm.get_model_for_task("task_A")
        model_b = mtm.get_model_for_task("task_B")
        np.testing.assert_array_equal(model_a.B[0], model_b.B[0])

    def test_shared_dynamics_preserves_task_specific_A(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="shared_dynamics"
        )
        model_a = mtm.get_model_for_task("task_A")
        model_b = mtm.get_model_for_task("task_B")
        # A matrices should differ
        assert not np.allclose(model_a.A[0], model_b.A[0])

    def test_compositional_mode_with_yang_tasks(self):
        mtm = build_compositional_battery(["Go", "Anti", "DelayGo"], T=1)
        assert mtm.mode == "compositional"
        assert mtm.num_tasks == 3
        assert "Go" in mtm.task_names
        assert "Anti" in mtm.task_names

    def test_invalid_mode_raises(self):
        gm = _make_simple_gm()
        with pytest.raises(ValueError, match="Invalid mode"):
            MultitaskGenerativeModel({"task": gm}, mode="invalid")

    def test_empty_task_models_raises(self):
        with pytest.raises(ValueError, match="at least one task"):
            MultitaskGenerativeModel({}, mode="independent")

    def test_custom_task_prior(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        prior = np.array([0.8, 0.2])
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b},
            mode="independent",
            task_prior=prior,
        )
        np.testing.assert_allclose(mtm.task_prior.sum(), 1.0, atol=1e-10)
        assert mtm.task_prior[0] > mtm.task_prior[1]


# ===================================================================
# 2. get_model_for_task returns valid GenerativeModel
# ===================================================================


class TestGetModelForTask:
    def test_returns_generative_model(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="independent"
        )
        model = mtm.get_model_for_task("task_A")
        assert isinstance(model, GenerativeModel)

    def test_model_has_valid_matrices(self):
        gm_a = _make_simple_gm(num_states=4, num_obs=3, seed=1)
        gm_b = _make_simple_gm(num_states=4, num_obs=3, seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="independent"
        )
        model = mtm.get_model_for_task("task_A")

        # A: columns sum to 1
        A = model.A[0]
        np.testing.assert_allclose(A.sum(axis=0), 1.0, atol=1e-10)

        # B: columns sum to 1 for each action
        B = model.B[0]
        for a in range(model.num_actions[0]):
            np.testing.assert_allclose(B[:, :, a].sum(axis=0), 1.0, atol=1e-10)

        # D: sums to 1
        np.testing.assert_allclose(model.D[0].sum(), 1.0, atol=1e-10)

    def test_unknown_task_raises(self):
        gm = _make_simple_gm()
        mtm = MultitaskGenerativeModel({"task_A": gm}, mode="independent")
        with pytest.raises(KeyError, match="Unknown task"):
            mtm.get_model_for_task("nonexistent")

    def test_compositional_model_returns_multifactor(self):
        mtm = build_compositional_battery(["Go", "Anti"], T=1)
        model = mtm.get_model_for_task("Go")
        assert model.num_factors == 3  # stimulus, phase, context
        assert model.num_modalities == 4  # fixation, stim1, stim2, feedback


# ===================================================================
# 3. CompositionalModel factored state space
# ===================================================================


class TestCompositionalModel:
    def test_construction_go_task(self):
        comp = CompositionalModel("Go")
        assert comp.task_name == "Go"
        assert comp.num_stimulus_states == NUM_DIRECTIONS
        assert comp.num_phase_states == NUM_PHASES

    def test_state_factor_sizes(self):
        comp = CompositionalModel("Go", num_context_states=2)
        sizes = comp.state_factor_sizes
        assert sizes == [NUM_DIRECTIONS, NUM_PHASES, 2]

    def test_obs_modality_sizes(self):
        comp = CompositionalModel("Go")
        sizes = comp.obs_modality_sizes
        assert sizes == [NUM_OBS_FIXATION, NUM_OBS_STIM, NUM_OBS_STIM, NUM_OBS_FEEDBACK]

    def test_A_matrices_shape(self):
        comp = CompositionalModel("Go", num_context_states=2)
        A = comp.A
        assert len(A) == 4  # 4 modalities
        assert A[0].shape == (NUM_OBS_FIXATION, NUM_DIRECTIONS, NUM_PHASES, 2)
        assert A[1].shape == (NUM_OBS_STIM, NUM_DIRECTIONS, NUM_PHASES, 2)
        assert A[2].shape == (NUM_OBS_STIM, NUM_DIRECTIONS, NUM_PHASES, 2)
        assert A[3].shape == (NUM_OBS_FEEDBACK, NUM_DIRECTIONS, NUM_PHASES, 2)

    def test_A_matrices_normalized(self):
        comp = CompositionalModel("Anti", num_context_states=2)
        for m, A_m in enumerate(comp.A):
            col_sums = A_m.sum(axis=0)
            np.testing.assert_allclose(
                col_sums,
                1.0,
                atol=1e-6,
                err_msg=f"Modality {m} A matrix columns not normalized",
            )

    def test_B_matrices_shape(self):
        comp = CompositionalModel("Go", num_context_states=2)
        B = comp.B
        assert len(B) == 3  # 3 factors
        assert B[0].shape == (NUM_DIRECTIONS, NUM_DIRECTIONS, 1)  # stimulus: identity
        assert B[1].shape == (NUM_PHASES, NUM_PHASES, 2)  # phase: hold/advance
        assert B[2].shape == (2, 2, 1)  # context: identity

    def test_B_matrices_normalized(self):
        comp = CompositionalModel("CtxGo1", num_context_states=2)
        for f, B_f in enumerate(comp.B):
            for a in range(B_f.shape[-1]):
                col_sums = B_f[:, :, a].sum(axis=0)
                np.testing.assert_allclose(
                    col_sums,
                    1.0,
                    atol=1e-10,
                    err_msg=f"Factor {f}, action {a} B matrix not normalized",
                )

    def test_D_priors_normalized(self):
        comp = CompositionalModel("DelayAnti", num_context_states=2)
        for f, D_f in enumerate(comp.D):
            np.testing.assert_allclose(
                D_f.sum(), 1.0, atol=1e-10, err_msg=f"Factor {f} D prior not normalized"
            )

    def test_D_phase_starts_at_fixation(self):
        comp = CompositionalModel("Go")
        D_phase = comp.D[1]
        assert D_phase[PHASE_FIXATION] == 1.0
        assert D_phase[PHASE_STIMULUS] == 0.0

    def test_D_context_reflects_attended_modality(self):
        comp1 = CompositionalModel("CtxGo1", num_context_states=2)
        comp2 = CompositionalModel("CtxGo2", num_context_states=2)
        # CtxGo1 attends modality 1 -> context state 0
        assert comp1.D[2][0] > comp1.D[2][1]
        # CtxGo2 attends modality 2 -> context state 1
        assert comp2.D[2][1] > comp2.D[2][0]

    def test_C_preferences_differ_go_vs_anti(self):
        comp_go = CompositionalModel("Go", stimulus_direction=0)
        comp_anti = CompositionalModel("Anti", stimulus_direction=0)
        # Go prefers direction 0 in stimulus modality
        assert comp_go.C[1][0] > 0
        # Anti prefers direction 4 (opposite of 0 for 8 directions)
        anti_dir = NUM_DIRECTIONS // 2
        assert comp_anti.C[1][anti_dir] > 0

    def test_to_generative_model(self):
        comp = CompositionalModel("Go", num_context_states=2)
        gm = comp.to_generative_model(T=2)
        assert isinstance(gm, GenerativeModel)
        assert gm.T == 2
        assert gm.num_factors == 3
        assert gm.num_modalities == 4

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            CompositionalModel("FakeTask")

    def test_all_20_tasks_construct(self):
        """Verify all 20 Yang tasks can be instantiated."""
        for task_name in YANG_TASK_DEFINITIONS:
            comp = CompositionalModel(task_name, num_context_states=2)
            gm = comp.to_generative_model(T=1)
            assert gm.num_factors == 3, f"Task {task_name} has wrong num_factors"
            assert gm.num_modalities == 4, f"Task {task_name} has wrong num_modalities"

    def test_stimulus_direction_prior(self):
        comp = CompositionalModel("Go", stimulus_direction=3)
        D_stim = comp.D[0]
        assert D_stim[3] > D_stim[0]
        np.testing.assert_allclose(D_stim.sum(), 1.0, atol=1e-10)


# ===================================================================
# 4. MultitaskAgent task switching
# ===================================================================


class TestMultitaskAgentSwitching:
    def test_set_task(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)
        agent.set_task("task_A")
        assert agent.current_task == "task_A"

    def test_set_task_resets_beliefs(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)
        agent.set_task("task_A")

        # Mutate beliefs
        agent.beliefs[0] = np.array([1.0, 0.0, 0.0, 0.0])

        # Switch task
        agent.set_task("task_B")
        # Beliefs should be reset to task_B's priors
        gm_b = mtm.get_model_for_task("task_B")
        np.testing.assert_array_almost_equal(agent.beliefs[0], gm_b.D[0])

    def test_switch_preserves_habits(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)

        # Run task_A and learn
        agent.set_task("task_A")
        mtm.get_model_for_task("task_A")
        obs = [0]
        agent.step(obs)
        agent.learn(1.0)  # positive outcome -> habit update

        habits_a_after = agent._task_habits["task_A"].copy()

        # Switch to task_B
        agent.set_task("task_B")
        assert agent.current_task == "task_B"

        # Switch back to task_A -- habits should be preserved
        agent.set_task("task_A")
        np.testing.assert_array_equal(agent._task_habits["task_A"], habits_a_after)

    def test_set_unknown_task_raises(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)
        with pytest.raises(KeyError):
            agent.set_task("nonexistent")

    def test_step_without_set_task_raises(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)
        with pytest.raises(RuntimeError, match="No task set"):
            agent.step([0])

    def test_reset_clears_histories(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm)
        agent.set_task("task_A")
        agent.step([0])
        agent.step([1])
        assert len(agent.action_history) == 2

        agent.reset()
        assert len(agent.action_history) == 0
        assert len(agent.belief_history) == 0


# ===================================================================
# 5. MultitaskAgent.run_trial
# ===================================================================


class TestMultitaskAgentRunTrial:
    def test_run_trial_completes(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        env = SimpleEnv(num_obs=4, num_steps=3, seed=42)

        result = agent.run_trial(env, "task_A")
        assert "actions" in result
        assert "total_reward" in result
        assert "task" in result
        assert result["task"] == "task_A"
        assert result["num_steps"] == 3

    def test_run_trial_records_performance(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        env = SimpleEnv(num_obs=4, num_steps=2, seed=42)

        agent.run_trial(env, "task_A")
        agent.run_trial(env, "task_B")

        assert len(agent.performance_log) == 2
        assert agent.performance_log[0]["task"] == "task_A"
        assert agent.performance_log[1]["task"] == "task_B"

    def test_run_trial_returns_observations(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        env = SimpleEnv(num_obs=4, num_steps=2, seed=42)

        result = agent.run_trial(env, "task_A")
        # Should have initial obs + one per step
        assert len(result["observations"]) == result["num_steps"] + 1

    def test_run_trial_with_compositional_model(self):
        mtm = build_compositional_battery(["Go", "Anti"], T=1)
        agent = MultitaskAgent(mtm, seed=42)
        obs_sizes = [NUM_OBS_FIXATION, NUM_OBS_STIM, NUM_OBS_STIM, NUM_OBS_FEEDBACK]
        env = SimpleMultiModalEnv(obs_sizes, num_steps=2, seed=42)

        result = agent.run_trial(env, "Go")
        assert result["task"] == "Go"
        assert result["num_steps"] <= 2


# ===================================================================
# 6. Task inference
# ===================================================================


class TestTaskInference:
    def test_infer_task_returns_distribution(self):
        gm_a = _make_simple_gm(seed=1)
        gm_b = _make_simple_gm(seed=2)
        mtm = MultitaskGenerativeModel(
            {"task_A": gm_a, "task_B": gm_b}, mode="independent"
        )
        # Single observation
        obs_history = [[0]]
        posterior = mtm.infer_task(obs_history)
        assert posterior.shape == (2,)
        np.testing.assert_allclose(posterior.sum(), 1.0, atol=1e-10)
        assert np.all(posterior >= 0)

    def test_infer_task_from_agent(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        agent.set_task("task_A")

        # Accumulate some observations
        for _ in range(3):
            agent.step([0])

        posterior = agent.infer_task()
        assert posterior.shape == (mtm.num_tasks,)
        np.testing.assert_allclose(posterior.sum(), 1.0, atol=1e-10)

    def test_empty_observation_history_returns_prior(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        agent.set_task("task_A")

        # No observations buffered after reset
        agent.reset()
        posterior = agent.infer_task()
        np.testing.assert_allclose(posterior, mtm.task_prior, atol=1e-10)


# ===================================================================
# 7. Multi-factor sequential EFE
# ===================================================================


class TestMultifactorEFE:
    def test_single_factor_matches_sequential_efe(self):
        """Multi-factor EFE with 1 factor should match standard sequential EFE."""
        from alf.sequential_efe import sequential_efe

        gm = _make_simple_gm(num_states=4, num_obs=4, num_actions=2, T=1)

        # Use the standard sequential_efe as reference
        for i, policy in enumerate(gm.policies):
            action_seq = policy[:, 0]
            g_standard = sequential_efe(gm.A[0], gm.B[0], gm.C[0], gm.D[0], action_seq)
            g_multi = multifactor_sequential_efe(gm.A, gm.B, gm.C, gm.D, policy)
            np.testing.assert_allclose(
                g_multi,
                g_standard,
                atol=1e-10,
                err_msg=f"Policy {i}: multi={g_multi:.6f} vs standard={g_standard:.6f}",
            )

    def test_multifactor_efe_finite(self):
        """EFE should be finite for compositional models."""
        comp = CompositionalModel("Go", num_context_states=2)
        gm = comp.to_generative_model(T=1)

        # Evaluate a single policy
        policy = gm.policies[0]
        g = multifactor_sequential_efe(gm.A, gm.B, gm.C, gm.D, policy)
        assert np.isfinite(g), f"EFE should be finite, got {g}"

    def test_evaluate_all_policies_multifactor(self):
        gm = _make_simple_gm(num_states=3, num_obs=3, num_actions=2, T=1)
        G = evaluate_all_policies_multifactor(gm)
        assert G.shape == (gm.num_policies,)
        assert np.all(np.isfinite(G))

    def test_evaluate_all_policies_compositional(self):
        comp = CompositionalModel("Go", num_context_states=2)
        gm = comp.to_generative_model(T=1)
        G = evaluate_all_policies_multifactor(gm)
        assert G.shape == (gm.num_policies,)
        assert np.all(np.isfinite(G)), f"Non-finite EFE values: {G}"


# ===================================================================
# 8. Battery execution
# ===================================================================


class TestBatteryExecution:
    def test_run_battery_blocked(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        envs = {
            "task_A": SimpleEnv(num_obs=4, num_steps=2, seed=1),
            "task_B": SimpleEnv(num_obs=4, num_steps=2, seed=2),
        }

        summary = agent.run_battery(envs, n_trials_per_task=3, interleaved=False)
        assert summary["n_tasks"] == 2
        assert summary["total_trials"] == 6
        assert "task_A" in summary["task_metrics"]
        assert "task_B" in summary["task_metrics"]
        assert summary["task_metrics"]["task_A"]["n_trials"] == 3

    def test_run_battery_interleaved(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        envs = {
            "task_A": SimpleEnv(num_obs=4, num_steps=2, seed=1),
            "task_B": SimpleEnv(num_obs=4, num_steps=2, seed=2),
        }

        summary = agent.run_battery(envs, n_trials_per_task=3, interleaved=True)
        assert summary["total_trials"] == 6

    def test_run_battery_custom_task_order(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        envs = {
            "task_A": SimpleEnv(num_obs=4, num_steps=2, seed=1),
            "task_B": SimpleEnv(num_obs=4, num_steps=2, seed=2),
        }

        summary = agent.run_battery(
            envs, n_trials_per_task=2, task_order=["task_B", "task_A"]
        )
        assert summary["task_order"] == ["task_B", "task_A"]

    def test_run_battery_no_matching_envs_raises(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        with pytest.raises(ValueError, match="No tasks to run"):
            agent.run_battery({}, n_trials_per_task=2)


# ===================================================================
# 9. build_simple_task_pair factory
# ===================================================================


class TestBuildSimpleTaskPair:
    def test_returns_multitask_model(self):
        mtm = build_simple_task_pair()
        assert isinstance(mtm, MultitaskGenerativeModel)
        assert mtm.num_tasks == 2

    def test_tasks_differ(self):
        mtm = build_simple_task_pair()
        model_a = mtm.get_model_for_task("task_A")
        model_b = mtm.get_model_for_task("task_B")
        # A and C should differ between tasks
        assert not np.allclose(model_a.A[0], model_b.A[0])
        assert not np.allclose(model_a.C[0], model_b.C[0])


# ===================================================================
# 10. Agent state summary
# ===================================================================


class TestAgentStateSummary:
    def test_get_state_summary(self):
        mtm = build_simple_task_pair()
        agent = MultitaskAgent(mtm, seed=42)
        agent.set_task("task_A")
        agent.step([0])

        summary = agent.get_state_summary()
        assert summary["current_task"] == "task_A"
        assert summary["gamma"] == 4.0
        assert summary["num_actions_taken"] == 1
        assert "beliefs" in summary
        assert "task_posterior" in summary


# ===================================================================
# 11. Build compositional battery factory
# ===================================================================


class TestBuildCompositionalBattery:
    def test_default_builds_all_20_tasks(self):
        mtm = build_compositional_battery(T=1)
        assert mtm.num_tasks == 20

    def test_subset_of_tasks(self):
        mtm = build_compositional_battery(["Go", "Anti", "DM1"], T=1)
        assert mtm.num_tasks == 3
        assert set(mtm.task_names) == {"Go", "Anti", "DM1"}

    def test_all_models_have_compatible_structure(self):
        mtm = build_compositional_battery(T=1)
        first = mtm.get_model_for_task(mtm.task_names[0])
        for name in mtm.task_names[1:]:
            model = mtm.get_model_for_task(name)
            assert model.num_factors == first.num_factors, (
                f"Task {name} has {model.num_factors} factors, "
                f"expected {first.num_factors}"
            )
            assert model.num_modalities == first.num_modalities, (
                f"Task {name} has {model.num_modalities} modalities, "
                f"expected {first.num_modalities}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
