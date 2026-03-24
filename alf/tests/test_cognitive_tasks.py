"""Tests for the cognitive task battery benchmarks (Yang et al. 2019).

Tests model construction (valid probability distributions), environment
correctness, and agent performance for:
    - Context-Dependent Decision Making
    - Delayed Match-to-Sample
    - Go/NoGo with Anti variant
    - Unified battery runner
"""

import numpy as np
import pytest

# --- Context-dependent DM imports ---
from alf.benchmarks.context_dm import (
    build_context_dm_model,
    ContextDMEnv,
    run_context_dm,
    NUM_STATES as CDM_NUM_STATES,
    NUM_OBS as CDM_NUM_OBS,
    NUM_ACTIONS as CDM_NUM_ACTIONS,
    ACT_FIXATE as CDM_ACT_FIXATE,
    ACT_RESPOND_LEFT as CDM_ACT_RESPOND_LEFT,
    ACT_RESPOND_RIGHT as CDM_ACT_RESPOND_RIGHT,
    OBS_CTX_ATTEND_MOD1,
    OBS_CTX_ATTEND_MOD2,
    OBS_STIM_LEFT_MOD1,
    OBS_STIM_RIGHT_MOD1,
    OBS_STIM_LEFT_MOD2,
    OBS_STIM_RIGHT_MOD2,
    _build_A as cdm_build_A,
    _build_B as cdm_build_B,
    _build_D as cdm_build_D,
    _state_index as cdm_state_index,
    _decode_state as cdm_decode_state,
    _correct_response as cdm_correct_response,
    PHASE_FIX as CDM_PHASE_FIX,
    PHASE_STIM as CDM_PHASE_STIM,
    CTX_MOD1,
    CTX_MOD2,
    DIR_LEFT as CDM_DIR_LEFT,
    DIR_RIGHT as CDM_DIR_RIGHT,
)

# --- Delayed match imports ---
from alf.benchmarks.delayed_match import (
    build_delayed_match_model,
    DelayedMatchEnv,
    run_delayed_match,
    NUM_STATES as DMS_NUM_STATES,
    NUM_OBS as DMS_NUM_OBS,
    NUM_ACTIONS as DMS_NUM_ACTIONS,
    ACT_FIXATE as DMS_ACT_FIXATE,
    ACT_RESPOND_MATCH,
    ACT_RESPOND_NONMATCH,
    OBS_FIXATION as DMS_OBS_FIXATION,
    OBS_SAMPLE_LEFT,
    OBS_SAMPLE_RIGHT,
    OBS_TEST_LEFT,
    OBS_TEST_RIGHT,
    OBS_REWARD as DMS_OBS_REWARD,
    _build_A as dms_build_A,
    _build_B as dms_build_B,
    _build_D as dms_build_D,
    _state_index as dms_state_index,
    _decode_state as dms_decode_state,
    _correct_response as dms_correct_response,
    PHASE_FIX as DMS_PHASE_FIX,
    PHASE_SAMPLE,
    PHASE_DELAY,
    PHASE_TEST,
    PHASE_RESP as DMS_PHASE_RESP,
    MATCH,
    NONMATCH,
    DIR_LEFT as DMS_DIR_LEFT,
    DIR_RIGHT as DMS_DIR_RIGHT,
)

# --- Go/NoGo imports ---
from alf.benchmarks.go_nogo import (
    build_go_nogo_model,
    GoNoGoEnv,
    run_go_nogo,
    NUM_STATES as GNG_NUM_STATES,
    NUM_OBS as GNG_NUM_OBS,
    NUM_ACTIONS as GNG_NUM_ACTIONS,
    ACT_FIXATE as GNG_ACT_FIXATE,
    ACT_RESPOND_LEFT as GNG_ACT_RESPOND_LEFT,
    ACT_RESPOND_RIGHT as GNG_ACT_RESPOND_RIGHT,
    OBS_STIM_LEFT,
    OBS_STIM_RIGHT,
    OBS_RULE_GO,
    OBS_RULE_NOGO,
    OBS_RULE_ANTI,
    _build_A as gng_build_A,
    _build_B as gng_build_B,
    _build_D as gng_build_D,
    _state_index as gng_state_index,
    _decode_state as gng_decode_state,
    _correct_response as gng_correct_response,
    PHASE_FIX as GNG_PHASE_FIX,
    PHASE_STIM as GNG_PHASE_STIM,
    RULE_GO,
    RULE_NOGO,
    RULE_ANTI,
    DIR_LEFT as GNG_DIR_LEFT,
    DIR_RIGHT as GNG_DIR_RIGHT,
)

# --- Battery imports ---
from alf.benchmarks.cognitive_battery import (
    run_battery,
    BATTERY,
    get_task_list,
    get_task_model,
)


# ===================================================================
# Helper: validate probability matrices
# ===================================================================


def _assert_valid_A(A, num_obs, num_states, name="A"):
    """Assert A is a valid likelihood matrix."""
    assert A.shape == (num_obs, num_states), (
        f"{name} shape {A.shape} != ({num_obs}, {num_states})"
    )
    assert np.all(A >= 0), f"{name} has negative entries"
    col_sums = A.sum(axis=0)
    np.testing.assert_allclose(
        col_sums,
        1.0,
        atol=1e-10,
        err_msg=f"{name} columns do not sum to 1",
    )


def _assert_valid_B(B, num_states, num_actions, name="B"):
    """Assert B is a valid transition matrix."""
    assert B.shape == (num_states, num_states, num_actions), (
        f"{name} shape {B.shape} != ({num_states}, {num_states}, {num_actions})"
    )
    assert np.all(B >= 0), f"{name} has negative entries"
    for a in range(num_actions):
        col_sums = B[:, :, a].sum(axis=0)
        np.testing.assert_allclose(
            col_sums,
            1.0,
            atol=1e-10,
            err_msg=f"{name}[:,:,{a}] columns do not sum to 1",
        )


def _assert_valid_D(D, num_states, name="D"):
    """Assert D is a valid prior distribution."""
    assert D.shape == (num_states,), f"{name} shape {D.shape} != ({num_states},)"
    assert np.all(D >= 0), f"{name} has negative entries"
    np.testing.assert_allclose(
        D.sum(),
        1.0,
        atol=1e-10,
        err_msg=f"{name} does not sum to 1",
    )


# ===================================================================
# Context-Dependent Decision Making tests
# ===================================================================


class TestContextDM:
    def test_state_encoding_roundtrip(self):
        """Test that state encoding and decoding are inverses."""
        for s in range(CDM_NUM_STATES):
            mod1, mod2, ctx, phase = cdm_decode_state(s)
            assert cdm_state_index(mod1, mod2, ctx, phase) == s

    def test_model_construction(self):
        """Test that the generative model builds with valid matrices."""
        gm = build_context_dm_model()
        assert gm.num_modalities == 1
        assert gm.num_factors == 1
        assert gm.num_obs == [CDM_NUM_OBS]
        assert gm.num_states == [CDM_NUM_STATES]
        assert gm.num_actions == [CDM_NUM_ACTIONS]
        assert gm.T == 3

    def test_A_matrix_valid(self):
        """Test that A matrix columns sum to 1."""
        A = cdm_build_A()
        _assert_valid_A(A, CDM_NUM_OBS, CDM_NUM_STATES, "context_dm A")

    def test_B_matrix_valid(self):
        """Test that B matrix columns sum to 1 for each action."""
        B = cdm_build_B()
        _assert_valid_B(B, CDM_NUM_STATES, CDM_NUM_ACTIONS, "context_dm B")

    def test_D_prior_valid(self):
        """Test that D prior sums to 1."""
        D = cdm_build_D()
        _assert_valid_D(D, CDM_NUM_STATES, "context_dm D")

    def test_D_prior_on_fixation_states(self):
        """Test that D prior is only on fixation-phase states."""
        D = cdm_build_D()
        for s in range(CDM_NUM_STATES):
            mod1, mod2, ctx, phase = cdm_decode_state(s)
            if phase == CDM_PHASE_FIX:
                assert D[s] > 0, f"Fixation state {s} should have nonzero prior"
            else:
                assert D[s] == 0.0, f"Non-fixation state {s} should have zero prior"

    def test_A_fixation_has_context_cue(self):
        """Test that fixation states emit context cue observations."""
        A = cdm_build_A()
        for mod1 in range(2):
            for mod2 in range(2):
                # Context = attend mod1
                s = cdm_state_index(mod1, mod2, CTX_MOD1, CDM_PHASE_FIX)
                assert A[OBS_CTX_ATTEND_MOD1, s] > 0, (
                    "Fixation state with ctx=mod1 should emit ctx_attend_mod1"
                )
                # Context = attend mod2
                s = cdm_state_index(mod1, mod2, CTX_MOD2, CDM_PHASE_FIX)
                assert A[OBS_CTX_ATTEND_MOD2, s] > 0, (
                    "Fixation state with ctx=mod2 should emit ctx_attend_mod2"
                )

    def test_A_stimulus_has_modality_obs(self):
        """Test that stimulus states emit modality observations."""
        A = cdm_build_A()
        for mod1 in range(2):
            for mod2 in range(2):
                for ctx in range(2):
                    s = cdm_state_index(mod1, mod2, ctx, CDM_PHASE_STIM)
                    if mod1 == CDM_DIR_LEFT:
                        assert A[OBS_STIM_LEFT_MOD1, s] > 0
                    else:
                        assert A[OBS_STIM_RIGHT_MOD1, s] > 0
                    if mod2 == CDM_DIR_LEFT:
                        assert A[OBS_STIM_LEFT_MOD2, s] > 0
                    else:
                        assert A[OBS_STIM_RIGHT_MOD2, s] > 0

    def test_B_fixate_advances_phase(self):
        """Test that fixating during fixation advances to stimulus phase."""
        B = cdm_build_B()
        for mod1 in range(2):
            for mod2 in range(2):
                for ctx in range(2):
                    s_fix = cdm_state_index(mod1, mod2, ctx, CDM_PHASE_FIX)
                    s_stim = cdm_state_index(mod1, mod2, ctx, CDM_PHASE_STIM)
                    assert B[s_stim, s_fix, CDM_ACT_FIXATE] == 1.0, (
                        "Fixating during fixation should advance to stimulus"
                    )

    def test_correct_response_logic(self):
        """Test the correct response mapping."""
        # Attend mod1: response follows mod1
        assert (
            cdm_correct_response(CDM_DIR_LEFT, CDM_DIR_RIGHT, CTX_MOD1)
            == CDM_ACT_RESPOND_LEFT
        )
        assert (
            cdm_correct_response(CDM_DIR_RIGHT, CDM_DIR_LEFT, CTX_MOD1)
            == CDM_ACT_RESPOND_RIGHT
        )
        # Attend mod2: response follows mod2
        assert (
            cdm_correct_response(CDM_DIR_LEFT, CDM_DIR_RIGHT, CTX_MOD2)
            == CDM_ACT_RESPOND_RIGHT
        )
        assert (
            cdm_correct_response(CDM_DIR_RIGHT, CDM_DIR_LEFT, CTX_MOD2)
            == CDM_ACT_RESPOND_LEFT
        )

    def test_environment_correct_response(self):
        """Test that the environment gives reward for correct response."""
        # Context = mod1, mod1 = left -> respond_left is correct
        env = ContextDMEnv(mod1_dir="left", mod2_dir="right", context="mod1", seed=0)
        obs = env.reset()
        obs, reward, done = env.step(CDM_ACT_FIXATE)  # fixation -> stimulus
        assert not done
        obs, reward, done = env.step(CDM_ACT_RESPOND_LEFT)  # correct
        assert done
        assert reward == 1.0

    def test_environment_wrong_response(self):
        """Test that the environment gives punishment for wrong response."""
        env = ContextDMEnv(mod1_dir="left", mod2_dir="right", context="mod1", seed=0)
        env.reset()
        env.step(CDM_ACT_FIXATE)  # fixation -> stimulus
        obs, reward, done = env.step(CDM_ACT_RESPOND_RIGHT)  # wrong
        assert done
        assert reward == -1.0

    def test_environment_context_switch(self):
        """Test that changing context changes correct response."""
        # mod1=left, mod2=right
        # context=mod1 -> correct is respond_left
        env1 = ContextDMEnv(mod1_dir="left", mod2_dir="right", context="mod1", seed=0)
        env1.reset()
        env1.step(CDM_ACT_FIXATE)
        _, r1, _ = env1.step(CDM_ACT_RESPOND_LEFT)

        # context=mod2 -> correct is respond_right
        env2 = ContextDMEnv(mod1_dir="left", mod2_dir="right", context="mod2", seed=0)
        env2.reset()
        env2.step(CDM_ACT_FIXATE)
        _, r2, _ = env2.step(CDM_ACT_RESPOND_RIGHT)

        assert r1 == 1.0
        assert r2 == 1.0

    def test_environment_produces_valid_observations(self):
        """Test that all observations are within valid range."""
        env = ContextDMEnv(mod1_dir="left", mod2_dir="right", context="mod1", seed=42)
        obs = env.reset()
        assert 0 <= obs < CDM_NUM_OBS

        for action in range(CDM_NUM_ACTIONS):
            env.reset()
            obs, _, _ = env.step(action)
            assert 0 <= obs < CDM_NUM_OBS

    def test_run_context_dm(self):
        """Run a short benchmark and check it completes."""
        results = run_context_dm(num_trials=5, gamma=4.0, seed=42, verbose=False)
        assert "accuracy" in results
        assert "congruent_accuracy" in results
        assert "incongruent_accuracy" in results
        assert "context_use_rate" in results
        assert len(results["trial_log"]) == 5


# ===================================================================
# Delayed Match-to-Sample tests
# ===================================================================


class TestDelayedMatch:
    def test_state_encoding_roundtrip(self):
        """Test that state encoding and decoding are inverses."""
        for s in range(DMS_NUM_STATES):
            sample, phase, test_matches = dms_decode_state(s)
            assert dms_state_index(sample, phase, test_matches) == s

    def test_model_construction(self):
        """Test that the generative model builds with valid matrices."""
        gm = build_delayed_match_model()
        assert gm.num_modalities == 1
        assert gm.num_factors == 1
        assert gm.num_obs == [DMS_NUM_OBS]
        assert gm.num_states == [DMS_NUM_STATES]
        assert gm.num_actions == [DMS_NUM_ACTIONS]
        assert gm.T == 5

    def test_A_matrix_valid(self):
        """Test that A matrix columns sum to 1."""
        A = dms_build_A()
        _assert_valid_A(A, DMS_NUM_OBS, DMS_NUM_STATES, "delayed_match A")

    def test_B_matrix_valid(self):
        """Test that B matrix columns sum to 1 for each action."""
        B = dms_build_B()
        _assert_valid_B(B, DMS_NUM_STATES, DMS_NUM_ACTIONS, "delayed_match B")

    def test_D_prior_valid(self):
        """Test that D prior sums to 1."""
        D = dms_build_D()
        _assert_valid_D(D, DMS_NUM_STATES, "delayed_match D")

    def test_D_prior_on_fixation_states(self):
        """Test that D prior is only on fixation-phase states."""
        D = dms_build_D()
        for s in range(DMS_NUM_STATES):
            sample, phase, test_matches = dms_decode_state(s)
            if phase == DMS_PHASE_FIX:
                assert D[s] > 0
            else:
                assert D[s] == 0.0

    def test_A_sample_phase_shows_sample(self):
        """Test that sample-phase states emit sample observations."""
        A = dms_build_A()
        for sample in range(2):
            for test_matches in range(2):
                s = dms_state_index(sample, PHASE_SAMPLE, test_matches)
                if sample == DMS_DIR_LEFT:
                    assert A[OBS_SAMPLE_LEFT, s] == 1.0
                else:
                    assert A[OBS_SAMPLE_RIGHT, s] == 1.0

    def test_A_delay_phase_shows_fixation(self):
        """Test that delay-phase states emit fixation observations."""
        A = dms_build_A()
        for sample in range(2):
            for test_matches in range(2):
                s = dms_state_index(sample, PHASE_DELAY, test_matches)
                assert A[DMS_OBS_FIXATION, s] == 1.0

    def test_A_test_phase_shows_test(self):
        """Test that test-phase states emit correct test observations."""
        A = dms_build_A()
        # sample=left, match -> test=left
        s = dms_state_index(DMS_DIR_LEFT, PHASE_TEST, MATCH)
        assert A[OBS_TEST_LEFT, s] == 1.0

        # sample=left, nonmatch -> test=right
        s = dms_state_index(DMS_DIR_LEFT, PHASE_TEST, NONMATCH)
        assert A[OBS_TEST_RIGHT, s] == 1.0

        # sample=right, match -> test=right
        s = dms_state_index(DMS_DIR_RIGHT, PHASE_TEST, MATCH)
        assert A[OBS_TEST_RIGHT, s] == 1.0

        # sample=right, nonmatch -> test=left
        s = dms_state_index(DMS_DIR_RIGHT, PHASE_TEST, NONMATCH)
        assert A[OBS_TEST_LEFT, s] == 1.0

    def test_B_fixate_advances_through_phases(self):
        """Test that fixating advances through all phases in order."""
        B = dms_build_B()
        for sample in range(2):
            for test_matches in range(2):
                # fix -> sample
                s0 = dms_state_index(sample, DMS_PHASE_FIX, test_matches)
                s1 = dms_state_index(sample, PHASE_SAMPLE, test_matches)
                assert B[s1, s0, DMS_ACT_FIXATE] == 1.0

                # sample -> delay
                s2 = dms_state_index(sample, PHASE_DELAY, test_matches)
                assert B[s2, s1, DMS_ACT_FIXATE] == 1.0

                # delay -> test
                s3 = dms_state_index(sample, PHASE_TEST, test_matches)
                assert B[s3, s2, DMS_ACT_FIXATE] == 1.0

                # test -> response
                s4 = dms_state_index(sample, DMS_PHASE_RESP, test_matches)
                assert B[s4, s3, DMS_ACT_FIXATE] == 1.0

    def test_correct_response_logic(self):
        """Test the correct response mapping."""
        assert dms_correct_response(MATCH) == ACT_RESPOND_MATCH
        assert dms_correct_response(NONMATCH) == ACT_RESPOND_NONMATCH

    def test_environment_match_trial(self):
        """Test environment on a match trial."""
        env = DelayedMatchEnv(sample_dir="left", test_matches=True, seed=0)
        obs = env.reset()
        assert obs == DMS_OBS_FIXATION

        # Fixate through to test phase
        obs, _, _ = env.step(DMS_ACT_FIXATE)  # -> sample
        assert obs == OBS_SAMPLE_LEFT
        obs, _, _ = env.step(DMS_ACT_FIXATE)  # -> delay
        assert obs == DMS_OBS_FIXATION
        obs, _, _ = env.step(DMS_ACT_FIXATE)  # -> test
        assert obs == OBS_TEST_LEFT  # match: same direction

        # Respond match -> correct
        obs, reward, done = env.step(ACT_RESPOND_MATCH)
        assert done
        assert reward == 1.0
        assert obs == DMS_OBS_REWARD

    def test_environment_nonmatch_trial(self):
        """Test environment on a nonmatch trial."""
        env = DelayedMatchEnv(sample_dir="left", test_matches=False, seed=0)
        env.reset()
        env.step(DMS_ACT_FIXATE)  # -> sample (left)
        env.step(DMS_ACT_FIXATE)  # -> delay
        obs, _, _ = env.step(DMS_ACT_FIXATE)  # -> test
        assert obs == OBS_TEST_RIGHT  # nonmatch: opposite direction

        # Respond nonmatch -> correct
        obs, reward, done = env.step(ACT_RESPOND_NONMATCH)
        assert done
        assert reward == 1.0

    def test_environment_wrong_response(self):
        """Test punishment for wrong response."""
        env = DelayedMatchEnv(sample_dir="left", test_matches=True, seed=0)
        env.reset()
        env.step(DMS_ACT_FIXATE)
        env.step(DMS_ACT_FIXATE)
        env.step(DMS_ACT_FIXATE)
        obs, reward, done = env.step(ACT_RESPOND_NONMATCH)  # wrong
        assert done
        assert reward == -1.0

    def test_environment_produces_valid_observations(self):
        """Test that all observations are within valid range."""
        env = DelayedMatchEnv(sample_dir="right", test_matches=False, seed=42)
        obs = env.reset()
        assert 0 <= obs < DMS_NUM_OBS

        for _ in range(5):
            obs, _, done = env.step(DMS_ACT_FIXATE)
            assert 0 <= obs < DMS_NUM_OBS
            if done:
                break

    def test_run_delayed_match(self):
        """Run a short benchmark and check it completes."""
        results = run_delayed_match(num_trials=5, gamma=4.0, seed=42, verbose=False)
        assert "accuracy" in results
        assert "match_accuracy" in results
        assert "nonmatch_accuracy" in results
        assert len(results["trial_log"]) == 5


# ===================================================================
# Go/NoGo with Anti tests
# ===================================================================


class TestGoNoGo:
    def test_state_encoding_roundtrip(self):
        """Test that state encoding and decoding are inverses."""
        for s in range(GNG_NUM_STATES):
            stim_dir, rule, phase = gng_decode_state(s)
            assert gng_state_index(stim_dir, rule, phase) == s

    def test_model_construction(self):
        """Test that the generative model builds with valid matrices."""
        gm = build_go_nogo_model()
        assert gm.num_modalities == 1
        assert gm.num_factors == 1
        assert gm.num_obs == [GNG_NUM_OBS]
        assert gm.num_states == [GNG_NUM_STATES]
        assert gm.num_actions == [GNG_NUM_ACTIONS]
        assert gm.T == 3

    def test_A_matrix_valid(self):
        """Test that A matrix columns sum to 1."""
        A = gng_build_A()
        _assert_valid_A(A, GNG_NUM_OBS, GNG_NUM_STATES, "go_nogo A")

    def test_B_matrix_valid(self):
        """Test that B matrix columns sum to 1 for each action."""
        B = gng_build_B()
        _assert_valid_B(B, GNG_NUM_STATES, GNG_NUM_ACTIONS, "go_nogo B")

    def test_D_prior_valid(self):
        """Test that D prior sums to 1."""
        D = gng_build_D()
        _assert_valid_D(D, GNG_NUM_STATES, "go_nogo D")

    def test_D_prior_on_fixation_states(self):
        """Test that D prior is only on fixation-phase states."""
        D = gng_build_D()
        for s in range(GNG_NUM_STATES):
            stim_dir, rule, phase = gng_decode_state(s)
            if phase == GNG_PHASE_FIX:
                assert D[s] > 0
            else:
                assert D[s] == 0.0

    def test_A_fixation_has_rule_cue(self):
        """Test that fixation states emit rule cue observations."""
        A = gng_build_A()
        for stim_dir in range(2):
            s_go = gng_state_index(stim_dir, RULE_GO, GNG_PHASE_FIX)
            assert A[OBS_RULE_GO, s_go] > 0

            s_nogo = gng_state_index(stim_dir, RULE_NOGO, GNG_PHASE_FIX)
            assert A[OBS_RULE_NOGO, s_nogo] > 0

            s_anti = gng_state_index(stim_dir, RULE_ANTI, GNG_PHASE_FIX)
            assert A[OBS_RULE_ANTI, s_anti] > 0

    def test_A_stimulus_shows_direction(self):
        """Test that stimulus states emit direction observations."""
        A = gng_build_A()
        for rule in range(3):
            s_left = gng_state_index(GNG_DIR_LEFT, rule, GNG_PHASE_STIM)
            assert A[OBS_STIM_LEFT, s_left] == 1.0

            s_right = gng_state_index(GNG_DIR_RIGHT, rule, GNG_PHASE_STIM)
            assert A[OBS_STIM_RIGHT, s_right] == 1.0

    def test_correct_response_go(self):
        """Test correct response for Go trials."""
        assert gng_correct_response(GNG_DIR_LEFT, RULE_GO) == GNG_ACT_RESPOND_LEFT
        assert gng_correct_response(GNG_DIR_RIGHT, RULE_GO) == GNG_ACT_RESPOND_RIGHT

    def test_correct_response_nogo(self):
        """Test correct response for NoGo trials: always fixate."""
        assert gng_correct_response(GNG_DIR_LEFT, RULE_NOGO) == GNG_ACT_FIXATE
        assert gng_correct_response(GNG_DIR_RIGHT, RULE_NOGO) == GNG_ACT_FIXATE

    def test_correct_response_anti(self):
        """Test correct response for Anti trials: opposite direction."""
        assert gng_correct_response(GNG_DIR_LEFT, RULE_ANTI) == GNG_ACT_RESPOND_RIGHT
        assert gng_correct_response(GNG_DIR_RIGHT, RULE_ANTI) == GNG_ACT_RESPOND_LEFT

    def test_environment_go_correct(self):
        """Test Go trial with correct response."""
        env = GoNoGoEnv(stim_dir="left", rule="go", seed=0)
        obs = env.reset()
        obs, _, done = env.step(GNG_ACT_FIXATE)  # fix -> stim
        assert not done
        obs, reward, done = env.step(GNG_ACT_RESPOND_LEFT)  # correct
        assert done
        assert reward == 1.0

    def test_environment_nogo_correct(self):
        """Test NoGo trial with correct withholding."""
        env = GoNoGoEnv(stim_dir="left", rule="nogo", seed=0)
        env.reset()
        env.step(GNG_ACT_FIXATE)  # fix -> stim
        obs, reward, done = env.step(GNG_ACT_FIXATE)  # withhold -> correct
        assert done
        assert reward == 1.0

    def test_environment_nogo_false_alarm(self):
        """Test NoGo trial with false alarm (responding when should withhold)."""
        env = GoNoGoEnv(stim_dir="left", rule="nogo", seed=0)
        env.reset()
        env.step(GNG_ACT_FIXATE)  # fix -> stim
        obs, reward, done = env.step(GNG_ACT_RESPOND_LEFT)  # false alarm
        assert done
        assert reward == -1.0

    def test_environment_anti_correct(self):
        """Test Anti trial with correct opposite response."""
        env = GoNoGoEnv(stim_dir="left", rule="anti", seed=0)
        env.reset()
        env.step(GNG_ACT_FIXATE)  # fix -> stim
        obs, reward, done = env.step(GNG_ACT_RESPOND_RIGHT)  # opposite = correct
        assert done
        assert reward == 1.0

    def test_environment_anti_wrong(self):
        """Test Anti trial with same-direction response (wrong)."""
        env = GoNoGoEnv(stim_dir="left", rule="anti", seed=0)
        env.reset()
        env.step(GNG_ACT_FIXATE)
        obs, reward, done = env.step(GNG_ACT_RESPOND_LEFT)  # same dir = wrong
        assert done
        assert reward == -1.0

    def test_environment_produces_valid_observations(self):
        """Test that all observations are within valid range."""
        for rule in ["go", "nogo", "anti"]:
            env = GoNoGoEnv(stim_dir="left", rule=rule, seed=42)
            obs = env.reset()
            assert 0 <= obs < GNG_NUM_OBS
            for _ in range(3):
                obs, _, done = env.step(GNG_ACT_FIXATE)
                assert 0 <= obs < GNG_NUM_OBS
                if done:
                    break

    def test_run_go_nogo(self):
        """Run a short benchmark and check it completes."""
        results = run_go_nogo(num_trials=5, gamma=4.0, seed=42, verbose=False)
        assert "accuracy" in results
        assert "go_accuracy" in results
        assert "nogo_accuracy" in results
        assert "anti_accuracy" in results
        assert "false_alarm_rate" in results
        assert len(results["trial_log"]) == 5


# ===================================================================
# Cognitive Battery tests
# ===================================================================


class TestCognitiveBattery:
    def test_battery_registry(self):
        """Test that the battery registry has all expected tasks."""
        assert "context_dm" in BATTERY
        assert "delayed_match" in BATTERY
        assert "go_nogo" in BATTERY

    def test_get_task_list(self):
        """Test that get_task_list returns all tasks."""
        tasks = get_task_list()
        assert "context_dm" in tasks
        assert "delayed_match" in tasks
        assert "go_nogo" in tasks

    def test_get_task_model(self):
        """Test building models via the battery interface."""
        for task_name in get_task_list():
            gm = get_task_model(task_name)
            assert gm.num_modalities == 1
            assert gm.num_factors == 1
            assert len(gm.A) == 1
            assert len(gm.B) == 1
            assert len(gm.C) == 1
            assert len(gm.D) == 1

    def test_get_task_model_unknown(self):
        """Test error on unknown task name."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_model("nonexistent_task")

    def test_run_battery_all_tasks(self):
        """Run full battery with minimal trials."""
        results = run_battery(n_trials=3, gamma=4.0, seed=42, verbose=False)
        assert "context_dm" in results
        assert "delayed_match" in results
        assert "go_nogo" in results

        for task_name, task_results in results.items():
            assert "trial_log" in task_results
            assert len(task_results["trial_log"]) == 3

    def test_run_battery_subset(self):
        """Test running only a subset of tasks."""
        results = run_battery(
            n_trials=3,
            seed=42,
            tasks=["context_dm", "go_nogo"],
        )
        assert "context_dm" in results
        assert "go_nogo" in results
        assert "delayed_match" not in results

    def test_run_battery_unknown_task(self):
        """Test error on unknown task in subset."""
        with pytest.raises(ValueError, match="Unknown task"):
            run_battery(n_trials=3, tasks=["fake_task"])


# ===================================================================
# Cross-task validation tests
# ===================================================================


class TestCrossTaskValidation:
    """Tests that apply uniformly across all tasks."""

    @pytest.mark.parametrize(
        "build_fn,n_states,n_obs,n_actions",
        [
            (build_context_dm_model, CDM_NUM_STATES, CDM_NUM_OBS, CDM_NUM_ACTIONS),
            (build_delayed_match_model, DMS_NUM_STATES, DMS_NUM_OBS, DMS_NUM_ACTIONS),
            (build_go_nogo_model, GNG_NUM_STATES, GNG_NUM_OBS, GNG_NUM_ACTIONS),
        ],
    )
    def test_all_models_valid_distributions(self, build_fn, n_states, n_obs, n_actions):
        """All models should have valid probability distributions."""
        gm = build_fn()
        _assert_valid_A(gm.A[0], n_obs, n_states)
        _assert_valid_B(gm.B[0], n_states, n_actions)
        _assert_valid_D(gm.D[0], n_states)

    @pytest.mark.parametrize(
        "build_fn",
        [
            build_context_dm_model,
            build_delayed_match_model,
            build_go_nogo_model,
        ],
    )
    def test_all_models_have_reward_preference(self, build_fn):
        """All models should prefer reward over punishment in C."""
        gm = build_fn()
        C = gm.C[0]
        # Find reward and punishment observations (last two)
        # All tasks have reward as second-to-last, punishment as last
        assert C.max() > 0, "C should have positive preference (reward)"
        assert C.min() < 0, "C should have negative preference (punishment)"

    @pytest.mark.parametrize(
        "build_fn",
        [
            build_context_dm_model,
            build_delayed_match_model,
            build_go_nogo_model,
        ],
    )
    def test_all_models_single_factor_single_modality(self, build_fn):
        """All task models should be single-factor, single-modality."""
        gm = build_fn()
        assert gm.num_factors == 1
        assert gm.num_modalities == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
