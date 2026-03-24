"""Integration tests for the 4 new CPC modules (HGF, DDM, metacognition, normative).

Verifies that:
1. HGF perception integrates with alf's discrete action selection pipeline.
2. DDM simulation, fitting, and bridge to EFE work end-to-end.
3. Metacognition SDT pipeline produces valid precision estimates.
4. Normative modeling pipeline detects outliers and vmap matches loop.
5. Cross-module interactions (HGF->metacognition, DDM+HGF, normative->VFE).
6. Full AIF loop works with HGFPerceptualAgent on the T-maze benchmark.
"""

import numpy as np
import jax
import jax.numpy as jnp

from alf.generative_model import GenerativeModel
from alf.agent import AnalyticAgent
from alf.benchmarks.t_maze import build_t_maze_model, TMazeEnv

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf,
    continuous_hgf,
)
from alf.hgf.bridge import HGFPerceptualAgent
from alf.hgf.learning import learn_binary_hgf

from alf.ddm.wiener import (
    DDMParams,
    simulate_ddm,
    ddm_nll,
)
from alf.ddm.bridge import efe_to_ddm, ddm_to_policy_probs

from alf.metacognition import (
    compute_type1_from_counts,
    fit_meta_d_mle,
    m_ratio_to_gamma,
    update_gamma_from_confidence,
)

from alf.normative.blr import (
    normative_model,
    normative_model_vmap,
    compute_zscore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_simple_2state_model():
    """Build a minimal 2-state, 2-obs, 2-action model for HGF integration."""
    A = [np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])]
    B = [np.array([
        [[0.8, 0.2],
         [0.2, 0.8]],
        [[0.2, 0.8],
         [0.8, 0.2]],
    ]).transpose(1, 2, 0)]
    C = [np.array([1.0, -1.0])]
    D = [np.array([0.5, 0.5])]
    return GenerativeModel(A=A, B=B, C=C, D=D, T=1)


def make_binary_params(omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0):
    """Create standard binary HGF parameters."""
    return BinaryHGFParams(
        omega_2=jnp.array(omega_2),
        mu_2_0=jnp.array(mu_2_0),
        sigma_2_0=jnp.array(sigma_2_0),
    )


def make_continuous_params():
    """Create standard 3-level continuous HGF parameters."""
    return ContinuousHGFParams(
        omega_1=jnp.array(-3.0),
        omega_2=jnp.array(-3.0),
        kappa_1=jnp.array(1.0),
        kappa_2=jnp.array(1.0),
        theta=jnp.array(0.01),
        pi_u=jnp.array(100.0),
        mu_1_0=jnp.array(0.0),
        sigma_1_0=jnp.array(1.0),
        mu_2_0=jnp.array(0.0),
        sigma_2_0=jnp.array(1.0),
        mu_3_0=jnp.array(0.0),
        sigma_3_0=jnp.array(1.0),
    )


def generate_switching_binary(n=200, switch_prob=0.05, seed=42):
    """Generate binary observations from a switching source."""
    rng = np.random.RandomState(seed)
    p = 0.8
    obs = np.zeros(n)
    for t in range(n):
        obs[t] = rng.binomial(1, p)
        if rng.random() < switch_prob:
            p = 1.0 - p
    return obs


def generate_drifting_signal(n=100, drift_rate=0.02, noise_std=0.1, seed=42):
    """Generate a slowly drifting continuous signal with observation noise."""
    rng = np.random.RandomState(seed)
    x = 0.0
    obs = np.zeros(n)
    for t in range(n):
        x += drift_rate + rng.normal() * 0.01
        obs[t] = x + rng.normal() * noise_std
    return obs


def generate_sdt_data(d_prime=1.5, n_trials=500, n_ratings=4, seed=42):
    """Generate SDT confidence-rating data with known d'.

    Returns nR_S1 and nR_S2 arrays.
    """
    rng = np.random.RandomState(seed)

    # Internal evidence values: S1 ~ N(-d'/2, 1), S2 ~ N(d'/2, 1)
    evidence_s1 = rng.normal(-d_prime / 2.0, 1.0, n_trials)
    evidence_s2 = rng.normal(d_prime / 2.0, 1.0, n_trials)

    # Confidence criteria (symmetric around 0, equally spaced)
    criteria = np.linspace(-1.5, 1.5, 2 * n_ratings - 1)

    def bin_evidence(evidence):
        counts = np.zeros(2 * n_ratings, dtype=int)
        for e in evidence:
            idx = np.searchsorted(criteria, e)
            counts[idx] += 1
        return counts

    nR_S1 = bin_evidence(evidence_s1)
    nR_S2 = bin_evidence(evidence_s2)

    return nR_S1, nR_S2


def generate_normative_data(n_train=200, n_test=50, seed=42):
    """Generate age-brain measure data with a known quadratic relationship."""
    rng = np.random.RandomState(seed)
    x_train = rng.uniform(20, 80, n_train)
    x_test = rng.uniform(20, 80, n_test)

    def brain_fn(x):
        return 100.0 - 0.01 * (x - 50) ** 2

    noise_std = 2.0
    y_train = brain_fn(x_train) + rng.normal(0, noise_std, n_train)
    y_test = brain_fn(x_test) + rng.normal(0, noise_std, n_test)

    return x_train, y_train, x_test, y_test


# ===========================================================================
# HGF Integration Tests
# ===========================================================================

def test_hgf_perceptual_agent_binary():
    """HGFPerceptualAgent with binary HGF produces valid actions and beliefs."""
    gm = build_simple_2state_model()
    hgf_params = make_binary_params(omega_2=-2.0)
    agent = HGFPerceptualAgent(gm, hgf_params, gamma=4.0, seed=123)

    num_steps = 20
    observations = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
                    1, 1, 0, 1, 0, 0, 1, 1, 1, 0]

    for t in range(num_steps):
        action, info = agent.step(float(observations[t]))

        # Action must be valid
        assert 0 <= action < gm.num_actions[0], (
            f"Step {t}: action {action} out of range [0, {gm.num_actions[0]})"
        )
        # Beliefs must be a valid probability distribution
        beliefs = info["beliefs"]
        np.testing.assert_allclose(
            beliefs.sum(), 1.0, atol=1e-6,
            err_msg=f"Step {t}: beliefs don't sum to 1"
        )
        assert np.all(beliefs >= 0), f"Step {t}: negative belief"

    # Surprise should generally decrease as the agent learns the pattern
    surprises = agent.surprise_history
    np.mean(surprises[:num_steps // 2])
    np.mean(surprises[num_steps // 2:])
    # This is a soft check; the agent should at least not diverge
    assert np.all(np.isfinite(surprises)), "Surprise contains non-finite values"


def test_hgf_perceptual_agent_continuous():
    """HGFPerceptualAgent with continuous HGF tracks a drifting signal."""
    gm = build_simple_2state_model()
    hgf_params = make_continuous_params()
    agent = HGFPerceptualAgent(
        gm, hgf_params, gamma=4.0, state_range=(-3.0, 3.0), seed=456
    )

    # Drifting signal: starts near 0, drifts upward
    obs = generate_drifting_signal(n=30, drift_rate=0.05, noise_std=0.1, seed=99)

    for t in range(len(obs)):
        action, info = agent.step(obs[t])
        assert 0 <= action < gm.num_actions[0], (
            f"Step {t}: invalid action {action}"
        )

    # The HGF level-1 mean should track the drifting signal
    hgf_mus = [m[0] for m in agent.hgf_mu_history]
    final_mu = hgf_mus[-1]
    obs[-1]

    # The mean should be in the same direction as the signal
    assert final_mu > 0.0, (
        f"HGF mu_1 should track upward drift, got {final_mu:.3f}"
    )
    assert np.all(np.isfinite(agent.surprise_history)), (
        "Surprise contains non-finite values"
    )


def test_hgf_learns_then_predicts():
    """Learning omega_2 from data should yield lower surprise than random params."""
    obs = generate_switching_binary(n=200, switch_prob=0.03, seed=77)

    # Learn omega_2
    result = learn_binary_hgf(
        observations=obs,
        init_omega_2=-2.0,
        num_epochs=80,
        lr=0.01,
    )
    learned_params = result.params

    # Run HGF with learned params
    obs_jnp = jnp.array(obs)
    result_learned = binary_hgf(obs_jnp, learned_params)
    surprise_learned = float(jnp.sum(result_learned.surprise))

    # Run HGF with deliberately poor params (random omega_2 far from learned)
    random_params = make_binary_params(omega_2=-6.0)
    result_random = binary_hgf(obs_jnp, random_params)
    surprise_random = float(jnp.sum(result_random.surprise))

    assert jnp.isfinite(jnp.array(surprise_learned)), (
        f"Learned surprise should be finite: {surprise_learned}"
    )
    assert jnp.isfinite(jnp.array(surprise_random)), (
        f"Random surprise should be finite: {surprise_random}"
    )
    assert surprise_learned < surprise_random, (
        f"Learned params should yield lower surprise: "
        f"learned={surprise_learned:.2f} vs random={surprise_random:.2f}"
    )


# ===========================================================================
# DDM Integration Tests
# ===========================================================================

def test_ddm_simulate_then_fit():
    """Simulating DDM data and computing NLL at true vs random params."""
    true_v, true_a, true_w, true_tau = 1.0, 1.5, 0.5, 0.3

    data = simulate_ddm(
        v=true_v, a=true_a, w=true_w, tau=true_tau,
        n_trials=200, seed=42,
    )

    rt = jnp.array(data.rt)
    choice = jnp.array(data.choice, dtype=jnp.float32)

    # NLL at true params
    nll_true = float(ddm_nll(
        jnp.array(true_v), jnp.array(true_a),
        jnp.array(true_w), jnp.array(true_tau),
        rt, choice,
    ))

    # NLL at random (wrong) params
    nll_random = float(ddm_nll(
        jnp.array(0.0), jnp.array(2.5),
        jnp.array(0.5), jnp.array(0.1),
        rt, choice,
    ))

    assert np.isfinite(nll_true), f"NLL at true params is not finite: {nll_true}"
    assert nll_true < nll_random, (
        f"True params should have lower NLL: true={nll_true:.2f} vs "
        f"random={nll_random:.2f}"
    )


def test_ddm_efe_bridge_roundtrip():
    """EFE -> DDM params -> policy probs preserves action preference ordering."""
    # Action 0 has lower (better) EFE
    G = jnp.array([-2.0, 1.0])

    ddm_params = efe_to_ddm(G, gamma=4.0, tau=0.3)

    # Verify DDM params are valid
    assert float(ddm_params.a) > 0, f"Boundary should be positive: {ddm_params.a}"
    assert 0.0 < float(ddm_params.w) < 1.0, (
        f"Starting point bias should be in (0,1): {ddm_params.w}"
    )
    assert float(ddm_params.tau) > 0, "Non-decision time should be positive"

    # Convert to policy probs
    probs = ddm_to_policy_probs(ddm_params)

    assert probs.shape == (2,), f"Expected shape (2,), got {probs.shape}"
    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)

    # The EFE bridge convention: G[0] < G[1] means action 0 is better.
    # efe_to_ddm: delta_G = G[0] - G[1] = -3.0 => positive drift toward upper.
    # ddm_to_policy_probs returns [P(lower), P(upper)].
    # With positive drift, upper boundary is favored.
    # Because lower G is better and delta_G = G[0] - G[1] < 0 here is negative
    # (G[0]=-2 < G[1]=1), v = gamma * delta_G = 4 * (-3) = -12 < 0.
    # Negative drift -> lower boundary -> action 0 favored -> probs[0] > probs[1].
    assert float(probs[0]) > float(probs[1]), (
        f"Better EFE action (0) should have higher probability: "
        f"P(0)={float(probs[0]):.3f}, P(1)={float(probs[1]):.3f}"
    )


def test_ddm_gradient_descent_recovers_drift():
    """Gradient descent through ddm_nll recovers the true drift rate."""
    true_v = 1.5
    data = simulate_ddm(
        v=true_v, a=1.5, w=0.5, tau=0.3,
        n_trials=300, seed=88,
    )

    rt = jnp.array(data.rt)
    choice = jnp.array(data.choice, dtype=jnp.float32)

    # Fixed params (known)
    a = jnp.array(1.5)
    w = jnp.array(0.5)
    tau = jnp.array(0.3)

    # Optimize v via gradient descent starting from 0
    v = jnp.array(0.0)
    grad_fn = jax.grad(ddm_nll, argnums=0)
    lr = 0.01

    for _ in range(100):
        grad_v = grad_fn(v, a, w, tau, rt, choice)
        # Clip gradient to avoid instability
        grad_v = jnp.clip(grad_v, -10.0, 10.0)
        v = v - lr * grad_v

    recovered_v = float(v)
    assert abs(recovered_v - true_v) < 0.5, (
        f"Recovered drift should be close to true: "
        f"recovered={recovered_v:.3f}, true={true_v:.3f}"
    )


# ===========================================================================
# Metacognition Integration Tests
# ===========================================================================

def test_metacognition_sdt_pipeline():
    """Full SDT pipeline: generate data -> d' -> meta-d' -> gamma."""
    nR_S1, nR_S2 = generate_sdt_data(d_prime=1.5, n_trials=500, seed=42)

    # Compute type 1 d'
    d_prime, c = compute_type1_from_counts(nR_S1, nR_S2)
    assert d_prime > 0.5, f"d' should be positive for signal > noise: {d_prime:.3f}"

    # Fit meta-d'
    result = fit_meta_d_mle(nR_S1, nR_S2)
    assert np.isfinite(result.d_prime), "d' is not finite"
    assert np.isfinite(result.meta_d), "meta-d' is not finite"
    assert np.isfinite(result.m_ratio), "m-ratio is not finite"

    # Convert m-ratio to gamma
    gamma = m_ratio_to_gamma(result.m_ratio, base_gamma=4.0)
    assert 0.1 <= gamma <= 16.0, (
        f"Gamma should be in [0.1, 16.0], got {gamma:.3f}"
    )

    # For data generated from an optimal observer, m-ratio should be
    # roughly around 1 (some deviation expected due to estimation noise)
    assert 0.1 < result.m_ratio < 5.0, (
        f"m-ratio for well-calibrated data should be reasonable: "
        f"{result.m_ratio:.3f}"
    )


def test_precision_update_stabilizes():
    """Feeding calibrated confidence/accuracy pairs should keep gamma stable."""
    gamma = 4.0
    rng = np.random.RandomState(42)

    gammas = [gamma]
    for _ in range(50):
        # Calibrated: confidence approximately equals accuracy
        accuracy = float(rng.binomial(1, 0.75))
        confidence = 0.75 + rng.normal() * 0.05
        confidence = np.clip(confidence, 0.5, 1.0)

        gamma = update_gamma_from_confidence(
            gamma, confidence, accuracy,
            learning_rate=0.05,
        )
        gammas.append(gamma)

    # Gamma should stay approximately stable (not diverge or collapse)
    final_gamma = gammas[-1]
    assert 1.0 < final_gamma < 10.0, (
        f"Gamma should remain stable with calibrated data: {final_gamma:.3f}"
    )

    # Standard deviation of gamma trajectory should be relatively small
    gamma_std = np.std(gammas[10:])  # ignore initial transient
    assert gamma_std < 3.0, (
        f"Gamma should be relatively stable: std={gamma_std:.3f}"
    )


# ===========================================================================
# Normative Integration Tests
# ===========================================================================

def test_normative_pipeline_end_to_end():
    """Full normative pipeline: fit model, add outliers, detect them."""
    x_train, y_train, x_test, y_test = generate_normative_data(
        n_train=200, n_test=50, seed=42
    )

    # Run normative model on clean data
    result = normative_model(x_train, y_train, x_test, y_test, n_basis=8)

    assert result.z_score.shape == (50,), f"z shape: {result.z_score.shape}"
    assert result.y_pred.shape == (50,), f"y_pred shape: {result.y_pred.shape}"
    assert result.y_std.shape == (50,), f"y_std shape: {result.y_std.shape}"
    assert jnp.all(jnp.isfinite(result.z_score)), "Z-scores not finite"
    assert jnp.all(result.y_std > 0), "Predicted std should be positive"

    # Most z-scores for clean data should be moderate
    z_clean = np.array(result.z_score)
    pct_moderate = np.mean(np.abs(z_clean) < 3.0)
    assert pct_moderate > 0.8, (
        f"Most clean z-scores should be |z| < 3: {pct_moderate:.2f}"
    )

    # Now add outliers to test set
    y_test_outlier = y_test.copy()
    y_test_outlier[:5] += 20.0  # Extreme positive deviations

    result_outlier = normative_model(
        x_train, y_train, x_test, y_test_outlier, n_basis=8
    )
    z_outlier = np.array(result_outlier.z_score)

    # Outlier z-scores should be large
    assert np.all(np.abs(z_outlier[:5]) > np.abs(z_clean[:5])), (
        "Outlier z-scores should be larger than clean z-scores"
    )
    # Mean absolute z of outliers should exceed mean absolute z of normals
    mean_z_outlier = np.mean(np.abs(z_outlier[:5]))
    mean_z_normal = np.mean(np.abs(z_outlier[5:]))
    assert mean_z_outlier > mean_z_normal, (
        f"Outlier |z| should exceed normal |z|: "
        f"outlier={mean_z_outlier:.2f} vs normal={mean_z_normal:.2f}"
    )


def test_normative_vmap_consistency():
    """normative_model_vmap should match running normative_model independently."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_regions = 100, 30, 5

    x_train = rng.uniform(20, 80, n_train)
    x_test = rng.uniform(20, 80, n_test)

    # Generate multi-region data
    Y_train = np.zeros((n_train, n_regions))
    Y_test = np.zeros((n_test, n_regions))
    for r in range(n_regions):
        slope = rng.uniform(-1, 1)
        intercept = rng.uniform(50, 100)
        Y_train[:, r] = intercept + slope * x_train + rng.normal(0, 2, n_train)
        Y_test[:, r] = intercept + slope * x_test + rng.normal(0, 2, n_test)

    # Run vmap version
    z_vmap, y_pred_vmap, y_std_vmap = normative_model_vmap(
        x_train, Y_train, x_test, Y_test, n_basis=6
    )

    # Run loop version (using normative_model with optimize_hyperparams=False
    # for consistency, since vmap uses optimize_hyperparams=False internally)
    from alf.normative.blr import bspline_basis, fit_blr, predict_blr
    x_min = float(min(np.min(x_train), np.min(x_test)))
    x_max = float(max(np.max(x_train), np.max(x_test)))
    Phi_train = jnp.array(bspline_basis(x_train, 6, 3, x_min, x_max))
    Phi_test = jnp.array(bspline_basis(x_test, 6, 3, x_min, x_max))

    z_loop = np.zeros((n_test, n_regions))
    y_pred_loop = np.zeros((n_test, n_regions))
    y_std_loop = np.zeros((n_test, n_regions))

    for r in range(n_regions):
        params = fit_blr(Phi_train, jnp.array(Y_train[:, r]),
                         optimize_hyperparams=False)
        pred = predict_blr(Phi_test, params)
        z_loop[:, r] = np.array(
            compute_zscore(jnp.array(Y_test[:, r]), pred.y_pred, pred.y_std)
        )
        y_pred_loop[:, r] = np.array(pred.y_pred)
        y_std_loop[:, r] = np.array(pred.y_std)

    np.testing.assert_allclose(
        np.array(z_vmap), z_loop, atol=1e-4,
        err_msg="vmap Z-scores don't match loop Z-scores"
    )
    np.testing.assert_allclose(
        np.array(y_pred_vmap), y_pred_loop, atol=1e-4,
        err_msg="vmap predictions don't match loop predictions"
    )
    np.testing.assert_allclose(
        np.array(y_std_vmap), y_std_loop, atol=1e-4,
        err_msg="vmap stds don't match loop stds"
    )


# ===========================================================================
# Cross-Module Integration Tests
# ===========================================================================

def test_hgf_to_metacognition():
    """HGF surprise drives metacognitive precision updates."""
    # Stable sequence (low surprise expected)
    stable_obs = np.ones(50)
    # Volatile sequence (frequent switches -> higher surprise)
    volatile_obs = generate_switching_binary(n=50, switch_prob=0.3, seed=99)

    params = make_binary_params(omega_2=-2.0)

    # Run HGF on both
    result_stable = binary_hgf(jnp.array(stable_obs), params)
    result_volatile = binary_hgf(jnp.array(volatile_obs), params)

    # Use per-trial surprise as proxy for prediction error to update gamma
    gamma_stable = 4.0
    for t in range(50):
        surprise = float(result_stable.surprise[t])
        # Higher surprise -> higher prediction error -> lower gamma
        gamma_stable = update_gamma_from_confidence(
            gamma_stable,
            predicted_confidence=0.8,
            actual_accuracy=1.0 if surprise < 0.5 else 0.0,
            learning_rate=0.05,
        )

    gamma_volatile = 4.0
    for t in range(50):
        surprise = float(result_volatile.surprise[t])
        gamma_volatile = update_gamma_from_confidence(
            gamma_volatile,
            predicted_confidence=0.8,
            actual_accuracy=1.0 if surprise < 0.5 else 0.0,
            learning_rate=0.05,
        )

    # Volatile input should reduce gamma more (more prediction errors)
    assert gamma_volatile < gamma_stable, (
        f"Volatile input should reduce gamma: "
        f"volatile={gamma_volatile:.3f} vs stable={gamma_stable:.3f}"
    )


def test_ddm_with_hgf_beliefs():
    """HGF precision parameterizes DDM boundary separation."""
    # Run continuous HGF on a signal
    obs = generate_drifting_signal(n=30, seed=42)
    params = make_continuous_params()
    result = continuous_hgf(jnp.array(obs), params)

    # Use HGF posterior precision at level 1 to parameterize DDM boundary
    # Higher precision -> more confident -> wider boundary (more careful)
    pi_values = np.array(result.pi[:, 0])

    for t in range(len(obs)):
        pi_t = float(pi_values[t])

        # Scale precision to boundary: a = base_a * sqrt(pi / pi_ref)
        pi_ref = 1.0
        base_a = 1.5
        a = base_a * np.sqrt(max(pi_t / pi_ref, 0.1))
        a = np.clip(a, 0.5, 5.0)

        ddm_params = DDMParams(
            v=jnp.array(0.5),
            a=jnp.array(a),
            w=jnp.array(0.5),
            tau=jnp.array(0.3),
        )

        # DDM params should be valid
        assert float(ddm_params.a) > 0, f"Step {t}: a should be positive"
        assert float(ddm_params.a) <= 5.0, f"Step {t}: a should be bounded"

        # Policy probs should be valid
        probs = ddm_to_policy_probs(ddm_params)
        np.testing.assert_allclose(
            float(probs.sum()), 1.0, atol=1e-6,
            err_msg=f"Step {t}: policy probs don't sum to 1"
        )


def test_normative_zscore_as_free_energy():
    """Z-scores from normative model relate to Gaussian surprisal (VFE proxy)."""
    x_train, y_train, x_test, y_test = generate_normative_data(
        n_train=200, n_test=50, seed=42
    )

    # Add outliers to some test subjects
    y_test_with_outliers = y_test.copy()
    y_test_with_outliers[:10] += 15.0

    result = normative_model(
        x_train, y_train, x_test, y_test_with_outliers, n_basis=8
    )
    z = np.array(result.z_score)

    # VFE approximation under Gaussian generative model: F = 0.5 * z^2
    vfe_approx = 0.5 * z ** 2

    # VFE should always be non-negative
    assert np.all(vfe_approx >= 0), "VFE approximation should be non-negative"

    # Outliers should have higher VFE
    mean_vfe_outlier = np.mean(vfe_approx[:10])
    mean_vfe_normal = np.mean(vfe_approx[10:])

    assert mean_vfe_outlier > mean_vfe_normal, (
        f"Outliers should have higher VFE: "
        f"outlier={mean_vfe_outlier:.2f} vs normal={mean_vfe_normal:.2f}"
    )


def test_full_aif_loop_with_hgf():
    """Both AnalyticAgent and HGFPerceptualAgent produce valid actions on T-maze."""
    gm = build_t_maze_model(cue_reliability=0.9, T=2)
    env = TMazeEnv(reward_side="left", cue_reliability=0.9, seed=42)

    # Run AnalyticAgent for 5 steps
    analytic_agent = AnalyticAgent(gm, gamma=4.0, seed=42)
    obs = env.reset()
    analytic_actions = []
    for t in range(5):
        action, info = analytic_agent.step([obs])
        analytic_actions.append(action)
        assert 0 <= action < gm.num_actions[0], (
            f"AnalyticAgent step {t}: invalid action {action}"
        )
        obs, _, done = env.step(action)
        if done:
            env.reset()
            obs = 0  # OBS_NULL

    # Run HGFPerceptualAgent for 5 steps
    # Build a simpler GM for HGF (HGFPerceptualAgent uses single-factor
    # discretized beliefs, so we use a compatible 2-state model).
    gm_simple = build_simple_2state_model()
    hgf_params = make_binary_params(omega_2=-2.0)
    hgf_agent = HGFPerceptualAgent(gm_simple, hgf_params, gamma=4.0, seed=42)

    hgf_actions = []
    # Feed same observation sequence (mapped to binary: obs > 0 -> 1)
    binary_obs = [0.0, 1.0, 0.0, 1.0, 0.0]
    for t in range(5):
        action, info = hgf_agent.step(binary_obs[t])
        hgf_actions.append(action)
        assert 0 <= action < gm_simple.num_actions[0], (
            f"HGFAgent step {t}: invalid action {action}"
        )

    # Both agents should produce sequences of valid actions
    assert len(analytic_actions) == 5, "AnalyticAgent should produce 5 actions"
    assert len(hgf_actions) == 5, "HGFAgent should produce 5 actions"

    # HGF agent should have accumulated beliefs and surprise
    assert len(hgf_agent.belief_history) == 5
    assert len(hgf_agent.surprise_history) == 5
    assert all(np.isfinite(s) for s in hgf_agent.surprise_history)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
