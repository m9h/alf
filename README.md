# ALF — Active inference/Learning Framework

[![CI](https://github.com/m9h/alf/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/alf/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A5%200.5.0-9cf.svg)](https://github.com/google/jax)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests: 439](https://img.shields.io/badge/tests-439-brightgreen.svg)](alf/tests/)

A standalone, JAX-native library for **Active Inference** on discrete POMDPs. Fully differentiable via `jax.grad`, vectorizable via `jax.vmap`, and GPU-acceleratable via `jax.jit`.

> **The key idea:** ALF parameterizes generative model matrices in unconstrained log-space, maps them through softmax, and computes the negative log-likelihood via `jax.lax.scan`-based forward filtering. The entire pipeline — from raw parameters to observation likelihood — is a single differentiable computation graph. This is something no other active inference framework provides.

---

## Why ALF?

| Library | Backend | Differentiable learning | Best for |
|---------|---------|------------------------|----------|
| [pymdp](https://github.com/infer-actively/pymdp) | JAX (v1.0) | Dirichlet counting | Standard AIF experiments |
| **ALF** | **JAX** | **`jax.grad` through forward algorithm** | **Learning, scaling, deep AIF, computational psychiatry** |
| [PGMax/aif](https://github.com/vicariousinc/PGMax) | PGMax BP | Message passing | Large state spaces |

<details>
<summary><b>What makes ALF different?</b></summary>

- **End-to-end differentiability** — `jax.grad` flows through the entire inference-to-learning pipeline, enabling gradient-based learning of A/B matrices
- **Batch scaling** — `jax.vmap` runs 1,000+ parallel agents with sub-linear GPU scaling
- **Minimal dependencies** — core needs only JAX + NumPy; no Flax, Haiku, or Equinox
- **Computational psychiatry toolkit** — HGF, DDM, metacognition, and normative modeling all in one JAX-native framework
- **Pure functional design** — all core functions are pure, so `jax.jit` works seamlessly

</details>

---

## Quick start

```bash
pip install alf-aif                    # core (JAX + NumPy only)
pip install "alf-aif[all]"             # all optional dependencies
```

```python
import alf
import numpy as np

# Define a POMDP generative model
A = [np.eye(3)]                        # likelihood: P(o|s)
B = [np.stack([                        # transitions: P(s'|s,a)
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),  # action 0: rotate
    np.eye(3),                                       # action 1: stay
], axis=-1)]
C = [np.array([1.0, 0.0, 0.0])]       # prefer observation 0
D = [np.array([1/3, 1/3, 1/3])]       # uniform prior

gm = alf.GenerativeModel(A=A, B=B, C=C, D=D)

# Single agent (NumPy — for clarity and debugging)
agent = alf.AnalyticAgent(gm, gamma=4.0)
action, info = agent.step([0])

# Batch of 1000 parallel agents (JAX vmap — for scaling)
batch = alf.BatchAgent(gm, batch_size=1000)
obs = np.zeros((1000,), dtype=int)
actions, info = batch.step_analytic(obs)
```

---

## Architecture

```
alf/
├── Core inference
│   ├── generative_model.py      GenerativeModel dataclass (A, B, C, D, E)
│   ├── agent.py                 AnalyticAgent — single-agent AIF loop
│   ├── jax_native.py            BatchAgent — vmap-parallel agents
│   ├── jax_core.py              Shared primitives: softmax, safe_log, entropy
│   ├── free_energy.py           Variational and expected free energy
│   ├── sequential_efe.py        Multi-step EFE via jax.lax.scan
│   └── policy.py                Action selection, habits, precision
│
├── Learning
│   ├── learning.py              Differentiable HMM (jax.grad through NLL)
│   └── deep_aif.py              Neural encoder/decoder generative models
│
├── Extended models
│   ├── hierarchical.py          Multi-level temporal abstraction
│   ├── multitask.py             Compositional cognitive task batteries
│   └── multitask_agent.py       MultitaskAgent with task switching
│
├── Computational psychiatry
│   ├── metacognition.py         Meta-d', m-ratio, precision calibration
│   ├── hgf/                     Hierarchical Gaussian Filter
│   │   ├── updates.py           Binary & continuous update equations
│   │   ├── graph.py             Generalized n-level graph HGF
│   │   ├── bridge.py            HGF → discrete action bridge
│   │   └── learning.py          Differentiable parameter learning
│   ├── ddm/                     Drift-Diffusion Models
│   │   ├── wiener.py            Navarro-Fuss density (pure JAX)
│   │   ├── bridge.py            EFE ↔ DDM parameter mapping
│   │   └── fitting.py           MLE / Bayesian / hierarchical fitting
│   └── normative/               Normative modeling
│       ├── blr.py               Bayesian Linear Regression + B-splines
│       ├── warping.py           SHASH likelihood warping
│       ├── combat.py            ComBat multi-site harmonization
│       └── bridge.py            Z-score → VFE bridge
│
├── Environments
│   ├── envs/cognitive_tasks.py  20 Yang et al. (2019) tasks
│   └── envs/neurogym_bridge.py  NeuroGym adapter
│
└── Benchmarks
    ├── t_maze.py                T-maze (8 states, 5 obs, 4 actions)
    ├── go_nogo.py               Go/no-go inhibitory control
    ├── delayed_match.py         Delayed match-to-sample
    ├── context_dm.py            Context-dependent matching
    ├── cognitive_battery.py     Full Yang et al. battery
    └── neuronav_wrappers.py     neuronav GridEnv → GenerativeModel
```

---

## Core concepts

### Generative model

A discrete POMDP defined by five matrices:

| Matrix | Shape | Meaning |
|--------|-------|---------|
| **A** | `(n_obs, n_states)` per modality | Likelihood: P(observation \| state) |
| **B** | `(n_states, n_states, n_actions)` per factor | Transition: P(next_state \| state, action) |
| **C** | `(n_obs,)` per modality | Log-preferences over observations |
| **D** | `(n_states,)` per factor | Prior beliefs over initial states |
| **E** | `(n_policies,)` | Policy prior (habits) |

Multi-factor and multi-modality models are supported — pass lists of arrays.

### Free energy

- **VFE** (Variational Free Energy) — divergence between beliefs and observations; drives *perception*
- **EFE** (Expected Free Energy) — expected VFE under future policies; drives *action*
  - **Pragmatic** term: preference satisfaction (`-E[ln P(o|C)]`)
  - **Epistemic** term: ambiguity reduction (`-E[H[P(o|s)]]`)

Lower G = better policy. The `EFEDecomposition` NamedTuple exposes both terms for analysis.

### JAX scaling pattern

```
1 agent       →  AnalyticAgent           NumPy, for clarity and debugging
N agents      →  BatchAgent + jax.vmap   sub-linear GPU scaling
1000+ regions →  normative_model_vmap    parallel population analysis
1000+ subjects → fit_ddm_hierarchical    Bayesian hierarchy via numpyro
```

---

## Modules

### Differentiable learning

Learn A and B matrices from observation-action sequences via gradient descent:

```python
from alf import learn_model

params, loss_history = learn_model(
    gm, observations, actions,
    learning_rate=1e-3, num_steps=500
)
```

Internally: parameterize in log-space → softmax → forward-filter NLL → `jax.grad` → optax/SGD update. The forward algorithm runs entirely in `jax.lax.scan`, so the full pipeline is JIT-compiled and differentiable.

### Deep Active Inference

Scale to high-dimensional observations with neural network A/B matrices:

```python
from alf import learn_decoder_model

result = learn_decoder_model(
    observations, actions, num_states=8,
    hidden_dims=[64, 32], learning_rate=1e-3
)
A_learned = result.A  # explicit likelihood matrix via Bayes' rule
```

Two architectures:
- **Encoder** (`obs → Q(s|o)`) — fast but prone to degenerate solutions
- **Decoder** (`state → P(o|s)`, preferred) — learns P(o|s) explicitly, uses Bayes' rule for inference

Pure JAX — no Flax, Haiku, or Equinox dependency. Parameters are `(weight, bias)` tuples for transparent pytree handling.

### Hierarchical models

Multi-level temporal abstraction with context-dependent likelihood:

```python
from alf import HierarchicalLevel, HierarchicalAgent

levels = [
    HierarchicalLevel(A=A_low, B=B_low, C=C_low, D=D_low, temporal_scale=1),
    HierarchicalLevel(A=A_high, B=B_high, C=C_high, D=D_high, temporal_scale=4),
]
agent = HierarchicalAgent(levels, gamma=4.0)
```

Upper levels modulate lower-level A matrices (top-down attention). Bottom-up and top-down belief propagation via alternating sweeps.

### Hierarchical Gaussian Filter (HGF)

Continuous-valued perception for volatile environments:

```python
from alf.hgf import binary_hgf, BinaryHGFParams

params = BinaryHGFParams(omega_2=-2.0, mu_2_0=0.0, sigma_2_0=1.0)
result = binary_hgf(observations, params)
# result.mu, result.pi, result.surprise — full trajectory
```

- Binary (2-level) and continuous (3-level) variants
- Generalized n-level graph HGF with arbitrary topology
- Bridge to discrete action selection via `HGFPerceptualAgent`
- Differentiable parameter learning via `jax.grad` through surprise

### Drift-Diffusion Models (DDM)

Reaction time modeling with a novel pure-JAX Navarro-Fuss implementation:

```python
from alf.ddm import DDMParams, fit_ddm_mle, efe_to_ddm

# Fit DDM to reaction time data
result = fit_ddm_mle(rt_data, accuracy_data)

# Map Active Inference quantities to DDM parameters
ddm_params = efe_to_ddm(delta_G=2.0, gamma=4.0, E_ratio=0.6)
# drift v = gamma * delta_G,  boundary a = gamma,  bias z = ln E
```

MLE, Bayesian (numpyro), and hierarchical Bayesian fitting for multi-subject data.

### Metacognition

Meta-d' estimation and precision calibration:

```python
from alf.metacognition import fit_meta_d_mle, MetacognitiveAgent

result = fit_meta_d_mle(hits=80, misses=20, FA=15, CR=85, nRatings=4)
# result.d_prime, result.meta_d, result.m_ratio

# Self-monitoring agent that adapts precision online
agent = MetacognitiveAgent(gm, gamma=4.0, window_size=20)
```

- `EFEMonitor` — online tracking of EFE prediction accuracy
- `MetacognitiveAgent` — wraps AnalyticAgent with confidence calibration
- `PopulationMetacognition` — aggregate stats across agent populations

### Normative modeling

Population-level deviation scoring for individual differences:

```python
from alf.normative import normative_model_vmap, combat_harmonize

# Fit normative models across 10,000 brain regions in parallel (jax.vmap)
results = normative_model_vmap(X_train, Y_train, X_test, Y_test)
# results.z_scores — individual deviation from population norm

# Harmonize multi-site data before normative modeling
Y_harmonized = combat_harmonize(Y_raw, batch_labels, covariates)
```

Z-score = surprisal under the population prior. The `bridge` module connects normative deviations directly to variational free energy.

### Multitask / cognitive battery

Compositional generative models for 20 cognitive neuroscience tasks (Yang et al. 2019):

```python
from alf import MultitaskAgent, build_compositional_battery

battery = build_compositional_battery()  # Go, Anti, Delay, Context tasks
agent = MultitaskAgent(battery, gamma=4.0)

agent.switch_task("DelayAnti")
action, info = agent.step([obs])
```

Three composition modes: `independent`, `shared_dynamics`, `compositional` (factored state spaces).

---

## Installation

Requires **Python >= 3.10**. Tested on **3.11** and **3.12** in CI.

```bash
pip install alf-aif                              # core only
pip install "alf-aif[learning]"                  # + optax for gradient descent
pip install "alf-aif[ddm]"                       # + scipy for SDT functions
pip install "alf-aif[metacognition]"             # + numpyro + scipy
pip install "alf-aif[normative]"                 # + scipy for B-splines
pip install "alf-aif[pymdp]"                     # + pymdp interop adapter
pip install "alf-aif[all]"                       # everything
pip install "alf-aif[dev]"                       # all + pytest + ruff + mypy
```

<details>
<summary><b>Dependency map</b></summary>

| Extra | Packages added | Unlocks |
|-------|---------------|---------|
| *(core)* | `jax >= 0.5.0`, `numpy >= 1.24` | GenerativeModel, agents, free energy, sequential EFE, HGF, deep AIF |
| `learning` | `optax >= 0.1.7` | Adam optimizer for `learn_model` (SGD fallback without) |
| `ddm` | `scipy >= 1.10` | Norm CDF for SDT, DDM fitting |
| `metacognition` | `numpyro >= 0.13.0`, `scipy >= 1.10` | Hierarchical Bayesian meta-d' |
| `normative` | `scipy >= 1.10` | B-spline basis for BLR |
| `pymdp` | `inferactively-pymdp >= 1.0.0` | `alf_to_pymdp`, `pymdp_to_alf` adapters |
| `test` | `pytest >= 7.0`, `pytest-xdist`, `scipy >= 1.10` | Test suite |
| `dev` | all + `ruff`, `mypy` | Linting and type checking |

</details>

---

## Testing

```bash
pytest alf/tests/ -v                             # full suite (439 tests)
pytest alf/tests/test_jax_native.py              # BatchAgent tests
pytest alf/tests/test_learning.py                # differentiable learning
pytest alf/tests/test_hgf.py                     # HGF tests
pytest alf/tests/test_ddm.py                     # DDM tests
pytest alf/tests/test_metacognition.py           # metacognition tests
pytest alf/tests/test_normative.py               # normative modeling
pytest alf/tests/test_integration.py             # cross-module integration
```

Oracle validation tests (`test_*_oracle.py`) benchmark against reference implementations: HDDM, metadPy, and pyhgf.

**CI** runs on every push to `main` and on all PRs: Python 3.11 + 3.12, ruff lint/format, no-mock enforcement, full pytest suite.

---

## Conventions

| Convention | Detail |
|-----------|--------|
| **EFE sign** | Lower G = better policy (minimize expected free energy) |
| **JAX functions** | Prefixed with `jax_` (e.g., `jax_variational_free_energy`) |
| **A matrix** | `(n_obs, n_states)` — P(observation \| state) |
| **B matrix** | `(n_states, n_states, n_actions)` — P(next_state \| state, action) |
| **B indexing** | `B[:, s, a]` = P(s' \| s, a) — destinations in rows, sources in columns |
| **Numerical stability** | `safe_log` (epsilon-clipped) and `safe_normalize` from `jax_core.py` |

---

## Used by

- [spinning-up-alf](https://github.com/m9h/spinning-up-alf) — educational curriculum covering RL, Active Inference, and computational neuroscience (Modules 08-16)

---

## References

- Smith, Friston & Whyte (2022). *A Step-by-Step Tutorial on Active Inference.* Journal of Mathematical Psychology.
- Mathys, Daunizeau, Friston & Stephan (2011). *A Bayesian foundation for individual learning under uncertainty.* Frontiers in Human Neuroscience.
- Navarro & Fuss (2009). *Fast and accurate calculations for first-passage times in Wiener diffusion models.* Journal of Mathematical Psychology.
- Maniscalco & Lau (2012). *A signal detection theoretic approach for estimating metacognitive sensitivity.* Consciousness and Cognition.
- Fleming (2017). *HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency.* Neuroscience of Consciousness.
- Marquand, Rezek, Buitelaar & Beckmann (2016). *Understanding heterogeneity in clinical cohorts using normative models.* Biological Psychiatry.
- Tschantz, Millidge et al. (2020). *Reinforcement Learning through Active Inference.* arXiv:2002.12636.
- Yang, Joglekar, Song et al. (2019). *Task representations in neural networks trained to perform many cognitive tasks.* Nature Neuroscience.

---

## License

Apache 2.0
