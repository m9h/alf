---
category: infrastructure
section: appendix
weight: 30
title: "ALF Module Architecture and Conventions"
status: draft
slide_summary: "ALF is a JAX-native active inference library with 11 modules spanning generative models, batch agents, hierarchical inference, and differentiable learning, all following consistent matrix conventions (A, B, C, D) with EFE minimization."
tags: [alf, active-inference, jax, architecture, module-map, conventions]
---

# ALF — Active inference/Learning Framework

## Quick start

```bash
cd /home/mhough/dev/alf
pytest alf/tests/ -v   # 87 tests + 1 xfail
```

## Module map

| Module | Purpose |
|--------|---------|
| `generative_model.py` | GenerativeModel (A, B, C, D matrices, policy enumeration) |
| `agent.py` | AnalyticAgent — single-agent AIF with sequential EFE |
| `free_energy.py` | VFE, EFE decomposition, JAX-native versions |
| `sequential_efe.py` | Multi-step EFE via forward rollout (jax.lax.scan) |
| `jax_native.py` | BatchAgent — vmapped batch agents (1-1000+) |
| `learning.py` | Differentiable HMM parameter learning |
| `deep_aif.py` | Neural network generative models |
| `hierarchical.py` | Multi-level models, context-dependent A, cross-level inference |
| `policy.py` | Softmax action selection, habit learning, precision dynamics |
| `benchmarks/t_maze.py` | T-maze benchmark (8 states, 5 obs, 4 actions) |
| `benchmarks/neuronav_wrappers.py` | Bridge neuronav GridEnv → GenerativeModel |
| `compat.py` | ALF ↔ pymdp v1.0.0 adapter (alf_to_pymdp, pymdp_to_alf, EFE conversion) |

## pymdp interop

ALF complements pymdp v1.0.0 (JAX-first). Install with `pip install inferactively-pymdp`.

- `alf.alf_to_pymdp(gm)` → pymdp Agent (adds batch dim, converts to float32)
- `alf.pymdp_to_alf(agent)` → ALF GenerativeModel (strips batch dim)
- `alf.neg_efe_to_G(neg_efe)` → G (pymdp higher=better → ALF lower=better)
- `alf.G_to_neg_efe(G)` → neg_efe (reverse)

## Key conventions

- A matrix: `(n_obs, n_states)` — P(observation | state)
- B matrix: `(n_states, n_states, n_actions)` — P(next_state | state, action)
- C vector: `(n_obs,)` — log-preferences over observations
- D vector: `(n_states,)` — prior beliefs
- EFE convention: lower G = better policy (minimize expected free energy)
- JAX functions prefixed with `jax_` (e.g., `jax_variational_free_energy`)

## Testing

```bash
pytest alf/tests/ -v              # all tests
pytest alf/tests/test_jax_native.py  # just BatchAgent tests
```

## Used by

- spinning-up-alf (m9h/spinning-up-alf) — educational curriculum, notebooks 08-16
