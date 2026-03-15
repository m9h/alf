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
