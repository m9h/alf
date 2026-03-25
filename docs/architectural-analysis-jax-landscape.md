# Architectural Analysis: ALF in the JAX Landscape

*pymdp vs. alf-native vs. the broader JAX ecosystem*

**Decision**: Keep alf-native. This document records the analysis.

---

## The Decision Space

| Option | What it means |
|--------|---------------|
| **A. Adopt pymdp** | Replace alf's core inference with `inferactively-pymdp` v1.0 (JAX-native since March 2026) |
| **B. Keep alf-native** | Maintain our own JAX primitives in `jax_core.py`, `jax_native.py`, `free_energy.py`, etc. |
| **C. Hybrid** | Use pymdp for discrete AIF core, keep alf's specialized modules (HGF, DDM, metacognition, normative) |
| **D. Broader JAX ecosystem** | Integrate with Gymnax/Brax/JaxMARL/PureJaxRL for environments and training loops |

---

## A. Adopt pymdp as Core Dependency

### Pros

- **Community and maintenance** — 653 stars, 126 forks, 1,411 commits, JOSS-published. Active maintainers (infer-actively group) means bug fixes and new features come for free.
- **SPM-validated** — Benchmarked against the gold-standard MATLAB SPM implementation; publishable without defending your own numerics.
- **JAX-native as of v1.0** — Now supports `jit`, `vmap`, batched agents (`batch_size` param), and functional PRNG handling — closing the execution gap that originally motivated alf.
- **Standard API** — Anyone who's read the active inference literature recognizes pymdp's `Agent.infer_states()` / `infer_policies()` / `sample_action()` interface.
- **Reduces maintenance surface** — ~1,200 lines of core inference code in alf (generative_model.py, agent.py, free_energy.py, sequential_efe.py, jax_native.py) that we'd no longer own.

### Cons

- **No `jax.grad` through inference** — This is the critical gap. pymdp's belief update is analytic Bayesian (conjugate updates), not a differentiable forward algorithm. Our `learning.py` differentiates through `jax.lax.scan`-based forward filtering to learn A/B matrices via gradient descent. pymdp cannot do this — it uses Dirichlet parameter counting, not gradient-based learning.
- **No hierarchical models** — pymdp has no equivalent to our `hierarchical.py` (multi-level temporal abstraction, context-dependent A matrices, cross-level epistemic value).
- **No HGF, DDM, metacognition, normative modeling** — These are alf-only. pymdp is strictly discrete state-space POMDP inference.
- **No deep generative models** — Our `deep_aif.py` (encoder/decoder neural A/B) has no pymdp equivalent.
- **Dependency coupling** — pymdp's API choices become our constraints. If they change matrix conventions or break backward compat, we absorb it.
- **Pedagogical cost** — spinning-up-alf explicitly teaches "why JAX-native AIF matters" by showing `jax.grad(NLL)` through the forward algorithm. Swapping to pymdp removes the most compelling demonstration.
- **Multi-step sequential EFE** — Our `sequential_efe.py` uses `jax.lax.scan` for T-step policy rollout. pymdp's EFE is computed differently; migrating would require validating numerical equivalence.

---

## B. Keep alf-native (Status Quo) — CHOSEN

### Pros

- **Full differentiability** — The killer feature. `jax.grad` flows through the entire inference → learning pipeline. This is what the blog post ecosystem (PureJaxRL, Gymnax) demonstrates: end-to-end compilation and differentiation is the JAX value proposition.
- **Architectural coherence** — One set of conventions (A/B/C/D matrices, `jax_core.py` primitives) shared across all modules: discrete AIF, HGF, DDM, metacognition, normative modeling.
- **Minimal dependencies** — Core alf needs only `jax >= 0.4.20` and `numpy >= 1.24`. Everything else is optional extras. This is the PureJaxRL philosophy: single-file, no framework overhead.
- **Educational control** — spinning-up-alf can teach every layer from matrix algebra to `vmap`-scaled batch agents because we own the code.
- **Performance tuning** — We control the `jit`/`vmap`/`scan` boundaries. No black-box calls through a third-party library's internal structure.
- **Specialized modules are alf-only anyway** — HGF, DDM, metacognition, normative, hierarchical, multitask (20 Yang tasks) — none of these exist in pymdp. We maintain these regardless.

### Cons

- **Maintenance burden** — We own all bugs. Numerical edge cases (underflow in belief updates, precision in sequential EFE) are on us.
- **Credibility gap** — Reviewers may ask "why not pymdp?" Requires justification in papers.
- **Smaller contributor pool** — pymdp has an established community; alf's bus factor is small.
- **API drift risk** — Without external pressure, our API may diverge from community conventions.

---

## C. Hybrid Approach (Evaluated, not chosen)

Use pymdp for what it does well; keep alf for what it can't:

| Layer | Source | Rationale |
|-------|--------|-----------|
| Discrete POMDP inference (non-learning) | pymdp | Mature, validated, community-maintained |
| Gradient-based learning (A/B fitting) | alf `learning.py` | Requires `jax.grad` through forward algorithm — pymdp can't |
| Deep generative models | alf `deep_aif.py` | Neural A/B, no pymdp equivalent |
| Hierarchical AIF | alf `hierarchical.py` | Multi-level, context-dependent — alf-only |
| HGF perception | alf `hgf/` | Continuous Gaussian filtering — orthogonal to pymdp |
| DDM bridge | alf `ddm/` | Reaction time modeling — orthogonal |
| Metacognition | alf `metacognition.py` | Meta-d', precision calibration — alf-only |
| Normative modeling | alf `normative/` | Population norms, ComBat — alf-only |

**Risk:** The interface boundary between pymdp's `Agent` and alf's learning/hierarchical modules becomes a maintenance seam. Matrix convention mismatches (pymdp may use different axis ordering or normalization) create subtle bugs.

**Why not chosen:** The boundary is in the worst possible place — right at the forward algorithm that learning needs to differentiate through. A hybrid would force us to maintain a compatibility shim between pymdp's inference output and our gradient-based learning input, which is more complex than just owning both.

---

## D. Broader JAX Ecosystem Integration

The Chris Lu blog post ([meta-disco](https://chrislu.page/blog/meta-disco/)) and the libraries used across spinning-up-alf point to a rich ecosystem of JAX-native tools. These are not alternatives to alf's inference core — they are **environment and training loop** libraries that alf agents can operate within.

### Environment libraries

| Library | What it gives us | Integration path | Priority |
|---------|-----------------|------------------|----------|
| [Gymnax](https://github.com/RobertTLange/gymnax) | 15+ JAX-native envs (CartPole, MetaMaze, MinAtar) | `gymnax.Env` → `GenerativeModel` wrapper (like our `neuronav_wrappers.py`) | **High** — expands benchmark suite |
| [Brax](https://github.com/google/brax) | Continuous control (HalfCheetah, Humanoid) | Would need continuous-state AIF extension | Medium — future work |
| [JaxMARL](https://github.com/FLAIROx/JaxMARL) | 11 multi-agent envs (Overcooked, SMAX, Hanabi, STORM) | Already relevant for SustainHub MARL comparison | **High** — directly serves OREL project |
| [Pgx](https://github.com/sotetsuk/pgx) | Board game environments (Go, Poker) | Strategic planning demos | Low |
| [Jumanji](https://github.com/instadeepai/jumanji) | Combinatorial optimization (TSP, Bin Packing) | Planning-heavy AIF | Low |
| [NeuroGym](https://github.com/neurogym/neurogym) | 30+ cognitive neuroscience tasks | Complements our `multitask.py` Yang battery; already referenced in spinning-up-alf A0 | Medium |

### Training and meta-learning libraries

| Library | What it gives us | Integration path | Priority |
|---------|-----------------|------------------|----------|
| [PureJaxRL](https://github.com/luchris429/purejaxrl) | End-to-end PPO training loop pattern | Architectural inspiration — we already follow this pattern with `BatchAgent` | Already adopted in spirit |
| [Evosax](https://github.com/RobertTLange/evosax) | Evolutionary strategies for meta-learning | Could meta-evolve agent hyperparameters (gamma, policy precision) | Speculative |

### Key ecosystem observations

1. **PureJaxRL's thesis applies to AIF**: the 1000x speedup from end-to-end JAX (no CPU-GPU transfer, full JIT compilation) is exactly what `BatchAgent` delivers for active inference. We're already following this architecture.

2. **Environment wrappers are the highest-value integration point**: adding Gymnax and JaxMARL wrappers (similar to our existing `neuronav_wrappers.py`) expands what alf agents can do without touching the inference core.

3. **Meta-evolution is the frontier**: Evosax + alf could meta-learn optimal precision schedules, EFE decomposition weights, or even generative model structure. This is where Chris Lu's "discovered environments" thesis meets active inference.

4. **No JAX-native AIF competitor exists at alf's scope**: pymdp covers discrete POMDP inference. No other library combines differentiable learning, deep generative models, HGF, DDM, metacognition, and normative modeling in a single JAX-native framework.

---

## Decision Matrix

| Criterion | pymdp | alf-native | Hybrid | Weight |
|-----------|-------|------------|--------|--------|
| Differentiable learning | No | **Yes** | Partial | **Critical** |
| Community/credibility | **Best** | Weakest | Good | High |
| Maintenance cost | **Lowest** | Highest | Medium | High |
| Hierarchical/deep models | No | **Yes** | Yes | High |
| HGF/DDM/metacog coverage | No | **Yes** | Yes | High |
| spinning-up-alf pedagogy | Weakens | **Strongest** | OK | Medium |
| API stability | Moderate risk | Full control | Seam risk | Medium |
| Performance control | Black box | **Full** | Mixed | Medium |

---

## Conclusion

**Keep alf-native for core inference. Use pymdp as a reference implementation for numerical validation, not as a runtime dependency.**

The fundamental reason: alf's value proposition is differentiability through inference. This is the property that makes the entire JAX ecosystem (PureJaxRL, Gymnax, Brax) powerful — end-to-end compilation and gradient flow. pymdp v1.0 is JAX-native for *execution* (jit, vmap) but not for *learning* (no `jax.grad` through belief updates). Adopting it would trade away the single most important architectural advantage.

The modules that justify alf's existence — gradient-based A/B learning, deep generative models, hierarchical inference, HGF, DDM, metacognition, normative modeling — have no pymdp equivalents and require the differentiable forward algorithm.

**Where pymdp helps:** Reference implementation for numerical validation. Cite it, benchmark against it, ensure our outputs match for standard discrete inference. But don't depend on it at runtime.

**Where to invest instead:** Environment wrappers for Gymnax and JaxMARL. These expand what alf agents can *do* without touching the inference core, and they directly serve the spinning-up-alf curriculum and the OREL SustainHub project.

---

## Appendix: ALF's JAX Usage Patterns

| JAX Transform | Where in ALF | Purpose |
|--------------|-------------|---------|
| `jax.jit` | `learning.py`, `ddm/wiener.py`, `hgf/updates.py` | Compile hot paths |
| `jax.vmap` | `jax_native.py` (BatchAgent), `normative/*.py` | Batch over agents, brain regions |
| `jax.grad` | `learning.py`, `ddm/fitting.py`, `hgf/learning.py`, `normative/warping.py` | Differentiate through forward algorithms |
| `jax.lax.scan` | `learning.py`, `sequential_efe.py`, `hgf/updates.py` | Temporal rollout (HMM forward, EFE, HGF) |
| `jax.random` | `jax_native.py`, `deep_aif.py` | Functional PRNG for reproducibility |

### Design principles

- **No Flax/Haiku/Equinox**: neural nets use raw `jax.numpy` + `(weight, bias)` tuple pytrees
- **Optional optax**: only in `[learning]` extra; core inference needs no optimizer
- **Optional numpyro**: only for advanced Bayesian fitting (DDM hierarchical, metacognition)
- **NumPy for single agents, JAX for batches**: `AnalyticAgent` stays NumPy for clarity; `BatchAgent` uses `vmap` for scaling
