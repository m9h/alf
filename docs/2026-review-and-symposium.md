# ALF maintenance and scientific review — November 2026

Reviewed November 8–9, 2026. Starting revision: `e1af0110d3b9dd32f55477fac1f7ce632b9b724e`.
The local checkout matched `origin/main` before edits. Changes are on
`maintenance/dependencies-symposium-2026`; this review does not imply a release.

## What changed

- Added a tracked uv lockfile and a Python 3.12 development default while retaining Python 3.10 support. Updated dependency floors for NumPy, JAX, SciPy, Optax, NumPyro, pytest and pymdp. JAX now selects the appropriate `jaxlib` itself.
- Added NumPyro to the DDM extra, whose public API includes Bayesian fitting.
- Replaced CI's missing `--group dev` and error-suppressing fallback with `uv sync --locked --extra dev`. Tests and lint are separate jobs; existing lint failures cannot prevent numerical tests from running.
- Updated GitHub Actions to verified releases: [checkout 7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1), [setup-uv 10.3.3](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.3). Ruff is pinned to 0.17.4 in the dev extra and pre-commit config. Fixed the no-mock hook's test-directory path.
- Removed tracked bytecode and egg-info from Git's index, retaining local generated files. Excluded test packages from distribution discovery.
- Corrected README call signatures for learning, decoder likelihood extraction, hierarchical agents, DDM mapping and meta-d' estimation. Removed unsupported exclusivity claims about JAX/differentiable inference and corrected the architecture comparison with pymdp.

The lock selects JAX/JAXlib **0.12.0.1**, NumPy **2.5.4**, SciPy **1.18.1**,
Optax **0.2.8**, NumPyro **0.22.0**, pymdp **1.0.5**, pytest **9.2.0**, and
Ruff **0.17.4** on Python 3.12. Python 3.10 resolves compatible older JAX,
NumPy and SciPy versions. This is intentional, not a claim that current JAX
supports Python 3.10.

NeuroGym 2.3.1 constrains NumPy below 2.3 and Matplotlib below 3.11. It was
tested in the curriculum's separate environment rather than added to ALF's
universal lock, where even an optional extra would constrain other resolutions.

## Correctness changes and migration

The follow-up used red–green TDD. `G = -E[C(o)] - I(s;o)` now agrees across
NumPy/JAX decomposition, analytic action scoring, sequential rollout and
multi-factor scoring. Hierarchical context-dependent levels use information
about the joint (state, context), preserving interactions such as XOR.
C remains the supplied log preference; no implicit softmax was added.

Independent tests enumerate observations and Bayesian posterior KL rather
than repeating the implementation's entropy formula. Full policy-score vectors
and posterior probabilities match pymdp 1.0.5 at horizons 1, 2 and 3. These
checks cover a single modality/factor; multi-modal scoring sums marginal
information gains, which can double-count correlated observations. Hierarchical
scores likewise sum local objectives, not global joint information.

Sequential planning remains open-loop. T-maze tests that asserted cue-first
must win were replaced with exact score checks: the arms reveal reward
condition perfectly, while a noisy cue reveals less. Repeated per-step
information terms can overlap. These limitations matter for scientific claims
about exploration and contingent planning.

The agent now predicts beliefs through the previously executed action before
conditioning on the next observation. Habit learning credits the sampled
policy, rather than the mode of the policy distribution. Both have dedicated
regressions including reset behavior.

Learning changes:

- `learn_model` adds `seed` and `num_restarts` (default 4). It selects the fit
  with lowest final training NLL. Fully supplied initial parameters are fit
  once. This costs more computation; use `num_restarts=1` for a single run.
  The former default could remain near a uniform local optimum. Restarts
  improve this case without changing its recovery threshold or choosing a
  hidden favorable seed. They do not guarantee a global optimum.
- `learn_deep_model` optimizes mean per-observation loss, with Adam when Optax
  is installed and SGD otherwise. History still reports total loss. This
  removes sequence-length scaling of the step size; training trajectories
  change and should be regenerated.
- **API correction:** `extract_A_matrix` now returns column-normalized P(o|s),
  matching `encoder_amatrix_nll`. Previously it returned row-normalized
  recognition probabilities. For recognition outputs use
  `jax.nn.softmax(encode(params, observation))`. Column-offset invariance is
  tested, because offsets that leave a likelihood unchanged can alter those
  recognition probabilities.
- The legacy `entropy_reg=0` neural encoder objective still weights filtering
  with recognition outputs; it is not a normalized observation likelihood.
  For categorical generative learning use the A-matrix objective (`entropy_reg
  > 0`) or the decoder architecture. Parameter identifiability and held-out
  validation remain separate from passing recovery tests.

References: [Da Costa et al.](https://arxiv.org/abs/2001.07203),
[pymdp control](https://github.com/infer-actively/pymdp/blob/main/pymdp/control.py).

## Red–green evidence

| Stage | Red, before the corresponding implementation change | Green |
|---|---|---|
| Single-step and sequential EFE | 18 failed, 5 passed; noisy sensor preferred to informative sensor | 23 passed |
| Multi-factor and hierarchical EFE | 5 failed, 24 passed, including XOR | 29 passed |
| EFE plus external/gradient validation | Complete score/posterior and finite-difference checks added | 33 passed |
| Learning restart and sequence-length regressions | 2 failed | Passed with existing recovery tests |
| Likelihood extraction | State-offset/normalization regression failed | Passed |
| Filtering and sampled-policy habits | 2 failed | 2 passed |
| Curriculum manual EFE and plot sign | 2 failed | 2 passed |

No recovery threshold was relaxed or expected-failure marker added. Incorrect
cue-dominance assertions and the old extraction row-normalization assertion
were corrected to match independently specified semantics.

## Validation

Local platform: macOS ARM64, CPU. No GPU or remote GitHub Actions validation
is implied by these local results.

| Check | Result |
|---|---|
| Initial dependency update, Python 3.12/JAX 0.12.0.1 | 436 passed, 2 failed, 27 skipped |
| Initial dependency update, Python 3.10/JAX 0.6.2 | 436 passed, 2 failed, 27 skipped |
| Legacy JAX 0.5.3/Optax 0.2.4 reproduction | Both original learning failures reproduced |
| Corrected full suite, Python 3.12.13/JAX 0.12.0.1 | **476 passed, 27 skipped**, 71.00 seconds |
| Corrected full suite, Python 3.10.20/JAX 0.6.2 | **476 passed, 27 skipped**, 80.28 seconds |
| NeuroGym adapter integration | **25 passed** with NeuroGym 2.3.1 after correctness fixes |
| Dependency consistency and packaging | Pip checks and rebuilt wheel/sdist passed; no tests or bytecode in wheel |
| Repository-wide lint/format | Pre-existing debt: initial 317 lint findings, 54 files needing formatting |

The full-suite skips are 25 NeuroGym tests and two external oracle checks.
New regression files are linted separately. Repository-wide lint is still a
known failing check; the numerical suite passing does not make all CI green.

The original recovery failure was A error 0.3611 (threshold 0.15). The neural
failure mapped three primary observations to only two states. Both now pass
without changing seeds, datasets, training epochs or recovery thresholds.

Further work should evaluate held-out likelihood over multiple datasets,
shared state permutations, posterior-predictive checks, and GPU scaling with
synchronized timings. The padded-action learning convention remains: T
observations use an equal-length action array whose final element is ignored.

## Priority 3: integration and maintenance

- Keep current pymdp adapters, but retain and broaden the new reference tests. [pymdp's
  current documentation](https://github.com/infer-actively/pymdp) provides a
  JAX-first interface and model-fitting extras; the old claim that ALF alone
  supports differentiable inference is not a defensible differentiator.
- Add a scheduled optional-integration CI job for NeuroGym using its constrained
  environment, independent of the current numerical dependency stack.
- Resolve lint and formatting in a dedicated change so the functional audit
  remains reviewable. Pinning Ruff prevents unexplained rule drift.
- Add CI artifact retention for parameter-recovery outputs, environment
  versions, seeds and benchmark metadata; GPU scaling claims need synchronized
  timings and declared hardware.

## Symposium fit

The [6th Applied Active Inference Symposium](https://activeinference.institute/projects/symposium/)
is online November 12–13, 2026. The companion curriculum has the
[complete program, Pacific-time shortlist and project follow-up plan](../../spinning-up-alf/docs/2026-symposium.md)
and an [importable calendar](../../spinning-up-alf/docs/2026-program.ics).

ALF fits Ryan Smith's computational psychiatry session through its HGF/DDM/
metacognition components; the curriculum fits Alison Gopnik's RL and Bayesian
learning session and the cpomdp continuous-control workshop. These are proposed
connections, not accepted presentations.

The strongest immediate contribution is a small comparison suite for the
[Field Working Group](https://activeinference.institute/working-groups/), whose
proposal explicitly seeks terminology correspondences and versioned numerical
comparisons. Follow [cpomdp](https://github.com/inferogenesis/cpomdp) for
continuous control and [Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
for inspectable model exchange. Treat both as candidates to evaluate, not new
runtime dependencies to add without a demonstrated use case.
