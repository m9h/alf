"""NeuroGym-to-ALF bridge adapter.

Wraps NeuroGym cognitive neuroscience task environments (Yang et al. 2019)
for use with ALF's Active Inference agents. NeuroGym provides Gymnasium-
compatible environments with continuous observation spaces and discrete
action spaces. This adapter discretizes continuous observations so they
can be consumed by ALF's discrete POMDP machinery.

Two discretization strategies are supported:
    - "bin":    Uniform binning per dimension with mixed-radix flattening.
    - "kmeans": KMeans clustering on sampled observations (requires fit()).

The adapter exposes the same minimal environment interface used by
TMazeEnv in alf.benchmarks.t_maze:
    reset()  -> int
    step(action: int) -> (int, float, bool)

It also provides build_generative_model() to automatically construct an
ALF GenerativeModel from interaction statistics.

References:
    Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., &
        Wang, X.-J. (2019). Task representations in neural networks
        trained to perform many cognitive tasks. Nature Neuroscience,
        22(2), 297-306.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from alf.generative_model import GenerativeModel

# NeuroGym is an optional dependency; defer the import error until use.
try:
    import neurogym  # noqa: F401

    _NEUROGYM_AVAILABLE = True
except ImportError:
    _NEUROGYM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Known trial phases in NeuroGym tasks.  The info dict returned by many
# tasks includes a "new_trial_info" or "gt" (ground truth) key; some tasks
# expose the current period name directly.  We define the canonical set of
# phase names here for convenience.
# ---------------------------------------------------------------------------
TRIAL_PHASES = ("fixation", "stimulus", "delay", "decision", "response")


def _require_neurogym() -> None:
    """Raise a clear error if neurogym is not installed."""
    if not _NEUROGYM_AVAILABLE:
        raise ImportError(
            "neurogym is required for NeurogymAdapter but is not installed. "
            "Install it with: pip install neurogym"
        )


# ---------------------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------------------


class BinDiscretizer:
    """Uniform binning of continuous observations.

    Each dimension of the observation vector is independently binned into
    *n_bins* uniform intervals spanning [low, high].  The per-dimension
    bin indices are then combined into a single flat index via mixed-radix
    encoding (like np.ravel_multi_index).

    Args:
        n_bins: Number of bins per observation dimension.
        low: Lower bound per dimension (array or scalar).
        high: Upper bound per dimension (array or scalar).
        obs_dim: Number of observation dimensions.
    """

    def __init__(
        self,
        n_bins: int,
        low: np.ndarray,
        high: np.ndarray,
        obs_dim: int,
    ):
        self.n_bins = n_bins
        self.obs_dim = obs_dim
        self.low = np.asarray(low, dtype=np.float64).flatten()[:obs_dim]
        self.high = np.asarray(high, dtype=np.float64).flatten()[:obs_dim]
        # Clamp degenerate ranges (constant dims) to avoid division by zero.
        span = self.high - self.low
        span[span < 1e-8] = 1.0
        self.span = span
        self.n_obs = n_bins ** obs_dim

    def discretize(self, obs: np.ndarray) -> int:
        """Map a continuous observation vector to a flat integer index."""
        obs = np.asarray(obs, dtype=np.float64).flatten()[: self.obs_dim]
        # Normalize to [0, 1] then to bin index.
        normed = (obs - self.low) / self.span
        normed = np.clip(normed, 0.0, 1.0 - 1e-10)
        bin_indices = (normed * self.n_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        # Mixed-radix encoding: index = sum_i(bin_i * n_bins^i)
        flat = int(np.ravel_multi_index(bin_indices, [self.n_bins] * self.obs_dim))
        return flat


class KMeansDiscretizer:
    """KMeans clustering of continuous observations.

    Observations are assigned to the nearest cluster centre.  Unlike the
    bin discretizer, this requires a fit() step on sampled observations
    before it can be used.

    Args:
        n_clusters: Number of discrete observation categories.
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.n_obs = n_clusters
        self.centres: Optional[np.ndarray] = None

    def fit(self, observations: np.ndarray, max_iter: int = 100) -> None:
        """Fit cluster centres using a simple KMeans (no sklearn needed).

        Args:
            observations: Array of shape (n_samples, obs_dim).
            max_iter: Maximum KMeans iterations.
        """
        observations = np.asarray(observations, dtype=np.float64)
        n = len(observations)
        k = min(self.n_clusters, n)

        # Initialise with k-means++ style: first centre random, then
        # proportional to squared distance.
        rng = np.random.RandomState(0)
        indices = [rng.randint(n)]
        for _ in range(1, k):
            dists = np.min(
                [np.sum((observations - observations[c]) ** 2, axis=1) for c in indices],
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-16)
            indices.append(rng.choice(n, p=probs))
        centres = observations[indices].copy()

        for _ in range(max_iter):
            # Assignment
            dists = np.stack(
                [np.sum((observations - c) ** 2, axis=1) for c in centres],
                axis=1,
            )
            labels = np.argmin(dists, axis=1)
            # Update
            new_centres = np.empty_like(centres)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centres[j] = observations[mask].mean(axis=0)
                else:
                    new_centres[j] = centres[j]
            if np.allclose(centres, new_centres, atol=1e-8):
                break
            centres = new_centres

        self.centres = centres

    @property
    def fitted(self) -> bool:
        return self.centres is not None

    def discretize(self, obs: np.ndarray) -> int:
        """Map a continuous observation to the nearest cluster index."""
        if self.centres is None:
            raise RuntimeError(
                "KMeansDiscretizer has not been fitted.  "
                "Call fit() with sampled observations first."
            )
        obs = np.asarray(obs, dtype=np.float64).flatten()
        dists = np.sum((self.centres - obs) ** 2, axis=1)
        return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class NeurogymAdapter:
    """Bridge between a NeuroGym environment and ALF's discrete-POMDP agents.

    Wraps a NeuroGym Gymnasium environment, discretises its continuous
    observations, and exposes the simple (reset, step) interface that
    ALF agents expect (matching TMazeEnv).

    Args:
        env: A NeuroGym (Gymnasium-compatible) environment instance.
        discretization: Strategy name -- ``"bin"`` or ``"kmeans"``.
        n_bins: Number of bins per dimension (for ``"bin"`` strategy).
        n_clusters: Number of clusters (for ``"kmeans"`` strategy).
        obs_low: Override for observation lower bounds.  If *None*,
            inferred from ``env.observation_space.low``.
        obs_high: Override for observation upper bounds.  If *None*,
            inferred from ``env.observation_space.high``.
        seed: Random seed for the adapter's internal RNG.

    Attributes:
        num_obs: Total number of discrete observation categories.
        num_actions: Number of discrete actions.
        current_phase: Current trial phase name (if detectable).
    """

    def __init__(
        self,
        env,
        discretization: str = "bin",
        n_bins: int = 5,
        n_clusters: int = 20,
        obs_low: Optional[np.ndarray] = None,
        obs_high: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        _require_neurogym()
        self.env = env
        self.rng = np.random.RandomState(seed)
        self._seed = seed
        self._discretization_name = discretization

        # Determine observation and action dimensionality.
        obs_space = env.observation_space
        self.obs_dim: int = int(np.prod(obs_space.shape)) if obs_space.shape else 1
        self.num_actions: int = env.action_space.n

        # Observation bounds -- clamp infinities to practical range.
        low = obs_low if obs_low is not None else np.asarray(obs_space.low).flatten()
        high = obs_high if obs_high is not None else np.asarray(obs_space.high).flatten()
        low = np.where(np.isfinite(low), low, -10.0)
        high = np.where(np.isfinite(high), high, 10.0)

        # Build discretizer.
        if discretization == "bin":
            self._discretizer = BinDiscretizer(n_bins, low, high, self.obs_dim)
        elif discretization == "kmeans":
            self._discretizer = KMeansDiscretizer(n_clusters)
        else:
            raise ValueError(
                f"Unknown discretization strategy '{discretization}'. "
                f"Choose 'bin' or 'kmeans'."
            )

        self.num_obs: int = self._discretizer.n_obs

        # Trial-phase tracking.
        self._current_phase: str = "unknown"
        self._timestep: int = 0

    # -- Properties ---------------------------------------------------------

    @property
    def current_phase(self) -> str:
        """Current trial phase, inferred from env info or timestep."""
        return self._current_phase

    # -- Gymnasium-like interface matching TMazeEnv -------------------------

    def reset(self) -> int:
        """Reset the environment and return the initial discrete observation.

        Returns:
            Integer observation index.
        """
        result = self.env.reset()
        # Gymnasium returns (obs, info); older API returns just obs.
        if isinstance(result, tuple):
            obs_cont, info = result
        else:
            obs_cont = result
            info = {}
        self._timestep = 0
        self._update_phase(info)
        return self._discretize(obs_cont)

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action and return (discrete_obs, reward, done).

        Args:
            action: Integer action index.

        Returns:
            Tuple of (obs_index, reward, done).
        """
        result = self.env.step(int(action))
        # Gymnasium: (obs, reward, terminated, truncated, info)
        # Older gym: (obs, reward, done, info)
        if len(result) == 5:
            obs_cont, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs_cont, reward, done, info = result
        else:
            raise ValueError(f"Unexpected step() return length: {len(result)}")

        self._timestep += 1
        self._update_phase(info)
        return self._discretize(obs_cont), float(reward), bool(done)

    # -- KMeans fitting -----------------------------------------------------

    def fit(
        self,
        n_samples: int = 5000,
        max_iter: int = 100,
    ) -> None:
        """Fit the KMeans discretizer by sampling random trajectories.

        Only needed (and only has effect) when ``discretization="kmeans"``.
        For the ``"bin"`` strategy this is a no-op.

        Args:
            n_samples: Number of observation samples to collect.
            max_iter: KMeans iterations.
        """
        if not isinstance(self._discretizer, KMeansDiscretizer):
            return  # bin discretizer needs no fitting

        observations = self._collect_observations(n_samples)
        self._discretizer.fit(observations, max_iter=max_iter)
        self.num_obs = self._discretizer.n_obs

    # -- Generative-model construction --------------------------------------

    def build_generative_model(
        self,
        n_episodes: int = 100,
        num_states: Optional[int] = None,
        T: int = 1,
    ) -> GenerativeModel:
        """Build an ALF GenerativeModel from interaction statistics.

        Runs the environment with random actions to estimate the A
        (likelihood), B (transition), C (preference), and D (prior)
        matrices.

        For the ``"kmeans"`` strategy, if the discretizer has not been
        fitted yet this method will fit it first using the same rollouts.

        Args:
            n_episodes: Number of episodes of random interaction.
            num_states: Number of hidden states for the model.  If *None*,
                defaults to ``self.num_obs`` (i.e. one hidden state per
                discrete observation -- fully observable assumption).
            T: Planning horizon for the GenerativeModel.

        Returns:
            An ``alf.GenerativeModel`` ready for use with ``AnalyticAgent``.
        """
        n_states = num_states if num_states is not None else self.num_obs
        n_obs = self.num_obs
        n_act = self.num_actions

        # ------------------------------------------------------------------
        # Collect interaction data
        # ------------------------------------------------------------------
        obs_list: list[int] = []
        transitions: list[tuple[int, int, int]] = []  # (obs, action, next_obs)
        obs_rewards: dict[int, list[float]] = {}

        # If kmeans and not fitted, collect raw observations for fitting.
        raw_obs_for_fit: list[np.ndarray] = []
        need_fit = isinstance(self._discretizer, KMeansDiscretizer) and not self._discretizer.fitted

        for _ in range(n_episodes):
            result = self.env.reset()
            if isinstance(result, tuple):
                obs_cont, _ = result
            else:
                obs_cont = result

            if need_fit:
                raw_obs_for_fit.append(np.asarray(obs_cont).flatten())

            prev_disc: Optional[int] = None
            prev_action: Optional[int] = None

            for _step_idx in range(500):  # upper bound per episode
                action = self.rng.randint(n_act)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs_cont, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs_cont, reward, done, _ = step_result

                if need_fit:
                    raw_obs_for_fit.append(np.asarray(obs_cont).flatten())

                # We delay discretization until after potential fit.
                if not need_fit:
                    disc = self._discretizer.discretize(obs_cont)
                    obs_list.append(disc)
                    obs_rewards.setdefault(disc, []).append(reward)
                    if prev_disc is not None and prev_action is not None:
                        transitions.append((prev_disc, prev_action, disc))
                    prev_disc = disc
                    prev_action = action

                if done:
                    break

        # If we needed to fit, do so now, then re-discretize.
        if need_fit and raw_obs_for_fit:
            self._discretizer.fit(np.array(raw_obs_for_fit))
            self.num_obs = self._discretizer.n_obs
            n_obs = self.num_obs
            if num_states is None:
                n_states = n_obs

            # Re-discretize all raw observations into transitions.
            # We need a second pass since we only stored raw obs.
            obs_list, transitions, obs_rewards = self._rediscretize_pass(
                n_episodes
            )

        # ------------------------------------------------------------------
        # Estimate matrices
        # ------------------------------------------------------------------

        # A matrix: P(o | s).  Under the fully-observable assumption
        # (state = obs), A is identity.  If num_states != num_obs we
        # estimate from co-occurrence counts.
        if n_states == n_obs:
            A = np.eye(n_obs)
        else:
            A = np.ones((n_obs, n_states)) / n_obs  # Laplace prior
            for obs_idx in obs_list:
                # Map observation to closest hidden state.
                state_idx = obs_idx % n_states
                A[obs_idx, state_idx] += 1.0
            A = A / A.sum(axis=0, keepdims=True)

        # B matrix: P(s' | s, a).
        B = np.ones((n_states, n_states, n_act)) / n_states  # Laplace prior
        for obs_prev, act, obs_next in transitions:
            s = obs_prev % n_states
            s_next = obs_next % n_states
            B[s_next, s, act] += 1.0
        for a in range(n_act):
            col_sums = B[:, :, a].sum(axis=0)
            col_sums[col_sums == 0] = 1.0
            B[:, :, a] /= col_sums

        # C vector: preference over observations derived from reward.
        C = np.zeros(n_obs)
        for obs_idx, rewards in obs_rewards.items():
            if obs_idx < n_obs:
                mean_r = np.mean(rewards)
                if mean_r > 0:
                    C[obs_idx] = 1.0 + np.log1p(abs(mean_r))
                elif mean_r < 0:
                    C[obs_idx] = -(1.0 + np.log1p(abs(mean_r)))

        # D: uniform prior over states.
        D = np.ones(n_states) / n_states

        return GenerativeModel(A=[A], B=[B], C=[C], D=[D], T=T)

    # -- Internal helpers ---------------------------------------------------

    def _discretize(self, obs_cont: np.ndarray) -> int:
        """Discretize a raw continuous observation."""
        return self._discretizer.discretize(obs_cont)

    def _update_phase(self, info: dict) -> None:
        """Infer the current trial phase from the env info dict."""
        if not info:
            self._current_phase = "unknown"
            return

        # NeuroGym tasks commonly expose the period in info.
        if "new_trial_info" in info:
            nti = info["new_trial_info"]
            if isinstance(nti, dict) and "periods" in nti:
                self._current_phase = str(nti["periods"][-1])
                return

        # Some tasks expose 'period' directly.
        if "period" in info:
            self._current_phase = str(info["period"])
            return

        # Fallback: use timestep-based heuristic for typical 3-period tasks.
        dt = getattr(self.env, "dt", None)
        timing = getattr(self.env, "timing", None)
        if timing is not None and isinstance(timing, dict) and dt is not None:
            elapsed = self._timestep * dt
            cumulative = 0
            for phase_name, duration in timing.items():
                cumulative += duration
                if elapsed < cumulative:
                    self._current_phase = str(phase_name)
                    return
            self._current_phase = list(timing.keys())[-1]
            return

        self._current_phase = "unknown"

    def _collect_observations(self, n_samples: int) -> np.ndarray:
        """Collect raw continuous observations via random rollouts."""
        observations: list[np.ndarray] = []
        while len(observations) < n_samples:
            result = self.env.reset()
            obs_cont = result[0] if isinstance(result, tuple) else result
            observations.append(np.asarray(obs_cont).flatten())
            for _ in range(500):
                action = self.rng.randint(self.num_actions)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs_cont, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs_cont, _, done, _ = step_result
                observations.append(np.asarray(obs_cont).flatten())
                if done or len(observations) >= n_samples:
                    break
        return np.array(observations[:n_samples])

    def _rediscretize_pass(
        self, n_episodes: int
    ) -> tuple[list[int], list[tuple[int, int, int]], dict[int, list[float]]]:
        """Re-run the environment and discretize with the now-fitted discretizer.

        Used when KMeans discretizer was fitted during build_generative_model.
        """
        obs_list: list[int] = []
        transitions: list[tuple[int, int, int]] = []
        obs_rewards: dict[int, list[float]] = {}

        for _ in range(n_episodes):
            result = self.env.reset()
            obs_cont = result[0] if isinstance(result, tuple) else result
            prev_disc = self._discretizer.discretize(obs_cont)
            obs_list.append(prev_disc)

            prev_action: Optional[int] = None

            for _ in range(500):
                action = self.rng.randint(self.num_actions)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs_cont, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs_cont, reward, done, _ = step_result

                disc = self._discretizer.discretize(obs_cont)
                obs_list.append(disc)
                obs_rewards.setdefault(disc, []).append(reward)
                if prev_action is not None:
                    transitions.append((prev_disc, prev_action, disc))
                prev_disc = disc
                prev_action = action

                if done:
                    break

        return obs_list, transitions, obs_rewards
