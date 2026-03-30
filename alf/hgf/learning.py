"""Differentiable parameter learning for HGF models.

Because all HGF computations are JAX-native, we can differentiate through
the full forward pass (surprise computation) with jax.grad and optimize
parameters via gradient descent. This is the same approach as alf.learning
for A/B matrices.

References:
    Mathys, Lomakina, Daunizeau et al. (2014). Uncertainty in perception and
        the Hierarchical Gaussian Filter. Frontiers in Human Neuroscience.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from alf.hgf.updates import (
    BinaryHGFParams,
    ContinuousHGFParams,
    binary_hgf_surprise,
    continuous_hgf_surprise,
)


# ---------------------------------------------------------------------------
# Learning results
# ---------------------------------------------------------------------------

class BinaryHGFLearningResult(NamedTuple):
    """Result of binary HGF parameter learning.

    Attributes:
        params: Learned HGF parameters.
        loss_history: NLL at each epoch.
    """
    params: BinaryHGFParams
    loss_history: list[float]


class ContinuousHGFLearningResult(NamedTuple):
    """Result of continuous HGF parameter learning.

    Attributes:
        params: Learned HGF parameters.
        loss_history: NLL at each epoch.
    """
    params: ContinuousHGFParams
    loss_history: list[float]


# ---------------------------------------------------------------------------
# Differentiable NLL wrappers
# ---------------------------------------------------------------------------

def binary_hgf_nll(
    omega_2: jnp.ndarray,
    observations: jnp.ndarray,
    mu_2_0: jnp.ndarray,
    sigma_2_0: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable NLL for binary HGF w.r.t. omega_2.

    Args:
        omega_2: Tonic log-volatility (the parameter to optimize).
        observations: Binary observation sequence, shape (T,).
        mu_2_0: Initial mean of level 2.
        sigma_2_0: Initial variance of level 2.

    Returns:
        Total negative log-likelihood (scalar).
    """
    params = BinaryHGFParams(
        omega_2=omega_2,
        mu_2_0=mu_2_0,
        sigma_2_0=sigma_2_0,
    )
    return binary_hgf_surprise(observations, params)


def continuous_hgf_nll(
    learnable: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                     jnp.ndarray, jnp.ndarray],
    observations: jnp.ndarray,
    fixed: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                 jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Differentiable NLL for continuous HGF w.r.t. learnable parameters.

    Separates learnable parameters (volatilities, couplings) from fixed
    ones (initial conditions, input precision) for clean gradient computation.

    Args:
        learnable: Tuple of (omega_1, omega_2, kappa_1, kappa_2, theta).
        observations: Continuous observation sequence, shape (T,).
        fixed: Tuple of (pi_u, mu_1_0, sigma_1_0, mu_2_0, sigma_2_0,
               mu_3_0, sigma_3_0).

    Returns:
        Total negative log-likelihood (scalar).
    """
    omega_1, omega_2, kappa_1, kappa_2, theta = learnable
    pi_u, mu_1_0, sigma_1_0, mu_2_0, sigma_2_0, mu_3_0, sigma_3_0 = fixed

    params = ContinuousHGFParams(
        omega_1=omega_1, omega_2=omega_2,
        kappa_1=kappa_1, kappa_2=kappa_2,
        theta=theta, pi_u=pi_u,
        mu_1_0=mu_1_0, sigma_1_0=sigma_1_0,
        mu_2_0=mu_2_0, sigma_2_0=sigma_2_0,
        mu_3_0=mu_3_0, sigma_3_0=sigma_3_0,
    )
    return continuous_hgf_surprise(observations, params)


# ---------------------------------------------------------------------------
# Gradient-based learning
# ---------------------------------------------------------------------------

def _try_import_optax():
    """Try to import optax, return None if unavailable."""
    try:
        import optax
        return optax
    except ImportError:
        return None


def learn_binary_hgf(
    observations: np.ndarray,
    mu_2_0: float = 0.0,
    sigma_2_0: float = 1.0,
    init_omega_2: float = -2.0,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> BinaryHGFLearningResult:
    """Learn omega_2 for binary HGF via gradient descent.

    Args:
        observations: Binary observation sequence, shape (T,).
        mu_2_0: Initial mean of level 2.
        sigma_2_0: Initial variance of level 2.
        init_omega_2: Starting value for omega_2 optimization.
        num_epochs: Number of gradient descent steps.
        lr: Learning rate.
        verbose: If True, print loss every 10 epochs.

    Returns:
        BinaryHGFLearningResult with learned parameters and loss history.
    """
    obs_jnp = jnp.array(observations, dtype=jnp.float32)
    mu_2_0_jnp = jnp.array(mu_2_0)
    sigma_2_0_jnp = jnp.array(sigma_2_0)
    omega_2 = jnp.array(init_omega_2)

    grad_fn = jax.grad(binary_hgf_nll, argnums=0)

    optax = _try_import_optax()
    loss_history = []

    if optax is not None:
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(omega_2)

        @jax.jit
        def update_step(omega_2, opt_state):
            loss = binary_hgf_nll(omega_2, obs_jnp, mu_2_0_jnp, sigma_2_0_jnp)
            grad = grad_fn(omega_2, obs_jnp, mu_2_0_jnp, sigma_2_0_jnp)
            updates, new_state = optimizer.update(grad, opt_state, omega_2)
            new_omega_2 = omega_2 + updates
            return new_omega_2, new_state, loss

        for epoch in range(num_epochs):
            omega_2, opt_state, loss = update_step(omega_2, opt_state)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")
    else:
        @jax.jit
        def sgd_step(omega_2):
            loss = binary_hgf_nll(omega_2, obs_jnp, mu_2_0_jnp, sigma_2_0_jnp)
            grad = grad_fn(omega_2, obs_jnp, mu_2_0_jnp, sigma_2_0_jnp)
            new_omega_2 = omega_2 - lr * grad
            return new_omega_2, loss

        for epoch in range(num_epochs):
            omega_2, loss = sgd_step(omega_2)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    learned_params = BinaryHGFParams(
        omega_2=omega_2,
        mu_2_0=mu_2_0_jnp,
        sigma_2_0=sigma_2_0_jnp,
    )
    return BinaryHGFLearningResult(params=learned_params, loss_history=loss_history)


def learn_continuous_hgf(
    observations: np.ndarray,
    pi_u: float = 100.0,
    mu_1_0: float = 0.0,
    sigma_1_0: float = 1.0,
    mu_2_0: float = 0.0,
    sigma_2_0: float = 1.0,
    mu_3_0: float = 0.0,
    sigma_3_0: float = 1.0,
    init_omega_1: float = -3.0,
    init_omega_2: float = -3.0,
    init_kappa_1: float = 1.0,
    init_kappa_2: float = 1.0,
    init_theta: float = 0.01,
    num_epochs: int = 100,
    lr: float = 0.001,
    verbose: bool = False,
) -> ContinuousHGFLearningResult:
    """Learn volatility parameters for continuous HGF via gradient descent.

    Optimizes omega_1, omega_2, kappa_1, kappa_2, theta while keeping
    initial conditions and input precision fixed.

    Args:
        observations: Continuous observation sequence, shape (T,).
        pi_u: Input precision (fixed).
        mu_1_0, sigma_1_0: Initial conditions for level 1 (fixed).
        mu_2_0, sigma_2_0: Initial conditions for level 2 (fixed).
        mu_3_0, sigma_3_0: Initial conditions for level 3 (fixed).
        init_omega_1, init_omega_2: Starting log-volatilities.
        init_kappa_1, init_kappa_2: Starting coupling strengths.
        init_theta: Starting meta-volatility.
        num_epochs: Number of gradient steps.
        lr: Learning rate.
        verbose: If True, print progress.

    Returns:
        ContinuousHGFLearningResult with learned parameters and loss history.
    """
    obs_jnp = jnp.array(observations, dtype=jnp.float32)

    learnable = (
        jnp.array(init_omega_1),
        jnp.array(init_omega_2),
        jnp.array(init_kappa_1),
        jnp.array(init_kappa_2),
        jnp.array(init_theta),
    )
    fixed = (
        jnp.array(pi_u),
        jnp.array(mu_1_0), jnp.array(sigma_1_0),
        jnp.array(mu_2_0), jnp.array(sigma_2_0),
        jnp.array(mu_3_0), jnp.array(sigma_3_0),
    )

    grad_fn = jax.grad(continuous_hgf_nll, argnums=0)

    optax = _try_import_optax()
    loss_history = []

    if optax is not None:
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(learnable)

        @jax.jit
        def update_step(learnable, opt_state):
            loss = continuous_hgf_nll(learnable, obs_jnp, fixed)
            grads = grad_fn(learnable, obs_jnp, fixed)
            updates, new_state = optimizer.update(grads, opt_state, learnable)
            new_learnable = jax.tree.map(
                lambda p, u: p + u, learnable, updates
            )
            return new_learnable, new_state, loss

        for epoch in range(num_epochs):
            learnable, opt_state, loss = update_step(learnable, opt_state)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")
    else:
        @jax.jit
        def sgd_step(learnable):
            loss = continuous_hgf_nll(learnable, obs_jnp, fixed)
            grads = grad_fn(learnable, obs_jnp, fixed)
            new_learnable = jax.tree.map(
                lambda p, g: p - lr * g, learnable, grads
            )
            return new_learnable, loss

        for epoch in range(num_epochs):
            learnable, loss = sgd_step(learnable)
            loss_val = float(loss)
            loss_history.append(loss_val)
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: NLL = {loss_val:.4f}")

    omega_1, omega_2, kappa_1, kappa_2, theta = learnable
    learned_params = ContinuousHGFParams(
        omega_1=omega_1, omega_2=omega_2,
        kappa_1=kappa_1, kappa_2=kappa_2,
        theta=theta, pi_u=fixed[0],
        mu_1_0=fixed[1], sigma_1_0=fixed[2],
        mu_2_0=fixed[3], sigma_2_0=fixed[4],
        mu_3_0=fixed[5], sigma_3_0=fixed[6],
    )
    return ContinuousHGFLearningResult(
        params=learned_params, loss_history=loss_history
    )
