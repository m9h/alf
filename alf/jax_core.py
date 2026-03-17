"""Shared JAX numerical primitives.

Small, foundational functions used across ALF and downstream JAX projects
(e.g., RatInABox JAX backend). Keeping them in one place ensures consistent
epsilon defaults and avoids subtle numerical divergence.

All functions are pure, jit-compatible, and vmap-friendly.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)


def safe_log(x, eps=1e-16):
    """Logarithm with clamped input to avoid -inf."""
    return jnp.log(jnp.clip(x, eps))


def safe_normalize(x, axis=-1, eps=1e-12):
    """Normalize array to sum to 1 along *axis*, with zero-division protection."""
    s = jnp.sum(x, axis=axis, keepdims=True)
    return x / jnp.maximum(s, eps)


def safe_divide(num, denom, eps=1e-12):
    """Element-wise division with zero-denominator protection."""
    safe_d = jnp.where(jnp.abs(denom) < eps, 1.0, denom)
    return jnp.where(jnp.abs(denom) < eps, 0.0, num / safe_d)


def entropy(p, axis=-1):
    """Shannon entropy H(p) = -sum(p * log(p)), with zero protection."""
    return -jnp.sum(p * safe_log(p), axis=axis)


def log_normalize(log_vals, axis=-1):
    """Normalize in log-space: log_vals - logsumexp(log_vals)."""
    return log_vals - logsumexp(log_vals, axis=axis, keepdims=True)
