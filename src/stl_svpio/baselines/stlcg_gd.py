from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp


@dataclass
class STLCGGradientDescentConfig:
    num_steps: int
    step_size: float
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    grad_clip_norm: Optional[float] = None


def _clip_controls(config: STLCGGradientDescentConfig, controls: jnp.ndarray) -> jnp.ndarray:
    if config.control_low is None or config.control_high is None:
        return controls
    return jnp.clip(controls, config.control_low, config.control_high)


def run_stlcg_gradient_descent(
    config: STLCGGradientDescentConfig,
    initial_controls: jnp.ndarray,
    initial_state: jnp.ndarray,
    dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    stl_formula,
    approx_method: str = "true",
    temperature: Optional[float] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Direct gradient descent on J_phi = -rho_phi, the STLCG++-style baseline."""

    def rollout(u_seq: jnp.ndarray) -> jnp.ndarray:
        def step_fn(x, u):
            x_next = dynamics_fn(x, u)
            return x_next, x_next

        _, trace = jax.lax.scan(step_fn, initial_state, u_seq)
        return trace

    def loss_fn(u_seq: jnp.ndarray):
        trace = rollout(u_seq)
        robustness = stl_formula.robustness(trace, approx_method=approx_method, temperature=temperature)
        robustness = jnp.nan_to_num(robustness, nan=-1e6, posinf=1e6, neginf=-1e6)
        return -robustness, trace

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def step(u_curr, _):
        (loss, trace), grad = value_and_grad(u_curr)
        if config.grad_clip_norm is not None:
            norm = jnp.linalg.norm(grad.reshape((-1,)))
            scale = jnp.minimum(1.0, config.grad_clip_norm / (norm + 1e-8))
            grad = grad * scale
        u_next = _clip_controls(config, u_curr - config.step_size * grad)
        return u_next, (trace, loss)

    iter_ids = jnp.arange(config.num_steps, dtype=jnp.int32)
    final_controls, (trace_history, loss_history) = jax.lax.scan(step, initial_controls, iter_ids)
    return final_controls, trace_history, loss_history

