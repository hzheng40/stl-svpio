from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController


@dataclass
class SVMPCConfig:
    horizon: int
    num_particles: int
    control_dim: int
    path_integral_temperature: float = 1.0
    sampling_distribution: str = "gaussian"
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    control_noise_sigma: Optional[jnp.ndarray] = None
    num_stein_steps: int = 10
    stein_step_size: float = 0.1
    finite_difference_dirs: int = 4
    finite_difference_delta: float = 1e-2
    finite_difference_alpha: Optional[float] = None
    use_kernel_repulsion: bool = True


class SVMPCOptimizer:
    """Finite-difference Stein MPC baseline."""

    def __init__(
        self,
        config: SVMPCConfig,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self.config = config
        self._controller = MPPIController(
            MPPIConfig(
                horizon=config.horizon,
                num_samples=config.num_particles,
                control_dim=config.control_dim,
                temperature=config.path_integral_temperature,
                sampling_mode=config.sampling_distribution,
                update_mode="svgd_fd",
                control_low=config.control_low,
                control_high=config.control_high,
                control_noise_sigma=config.control_noise_sigma,
                svgd_iters=config.num_stein_steps,
                svgd_step_size=config.stein_step_size,
                svgd_fd_dirs=config.finite_difference_dirs,
                svgd_fd_delta=config.finite_difference_delta,
                svgd_fd_alpha=config.finite_difference_alpha,
                svgd_fd_use_kernel_repulsion=config.use_kernel_repulsion,
            ),
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
        )

    def init_state(self, key: jax.Array):
        return self._controller.init_state(key)

    def optimize(self, state, initial_state: jnp.ndarray):
        return self._controller.command(state, initial_state)

    command = optimize
