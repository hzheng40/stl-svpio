from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController


@dataclass
class DPIConfig:
    horizon: int
    num_particles: int
    control_dim: int
    path_integral_temperature: float = 1.0
    sampling_distribution: str = "gaussian"
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    control_noise_sigma: Optional[jnp.ndarray] = None
    num_update_steps: int = 10
    shrink_factor: float = 0.95
    min_temperature: float = 1e-3
    augmented_mode: str = "implicit"


class DPIOptimizer:
    """Deterministic path integral baseline."""

    def __init__(
        self,
        config: DPIConfig,
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
                update_mode="deterministic_pi",
                control_low=config.control_low,
                control_high=config.control_high,
                control_noise_sigma=config.control_noise_sigma,
                svgd_iters=config.num_update_steps,
                dpi_iters=config.num_update_steps,
                dpi_shrink_factor=config.shrink_factor,
                dpi_min_temperature=config.min_temperature,
                dpi_augmented_mode=config.augmented_mode,
            ),
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
        )

    def init_state(self, key: jax.Array):
        return self._controller.init_state(key)

    def optimize(self, state, initial_state: jnp.ndarray):
        return self._controller.command(state, initial_state)

    command = optimize
