from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController


def make_reach_avoid_heuristic_cost(
    goal_center, obstacle_centers, obstacle_radii, *,
    stage_goal_weight=0.1, stage_obstacle_weight=25.0,
    terminal_goal_weight=8.0, terminal_obstacle_weight=100.0, margin=0.0,
):
    """Stage and terminal distance costs from the original Table I comparison."""
    def single_cost(trace):
        positions = trace[:, :2]
        goal_distance = jnp.linalg.norm(positions - goal_center, axis=-1)
        distance = jnp.linalg.norm(positions[:, None, :] - obstacle_centers, axis=-1)
        violation = jnp.sum(jnp.maximum(obstacle_radii + margin - distance, 0.0), axis=-1)
        return (
            jnp.sum(stage_goal_weight * goal_distance + stage_obstacle_weight * violation)
            + terminal_goal_weight * goal_distance[-1]
            + terminal_obstacle_weight * violation[-1]
        )

    return jax.vmap(single_cost)


@dataclass
class MPPIBaselineConfig:
    horizon: int
    num_particles: int
    control_dim: int
    path_integral_temperature: float = 1.0
    sampling_distribution: str = "gaussian"
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    control_noise_sigma: Optional[jnp.ndarray] = None
    num_update_steps: int = 1


class MPPIBaselineOptimizer:
    """Importance-sampling MPPI baseline used in the reach-avoid comparison."""

    def __init__(
        self,
        config: MPPIBaselineConfig,
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
                update_mode="importance_sampling",
                control_low=config.control_low,
                control_high=config.control_high,
                control_noise_sigma=config.control_noise_sigma,
                svgd_iters=config.num_update_steps,
            ),
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
        )

    def init_state(self, key: jax.Array):
        return self._controller.init_state(key)

    def optimize(self, state, initial_state: jnp.ndarray):
        return self._controller.command(state, initial_state)

    command = optimize
