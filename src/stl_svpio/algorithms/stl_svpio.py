from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController, MPPIState, make_stl_cost_fn

SamplingDistribution = Literal["gaussian", "uniform", "truncated_gaussian", "squashed_gaussian"]
AnnealingSchedule = Literal["none", "linear", "exp", "cosine"]
ParticleSelection = Literal["mean", "best", "weighted_mean"]
GradientMode = Literal["reverse", "forward"]

STLSVPIOState = MPPIState


@dataclass
class STLSVPIOConfig:
    """Paper-facing configuration for STL-SVPIO.

    STL-SVPIO transports a population of bounded control-sequence particles
    using exact gradients of negative STL robustness and an RBF Stein kernel.
    """

    horizon: int
    num_particles: int
    control_dim: int
    path_integral_temperature: float = 1.0
    sampling_distribution: SamplingDistribution = "uniform"
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    control_noise_sigma: Optional[jnp.ndarray] = None
    clip_controls: bool = True
    stein_step_size: float = 0.1
    num_stein_steps: int = 10
    stein_kernel_bandwidth: Optional[float] = None
    stein_step_size_anneal: AnnealingSchedule = "none"
    stein_step_size_final: Optional[float] = None
    stein_repulsion_coef: float = 1.0
    stein_repulsion_anneal: AnnealingSchedule = "none"
    stein_repulsion_final: Optional[float] = None
    particle_selection: ParticleSelection = "best"
    particle_selection_temperature: Optional[float] = None
    resample_enabled: bool = False
    resample_ess_threshold: float = 0.5
    resample_temperature: Optional[float] = None
    resample_jitter_scale: float = 0.0
    gradient_mode: GradientMode = "reverse"
    record_optimization_history: bool = False
    show_progress: bool = False

    def to_legacy_config(self) -> MPPIConfig:
        return MPPIConfig(
            horizon=self.horizon,
            num_samples=self.num_particles,
            control_dim=self.control_dim,
            temperature=self.path_integral_temperature,
            sampling_mode=self.sampling_distribution,
            update_mode="svgd",
            control_low=self.control_low,
            control_high=self.control_high,
            control_noise_sigma=self.control_noise_sigma,
            clip_controls=self.clip_controls,
            svgd_step_size=self.stein_step_size,
            svgd_iters=self.num_stein_steps,
            svgd_kernel_bandwidth=self.stein_kernel_bandwidth,
            svgd_step_size_anneal=self.stein_step_size_anneal,
            svgd_step_size_final=self.stein_step_size_final,
            svgd_repulsion_coef=self.stein_repulsion_coef,
            svgd_repulsion_anneal=self.stein_repulsion_anneal,
            svgd_repulsion_final=self.stein_repulsion_final,
            svgd_selection_mode=self.particle_selection,
            svgd_selection_temperature=self.particle_selection_temperature,
            svgd_resample_enabled=self.resample_enabled,
            svgd_resample_ess_threshold=self.resample_ess_threshold,
            svgd_resample_temperature=self.resample_temperature,
            svgd_resample_jitter_scale=self.resample_jitter_scale,
            svgd_grad_mode=self.gradient_mode,
            record_svgd_history=self.record_optimization_history,
            show_svgd_progress=self.show_progress,
        )


def make_negative_robustness_cost(
    stl_formula,
    approx_method: str = "true",
    temperature: Optional[float] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return J_phi(u) = -rho_phi(x_0:H(u)) for batched trajectories."""

    return make_stl_cost_fn(stl_formula, approx_method=approx_method, temperature=temperature)


class STLSVPIOOptimizer:
    """Exact-gradient Stein variational optimizer for STL robustness."""

    def __init__(
        self,
        config: STLSVPIOConfig,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
        rollout_batch_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    ) -> None:
        self.config = config
        self._controller = MPPIController(
            config=config.to_legacy_config(),
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            rollout_batch_fn=rollout_batch_fn,
        )

    @staticmethod
    def pointmass2d_dynamics(dt: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        return MPPIController.pointmass2d_dynamics(dt)

    @staticmethod
    def pointmass2d_multiagent_dynamics(
        num_agents: int,
        dt: float,
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        return MPPIController.pointmass2d_multiagent_dynamics(num_agents=num_agents, dt=dt)

    def init_state(
        self,
        key: jax.Array,
        mean_controls: Optional[jnp.ndarray] = None,
    ) -> STLSVPIOState:
        return self._controller.init_state(key, mean_controls=mean_controls)

    def optimize(
        self,
        state: STLSVPIOState,
        initial_state: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, STLSVPIOState, dict]:
        """Optimize and return the first action, warm-started state, and diagnostics."""

        return self._controller.command(state, initial_state)

    command = optimize

