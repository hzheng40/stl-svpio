from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.scipy.special import erf, erfinv
from tqdm.auto import tqdm

from .envs import quadrotor_step

SamplingMode = Literal["gaussian", "uniform", "truncated_gaussian", "squashed_gaussian"]
UpdateMode = Literal["importance_sampling", "svgd", "deterministic_pi", "svgd_fd"]
AnnealMode = Literal["none", "linear", "exp", "cosine"]
SVGDSelectionMode = Literal["mean", "best", "weighted_mean"]
SVGDGradMode = Literal["reverse", "forward"]
DPIAugmentedMode = Literal["implicit", "explicit"]
SVGDFDNanFallback = Literal["zero_grad"]


@dataclass
class MPPIConfig:
    horizon: int
    num_samples: int
    control_dim: int
    temperature: float = 1.0
    sampling_mode: SamplingMode = "gaussian"
    update_mode: UpdateMode = "importance_sampling"
    control_low: Optional[jnp.ndarray] = None
    control_high: Optional[jnp.ndarray] = None
    control_noise_sigma: Optional[jnp.ndarray] = None
    clip_controls: bool = True
    svgd_step_size: float = 0.1
    svgd_iters: int = 10
    svgd_kernel_bandwidth: Optional[float] = None
    svgd_step_size_anneal: AnnealMode = "none"
    svgd_step_size_final: Optional[float] = None
    svgd_repulsion_coef: float = 1.0
    svgd_repulsion_anneal: AnnealMode = "none"
    svgd_repulsion_final: Optional[float] = None
    svgd_selection_mode: SVGDSelectionMode = "best"
    svgd_selection_temperature: Optional[float] = None
    svgd_resample_enabled: bool = False
    svgd_resample_ess_threshold: float = 0.5
    svgd_resample_temperature: Optional[float] = None
    svgd_resample_jitter_scale: float = 0.0
    svgd_grad_mode: SVGDGradMode = "reverse"
    record_svgd_history: bool = False
    show_svgd_progress: bool = False
    svgd_fd_dirs: int = 4
    svgd_fd_delta: float = 1e-2
    svgd_fd_alpha: Optional[float] = None
    svgd_fd_use_kernel_repulsion: bool = True
    svgd_fd_nan_fallback: SVGDFDNanFallback = "zero_grad"
    dpi_iters: Optional[int] = None
    dpi_shrink_factor: float = 0.95
    dpi_min_temperature: float = 1e-3
    dpi_augmented_mode: DPIAugmentedMode = "implicit"


_TQDM_BARS: dict[int, tqdm] = {}


def _tqdm_svgd_update_callback(
    iter_idx: jax.Array,
    total_iters: jax.Array,
    best_robustness: jax.Array,
) -> np.int32:
    i = int(np.asarray(iter_idx))
    total = int(np.asarray(total_iters))
    best_rob = float(np.asarray(best_robustness))
    bar = _TQDM_BARS.get(total)
    if bar is None:
        bar = tqdm(total=total, desc="SVGD iterations", leave=False)
        _TQDM_BARS[total] = bar

    target_n = i + 1
    delta = target_n - bar.n
    if delta > 0:
        bar.update(delta)
    bar.set_postfix_str(f"best STL rob={best_rob:.4f}", refresh=False)

    if target_n >= total:
        bar.close()
        _TQDM_BARS.pop(total, None)

    return np.int32(0)


@dataclass
@jax.tree_util.register_pytree_node_class
class MPPIState:
    key: jax.Array
    mean_controls: jnp.ndarray

    def tree_flatten(self):
        return (self.key, self.mean_controls), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        key, mean_controls = children
        return cls(key=key, mean_controls=mean_controls)


def pointmass2d_dynamics_step(state: jnp.ndarray, control: jnp.ndarray, dt: float = 0.1) -> jnp.ndarray:
    """Single-step point mass dynamics.

    state: [..., 4] = [x, y, vx, vy]
    control: [..., 2] = [ax, ay]
    """
    pos = state[..., :2]
    vel = state[..., 2:]
    pos_next = pos + dt * vel + 0.5 * (dt**2) * control
    vel_next = vel + dt * control
    return jnp.concatenate([pos_next, vel_next], axis=-1)


def pointmass2d_multiagent_dynamics_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    num_agents: int,
    dt: float = 0.1,
) -> jnp.ndarray:
    """Single-step joint dynamics for concatenated multi-agent point masses.

    state: [..., 4 * num_agents]
    control: [..., 2 * num_agents]
    """
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    expected_state_dim = 4 * num_agents
    expected_control_dim = 2 * num_agents
    if state.shape[-1] != expected_state_dim:
        raise ValueError(
            f"state must have trailing dim {expected_state_dim}, got {state.shape[-1]}"
        )
    if control.shape[-1] != expected_control_dim:
        raise ValueError(
            f"control must have trailing dim {expected_control_dim}, got {control.shape[-1]}"
        )

    batch_shape = state.shape[:-1]
    flat_state = state.reshape(batch_shape + (num_agents, 4))
    flat_control = control.reshape(batch_shape + (num_agents, 2))
    next_state = pointmass2d_dynamics_step(flat_state, flat_control, dt=dt)
    return next_state.reshape(batch_shape + (expected_state_dim,))


def make_stl_cost_fn(
    stl_formula,
    approx_method: str = "true",
    temperature: Optional[float] = None,
    large_number: Optional[float] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create batched MPPI cost from an STL formula.

    Minimizes negative robustness, i.e. maximizing STL robustness.

    trajectories: [num_samples, horizon, state_dim]
    returns costs: [num_samples]
    """

    def _single_cost(trace: jnp.ndarray) -> jnp.ndarray:
        kwargs = {} if large_number is None else {"large_number": large_number}
        return -stl_formula.robustness(
            trace,
            approx_method=approx_method,
            temperature=temperature,
            **kwargs,
        )

    return jax.vmap(_single_cost)


class MPPIController:
    """JAX-compatible MPPI controller with configurable sampling and update modes.

    Required callables:
    - dynamics_fn(state, control) -> next_state
      state: [..., state_dim], control: [..., control_dim]
    - cost_fn(trajectories) -> costs
      trajectories: [num_samples, horizon, state_dim], costs: [num_samples]
    Optional callable:
    - rollout_batch_fn(x0, controls) -> trajectories
      x0: planner state, controls: [num_samples, horizon, control_dim],
      trajectories: [num_samples, horizon, state_dim]
    """

    def __init__(
        self,
        config: MPPIConfig,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
        rollout_batch_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    ) -> None:
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")
        if config.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if config.control_dim <= 0:
            raise ValueError("control_dim must be positive")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if config.sampling_mode not in ("gaussian", "uniform", "truncated_gaussian", "squashed_gaussian"):
            raise ValueError(f"Unsupported sampling_mode={config.sampling_mode}")
        if config.update_mode not in ("importance_sampling", "svgd", "deterministic_pi", "svgd_fd"):
            raise ValueError(f"Unsupported update_mode={config.update_mode}")
        if config.svgd_selection_mode not in ("mean", "best", "weighted_mean"):
            raise ValueError(f"Unsupported svgd_selection_mode={config.svgd_selection_mode}")
        if config.svgd_grad_mode not in ("reverse", "forward"):
            raise ValueError(f"Unsupported svgd_grad_mode={config.svgd_grad_mode}")
        if config.svgd_repulsion_coef < 0:
            raise ValueError("svgd_repulsion_coef must be non-negative")
        if config.svgd_selection_temperature is not None and config.svgd_selection_temperature <= 0:
            raise ValueError("svgd_selection_temperature must be positive when provided")
        if config.svgd_resample_ess_threshold < 0:
            raise ValueError("svgd_resample_ess_threshold must be non-negative")
        if config.svgd_resample_temperature is not None and config.svgd_resample_temperature <= 0:
            raise ValueError("svgd_resample_temperature must be positive when provided")
        if config.svgd_resample_jitter_scale < 0:
            raise ValueError("svgd_resample_jitter_scale must be non-negative")
        if config.svgd_fd_dirs <= 0:
            raise ValueError("svgd_fd_dirs must be positive")
        if config.svgd_fd_delta <= 0:
            raise ValueError("svgd_fd_delta must be positive")
        if config.svgd_fd_alpha is not None and config.svgd_fd_alpha <= 0:
            raise ValueError("svgd_fd_alpha must be positive when provided")
        if config.svgd_fd_nan_fallback != "zero_grad":
            raise ValueError(f"Unsupported svgd_fd_nan_fallback={config.svgd_fd_nan_fallback}")
        if config.dpi_iters is not None and config.dpi_iters <= 0:
            raise ValueError("dpi_iters must be positive when provided")
        if config.dpi_shrink_factor <= 0 or config.dpi_shrink_factor > 1:
            raise ValueError("dpi_shrink_factor must be in (0, 1]")
        if config.dpi_min_temperature <= 0:
            raise ValueError("dpi_min_temperature must be positive")
        if config.dpi_augmented_mode not in ("implicit", "explicit"):
            raise ValueError(f"Unsupported dpi_augmented_mode={config.dpi_augmented_mode}")

        self.config = config
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.rollout_batch_fn = rollout_batch_fn

        if self.config.control_noise_sigma is None:
            self.control_noise_sigma = jnp.ones((self.config.control_dim,), dtype=jnp.float32)
        else:
            sigma = jnp.asarray(self.config.control_noise_sigma, dtype=jnp.float32)
            if sigma.shape != (self.config.control_dim,):
                raise ValueError(
                    f"control_noise_sigma must have shape ({self.config.control_dim},), got {sigma.shape}"
                )
            self.control_noise_sigma = sigma

        if (self.config.control_low is None) != (self.config.control_high is None):
            raise ValueError("control_low and control_high must be provided together")

        if self.config.control_low is not None:
            self.control_low = jnp.asarray(self.config.control_low, dtype=jnp.float32)
            self.control_high = jnp.asarray(self.config.control_high, dtype=jnp.float32)
            if self.control_low.shape != (self.config.control_dim,):
                raise ValueError(
                    f"control_low must have shape ({self.config.control_dim},), got {self.control_low.shape}"
                )
            if self.control_high.shape != (self.config.control_dim,):
                raise ValueError(
                    f"control_high must have shape ({self.config.control_dim},), got {self.control_high.shape}"
                )
            if bool(jnp.any(self.control_low >= self.control_high)):
                raise ValueError("control_low must be strictly less than control_high")
        else:
            self.control_low = None
            self.control_high = None

        if self.config.sampling_mode != "gaussian" and self.control_low is None:
            raise ValueError(
                "uniform, truncated_gaussian, and squashed_gaussian require control_low/control_high"
            )

    @staticmethod
    def quadrotor_dynamics(dt: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def _dyn(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
            return quadrotor_step(state[None, :], control[None, :], dt=dt)[0]

        return _dyn

    @staticmethod
    def pointmass2d_dynamics(dt: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def _dyn(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
            return pointmass2d_dynamics_step(state, control, dt=dt)

        return _dyn

    @staticmethod
    def pointmass2d_multiagent_dynamics(
        num_agents: int,
        dt: float,
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def _dyn(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
            return pointmass2d_multiagent_dynamics_step(
                state,
                control,
                num_agents=num_agents,
                dt=dt,
            )

        return _dyn

    def init_state(
        self,
        key: jax.Array,
        mean_controls: Optional[jnp.ndarray] = None,
    ) -> MPPIState:
        if mean_controls is None:
            mean_controls = jnp.zeros(
                (self.config.horizon, self.config.control_dim),
                dtype=jnp.float32,
            )
        else:
            mean_controls = jnp.asarray(mean_controls, dtype=jnp.float32)
            if mean_controls.shape != (self.config.horizon, self.config.control_dim):
                raise ValueError(
                    f"mean_controls must have shape ({self.config.horizon}, {self.config.control_dim}), got {mean_controls.shape}"
                )
            mean_controls = self._apply_control_limits(mean_controls)

        return MPPIState(key=key, mean_controls=mean_controls)

    def command(self, state: MPPIState, x0: jnp.ndarray) -> Tuple[jnp.ndarray, MPPIState, dict]:
        key, sample_key, svgd_key = jax.random.split(state.key, 3)

        if self.config.update_mode == "importance_sampling":
            (
                updated_mean,
                costs,
                weights,
                sampled_controls,
                trajectories,
                imp_hist_controls_pre,
                imp_hist_controls_post,
                imp_hist_trajs,
                imp_hist_costs,
                imp_hist_weights,
                imp_hist_selected_controls,
                imp_hist_selected_trajs,
            ) = self._importance_update_iterative(state.mean_controls, x0, sample_key)
            selected_trajectory = self._rollout_batch(x0, updated_mean[None, ...])[0]
            info = {
                "costs": costs,
                "weights": weights,
                "sampled_controls": sampled_controls,
                "trajectories": trajectories,
                "selected_controls": updated_mean,
                "selected_trajectory": selected_trajectory,
                "importance_iters": max(int(self.config.svgd_iters), 1),
            }
            if (
                imp_hist_controls_pre is not None
                and imp_hist_controls_post is not None
                and imp_hist_trajs is not None
                and imp_hist_costs is not None
                and imp_hist_weights is not None
                and imp_hist_selected_controls is not None
                and imp_hist_selected_trajs is not None
            ):
                info["importance_iter_sampled_controls_pre"] = imp_hist_controls_pre
                info["importance_iter_sampled_controls_post"] = imp_hist_controls_post
                info["importance_iter_trajectories"] = imp_hist_trajs
                info["importance_iter_costs"] = imp_hist_costs
                info["importance_iter_weights"] = imp_hist_weights
                info["importance_iter_selected_controls"] = imp_hist_selected_controls
                info["importance_iter_selected_trajectories"] = imp_hist_selected_trajs
        elif self.config.update_mode == "svgd":
            sampled_controls = self._sample_control_sequences(sample_key, state.mean_controls)
            (
                updated_particles,
                costs,
                svgd_hist_particles,
                svgd_hist_costs,
                svgd_hist_ess,
                svgd_hist_resampled,
            ) = self._svgd_update(
                sampled_controls,
                x0,
                svgd_key,
            )
            updated_mean = self._select_svgd_controls(updated_particles, costs)
            selected_trajectory = self._rollout_batch(x0, updated_mean[None, ...])[0]
            info = {
                "costs": costs,
                "sampled_controls": updated_particles,
                "trajectories": self._rollout_batch(x0, updated_particles),
                "selected_controls": updated_mean,
                "selected_trajectory": selected_trajectory,
            }
            if svgd_hist_particles is not None and svgd_hist_costs is not None:
                hist_trajs = jax.vmap(lambda p: self._rollout_batch(x0, p))(svgd_hist_particles)
                hist_selected_controls = jax.vmap(self._select_svgd_controls)(
                    svgd_hist_particles,
                    svgd_hist_costs,
                )
                hist_particles_pre = jnp.concatenate(
                    [sampled_controls[None, ...], svgd_hist_particles[:-1]],
                    axis=0,
                )
                hist_selected_trajs = jax.vmap(
                    lambda u: self._rollout_batch(x0, u[None, ...])[0]
                )(hist_selected_controls)
                info["svgd_iter_sampled_controls"] = svgd_hist_particles
                info["svgd_iter_sampled_controls_pre"] = hist_particles_pre
                info["svgd_iter_sampled_controls_post"] = svgd_hist_particles
                info["svgd_iter_costs"] = svgd_hist_costs
                info["svgd_iter_trajectories"] = hist_trajs
                info["svgd_iter_selected_controls"] = hist_selected_controls
                info["svgd_iter_selected_trajectories"] = hist_selected_trajs
                if svgd_hist_ess is not None and svgd_hist_resampled is not None:
                    info["svgd_iter_ess"] = svgd_hist_ess
                    info["svgd_iter_resampled"] = svgd_hist_resampled
        elif self.config.update_mode == "svgd_fd":
            sampled_controls = self._sample_control_sequences(sample_key, state.mean_controls)
            (
                updated_particles,
                costs,
                svgd_fd_hist_particles,
                svgd_fd_hist_costs,
            ) = self._svgd_fd_update(sampled_controls, x0, svgd_key)
            updated_mean = self._select_svgd_controls(updated_particles, costs)
            selected_trajectory = self._rollout_batch(x0, updated_mean[None, ...])[0]
            info = {
                "costs": costs,
                "sampled_controls": updated_particles,
                "trajectories": self._rollout_batch(x0, updated_particles),
                "selected_controls": updated_mean,
                "selected_trajectory": selected_trajectory,
                "svgd_fd_dirs": int(self.config.svgd_fd_dirs),
                "svgd_fd_delta": float(self.config.svgd_fd_delta),
            }
            if svgd_fd_hist_particles is not None and svgd_fd_hist_costs is not None:
                hist_trajs = jax.vmap(lambda p: self._rollout_batch(x0, p))(svgd_fd_hist_particles)
                hist_selected_controls = jax.vmap(self._select_svgd_controls)(
                    svgd_fd_hist_particles,
                    svgd_fd_hist_costs,
                )
                hist_particles_pre = jnp.concatenate(
                    [sampled_controls[None, ...], svgd_fd_hist_particles[:-1]],
                    axis=0,
                )
                hist_selected_trajs = jax.vmap(
                    lambda u: self._rollout_batch(x0, u[None, ...])[0]
                )(hist_selected_controls)
                info["svgd_iter_sampled_controls"] = svgd_fd_hist_particles
                info["svgd_iter_sampled_controls_pre"] = hist_particles_pre
                info["svgd_iter_sampled_controls_post"] = svgd_fd_hist_particles
                info["svgd_iter_costs"] = svgd_fd_hist_costs
                info["svgd_iter_trajectories"] = hist_trajs
                info["svgd_iter_selected_controls"] = hist_selected_controls
                info["svgd_iter_selected_trajectories"] = hist_selected_trajs
        else:
            (
                updated_mean,
                costs,
                sampled_controls,
                trajectories,
                dpi_hist_controls_pre,
                dpi_hist_controls_post,
                dpi_hist_trajs,
                dpi_hist_costs,
                dpi_hist_selected_controls,
                dpi_hist_selected_trajs,
            ) = self._deterministic_pi_update_iterative(
                state.mean_controls,
                x0,
                sample_key,
            )
            selected_trajectory = self._rollout_batch(x0, updated_mean[None, ...])[0]
            info = {
                "costs": costs,
                "sampled_controls": sampled_controls,
                "trajectories": trajectories,
                "selected_controls": updated_mean,
                "selected_trajectory": selected_trajectory,
                "dpi_iters": (
                    int(self.config.dpi_iters)
                    if self.config.dpi_iters is not None
                    else max(int(self.config.svgd_iters), 1)
                ),
            }
            if (
                dpi_hist_controls_pre is not None
                and dpi_hist_controls_post is not None
                and dpi_hist_trajs is not None
                and dpi_hist_costs is not None
                and dpi_hist_selected_controls is not None
                and dpi_hist_selected_trajs is not None
            ):
                info["dpi_iter_sampled_controls_pre"] = dpi_hist_controls_pre
                info["dpi_iter_sampled_controls_post"] = dpi_hist_controls_post
                info["dpi_iter_trajectories"] = dpi_hist_trajs
                info["dpi_iter_costs"] = dpi_hist_costs
                info["dpi_iter_selected_controls"] = dpi_hist_selected_controls
                info["dpi_iter_selected_trajectories"] = dpi_hist_selected_trajs
                info["importance_iter_sampled_controls_pre"] = dpi_hist_controls_pre
                info["importance_iter_sampled_controls_post"] = dpi_hist_controls_post
                info["importance_iter_trajectories"] = dpi_hist_trajs
                info["importance_iter_costs"] = dpi_hist_costs
                info["importance_iter_selected_controls"] = dpi_hist_selected_controls
                info["importance_iter_selected_trajectories"] = dpi_hist_selected_trajs

        updated_mean = self._apply_control_limits(updated_mean)
        action = updated_mean[0]
        warm_started = jnp.concatenate([updated_mean[1:], updated_mean[-1:]], axis=0)
        next_state = MPPIState(key=key, mean_controls=warm_started)
        return action, next_state, info

    def _apply_control_limits(self, controls: jnp.ndarray) -> jnp.ndarray:
        if self.control_low is None or self.control_high is None:
            return controls
        if self.config.clip_controls:
            return jnp.clip(controls, self.control_low, self.control_high)
        return controls

    def _sample_control_sequences(self, key: jax.Array, mean_controls: jnp.ndarray) -> jnp.ndarray:
        k = self.config.num_samples
        h = self.config.horizon
        u = self.config.control_dim

        loc = jnp.broadcast_to(mean_controls[None, :, :], (k, h, u))
        scale = jnp.broadcast_to(self.control_noise_sigma[None, None, :], (k, h, u))

        if self.config.sampling_mode == "gaussian":
            dist = distrax.Normal(loc=loc, scale=scale)
            controls = dist.sample(seed=key)
            return self._apply_control_limits(controls)

        if self.config.sampling_mode == "uniform":
            assert self.control_low is not None and self.control_high is not None
            low = jnp.broadcast_to(self.control_low[None, None, :], (k, h, u))
            high = jnp.broadcast_to(self.control_high[None, None, :], (k, h, u))
            return jax.random.uniform(key, shape=(k, h, u), minval=low, maxval=high)

        if self.config.sampling_mode == "truncated_gaussian":
            assert self.control_low is not None and self.control_high is not None
            low = jnp.broadcast_to(self.control_low[None, None, :], (k, h, u))
            high = jnp.broadcast_to(self.control_high[None, None, :], (k, h, u))
            return self._sample_truncated_normal(key, loc=loc, scale=scale, low=low, high=high)

        # squashed_gaussian
        assert self.control_low is not None and self.control_high is not None
        dist = distrax.Transformed(distribution=distrax.Normal(loc=loc, scale=scale), bijector=distrax.Tanh())
        squashed = dist.sample(seed=key)
        low = self.control_low[None, None, :]
        high = self.control_high[None, None, :]
        controls = low + 0.5 * (squashed + 1.0) * (high - low)
        return controls

    def _sample_truncated_normal(
        self,
        key: jax.Array,
        loc: jnp.ndarray,
        scale: jnp.ndarray,
        low: jnp.ndarray,
        high: jnp.ndarray,
    ) -> jnp.ndarray:
        # Inverse-CDF sampling for elementwise truncated Gaussians.
        sqrt_2 = jnp.sqrt(jnp.array(2.0, dtype=loc.dtype))
        alpha = (low - loc) / scale
        beta = (high - loc) / scale

        cdf_alpha = 0.5 * (1.0 + erf(alpha / sqrt_2))
        cdf_beta = 0.5 * (1.0 + erf(beta / sqrt_2))
        interval_mass = jnp.maximum(cdf_beta - cdf_alpha, 1e-8)

        u = jax.random.uniform(key, shape=loc.shape, minval=0.0, maxval=1.0, dtype=loc.dtype)
        p = cdf_alpha + u * interval_mass
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
        z = sqrt_2 * erfinv(2.0 * p - 1.0)
        x = loc + scale * z
        return jnp.clip(x, low, high)

    def _rollout_batch(self, x0: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
        """Rollout sampled controls.

        x0: [state_dim]
        controls: [num_samples, horizon, control_dim]
        returns: [num_samples, horizon, state_dim]
        """
        if self.rollout_batch_fn is not None:
            return self.rollout_batch_fn(x0, controls)

        k = controls.shape[0]
        init_states = jnp.broadcast_to(x0[None, :], (k, x0.shape[0]))

        def scan_step(carry_state, u_t):
            next_state = jax.vmap(self.dynamics_fn)(carry_state, u_t)
            return next_state, next_state

        u_time_major = jnp.swapaxes(controls, 0, 1)
        _, states_time_major = jax.lax.scan(scan_step, init_states, u_time_major)
        return jnp.swapaxes(states_time_major, 0, 1)

    def _importance_update(
        self,
        controls: jnp.ndarray,
        costs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        invalid_cost = jnp.asarray(1e12, dtype=costs.dtype)
        finite_costs = jnp.where(jnp.isfinite(costs), costs, invalid_cost)
        shifted = finite_costs - jnp.min(finite_costs)
        logits = -shifted / self.config.temperature
        weights = jax.nn.softmax(logits)
        updated_mean = jnp.einsum("k,khu->hu", weights, controls)
        return updated_mean, weights

    def _importance_update_iterative(
        self,
        mean_controls: jnp.ndarray,
        x0: jnp.ndarray,
        rng_key: jax.Array,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        """Run iterative importance-sampling MPPI updates.

        Uses the same iteration budget as SVGD (`svgd_iters`) to keep update
        counts aligned for baseline comparisons.
        """
        num_iters = max(int(self.config.svgd_iters), 1)
        dummy_controls = jnp.broadcast_to(
            mean_controls[None, ...],
            (self.config.num_samples, self.config.horizon, self.config.control_dim),
        )
        dummy_traj = self._rollout_batch(x0, dummy_controls)
        dummy_costs = self.cost_fn(dummy_traj)
        dummy_costs = jnp.nan_to_num(dummy_costs, nan=1e12, posinf=1e12, neginf=1e12)
        _, dummy_weights = self._importance_update(dummy_controls, dummy_costs)

        def body(_i, carry):
            curr_mean, curr_key, _last_costs, _last_weights, _last_controls, _last_traj = carry
            curr_key, k_sample = jax.random.split(curr_key)
            controls = self._sample_control_sequences(k_sample, curr_mean)
            traj = self._rollout_batch(x0, controls)
            costs = self.cost_fn(traj)
            costs = jnp.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12)
            next_mean, weights = self._importance_update(controls, costs)
            next_mean = self._apply_control_limits(next_mean)
            return next_mean, curr_key, costs, weights, controls, traj

        if self.config.record_svgd_history:
            iter_ids = jnp.arange(num_iters, dtype=jnp.int32)

            def scan_step(carry, _iter_idx):
                curr_mean, curr_key = carry
                curr_key, k_sample, k_post = jax.random.split(curr_key, 3)
                controls = self._sample_control_sequences(k_sample, curr_mean)
                traj = self._rollout_batch(x0, controls)
                costs = self.cost_fn(traj)
                costs = jnp.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12)
                next_mean, weights = self._importance_update(controls, costs)
                next_mean = self._apply_control_limits(next_mean)

                # For history visualization parity with SVGD, record post-update samples.
                post_controls = self._sample_control_sequences(k_post, next_mean)
                post_traj = self._rollout_batch(x0, post_controls)
                post_costs = self.cost_fn(post_traj)
                post_costs = jnp.nan_to_num(post_costs, nan=1e12, posinf=1e12, neginf=1e12)
                selected_traj = self._rollout_batch(x0, next_mean[None, ...])[0]
                return (
                    (next_mean, curr_key),
                    (
                        controls,
                        post_controls,
                        post_traj,
                        post_costs,
                        traj,
                        costs,
                        weights,
                        next_mean,
                        selected_traj,
                    ),
                )

            (
                (updated_mean, _),
                (
                    hist_controls,
                    hist_post_controls,
                    hist_post_trajs,
                    hist_post_costs,
                    hist_trajs,
                    hist_costs,
                    hist_weights,
                    hist_selected_controls,
                    hist_selected_trajs,
                ),
            ) = jax.lax.scan(
                scan_step,
                (mean_controls, rng_key),
                iter_ids,
            )
            sampled_controls = hist_controls[-1]
            trajectories = hist_trajs[-1]
            costs = hist_costs[-1]
            weights = hist_weights[-1]
            return (
                updated_mean,
                costs,
                weights,
                sampled_controls,
                trajectories,
                hist_controls,
                hist_post_controls,
                hist_post_trajs,
                hist_post_costs,
                hist_weights,
                hist_selected_controls,
                hist_selected_trajs,
            )

        updated_mean, _, costs, weights, sampled_controls, trajectories = jax.lax.fori_loop(
            0,
            num_iters,
            body,
            (mean_controls, rng_key, dummy_costs, dummy_weights, dummy_controls, dummy_traj),
        )
        return updated_mean, costs, weights, sampled_controls, trajectories, None, None, None, None, None, None, None

    def _augment_trajectories_explicit(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        """Build augmented-state trajectories as in the paper's Eq. (9).

        trajectories: [num_samples, horizon, state_dim]
        returns: [num_samples, horizon, horizon * state_dim]
        """

        def augment_single(traj: jnp.ndarray) -> jnp.ndarray:
            horizon = traj.shape[0]
            state_dim = traj.shape[1]

            def step_fn(hist: jnp.ndarray, x_t: jnp.ndarray):
                next_hist = jnp.concatenate([x_t[None, :], hist[:-1]], axis=0)
                return next_hist, next_hist.reshape((horizon * state_dim,))

            init_hist = jnp.zeros((horizon, state_dim), dtype=traj.dtype)
            _, aug = jax.lax.scan(step_fn, init_hist, traj)
            return aug

        return jax.vmap(augment_single)(trajectories)

    def _evaluate_dpi_costs(
        self,
        trajectories: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.config.dpi_augmented_mode == "explicit":
            aug = self._augment_trajectories_explicit(trajectories)
            state_dim = trajectories.shape[-1]
            physical = aug[..., :state_dim]
            return self.cost_fn(physical)
        return self.cost_fn(trajectories)

    def _deterministic_pi_update_iterative(
        self,
        mean_controls: jnp.ndarray,
        x0: jnp.ndarray,
        rng_key: jax.Array,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        """Deterministic PI update with shrinking temperature and covariance.

        Implements the paper-style iterative update with MPPI correction term:
          lambda * eps^T * Sigma^{-1} * u_hat
        """
        num_iters = (
            max(int(self.config.dpi_iters), 1)
            if self.config.dpi_iters is not None
            else max(int(self.config.svgd_iters), 1)
        )
        shrink = jnp.asarray(self.config.dpi_shrink_factor, dtype=jnp.float32)
        sqrt_shrink = jnp.sqrt(shrink)
        min_temp = jnp.asarray(self.config.dpi_min_temperature, dtype=jnp.float32)

        sigma0 = self.control_noise_sigma.astype(jnp.float32)
        lambda0 = jnp.maximum(jnp.asarray(self.config.temperature, dtype=jnp.float32), min_temp)

        dummy_controls = jnp.broadcast_to(
            mean_controls[None, ...],
            (self.config.num_samples, self.config.horizon, self.config.control_dim),
        )
        dummy_traj = self._rollout_batch(x0, dummy_controls)
        dummy_costs = self._evaluate_dpi_costs(dummy_traj)
        dummy_costs = jnp.nan_to_num(dummy_costs, nan=1e12, posinf=1e12, neginf=1e12)
        init = (mean_controls, rng_key, lambda0, sigma0, dummy_costs, dummy_controls, dummy_traj)

        def sample_controls_from_mean_sigma(key: jax.Array, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
            eps = (
                jax.random.normal(
                    key,
                    shape=(self.config.num_samples, self.config.horizon, self.config.control_dim),
                    dtype=mean.dtype,
                )
                * sigma[None, None, :]
            )
            return self._apply_control_limits(mean[None, ...] + eps)

        def body(_i, carry):
            curr_mean, curr_key, curr_lambda, curr_sigma, _last_costs, _last_controls, _last_traj = carry
            curr_key, k_eps = jax.random.split(curr_key)
            controls = sample_controls_from_mean_sigma(k_eps, curr_mean, curr_sigma)
            # Use the realized perturbation after clipping for both weighting and update.
            eps = controls - curr_mean[None, ...]
            traj = self._rollout_batch(x0, controls)
            base_costs = self._evaluate_dpi_costs(traj)
            base_costs = jnp.nan_to_num(base_costs, nan=1e12, posinf=1e12, neginf=1e12)

            inv_var = 1.0 / jnp.maximum(curr_sigma * curr_sigma, 1e-8)
            correction = curr_lambda * jnp.sum(eps * inv_var[None, None, :] * curr_mean[None, :, :], axis=(1, 2))
            total_costs = base_costs + correction
            total_costs = jnp.nan_to_num(total_costs, nan=1e12, posinf=1e12, neginf=1e12)

            shifted = total_costs - jnp.min(total_costs)
            logits = -shifted / jnp.maximum(curr_lambda, min_temp)
            weights = jax.nn.softmax(logits)
            delta = jnp.einsum("k,khu->hu", weights, eps)
            next_mean = self._apply_control_limits(curr_mean + delta)

            next_lambda = jnp.maximum(curr_lambda * shrink, min_temp)
            next_sigma = curr_sigma * sqrt_shrink
            return next_mean, curr_key, next_lambda, next_sigma, total_costs, controls, traj

        if self.config.record_svgd_history:
            iter_ids = jnp.arange(num_iters, dtype=jnp.int32)

            def scan_step(carry, _iter_idx):
                curr_mean, curr_key, curr_lambda, curr_sigma = carry
                (
                    next_mean,
                    curr_key,
                    next_lambda,
                    next_sigma,
                    total_costs,
                    controls,
                    traj,
                ) = body(
                    _iter_idx,
                    (
                        curr_mean,
                        curr_key,
                        curr_lambda,
                        curr_sigma,
                        dummy_costs,
                        dummy_controls,
                        dummy_traj,
                    ),
                )
                # Keep history-only diagnostics from perturbing the optimization RNG stream.
                # Using fold_in derives a deterministic side key without advancing curr_key.
                k_post = jax.random.fold_in(curr_key, 1)
                post_controls = sample_controls_from_mean_sigma(k_post, next_mean, next_sigma)
                post_traj = self._rollout_batch(x0, post_controls)
                post_costs = self._evaluate_dpi_costs(post_traj)
                post_costs = jnp.nan_to_num(post_costs, nan=1e12, posinf=1e12, neginf=1e12)
                selected_traj = self._rollout_batch(x0, next_mean[None, ...])[0]
                return (
                    (next_mean, curr_key, next_lambda, next_sigma),
                    (
                        controls,
                        post_controls,
                        post_traj,
                        post_costs,
                        traj,
                        total_costs,
                        next_mean,
                        selected_traj,
                    ),
                )

            (
                (updated_mean, _, _, _),
                (
                    hist_controls,
                    hist_post_controls,
                    hist_post_trajs,
                    hist_post_costs,
                    hist_trajs,
                    hist_costs,
                    hist_selected_controls,
                    hist_selected_trajs,
                ),
            ) = jax.lax.scan(
                scan_step,
                (mean_controls, rng_key, lambda0, sigma0),
                iter_ids,
            )
            sampled_controls = hist_controls[-1]
            trajectories = hist_trajs[-1]
            costs = hist_costs[-1]
            return (
                updated_mean,
                costs,
                sampled_controls,
                trajectories,
                hist_controls,
                hist_post_controls,
                hist_post_trajs,
                hist_post_costs,
                hist_selected_controls,
                hist_selected_trajs,
            )

        updated_mean, _, _, _, costs, sampled_controls, trajectories = jax.lax.fori_loop(
            0,
            num_iters,
            body,
            init,
        )
        return updated_mean, costs, sampled_controls, trajectories, None, None, None, None, None, None

    def _annealed_value(
        self,
        iter_idx: jnp.ndarray,
        init_value: float,
        final_value: Optional[float],
        mode: AnnealMode,
    ) -> jnp.ndarray:
        init = jnp.asarray(init_value, dtype=jnp.float32)
        final = jnp.asarray(init_value if final_value is None else final_value, dtype=jnp.float32)
        denom = jnp.maximum(self.config.svgd_iters - 1, 1)
        frac = jnp.asarray(iter_idx, dtype=jnp.float32) / jnp.asarray(denom, dtype=jnp.float32)

        if mode == "none":
            return init
        if mode == "linear":
            return init + frac * (final - init)
        if mode == "exp":
            ratio = final / jnp.maximum(init, 1e-8)
            return init * jnp.power(jnp.maximum(ratio, 1e-8), frac)
        if mode == "cosine":
            w = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
            return final + w * (init - final)
        raise ValueError(f"Unsupported anneal mode={mode}")

    def _select_svgd_controls(self, particles: jnp.ndarray, costs: jnp.ndarray) -> jnp.ndarray:
        invalid_cost = jnp.asarray(1e12, dtype=costs.dtype)
        finite_costs = jnp.where(jnp.isfinite(costs), costs, invalid_cost)
        mode = self.config.svgd_selection_mode
        if mode == "mean":
            return jnp.mean(particles, axis=0)
        if mode == "best":
            return particles[jnp.argmin(finite_costs)]

        sel_temp = (
            self.config.temperature
            if self.config.svgd_selection_temperature is None
            else self.config.svgd_selection_temperature
        )
        shifted = finite_costs - jnp.min(finite_costs)
        logits = -shifted / sel_temp
        weights = jax.nn.softmax(logits)
        return jnp.einsum("k,khu->hu", weights, particles)

    def _svgd_fd_update(
        self,
        particles: jnp.ndarray,
        x0: jnp.ndarray,
        rng_key: jax.Array,
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        temperature = self.config.temperature
        flat_dim = self.config.horizon * self.config.control_dim
        alpha = (
            (1.0 / temperature)
            if self.config.svgd_fd_alpha is None
            else self.config.svgd_fd_alpha
        )
        delta = self.config.svgd_fd_delta

        def rbf_kernel(flat_particles: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            diff = flat_particles[:, None, :] - flat_particles[None, :, :]
            sq_dist = jnp.sum(diff * diff, axis=-1)
            sq_dist = sq_dist / jnp.maximum(float(flat_dim), 1.0)
            if self.config.svgd_kernel_bandwidth is None:
                median = jnp.median(sq_dist)
                bandwidth = median / jnp.log(self.config.num_samples + 1.0)
            else:
                bandwidth = self.config.svgd_kernel_bandwidth
            bandwidth = jnp.maximum(bandwidth, 1e-6)
            kxy = jnp.exp(-sq_dist / bandwidth)
            return kxy, bandwidth

        def annealed_step_size(iter_idx: jnp.ndarray) -> jnp.ndarray:
            return self._annealed_value(
                iter_idx=iter_idx,
                init_value=self.config.svgd_step_size,
                final_value=self.config.svgd_step_size_final,
                mode=self.config.svgd_step_size_anneal,
            )

        def annealed_repulsion_coef(iter_idx: jnp.ndarray) -> jnp.ndarray:
            return self._annealed_value(
                iter_idx=iter_idx,
                init_value=self.config.svgd_repulsion_coef,
                final_value=self.config.svgd_repulsion_final,
                mode=self.config.svgd_repulsion_anneal,
            )

        def finite_costs_for_controls(controls: jnp.ndarray) -> jnp.ndarray:
            traj = self._rollout_batch(x0, controls)
            costs = self.cost_fn(traj)
            return jnp.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12)

        def spsa_grad(curr_particles: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
            k = self.config.num_samples
            h = self.config.horizon
            u = self.config.control_dim
            r = int(self.config.svgd_fd_dirs)

            key, k_sign = jax.random.split(key)
            signs = jax.random.rademacher(
                k_sign,
                shape=(k, r, h, u),
                dtype=curr_particles.dtype,
            )
            plus = self._apply_control_limits(curr_particles[:, None, :, :] + delta * signs)
            minus = self._apply_control_limits(curr_particles[:, None, :, :] - delta * signs)
            plus_flat = plus.reshape((k * r, h, u))
            minus_flat = minus.reshape((k * r, h, u))
            plus_costs = finite_costs_for_controls(plus_flat).reshape((k, r))
            minus_costs = finite_costs_for_controls(minus_flat).reshape((k, r))

            diff = ((plus_costs - minus_costs) / (2.0 * delta)).astype(curr_particles.dtype)
            grads = diff[:, :, None, None] * signs
            grad = jnp.mean(grads, axis=1)
            if self.config.svgd_fd_nan_fallback == "zero_grad":
                grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            return grad

        def one_svgd_fd_step(curr_particles: jnp.ndarray, key: jax.Array, iter_idx: jnp.ndarray) -> jnp.ndarray:
            grad = spsa_grad(curr_particles, key)
            flat = curr_particles.reshape((self.config.num_samples, flat_dim))
            flat_grad = grad.reshape((self.config.num_samples, flat_dim))
            kxy, h_bw = rbf_kernel(flat)
            attraction = -(alpha) * flat_grad

            if self.config.svgd_fd_use_kernel_repulsion:
                flat_scale = jnp.maximum(float(flat_dim), 1.0)
                grad_k = (-2.0 / (h_bw * flat_scale)) * (flat[:, None, :] - flat[None, :, :]) * kxy[:, :, None]
                repulsion = annealed_repulsion_coef(iter_idx) * jnp.sum(grad_k, axis=0)
            else:
                repulsion = jnp.zeros_like(attraction)

            phi_flat = (kxy @ attraction + repulsion) / self.config.num_samples
            phi = phi_flat.reshape(curr_particles.shape)
            phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
            step_size = annealed_step_size(iter_idx)
            next_particles = curr_particles + step_size * phi
            return self._apply_control_limits(next_particles)

        def body(i, carry):
            curr_particles, curr_key = carry
            curr_key, k_step = jax.random.split(curr_key)
            next_particles = one_svgd_fd_step(curr_particles, k_step, i)
            return next_particles, curr_key

        if self.config.record_svgd_history:
            iter_ids = jnp.arange(self.config.svgd_iters, dtype=jnp.int32)

            def scan_step(carry, iter_idx: jnp.ndarray):
                curr_particles, curr_key = carry
                curr_key, k_step = jax.random.split(curr_key)
                next_particles = one_svgd_fd_step(curr_particles, k_step, iter_idx)
                step_costs = finite_costs_for_controls(next_particles)
                return (next_particles, curr_key), (next_particles, step_costs)

            (updated, _), (hist_particles, hist_costs) = jax.lax.scan(
                scan_step,
                (particles, rng_key),
                iter_ids,
            )
            final_costs = hist_costs[-1]
            return updated, final_costs, hist_particles, hist_costs

        updated, _ = jax.lax.fori_loop(0, self.config.svgd_iters, body, (particles, rng_key))
        final_costs = finite_costs_for_controls(updated)
        return updated, final_costs, None, None

    def _svgd_update(
        self, particles: jnp.ndarray, x0: jnp.ndarray, rng_key: jax.Array
    ) -> Tuple[
        jnp.ndarray,
        jnp.ndarray,
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        temperature = self.config.temperature
        flat_dim = self.config.horizon * self.config.control_dim

        def single_cost(u_seq: jnp.ndarray) -> jnp.ndarray:
            traj = self._rollout_batch(x0, u_seq[None, ...])[0]
            return self.cost_fn(traj[None, ...])[0]

        grad_cost = (
            jax.vmap(jax.grad(single_cost))
            if self.config.svgd_grad_mode == "reverse"
            else jax.vmap(jax.jacfwd(single_cost))
        )

        def rbf_kernel(flat_particles: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
            diff = flat_particles[:, None, :] - flat_particles[None, :, :]
            sq_dist = jnp.sum(diff * diff, axis=-1)
            sq_dist = sq_dist / jnp.maximum(float(flat_dim), 1.0)
            if self.config.svgd_kernel_bandwidth is None:
                median = jnp.median(sq_dist)
                bandwidth = median / jnp.log(self.config.num_samples + 1.0)
            else:
                bandwidth = self.config.svgd_kernel_bandwidth
            bandwidth = jnp.maximum(bandwidth, 1e-6)
            kxy = jnp.exp(-sq_dist / bandwidth)
            return kxy, bandwidth

        def annealed_step_size(iter_idx: jnp.ndarray) -> jnp.ndarray:
            return self._annealed_value(
                iter_idx=iter_idx,
                init_value=self.config.svgd_step_size,
                final_value=self.config.svgd_step_size_final,
                mode=self.config.svgd_step_size_anneal,
            )

        def annealed_repulsion_coef(iter_idx: jnp.ndarray) -> jnp.ndarray:
            return self._annealed_value(
                iter_idx=iter_idx,
                init_value=self.config.svgd_repulsion_coef,
                final_value=self.config.svgd_repulsion_final,
                mode=self.config.svgd_repulsion_anneal,
            )

        def resample_with_jitter(
            curr_particles: jnp.ndarray,
            curr_costs: jnp.ndarray,
            key: jax.Array,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if not self.config.svgd_resample_enabled:
                return curr_particles, jnp.asarray(float(self.config.num_samples), dtype=jnp.float32), jnp.asarray(
                    False
                )

            temp = (
                self.config.temperature
                if self.config.svgd_resample_temperature is None
                else self.config.svgd_resample_temperature
            )
            invalid_cost = jnp.asarray(1e12, dtype=curr_costs.dtype)
            finite_costs = jnp.where(jnp.isfinite(curr_costs), curr_costs, invalid_cost)
            shifted = finite_costs - jnp.min(finite_costs)
            logits = -shifted / temp
            weights = jax.nn.softmax(logits)
            ess = 1.0 / jnp.sum(jnp.square(weights))

            thresh_cfg = jnp.asarray(self.config.svgd_resample_ess_threshold, dtype=jnp.float32)
            n = jnp.asarray(self.config.num_samples, dtype=jnp.float32)
            ess_thresh = jnp.where(thresh_cfg <= 1.0, thresh_cfg * n, thresh_cfg)
            need_resample = ess < ess_thresh

            def do_resample(_: None) -> jnp.ndarray:
                k_idx, k_noise = jax.random.split(key)
                idx = jax.random.choice(k_idx, self.config.num_samples, shape=(self.config.num_samples,), p=weights)
                resampled = curr_particles[idx]
                if self.config.svgd_resample_jitter_scale <= 0:
                    return resampled
                sigma = self.control_noise_sigma[None, None, :]
                noise = (
                    jax.random.normal(k_noise, shape=curr_particles.shape, dtype=curr_particles.dtype)
                    * sigma
                    * self.config.svgd_resample_jitter_scale
                )
                jittered = resampled + noise
                return self._apply_control_limits(jittered)

            next_particles = jax.lax.cond(
                need_resample,
                do_resample,
                lambda _: curr_particles,
                operand=None,
            )
            return next_particles, ess, need_resample

        def one_svgd_step(
            curr_particles: jnp.ndarray,
            step_size: jnp.ndarray,
            repulsion_coef: jnp.ndarray,
        ) -> jnp.ndarray:
            score = -(1.0 / temperature) * grad_cost(curr_particles)
            score = jnp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

            flat = curr_particles.reshape((self.config.num_samples, flat_dim))
            flat_score = score.reshape((self.config.num_samples, flat_dim))

            kxy, h = rbf_kernel(flat)
            flat_scale = jnp.maximum(float(flat_dim), 1.0)
            grad_k = (-2.0 / (h * flat_scale)) * (flat[:, None, :] - flat[None, :, :]) * kxy[:, :, None]
            phi_flat = (kxy @ flat_score + repulsion_coef * jnp.sum(grad_k, axis=0)) / self.config.num_samples

            phi = phi_flat.reshape(curr_particles.shape)
            phi = jnp.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
            next_particles = curr_particles + step_size * phi
            next_particles = self._apply_control_limits(next_particles)
            return next_particles

        if self.config.record_svgd_history or self.config.show_svgd_progress:
            total_iters = jnp.asarray(self.config.svgd_iters, dtype=jnp.int32)
            iter_ids = jnp.arange(self.config.svgd_iters, dtype=jnp.int32)

            def scan_step(carry, iter_idx: jnp.ndarray):
                curr_particles, curr_key = carry
                step_size = annealed_step_size(iter_idx)
                repulsion_coef = annealed_repulsion_coef(iter_idx)
                next_particles = one_svgd_step(curr_particles, step_size, repulsion_coef)
                step_costs = jax.vmap(single_cost)(next_particles)
                step_costs = jnp.nan_to_num(step_costs, nan=1e12, posinf=1e12, neginf=1e12)
                curr_key, k_resample = jax.random.split(curr_key)
                next_particles, ess, did_resample = resample_with_jitter(next_particles, step_costs, k_resample)
                step_costs = jax.vmap(single_cost)(next_particles)
                step_costs = jnp.nan_to_num(step_costs, nan=1e12, posinf=1e12, neginf=1e12)
                if self.config.show_svgd_progress:
                    best_robustness = -jnp.min(step_costs)
                    _ = io_callback(
                        _tqdm_svgd_update_callback,
                        jax.ShapeDtypeStruct((), jnp.int32),
                        iter_idx,
                        total_iters,
                        best_robustness,
                    )
                return (next_particles, curr_key), (next_particles, step_costs, ess, did_resample)

            (updated, _), (hist_particles, hist_costs, hist_ess, hist_resampled) = jax.lax.scan(
                scan_step, (particles, rng_key), iter_ids
            )
            final_costs = hist_costs[-1]
            if self.config.record_svgd_history:
                return updated, final_costs, hist_particles, hist_costs, hist_ess, hist_resampled
            return updated, final_costs, None, None, None, None

        def body(i, carry):
            p, key = carry
            p = one_svgd_step(p, annealed_step_size(i), annealed_repulsion_coef(i))
            c = jax.vmap(single_cost)(p)
            c = jnp.nan_to_num(c, nan=1e12, posinf=1e12, neginf=1e12)
            key, k_resample = jax.random.split(key)
            p, _, _ = resample_with_jitter(p, c, k_resample)
            return p, key

        updated, _ = jax.lax.fori_loop(0, self.config.svgd_iters, body, (particles, rng_key))
        final_costs = jax.vmap(single_cost)(updated)
        final_costs = jnp.nan_to_num(final_costs, nan=1e12, posinf=1e12, neginf=1e12)
        return updated, final_costs, None, None, None, None


__all__ = [
    "MPPIConfig",
    "MPPIState",
    "MPPIController",
    "pointmass2d_dynamics_step",
    "pointmass2d_multiagent_dynamics_step",
    "make_stl_cost_fn",
]
