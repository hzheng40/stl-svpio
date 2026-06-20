from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import jax
import jax.numpy as jnp
from stljax.formula import Always, And, Eventually, Predicate

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController, make_stl_cost_fn
from stl_svpio.baselines.stlcg_gd import STLCGGradientDescentConfig, run_stlcg_gradient_descent
from stl_svpio.envs import MultiAgentPointMass2DEnv, PointMass2DEnv
from stl_svpio.specifications import (
    pointmass_full_task_spec,
    pointmass_multiagent_corridor_spec,
    pointmass_multiagent_synchronized_goals_spec,
    pointmass_two_agent_button_task_spec,
    pointmass_visit_all_zones_task_spec,
)

PAPER_POINTMASS_TASKS = {
    "single_default": "Table I reach-avoid",
    "single_visit_goals": "Figure 3 long-horizon clutter",
    "multiagent_button": "Figure 3 button ordering",
    "multiagent_sync_goals": "Figure 3 synchronized goals",
    "multiagent_corridor": "Figure 3 corridor queuing",
}


@dataclass
class PointMassProblem:
    env: PointMass2DEnv | MultiAgentPointMass2DEnv
    initial_state: jnp.ndarray
    stl_specification: Any
    dynamics_fn: Any
    control_low: jnp.ndarray
    control_high: jnp.ndarray
    control_noise_sigma: jnp.ndarray
    control_dim: int
    num_agents: int
    horizon: int
    episode_steps: int


@dataclass
class PointMassTrialResult:
    task_id: str
    method: str
    seed: int
    sampling_seed: int
    runtime_ms: float
    robustness: float
    satisfied: bool
    num_particles: int
    num_iterations: int


def _conjunction(formulas):
    if not formulas:
        return Predicate("true", lambda x: jnp.ones((x.shape[0],), dtype=x.dtype)) > 0.0
    out = formulas[0]
    for f in formulas[1:]:
        out = And(out, f)
    return out


def _constructed_reach_avoid_scene(env: PointMass2DEnv) -> None:
    env.obstacles.centers = jnp.array([[2.0, 2.0]], dtype=jnp.float32)
    env.obstacles.radii = jnp.array([1.0], dtype=jnp.float32)
    env.zones.centers = jnp.array([[4.0, 4.0]], dtype=jnp.float32)
    env.zones.radii = jnp.array([0.5], dtype=jnp.float32)
    env.goal_square_center = jnp.array([4.0, 4.0], dtype=jnp.float32)  # noqa: B010
    env.goal_square_side = 1.0  # noqa: B010
    env.state = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)


def _constructed_reach_avoid_spec(env: PointMass2DEnv, horizon: int, stay_steps: int = 1):
    safe_terms = []
    for i in range(env.obstacles.centers.shape[0]):
        center = env.obstacles.centers[i]
        radius = float(env.obstacles.radii[i])
        outside = Predicate(
            f"reach_avoid_outside_obstacle_{i}",
            lambda tr, c=center, r=radius: jnp.linalg.norm(tr[:, :2] - c, axis=-1) - r,
        )
        safe_terms.append(Always(outside > 0.0, interval=[0, horizon - 1]))

    goal_center = env.goal_square_center
    goal_half_side = 0.5 * float(env.goal_square_side)
    in_goal = Predicate(
        "reach_avoid_inside_square_goal",
        lambda tr, c=goal_center, h=goal_half_side: jnp.min(h - jnp.abs(tr[:, :2] - c), axis=-1),
    )
    dwell = Always(in_goal > 0.0, interval=[0, stay_steps - 1])
    goal = Eventually(dwell, interval=[0, horizon - stay_steps])
    return _conjunction([_conjunction(safe_terms), goal])


def _corridor_scene(
    env: MultiAgentPointMass2DEnv,
    key: jax.Array,
    collision_radius: float,
    corridor_half_extent: float,
) -> jnp.ndarray:
    center = 0.5 * (env.world_low + env.world_high)
    half_x = float(corridor_half_extent)
    half_y = 3.0 * half_x
    env.corridor_center = center  # noqa: B010
    env.corridor_half_extent_x = half_x  # noqa: B010
    env.corridor_half_extent_y = half_y  # noqa: B010
    env.wall_half_thickness = half_x  # noqa: B010
    env.agent_collision_radius = float(collision_radius)  # noqa: B010
    env.wall_x = float(center[0])  # noqa: B010
    env.obstacles.centers = jnp.zeros((0, 2), dtype=jnp.float32)
    env.obstacles.radii = jnp.zeros((0,), dtype=jnp.float32)
    env.zones.centers = center[None, :]
    env.zones.radii = jnp.array([half_x], dtype=jnp.float32)

    min_sep = 2.2 * collision_radius
    pts: list[jnp.ndarray] = []
    x_low = float(env.world_low[0] + 0.2)
    x_high = float(min(-0.2, center[0] - half_x - 3.2))
    y_low = float(env.world_low[1] + 0.2)
    y_high = float(env.world_high[1] - 0.2)
    for _ in range(env.num_agents):
        placed = False
        for _attempt in range(6000):
            key, kp = jax.random.split(key)
            cand = jax.random.uniform(
                kp,
                shape=(2,),
                minval=jnp.array([x_low, y_low], dtype=jnp.float32),
                maxval=jnp.array([x_high, y_high], dtype=jnp.float32),
            )
            if pts:
                d = jnp.linalg.norm(cand[None, :] - jnp.stack(pts), axis=-1)
                if bool(jnp.any(d < min_sep)):
                    continue
            pts.append(cand)
            placed = True
            break
        if not placed:
            raise RuntimeError("Failed to sample separated corridor starts.")

    pos = jnp.stack(pts, axis=0)
    vel = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    env.state = jnp.concatenate([pos, vel], axis=-1).reshape((-1,))
    return env.state


def _sync_goals_scene(
    env: MultiAgentPointMass2DEnv,
    key: jax.Array,
    collision_radius: float,
) -> jnp.ndarray:
    center = 0.5 * (env.world_low + env.world_high)
    env.agent_collision_radius = float(collision_radius)  # noqa: B010
    env.sync_goal_mode = True  # noqa: B010
    env.obstacles.centers = jnp.zeros((0, 2), dtype=jnp.float32)
    env.obstacles.radii = jnp.zeros((0,), dtype=jnp.float32)

    goal_radius = 0.6
    cols = int(jnp.ceil(jnp.sqrt(env.num_agents)))
    rows = int(jnp.ceil(env.num_agents / cols))
    spacing = 1.2 * (2.0 * goal_radius + 0.1)
    xs = jnp.linspace(center[0] - 0.5 * spacing * max(cols - 1, 0), center[0] + 0.5 * spacing * max(cols - 1, 0), cols)
    ys = jnp.linspace(center[1] - 0.5 * spacing * max(rows - 1, 0), center[1] + 0.5 * spacing * max(rows - 1, 0), rows)
    env.zones.centers = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape((-1, 2))[: env.num_agents]
    env.zones.radii = jnp.full((env.num_agents,), goal_radius, dtype=jnp.float32)

    spawn_radius = 5.0
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, env.num_agents, endpoint=False, dtype=jnp.float32)
    pos = center[None, :] + spawn_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    key, k_perm = jax.random.split(key)
    pos = pos[jax.random.permutation(k_perm, env.num_agents)]
    vel = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    env.state = jnp.concatenate([pos, vel], axis=-1).reshape((-1,))
    return env.state


def build_pointmass_problem(config: dict[str, Any], seed: int) -> PointMassProblem:
    task = str(config.get("task", config.get("task_id", "single_default")))
    horizon = int(config.get("horizon", 20))
    episode_steps = int(config.get("episode_steps", horizon))
    dt = float(config.get("dt", 0.1))
    stay_steps = int(config.get("stay_steps", 1))
    num_agents = int(
        config.get(
            "num_agents",
            10 if task == "multiagent_corridor" else 9 if task == "multiagent_sync_goals" else 2 if task == "multiagent_button" else 1,
        )
    )
    is_multiagent = task.startswith("multiagent_")
    reset_key = jax.random.PRNGKey(seed + 123)

    if is_multiagent:
        world = (-10.0, 10.0) if task in {"multiagent_corridor", "multiagent_sync_goals"} else (-5.0, 5.0)
        env: PointMass2DEnv | MultiAgentPointMass2DEnv = MultiAgentPointMass2DEnv(
            num_agents=num_agents,
            world_low=(world[0], world[0]),
            world_high=(world[1], world[1]),
            dt=dt,
            seed=seed,
            num_obstacles=int(config.get("num_obstacles", 0)),
            num_zones=int(config.get("num_zones", 3 if task == "multiagent_button" else 1)),
        )
    else:
        env = PointMass2DEnv(
            dt=dt,
            seed=seed,
            num_obstacles=int(config.get("num_obstacles", 4)),
            num_zones=int(config.get("num_zones", 2)),
        )

    initial_state = env.reset(
        key=reset_key,
        landmark_sampling_mode=str(config.get("landmark_sampling_mode", "sequential")),
    )
    if task == "single_default" and str(config.get("scene", "constructed_pointmass_diag")) == "constructed_pointmass_diag":
        _constructed_reach_avoid_scene(env)
        initial_state = env.state
        spec = _constructed_reach_avoid_spec(env, horizon=horizon, stay_steps=stay_steps)
    elif task == "single_visit_goals":
        env.state = jnp.concatenate([initial_state[:2], jnp.zeros((2,), dtype=initial_state.dtype)])
        initial_state = env.state
        spec = pointmass_visit_all_zones_task_spec(env, horizon=horizon, stay_steps=stay_steps)
    elif task == "single_default":
        env.state = jnp.concatenate([initial_state[:2], jnp.zeros((2,), dtype=initial_state.dtype)])
        initial_state = env.state
        spec = pointmass_full_task_spec(env, horizon=horizon, stay_steps=stay_steps)
    elif task == "multiagent_button":
        x0_agents = initial_state.reshape((num_agents, 4))
        env.state = jnp.concatenate([x0_agents[:, :2], jnp.zeros((num_agents, 2), dtype=initial_state.dtype)], axis=-1).reshape((-1,))
        initial_state = env.state
        spec = pointmass_two_agent_button_task_spec(env, horizon=horizon, stay_steps=stay_steps)
    elif task == "multiagent_sync_goals":
        initial_state = _sync_goals_scene(env, reset_key, float(config.get("agent_collision_radius", 0.12)))
        spec = pointmass_multiagent_synchronized_goals_spec(
            env,
            horizon=horizon,
            sync_delta_steps=int(config.get("sync_delta_steps", 2)),
            collision_radius=float(config.get("agent_collision_radius", 0.12)),
        )
    elif task == "multiagent_corridor":
        initial_state = _corridor_scene(
            env,
            reset_key,
            collision_radius=float(config.get("agent_collision_radius", 0.12)),
            corridor_half_extent=float(config.get("corridor_half_extent", 0.6)),
        )
        spec = pointmass_multiagent_corridor_spec(
            env,
            horizon=horizon,
            corridor_center=env.corridor_center,
            corridor_half_extent_x=float(env.corridor_half_extent_x),
            corridor_half_extent_y=float(env.corridor_half_extent_y),
            collision_radius=float(config.get("agent_collision_radius", 0.12)),
        )
    else:
        raise ValueError(f"Unsupported paper point-mass task: {task}")

    num_agents_eff = num_agents if is_multiagent else 1
    control_dim = 2 * num_agents_eff
    control_low = jnp.tile(jnp.array([-5.0, -5.0], dtype=jnp.float32), (num_agents_eff,))
    control_high = jnp.tile(jnp.array([5.0, 5.0], dtype=jnp.float32), (num_agents_eff,))
    control_noise_sigma = jnp.tile(
        jnp.array([float(config.get("noise_sigma", 0.3)), float(config.get("noise_sigma", 0.3))], dtype=jnp.float32),
        (num_agents_eff,),
    )
    dynamics_fn = (
        MPPIController.pointmass2d_multiagent_dynamics(num_agents=num_agents_eff, dt=dt)
        if is_multiagent
        else MPPIController.pointmass2d_dynamics(dt=dt)
    )
    return PointMassProblem(
        env=env,
        initial_state=initial_state,
        stl_specification=spec,
        dynamics_fn=dynamics_fn,
        control_low=control_low,
        control_high=control_high,
        control_noise_sigma=control_noise_sigma,
        control_dim=control_dim,
        num_agents=num_agents_eff,
        horizon=horizon,
        episode_steps=episode_steps,
    )


def _method_to_update_mode(method: str) -> str:
    aliases = {
        "stl_svpio": "svgd",
        "svgd": "svgd",
        "mppi": "importance_sampling",
        "importance_sampling": "importance_sampling",
        "svmpc": "svgd_fd",
        "svgd_fd": "svgd_fd",
        "dpi": "deterministic_pi",
        "deterministic_pi": "deterministic_pi",
    }
    if method not in aliases:
        raise ValueError(f"Unsupported controller method: {method}")
    return aliases[method]


def _rollout(dynamics_fn, initial_state: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
    def step_fn(x, u):
        x_next = dynamics_fn(x, u)
        return x_next, x_next

    _, trace = jax.lax.scan(step_fn, initial_state, controls)
    return trace


def _final_trace(problem: PointMassProblem, planned_controls: jnp.ndarray) -> jnp.ndarray:
    state_hist = _rollout(problem.dynamics_fn, problem.initial_state, planned_controls)
    states = jnp.concatenate([problem.initial_state[None, :], state_hist], axis=0)
    if states.shape[0] >= problem.horizon:
        return states[: problem.horizon]
    pad = jnp.repeat(states[-1][None, :], problem.horizon - states.shape[0], axis=0)
    return jnp.concatenate([states, pad], axis=0)


def run_pointmass_trial(
    task_id: str,
    method: str,
    config: dict[str, Any],
    seed: int,
    sampling_seed: Optional[int] = None,
    jit: bool = True,
) -> PointMassTrialResult:
    cfg = dict(config)
    cfg["task"] = cfg.get("task", task_id)
    sampling_seed = seed + 1000 if sampling_seed is None else int(sampling_seed)
    problem = build_pointmass_problem(cfg, seed=seed)
    approx_method = str(cfg.get("stl_approx_method", "true"))
    stl_temperature = cfg.get("stl_temperature", None)

    if method in {"stlcg_gradient_descent", "stl_gd"}:
        init = jax.random.uniform(
            jax.random.PRNGKey(sampling_seed),
            shape=(problem.horizon, problem.control_dim),
            minval=problem.control_low,
            maxval=problem.control_high,
        )
        start = time.perf_counter()
        controls, _, _ = run_stlcg_gradient_descent(
            STLCGGradientDescentConfig(
                num_steps=int(cfg.get("stl_gd_iters", cfg.get("svgd_iters", 200))),
                step_size=float(cfg.get("stl_gd_step_size", cfg.get("svgd_step_size", 0.05))),
                control_low=problem.control_low,
                control_high=problem.control_high,
                grad_clip_norm=cfg.get("stl_gd_grad_clip_norm", None),
            ),
            init,
            problem.initial_state,
            problem.dynamics_fn,
            problem.stl_specification,
            approx_method=approx_method,
            temperature=stl_temperature,
        )
        jax.block_until_ready(controls)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        num_iterations = int(cfg.get("stl_gd_iters", cfg.get("svgd_iters", 200)))
        num_particles = 1
    else:
        update_mode = _method_to_update_mode(method)
        controller_cfg = MPPIConfig(
            horizon=problem.horizon,
            num_samples=int(cfg.get("num_samples", 10)),
            control_dim=problem.control_dim,
            temperature=float(cfg.get("temperature", 0.8)),
            sampling_mode=str(cfg.get("sampling_mode", "uniform")),
            update_mode=update_mode,
            control_low=problem.control_low,
            control_high=problem.control_high,
            control_noise_sigma=problem.control_noise_sigma,
            svgd_iters=int(cfg.get("svgd_iters", 20)),
            svgd_step_size=float(cfg.get("svgd_step_size", 0.1)),
            svgd_step_size_anneal=str(cfg.get("svgd_step_anneal", "none")),
            svgd_step_size_final=cfg.get("svgd_step_final", None),
            svgd_repulsion_coef=float(cfg.get("svgd_repulsion_coef", 1.0)),
            svgd_repulsion_anneal=str(cfg.get("svgd_repulsion_anneal", "none")),
            svgd_repulsion_final=cfg.get("svgd_repulsion_final", None),
            svgd_selection_mode=str(cfg.get("svgd_selection_mode", "best")),
            svgd_resample_enabled=bool(cfg.get("svgd_resample_enabled", False)),
            svgd_resample_ess_threshold=float(cfg.get("svgd_resample_ess_threshold", 0.5)),
            svgd_resample_temperature=cfg.get("svgd_resample_temperature", None),
            svgd_resample_jitter_scale=float(cfg.get("svgd_resample_jitter_scale", 0.0)),
            svgd_fd_dirs=int(cfg.get("svgd_fd_dirs", 4)),
            svgd_fd_delta=float(cfg.get("svgd_fd_delta", 1e-2)),
            svgd_fd_alpha=cfg.get("svgd_fd_alpha", None),
            svgd_fd_use_kernel_repulsion=bool(cfg.get("svgd_fd_use_kernel_repulsion", True)),
            dpi_iters=cfg.get("dpi_iters", None),
            dpi_shrink_factor=float(cfg.get("dpi_shrink_factor", 0.95)),
            dpi_min_temperature=float(cfg.get("dpi_min_temperature", 1e-3)),
            dpi_augmented_mode=str(cfg.get("dpi_augmented_mode", "implicit")),
        )
        cost_fn = make_stl_cost_fn(problem.stl_specification, approx_method=approx_method, temperature=stl_temperature)
        controller = MPPIController(config=controller_cfg, dynamics_fn=problem.dynamics_fn, cost_fn=cost_fn)
        state = controller.init_state(jax.random.PRNGKey(sampling_seed))
        command = jax.jit(controller.command) if jit else controller.command
        start = time.perf_counter()
        _, _, info = command(state, problem.initial_state)
        jax.block_until_ready(info["selected_controls"])
        runtime_ms = (time.perf_counter() - start) * 1000.0
        controls = info["selected_controls"]
        num_iterations = int(controller_cfg.dpi_iters if update_mode == "deterministic_pi" and controller_cfg.dpi_iters else controller_cfg.svgd_iters)
        num_particles = int(controller_cfg.num_samples)

    trace = _final_trace(problem, controls[: problem.episode_steps])
    robustness = float(
        problem.stl_specification.robustness(trace, approx_method=approx_method, temperature=stl_temperature)
    )
    satisfied = bool(problem.stl_specification.eval(trace))
    return PointMassTrialResult(
        task_id=task_id,
        method=method,
        seed=seed,
        sampling_seed=sampling_seed,
        runtime_ms=float(runtime_ms),
        robustness=robustness,
        satisfied=satisfied,
        num_particles=num_particles,
        num_iterations=num_iterations,
    )


def summarize_trials(results: Iterable[PointMassTrialResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[PointMassTrialResult]] = {}
    for result in results:
        groups.setdefault((result.task_id, result.method), []).append(result)
    rows = []
    for (task_id, method), items in sorted(groups.items()):
        runtimes = [x.runtime_ms for x in items]
        robustness = [x.robustness for x in items]
        rows.append(
            {
                "config_id": task_id,
                "method": method,
                "n_success": len(items),
                "runtime_ms_mean": statistics.mean(runtimes),
                "runtime_ms_std": statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0,
                "robustness_mean": statistics.mean(robustness),
                "robustness_std": statistics.pstdev(robustness) if len(robustness) > 1 else 0.0,
                "satisfaction_rate": sum(1 for x in items if x.satisfied) / len(items),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "config_id",
        "method",
        "n_success",
        "runtime_ms_mean",
        "runtime_ms_std",
        "robustness_mean",
        "robustness_std",
        "satisfaction_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

