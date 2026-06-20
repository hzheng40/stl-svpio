from __future__ import annotations

from functools import reduce
from typing import List

import jax.numpy as jnp
from stljax.formula import Always, And, Eventually, Predicate

from .envs import LinearizedQuadrotorEnv, MultiAgentPointMass2DEnv, PointMass2DEnv


def _conjunction(formulas: List):
    if not formulas:
        return Predicate("true", lambda x: jnp.ones((x.shape[0],), dtype=x.dtype)) > 0.0
    return reduce(lambda a, b: And(a, b), formulas)


def _outside_box_trace(trace: jnp.ndarray, center: jnp.ndarray, half: jnp.ndarray, margin: float) -> jnp.ndarray:
    pos = trace[:, 3:6]
    return jnp.max(jnp.abs(pos - center) - (half + margin), axis=-1)


def _inside_box_trace(trace: jnp.ndarray, center: jnp.ndarray, half: jnp.ndarray, margin: float) -> jnp.ndarray:
    pos = trace[:, 3:6]
    return jnp.min((half - margin) - jnp.abs(pos - center), axis=-1)


def _outside_circle_trace(trace: jnp.ndarray, center: jnp.ndarray, radius: float, margin: float) -> jnp.ndarray:
    pos = trace[:, 0:2]
    return jnp.linalg.norm(pos - center, axis=-1) - (radius + margin)


def _inside_circle_trace(trace: jnp.ndarray, center: jnp.ndarray, radius: float, margin: float) -> jnp.ndarray:
    pos = trace[:, 0:2]
    return (radius - margin) - jnp.linalg.norm(pos - center, axis=-1)


def _joint_agent_pos_trace(trace: jnp.ndarray, agent_id: int) -> jnp.ndarray:
    start = 4 * int(agent_id)
    return trace[:, start : start + 2]


def _outside_circle_trace_joint(
    trace: jnp.ndarray,
    center: jnp.ndarray,
    radius: float,
    margin: float,
    agent_id: int,
) -> jnp.ndarray:
    pos = _joint_agent_pos_trace(trace, agent_id=agent_id)
    return jnp.linalg.norm(pos - center, axis=-1) - (radius + margin)


def _inside_circle_trace_joint(
    trace: jnp.ndarray,
    center: jnp.ndarray,
    radius: float,
    margin: float,
    agent_id: int,
) -> jnp.ndarray:
    pos = _joint_agent_pos_trace(trace, agent_id=agent_id)
    return (radius - margin) - jnp.linalg.norm(pos - center, axis=-1)


def _validate_joint_agent_id(env: MultiAgentPointMass2DEnv, agent_id: int) -> int:
    a = int(agent_id)
    if not 0 <= a < env.num_agents:
        raise ValueError(f"agent_id must be in [0, {env.num_agents - 1}], got {agent_id}.")
    return a


def _validate_zone_id(env: MultiAgentPointMass2DEnv, zone_id: int) -> int:
    z = int(zone_id)
    if not 0 <= z < env.zones.centers.shape[0]:
        raise ValueError(
            f"zone_id must be in [0, {env.zones.centers.shape[0] - 1}], got {zone_id}."
        )
    return z


def _validate_obstacle_id(env: MultiAgentPointMass2DEnv, obstacle_id: int) -> int:
    o = int(obstacle_id)
    if not 0 <= o < env.obstacles.centers.shape[0]:
        raise ValueError(
            f"obstacle_id must be in [0, {env.obstacles.centers.shape[0] - 1}], got {obstacle_id}."
        )
    return o


def _inside_square_trace_joint(
    trace: jnp.ndarray,
    center: jnp.ndarray,
    half_extent: float,
    agent_id: int,
) -> jnp.ndarray:
    pos = _joint_agent_pos_trace(trace, agent_id=agent_id)
    return jnp.min(half_extent - jnp.abs(pos - center), axis=-1)


def _inside_rectangle_trace_joint(
    trace: jnp.ndarray,
    center: jnp.ndarray,
    half_extents: jnp.ndarray,
    agent_id: int,
) -> jnp.ndarray:
    pos = _joint_agent_pos_trace(trace, agent_id=agent_id)
    return jnp.min(half_extents - jnp.abs(pos - center), axis=-1)


def _outside_agent_collision_trace(
    trace: jnp.ndarray,
    agent_a_id: int,
    agent_b_id: int,
    collision_radius: float,
) -> jnp.ndarray:
    pos_a = _joint_agent_pos_trace(trace, agent_id=agent_a_id)
    pos_b = _joint_agent_pos_trace(trace, agent_id=agent_b_id)
    return jnp.linalg.norm(pos_a - pos_b, axis=-1) - (2.0 * collision_radius)


def _outside_corridor_walls_trace_joint(
    trace: jnp.ndarray,
    corridor_center: jnp.ndarray,
    corridor_half_height: float,
    wall_half_thickness: float,
    agent_id: int,
) -> jnp.ndarray:
    """Positive when agent is outside corridor wall regions.

    Wall regions are the vertical slab |x-cx| < wall_half_thickness outside the
    opening |y-cy| <= corridor_half_height. The robust predicate encodes:
    outside_x OR inside_opening_y.
    """
    pos = _joint_agent_pos_trace(trace, agent_id=agent_id)
    center = jnp.asarray(corridor_center, dtype=pos.dtype)
    wall_half_thickness = jnp.asarray(wall_half_thickness, dtype=pos.dtype)
    corridor_half_height = jnp.asarray(corridor_half_height, dtype=pos.dtype)
    dx = jnp.abs(pos[:, 0] - center[0])
    dy = jnp.abs(pos[:, 1] - center[1])
    outside_x = dx - wall_half_thickness
    inside_opening_y = corridor_half_height - dy
    return jnp.maximum(outside_x, inside_opening_y)


def quad_always_outside_obstacles(
    env: LinearizedQuadrotorEnv,
    horizon: int,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    obs = env.get_obstacle_boxes()
    centers = obs["centers"]
    half_extents = obs["half_extents"]

    safe_formulas = []
    for i in range(centers.shape[0]):
        c_i = centers[i]
        h_i = half_extents[i]
        pred = Predicate(
            f"quad_outside_obstacle_{i}",
            lambda tr, c=c_i, h=h_i: _outside_box_trace(tr, c, h, margin),
        )
        safe_formulas.append(Always(pred > 0.0, interval=[0, horizon - 1]))
    return _conjunction(safe_formulas)


def quad_eventually_goal1_first_half(
    env: LinearizedQuadrotorEnv,
    horizon: int,
    stay_steps: int = 5,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    if env.goals.centers.shape[0] < 1:
        raise ValueError("Environment must contain at least one goal box.")

    first_half = horizon // 2
    latest_start = first_half - stay_steps
    if latest_start < 0:
        raise ValueError(
            f"Horizon {horizon} too short for stay_steps={stay_steps} in first half."
        )

    c0 = env.goals.centers[0]
    h0 = env.goals.half_extents[0]
    in_goal_1 = Predicate(
        "quad_in_goal_1",
        lambda tr, c=c0, h=h0: _inside_box_trace(tr, c, h, margin),
    )

    dwell_goal_1 = Always(in_goal_1 > 0.0, interval=[0, stay_steps - 1])
    return Eventually(dwell_goal_1, interval=[0, latest_start])


def quad_eventually_goal2_second_half(
    env: LinearizedQuadrotorEnv,
    horizon: int,
    stay_steps: int = 5,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    if env.goals.centers.shape[0] < 2:
        raise ValueError("Environment must contain at least two goal boxes.")

    second_half_start = horizon // 2
    latest_start = horizon - stay_steps
    if latest_start < second_half_start:
        raise ValueError(
            f"Horizon {horizon} too short for stay_steps={stay_steps} in second half."
        )

    c1 = env.goals.centers[1]
    h1 = env.goals.half_extents[1]
    in_goal_2 = Predicate(
        "quad_in_goal_2",
        lambda tr, c=c1, h=h1: _inside_box_trace(tr, c, h, margin),
    )

    dwell_goal_2 = Always(in_goal_2 > 0.0, interval=[0, stay_steps - 1])
    return Eventually(dwell_goal_2, interval=[second_half_start, latest_start])


def quad_full_task_spec(
    env: LinearizedQuadrotorEnv,
    horizon: int,
    stay_steps: int = 5,
    obstacle_margin: float = 0.0,
    goal_margin: float = 0.0,
):
    safe = quad_always_outside_obstacles(env, horizon=horizon, margin=obstacle_margin)
    goal1 = quad_eventually_goal1_first_half(
        env,
        horizon=horizon,
        stay_steps=stay_steps,
        margin=goal_margin,
    )
    goal2 = quad_eventually_goal2_second_half(
        env,
        horizon=horizon,
        stay_steps=stay_steps,
        margin=goal_margin,
    )
    return _conjunction([safe, goal1, goal2])


def pointmass_always_outside_obstacles(
    env: PointMass2DEnv,
    horizon: int,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    obs = env.get_obstacle_circles()
    centers = obs["centers"]
    radii = obs["radii"]

    safe_formulas = []
    for i in range(centers.shape[0]):
        c_i = centers[i]
        r_i = float(radii[i])
        pred = Predicate(
            f"pointmass_outside_obstacle_{i}",
            lambda tr, c=c_i, r=r_i: _outside_circle_trace(tr, c, r, margin),
        )
        safe_formulas.append(Always(pred > 0.0, interval=[0, horizon - 1]))
    return _conjunction(safe_formulas)


def pointmass_eventually_zone1_first_half(
    env: PointMass2DEnv,
    horizon: int,
    stay_steps: int = 5,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    if env.zones.centers.shape[0] < 1:
        raise ValueError("Environment must contain at least one zone.")

    first_half = horizon // 2
    latest_start = first_half - stay_steps
    if latest_start < 0:
        raise ValueError(
            f"Horizon {horizon} too short for stay_steps={stay_steps} in first half."
        )

    c0 = env.zones.centers[0]
    r0 = float(env.zones.radii[0])
    in_zone_1 = Predicate(
        "pointmass_in_zone_1",
        lambda tr, c=c0, r=r0: _inside_circle_trace(tr, c, r, margin),
    )

    dwell_zone_1 = Always(in_zone_1 > 0.0, interval=[0, stay_steps - 1])
    return Eventually(dwell_zone_1, interval=[0, latest_start])


def pointmass_eventually_zone2_second_half(
    env: PointMass2DEnv,
    horizon: int,
    stay_steps: int = 5,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    if env.zones.centers.shape[0] < 2:
        raise ValueError("Environment must contain at least two zones.")

    second_half_start = horizon // 2
    latest_start = horizon - stay_steps
    if latest_start < second_half_start:
        raise ValueError(
            f"Horizon {horizon} too short for stay_steps={stay_steps} in second half."
        )

    c1 = env.zones.centers[1]
    r1 = float(env.zones.radii[1])
    in_zone_2 = Predicate(
        "pointmass_in_zone_2",
        lambda tr, c=c1, r=r1: _inside_circle_trace(tr, c, r, margin),
    )

    dwell_zone_2 = Always(in_zone_2 > 0.0, interval=[0, stay_steps - 1])
    return Eventually(dwell_zone_2, interval=[second_half_start, latest_start])


def pointmass_full_task_spec(
    env: PointMass2DEnv,
    horizon: int,
    stay_steps: int = 5,
    obstacle_margin: float = 0.0,
    zone_margin: float = 0.0,
):
    safe = pointmass_always_outside_obstacles(env, horizon=horizon, margin=obstacle_margin)
    zone1 = pointmass_eventually_zone1_first_half(
        env,
        horizon=horizon,
        stay_steps=stay_steps,
        margin=zone_margin,
    )
    zone2 = pointmass_eventually_zone2_second_half(
        env,
        horizon=horizon,
        stay_steps=stay_steps,
        margin=zone_margin,
    )
    return _conjunction([safe, zone1, zone2])


def quadruped_reach_avoid_periodic_gait_spec(
    horizon: int,
    goal_xy: jnp.ndarray,
    obstacle_xy: jnp.ndarray,
    goal_radius: float = 0.35,
    obstacle_radius: float = 0.3,
    robot_margin: float = 0.3,
    z_min: float = 0.20,
    z_max: float = 0.55,
    tilt_limit: float = 0.7,
    gait_period_steps: int = 20,
    gait_amplitude: float = 0.55,
    gait_offset: float = -1.35,
    gait_eps: float = 0.8,
    foot_contact_z: float = 0.04,
    foot_lift_margin: float = 0.02,
    diagonal_sync_eps: float = 0.02,
):
    """Quadruped reach-avoid with periodic trot gait.

    Expected trace layout per timestep:
    [base_x, base_y, base_z, roll, pitch,
     FR_thigh, FR_calf, FL_thigh, FL_calf, RR_thigh, RR_calf, RL_thigh, RL_calf,
     FR_foot_z, FL_foot_z, RR_foot_z, RL_foot_z]
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if gait_period_steps <= 0:
        raise ValueError("gait_period_steps must be positive.")
    if goal_radius <= 0 or obstacle_radius <= 0 or robot_margin < 0:
        raise ValueError("goal_radius and obstacle_radius must be positive; robot_margin must be non-negative.")
    if z_min >= z_max:
        raise ValueError("z_min must be less than z_max.")
    if tilt_limit <= 0:
        raise ValueError("tilt_limit must be positive.")
    if foot_contact_z <= 0:
        raise ValueError("foot_contact_z must be positive.")
    if foot_lift_margin <= 0:
        raise ValueError("foot_lift_margin must be positive.")
    if diagonal_sync_eps <= 0:
        raise ValueError("diagonal_sync_eps must be positive.")

    goal_xy = jnp.asarray(goal_xy, dtype=jnp.float32)
    obstacle_xy = jnp.asarray(obstacle_xy, dtype=jnp.float32)
    avoid_radius = obstacle_radius + robot_margin

    reach = Predicate(
        "quadruped_reach_goal",
        lambda tr, g=goal_xy, r=goal_radius: r - jnp.linalg.norm(tr[:, 0:2] - g, axis=-1),
    )
    avoid = Predicate(
        "quadruped_avoid_obstacle",
        lambda tr, c=obstacle_xy, r=avoid_radius: jnp.linalg.norm(tr[:, 0:2] - c, axis=-1) - r,
    )
    upright = Predicate(
        "quadruped_upright_safe",
        lambda tr, lo=z_min, hi=z_max, tl=tilt_limit: jnp.min(
            jnp.stack(
                [
                    tr[:, 2] - lo,
                    hi - tr[:, 2],
                    tl - jnp.abs(tr[:, 3]),
                    tl - jnp.abs(tr[:, 4]),
                ],
                axis=-1,
            ),
            axis=-1,
        ),
    )

    gait = Predicate(
        "quadruped_periodic_trot",
        lambda tr, p=gait_period_steps, a=gait_amplitude, o=gait_offset, eps=gait_eps: (
            eps
            - jnp.mean(
                jnp.abs(
                    tr[:, 5:13]
                    - jnp.stack(
                        [
                            o + a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p),
                            o - a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p),
                            o + a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p + jnp.pi),
                            o - a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p + jnp.pi),
                            o + a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p + jnp.pi),
                            o - a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p + jnp.pi),
                            o + a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p),
                            o - a * jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p),
                        ],
                        axis=-1,
                    )
                ),
                axis=-1,
            )
        ),
    )
    contact_gait = Predicate(
        "quadruped_diagonal_contact_cycle",
        lambda tr, p=gait_period_steps, zc=foot_contact_z, lm=foot_lift_margin, se=diagonal_sync_eps: (
            jnp.where(
                jnp.sin(2.0 * jnp.pi * jnp.arange(tr.shape[0], dtype=tr.dtype) / p) >= 0.0,
                jnp.minimum(
                    jnp.minimum(
                        jnp.minimum(zc - tr[:, 13], zc - tr[:, 16]),
                        jnp.minimum(
                            tr[:, 14] - (zc + lm),
                            tr[:, 15] - (zc + lm),
                        ),
                    ),
                    se - jnp.abs(tr[:, 13] - tr[:, 16]),
                ),
                jnp.minimum(
                    jnp.minimum(
                        jnp.minimum(zc - tr[:, 14], zc - tr[:, 15]),
                        jnp.minimum(
                            tr[:, 13] - (zc + lm),
                            tr[:, 16] - (zc + lm),
                        ),
                    ),
                    se - jnp.abs(tr[:, 14] - tr[:, 15]),
                ),
            )
        ),
    )

    return _conjunction(
        [
            Eventually(reach > 0.0, interval=[0, horizon - 1]),
            Always(avoid > 0.0, interval=[0, horizon - 1]),
            Always(upright > 0.0, interval=[0, horizon - 1]),
            Always(gait > 0.0, interval=[0, horizon - 1]),
            Always(contact_gait > 0.0, interval=[0, horizon - 1]),
        ]
    )


def quadruped_teleop_tracking_spec(
    horizon: int,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw_rate: float,
    warmup_steps: int = 60,
    hold_steps: int = 90,
    vx_tol: float = 0.20,
    vy_tol: float = 0.20,
    yaw_rate_tol: float = 0.30,
    z_min: float = 0.20,
    z_max: float = 0.55,
    roll_limit: float = 0.6,
    pitch_limit: float = 0.6,
):
    """Quadruped teleop command-tracking STL specification.

    Expected trace layout per timestep:
    [z, roll, pitch, vx_body, vy_body, wz, cmd_vx, cmd_vy, cmd_yaw_rate]
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive.")
    if warmup_steps >= horizon:
        raise ValueError("warmup_steps must be smaller than horizon.")
    if hold_steps > horizon:
        raise ValueError("hold_steps must be less than or equal to horizon.")
    if vx_tol <= 0 or vy_tol <= 0 or yaw_rate_tol <= 0:
        raise ValueError("tracking tolerances must be positive.")
    if z_min >= z_max:
        raise ValueError("z_min must be less than z_max.")
    if roll_limit <= 0 or pitch_limit <= 0:
        raise ValueError("roll_limit and pitch_limit must be positive.")

    latest_start = horizon - hold_steps
    if warmup_steps > latest_start:
        raise ValueError(
            "warmup_steps leaves no room for the requested hold_steps within the horizon."
        )

    cmd_vx = jnp.asarray(cmd_vx, dtype=jnp.float32)
    cmd_vy = jnp.asarray(cmd_vy, dtype=jnp.float32)
    cmd_yaw_rate = jnp.asarray(cmd_yaw_rate, dtype=jnp.float32)

    healthy = Predicate(
        "quadruped_teleop_healthy",
        lambda tr, lo=z_min, hi=z_max, rl=roll_limit, pl=pitch_limit: jnp.min(
            jnp.stack(
                [
                    tr[:, 0] - lo,
                    hi - tr[:, 0],
                    rl - jnp.abs(tr[:, 1]),
                    pl - jnp.abs(tr[:, 2]),
                ],
                axis=-1,
            ),
            axis=-1,
        ),
    )
    tracking = Predicate(
        "quadruped_teleop_tracking",
        lambda tr, vx=cmd_vx, vy=cmd_vy, wz=cmd_yaw_rate, vx_eps=vx_tol, vy_eps=vy_tol, wz_eps=yaw_rate_tol: jnp.min(
            jnp.stack(
                [
                    vx_eps - jnp.abs(tr[:, 3] - vx),
                    vy_eps - jnp.abs(tr[:, 4] - vy),
                    wz_eps - jnp.abs(tr[:, 5] - wz),
                ],
                axis=-1,
            ),
            axis=-1,
        ),
    )

    tracking_hold = Always(tracking > 0.0, interval=[0, hold_steps - 1])
    return _conjunction(
        [
            Always(healthy > 0.0, interval=[0, horizon - 1]),
            Eventually(tracking_hold, interval=[warmup_steps, latest_start]),
        ]
    )


def quadruped_backflip_spec(
    horizon: int,
    completion_style: str = "terminal",
    target_rotation: float = 2.0 * jnp.pi,
    rotation_tolerance: float = 0.35,
    final_window_steps: int = 8,
    max_abs_pitch_rate: float = 1.2,
    stabilization_steps: int = 4,
    stabilization_pitch_rate: float = 0.5,
):
    """Quadruped backflip STL specification.

    Expected trace layout per timestep:
    [base_x, base_y, base_z, roll, pitch,
     FR_thigh, FR_calf, FL_thigh, FL_calf, RR_thigh, RR_calf, RL_thigh, RL_calf,
     FR_foot_z, FL_foot_z, RR_foot_z, RL_foot_z]

    ``completion_style`` options:
    - ``terminal``: complete one backward rotation near the end.
    - ``staged``: pass quarter-turn checkpoints in order, then complete.
    - ``stabilized``: complete and hold near target with low angular rate.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if final_window_steps <= 0:
        raise ValueError("final_window_steps must be positive.")
    if rotation_tolerance <= 0:
        raise ValueError("rotation_tolerance must be positive.")
    if target_rotation <= 0:
        raise ValueError("target_rotation must be positive.")
    if max_abs_pitch_rate <= 0:
        raise ValueError("max_abs_pitch_rate must be positive.")
    if stabilization_steps <= 0:
        raise ValueError("stabilization_steps must be positive.")
    if stabilization_pitch_rate <= 0:
        raise ValueError("stabilization_pitch_rate must be positive.")

    completion_style = str(completion_style).lower()
    if completion_style not in {"terminal", "staged", "stabilized"}:
        raise ValueError(
            "completion_style must be one of "
            "['terminal', 'staged', 'stabilized'], "
            f"got {completion_style!r}."
        )

    final_start = max(0, horizon - final_window_steps)
    backward_rotation = lambda tr: -(tr[:, 4] - tr[0, 4])
    pitch_rate = lambda tr: jnp.concatenate(
        [jnp.zeros((1,), dtype=tr.dtype), jnp.diff(tr[:, 4])]
    )

    base_predicates = [
        Always(
            Predicate(
                "quadruped_pitch_rate_bounded",
                lambda tr, w=max_abs_pitch_rate: w - jnp.abs(pitch_rate(tr)),
            )
            > 0.0,
            interval=[0, horizon - 1],
        ),
    ]

    completed = Predicate(
        "quadruped_completed_backflip",
        lambda tr, target=target_rotation, tol=rotation_tolerance: (
            backward_rotation(tr) - (target - tol)
        ),
    )
    completion = Eventually(completed > 0.0, interval=[final_start, horizon - 1])

    if completion_style == "terminal":
        return _conjunction(base_predicates + [completion])

    if completion_style == "staged":
        q1 = Predicate(
            "quadruped_backflip_quarter_1",
            lambda tr: backward_rotation(tr) - (0.5 * jnp.pi),
        )
        q2 = Predicate(
            "quadruped_backflip_quarter_2",
            lambda tr: backward_rotation(tr) - jnp.pi,
        )
        q3 = Predicate(
            "quadruped_backflip_quarter_3",
            lambda tr: backward_rotation(tr) - (1.5 * jnp.pi),
        )
        staged_progress = Eventually(
            q1 > 0.0,
            interval=[0, horizon - 1],
        )
        staged_progress = And(
            staged_progress,
            Eventually(
                And(
                    q2 > 0.0,
                    Eventually(And(q3 > 0.0, completion), interval=[0, horizon - 1]),
                ),
                interval=[0, horizon - 1],
            ),
        )
        return _conjunction(base_predicates + [staged_progress])

    # stabilized
    in_target_band = Predicate(
        "quadruped_rotation_in_target_band",
        lambda tr, target=target_rotation, tol=rotation_tolerance: tol
        - jnp.abs(backward_rotation(tr) - target),
    )
    low_pitch_rate = Predicate(
        "quadruped_low_pitch_rate",
        lambda tr, w=stabilization_pitch_rate: w - jnp.abs(pitch_rate(tr)),
    )
    hold = And(
        Always(in_target_band > 0.0, interval=[0, stabilization_steps - 1]),
        Always(low_pitch_rate > 0.0, interval=[0, stabilization_steps - 1]),
    )
    hold_latest_start = max(0, horizon - stabilization_steps)
    hold_near_end = Eventually(hold, interval=[final_start, hold_latest_start])
    return _conjunction(base_predicates + [completion, hold_near_end])


def crazyflie_backflip_spec(
    horizon: int,
    completion_style: str = "staged",
    target_rotation: float = 2.0 * jnp.pi,
    rotation_tolerance: float = 0.35,
    final_window_steps: int = 50,
    min_height: float = 0.10,
    max_height: float = 4.0,
    max_abs_pitch_rate: float = 40.0,
    stabilization_steps: int = 25,
    stabilization_pitch_rate: float = 6.0,
    timestep: float = 0.002,
    phased_total_time: float = 2.0,
    phased_accel_end_time: float = 0.5,
    phased_rotation_start_end_time: float = 0.8,
    phased_coast_end_time: float = 1.4,
    phased_stop_end_time: float = 1.7,
    phased_min_height: float = 0.5,
    phased_collective_thrust_high: float = 0.7,
    phased_collective_thrust_low: float = -0.2,
    phased_collective_thrust_recover_high: float = 0.7,
    phased_pitch_diff_pos_high: float = 0.8,
    phased_pitch_diff_neg_high: float = 0.8,
    phased_climb_rate_min: float = 0.1,
    phased_pitch_rate_accel_min: float = 0.1,
    phased_pitch_rate_start_min: float = 1.0,
    phased_pitch_rate_coast_min: float = 1.0,
    phased_pitch_rate_stop_max: float = 0.5,
    phased_pitch_rate_recover_max: float = 0.5,
    phased_theta_stop_eps: float = 0.4,
    phased_theta_recover_eps: float = 0.2,
    phased_recovery_zdot_eps: float = 0.25,
    phased_start_hold_steps: int = 10,
    phased_coast_hold_steps: int = 25,
    phased_stop_hold_steps: int = 10,
    phased_recover_hold_steps: int = 25,
):
    """Crazyflie backflip STL specification.

    Expected trace layout per timestep:
    [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, roll, pitch, yaw,
     theta_unwrapped, collective_thrust_cmd, pitch_diff_cmd]

    ``completion_style`` options:
    - ``terminal``: complete one backward rotation near the end.
    - ``staged``: pass quarter-turn checkpoints in order, then complete.
    - ``stabilized``: complete and hold near target with low pitch rate.
    - ``phased``: satisfy a five-phase backflip using altitude, pitch state, and normalized thrust/pitch control commands.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if final_window_steps <= 0:
        raise ValueError("final_window_steps must be positive.")
    if rotation_tolerance <= 0:
        raise ValueError("rotation_tolerance must be positive.")
    if target_rotation <= 0:
        raise ValueError("target_rotation must be positive.")
    if min_height < 0:
        raise ValueError("min_height must be non-negative.")
    if max_height <= min_height:
        raise ValueError("max_height must be greater than min_height.")
    if max_abs_pitch_rate <= 0:
        raise ValueError("max_abs_pitch_rate must be positive.")
    if stabilization_steps <= 0:
        raise ValueError("stabilization_steps must be positive.")
    if stabilization_pitch_rate <= 0:
        raise ValueError("stabilization_pitch_rate must be positive.")
    if timestep <= 0:
        raise ValueError("timestep must be positive.")
    if phased_total_time <= 0:
        raise ValueError("phased_total_time must be positive.")
    if phased_accel_end_time < 0:
        raise ValueError("phased_accel_end_time must be non-negative.")
    if phased_rotation_start_end_time <= phased_accel_end_time:
        raise ValueError("phased_rotation_start_end_time must be greater than phased_accel_end_time.")
    if phased_coast_end_time <= phased_rotation_start_end_time:
        raise ValueError("phased_coast_end_time must be greater than phased_rotation_start_end_time.")
    if phased_stop_end_time <= phased_coast_end_time:
        raise ValueError("phased_stop_end_time must be greater than phased_coast_end_time.")
    if phased_total_time < phased_stop_end_time:
        raise ValueError("phased_total_time must be greater than or equal to phased_stop_end_time.")
    if phased_min_height < 0:
        raise ValueError("phased_min_height must be non-negative.")
    if phased_pitch_rate_accel_min < 0:
        raise ValueError("phased_pitch_rate_accel_min must be non-negative.")
    if phased_pitch_diff_neg_high <= 0:
        raise ValueError("phased_pitch_diff_neg_high must be positive.")
    if phased_theta_stop_eps <= 0:
        raise ValueError("phased_theta_stop_eps must be positive.")
    if phased_theta_recover_eps <= 0:
        raise ValueError("phased_theta_recover_eps must be positive.")
    if phased_start_hold_steps <= 0:
        raise ValueError("phased_start_hold_steps must be positive.")
    if phased_coast_hold_steps <= 0:
        raise ValueError("phased_coast_hold_steps must be positive.")
    if phased_stop_hold_steps <= 0:
        raise ValueError("phased_stop_hold_steps must be positive.")
    if phased_recover_hold_steps <= 0:
        raise ValueError("phased_recover_hold_steps must be positive.")

    completion_style = str(completion_style).lower()
    if completion_style not in {"terminal", "staged", "stabilized", "phased"}:
        raise ValueError(
            "completion_style must be one of "
            "['terminal', 'staged', 'stabilized', 'phased'], "
            f"got {completion_style!r}."
        )

    final_start = max(0, horizon - final_window_steps)
    backward_rotation = lambda tr: -(tr[:, 14] - tr[0, 14])

    base_predicates = [
        Always(
            Predicate(
                "crazyflie_altitude_above_min",
                lambda tr, zmin=min_height: tr[:, 2] - zmin,
            )
            > 0.0,
            interval=[0, horizon - 1],
        ),
    ]
    if completion_style == "stabilized":
        base_predicates.extend(
            [
                Always(
                    Predicate(
                        "crazyflie_altitude_below_max",
                        lambda tr, zmax=max_height: zmax - tr[:, 2],
                    )
                    > 0.0,
                    interval=[0, horizon - 1],
                ),
                Always(
                    Predicate(
                        "crazyflie_pitch_rate_bounded",
                        lambda tr, wmax=max_abs_pitch_rate: wmax - jnp.abs(tr[:, 11]),
                    )
                    > 0.0,
                    interval=[0, horizon - 1],
                ),
            ]
        )

    completed = Predicate(
        "crazyflie_completed_backflip",
        lambda tr, target=target_rotation, tol=rotation_tolerance: (
            backward_rotation(tr) - (target - tol)
        ),
    )
    completion = Eventually(completed > 0.0, interval=[final_start, horizon - 1])

    if completion_style == "terminal":
        return _conjunction(base_predicates + [completion])

    if completion_style == "phased":
        total_steps = min(horizon, max(1, int(round(phased_total_time / timestep))))
        accel_end_step = min(max(int(round(phased_accel_end_time / timestep)), 0), total_steps - 1)
        rotation_start_end_step = min(
            max(int(round(phased_rotation_start_end_time / timestep)), accel_end_step + 1),
            total_steps - 1,
        )
        coast_end_step = min(
            max(int(round(phased_coast_end_time / timestep)), rotation_start_end_step + 1),
            total_steps - 1,
        )
        stop_end_step = min(
            max(int(round(phased_stop_end_time / timestep)), coast_end_step + 1),
            total_steps - 1,
        )

        def _window_latest_start(start: int, end: int, hold_steps: int) -> int:
            return max(start, end - hold_steps + 1)

        theta_signal = lambda tr: tr[:, 16]
        collective_cmd = lambda tr: tr[:, 17]
        pitch_diff_cmd = lambda tr: tr[:, 18]
        z_dot = lambda tr: tr[:, 9]
        pitch_rate = lambda tr: tr[:, 11]

        safety = Always(
            Predicate(
                "crazyflie_phased_altitude_above_min",
                lambda tr, zmin=phased_min_height: tr[:, 2] - zmin,
            )
            >= 0.0,
            interval=[0, total_steps - 1],
        )

        accelerate = _conjunction(
            [
                Always(
                    Predicate(
                        "crazyflie_phased_accel_collective_high",
                        lambda tr, u=phased_collective_thrust_high: collective_cmd(tr) - u,
                    )
                    >= 0.0,
                    interval=[0, accel_end_step],
                ),
                Always(
                    Predicate(
                        "crazyflie_phased_accel_climb_rate_min",
                        lambda tr, v=phased_climb_rate_min: z_dot(tr) - v,
                    )
                    >= 0.0,
                    interval=[0, accel_end_step],
                ),
                Always(
                    Predicate(
                        "crazyflie_phased_accel_pitch_rate_positive",
                        lambda tr, w=phased_pitch_rate_accel_min: pitch_rate(tr) - w,
                    )
                    >= 0.0,
                    interval=[0, accel_end_step],
                ),
            ]
        )

        start_rotation_core = _conjunction(
            [
                Predicate(
                    "crazyflie_phased_start_rotation_diff_positive",
                    lambda tr, u=phased_pitch_diff_pos_high: pitch_diff_cmd(tr) - u,
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_start_rotation_pitch_rate_high",
                    lambda tr, w=phased_pitch_rate_start_min: pitch_rate(tr) - w,
                )
                >= 0.0,
            ]
        )
        start_rotation = Eventually(
            Always(start_rotation_core, interval=[0, phased_start_hold_steps - 1]),
            interval=[
                accel_end_step,
                _window_latest_start(accel_end_step, rotation_start_end_step, phased_start_hold_steps),
            ],
        )

        coast_core = _conjunction(
            [
                Predicate(
                    "crazyflie_phased_coast_collective_low",
                    lambda tr, u=phased_collective_thrust_low: u - collective_cmd(tr),
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_coast_pitch_rate_high",
                    lambda tr, w=phased_pitch_rate_coast_min: pitch_rate(tr) - w,
                )
                >= 0.0,
            ]
        )
        coast = Eventually(
            Always(coast_core, interval=[0, phased_coast_hold_steps - 1]),
            interval=[
                rotation_start_end_step,
                _window_latest_start(rotation_start_end_step, coast_end_step, phased_coast_hold_steps),
            ],
        )

        stop_rotation_core = _conjunction(
            [
                Predicate(
                    "crazyflie_phased_stop_rotation_diff_negative",
                    lambda tr, u=phased_pitch_diff_neg_high: -pitch_diff_cmd(tr) - u,
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_stop_rotation_pitch_rate_low",
                    lambda tr, w=phased_pitch_rate_stop_max: w - jnp.abs(pitch_rate(tr)),
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_stop_rotation_near_full_turn",
                    lambda tr, eps=phased_theta_stop_eps: eps - jnp.abs(theta_signal(tr) - (2.0 * jnp.pi)),
                )
                >= 0.0,
            ]
        )
        stop_rotation = Eventually(
            Always(stop_rotation_core, interval=[0, phased_stop_hold_steps - 1]),
            interval=[
                coast_end_step,
                _window_latest_start(coast_end_step, stop_end_step, phased_stop_hold_steps),
            ],
        )

        recover_core = _conjunction(
            [
                Predicate(
                    "crazyflie_phased_recover_theta_near_full_turn",
                    lambda tr, eps=phased_theta_recover_eps: eps - jnp.abs(theta_signal(tr) - (2.0 * jnp.pi)),
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_recover_collective_high",
                    lambda tr, u=phased_collective_thrust_recover_high: collective_cmd(tr) - u,
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_recover_vertical_speed_small",
                    lambda tr, eps=phased_recovery_zdot_eps: eps - jnp.abs(z_dot(tr)),
                )
                >= 0.0,
                Predicate(
                    "crazyflie_phased_recover_pitch_rate_small",
                    lambda tr, w=phased_pitch_rate_recover_max: w - jnp.abs(pitch_rate(tr)),
                )
                >= 0.0,
            ]
        )
        recover = Eventually(
            Always(recover_core, interval=[0, phased_recover_hold_steps - 1]),
            interval=[
                stop_end_step,
                _window_latest_start(stop_end_step, total_steps - 1, phased_recover_hold_steps),
            ],
        )

        return _conjunction(
            [
                safety,
                accelerate,
                start_rotation,
                coast,
                stop_rotation,
                recover,
            ]
        )

    if completion_style == "staged":
        q1 = Predicate(
            "crazyflie_backflip_quarter_1",
            lambda tr: backward_rotation(tr) - (0.5 * jnp.pi),
        )
        q2 = Predicate(
            "crazyflie_backflip_quarter_2",
            lambda tr: backward_rotation(tr) - jnp.pi,
        )
        q3 = Predicate(
            "crazyflie_backflip_quarter_3",
            lambda tr: backward_rotation(tr) - (1.5 * jnp.pi),
        )
        staged_progress = Eventually(q1 > 0.0, interval=[0, horizon - 1])
        staged_progress = And(
            staged_progress,
            Eventually(
                And(
                    q2 > 0.0,
                    Eventually(And(q3 > 0.0, completion), interval=[0, horizon - 1]),
                ),
                interval=[0, horizon - 1],
            ),
        )
        return _conjunction(base_predicates + [staged_progress])

    in_target_band = Predicate(
        "crazyflie_rotation_in_target_band",
        lambda tr, target=target_rotation, tol=rotation_tolerance: tol
        - jnp.abs(backward_rotation(tr) - target),
    )
    low_pitch_rate = Predicate(
        "crazyflie_low_pitch_rate",
        lambda tr, w=stabilization_pitch_rate: w - jnp.abs(tr[:, 11]),
    )
    hold = And(
        Always(in_target_band > 0.0, interval=[0, stabilization_steps - 1]),
        Always(low_pitch_rate > 0.0, interval=[0, stabilization_steps - 1]),
    )
    hold_latest_start = max(0, horizon - stabilization_steps)
    hold_near_end = Eventually(hold, interval=[final_start, hold_latest_start])
    return _conjunction(base_predicates + [completion, hold_near_end])


def halfcheetah_backflip_spec(
    horizon: int,
    completion_style: str = "terminal",
    target_rotation: float = 2.0 * jnp.pi,
    rotation_tolerance: float = 0.35,
    final_window_steps: int = 8,
    min_torso_height: float = -0.6,
    max_abs_pitch_rate: float = 40.0,
    stabilization_steps: int = 4,
    stabilization_pitch_rate: float = 6.0,
):
    """HalfCheetah backflip STL specification.

    Expected trace layout per timestep:
    [root_x, root_z, root_pitch, root_xd, root_zd, root_pitchd, joint_0..joint_5]

    ``completion_style`` options:
    - ``terminal``: complete one backward rotation near the end.
    - ``staged``: pass quarter-turn checkpoints in order, then complete.
    - ``stabilized``: complete and hold near target with low angular rate.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if final_window_steps <= 0:
        raise ValueError("final_window_steps must be positive.")
    if rotation_tolerance <= 0:
        raise ValueError("rotation_tolerance must be positive.")
    if target_rotation <= 0:
        raise ValueError("target_rotation must be positive.")
    if max_abs_pitch_rate <= 0:
        raise ValueError("max_abs_pitch_rate must be positive.")
    if stabilization_steps <= 0:
        raise ValueError("stabilization_steps must be positive.")
    if stabilization_pitch_rate <= 0:
        raise ValueError("stabilization_pitch_rate must be positive.")

    completion_style = str(completion_style).lower()
    if completion_style not in {"terminal", "staged", "stabilized", "feet_head_order"}:
        raise ValueError(
            "completion_style must be one of "
            "['terminal', 'staged', 'stabilized', 'feet_head_order'], "
            f"got {completion_style!r}."
        )

    final_start = max(0, horizon - final_window_steps)
    base_predicates = [
        Always(
            Predicate(
                "halfcheetah_torso_height_safe",
                lambda tr, zmin=min_torso_height: tr[:, 1] - zmin,
            )
            > 0.0,
            interval=[0, horizon - 1],
        ),
        Always(
            Predicate(
                "halfcheetah_pitch_rate_bounded",
                lambda tr, wmax=max_abs_pitch_rate: wmax - jnp.abs(tr[:, 5]),
            )
            > 0.0,
            interval=[0, horizon - 1],
        ),
    ]

    completed = Predicate(
        "halfcheetah_completed_backflip",
        lambda tr, target=target_rotation, tol=rotation_tolerance: (
            -(tr[:, 2] - tr[0, 2]) - (target - tol)
        ),
    )
    completion = Eventually(completed > 0.0, interval=[final_start, horizon - 1])

    if completion_style == "terminal":
        return _conjunction(base_predicates + [completion])

    if completion_style == "staged":
        q1 = Predicate(
            "halfcheetah_backflip_quarter_1",
            lambda tr: -(tr[:, 2] - tr[0, 2]) - (0.5 * jnp.pi),
        )
        q2 = Predicate(
            "halfcheetah_backflip_quarter_2",
            lambda tr: -(tr[:, 2] - tr[0, 2]) - jnp.pi,
        )
        q3 = Predicate(
            "halfcheetah_backflip_quarter_3",
            lambda tr: -(tr[:, 2] - tr[0, 2]) - (1.5 * jnp.pi),
        )
        staged_progress = Eventually(
            q1 > 0.0,
            interval=[
                0,
                horizon - 1,
            ],
        )
        staged_progress = And(
            staged_progress,
            Eventually(
                And(
                    q2 > 0.0,
                    Eventually(And(q3 > 0.0, completion), interval=[0, horizon - 1]),
                ),
                interval=[0, horizon - 1],
            ),
        )
        return _conjunction(base_predicates + [staged_progress])

    if completion_style == "feet_head_order":
        # Uses trace layout:
        # [root_x, root_z, root_pitch, root_xd, root_zd, root_pitchd, joints..., bfoot_z, ffoot_z]
        feet_above_head = Predicate(
            "halfcheetah_feet_above_head",
            lambda tr: jnp.minimum(tr[:, -2] - tr[:, 1], tr[:, -1] - tr[:, 1]),
        )
        head_above_feet = Predicate(
            "halfcheetah_head_above_feet",
            lambda tr: tr[:, 1] - jnp.maximum(tr[:, -2], tr[:, -1]),
        )
        ordered_inversion = Eventually(
            And(
                feet_above_head > 0.0,
                Eventually(head_above_feet > 0.0, interval=[0, horizon - 1]),
            ),
            interval=[0, horizon - 1],
        )
        return _conjunction(base_predicates + [ordered_inversion])

    # stabilized
    in_target_band = Predicate(
        "halfcheetah_rotation_in_target_band",
        lambda tr, target=target_rotation, tol=rotation_tolerance: tol
        - jnp.abs((-(tr[:, 2] - tr[0, 2])) - target),
    )
    low_pitch_rate = Predicate(
        "halfcheetah_low_pitch_rate",
        lambda tr, w=stabilization_pitch_rate: w - jnp.abs(tr[:, 5]),
    )
    hold = And(
        Always(in_target_band > 0.0, interval=[0, stabilization_steps - 1]),
        Always(low_pitch_rate > 0.0, interval=[0, stabilization_steps - 1]),
    )
    hold_latest_start = max(0, horizon - stabilization_steps)
    hold_near_end = Eventually(hold, interval=[final_start, hold_latest_start])
    return _conjunction(base_predicates + [completion, hold_near_end])


def pointmass_visit_all_zones_task_spec(
    env: PointMass2DEnv,
    horizon: int,
    stay_steps: int = 1,
    obstacle_margin: float = 0.0,
    zone_margin: float = 0.0,
):
    """Single-agent task: eventually visit all zones (order-free)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    num_zones = int(env.zones.centers.shape[0])
    if num_zones <= 0:
        raise ValueError("Environment must contain at least one zone.")
    latest_start = horizon - stay_steps
    if latest_start < 0:
        raise ValueError(f"Horizon {horizon} too short for stay_steps={stay_steps}.")

    formulas = [
        pointmass_always_outside_obstacles(env, horizon=horizon, margin=obstacle_margin),
    ]

    for zone_id in range(num_zones):
        c_i = env.zones.centers[zone_id]
        r_i = float(env.zones.radii[zone_id])
        in_zone_i = Predicate(
            f"pointmass_in_zone_{zone_id}",
            lambda tr, c=c_i, r=r_i: _inside_circle_trace(tr, c, r, zone_margin),
        )
        dwell_i = Always(in_zone_i > 0.0, interval=[0, stay_steps - 1])
        formulas.append(Eventually(dwell_i, interval=[0, latest_start]))

    return _conjunction(formulas)


def pointmass_joint_outside_obstacle_predicate(
    env: MultiAgentPointMass2DEnv,
    obstacle_id: int,
    agent_id: int,
    margin: float = 0.0,
):
    obs_i = _validate_obstacle_id(env, obstacle_id)
    agent_i = _validate_joint_agent_id(env, agent_id)
    c = env.obstacles.centers[obs_i]
    r = float(env.obstacles.radii[obs_i])
    return Predicate(
        f"pointmass_joint_agent{agent_i}_outside_obstacle_{obs_i}",
        lambda tr, c=c, r=r, a=agent_i: _outside_circle_trace_joint(tr, c, r, margin, a),
    )


def pointmass_joint_inside_zone_predicate(
    env: MultiAgentPointMass2DEnv,
    zone_id: int,
    agent_id: int,
    margin: float = 0.0,
):
    z_i = _validate_zone_id(env, zone_id)
    agent_i = _validate_joint_agent_id(env, agent_id)
    c = env.zones.centers[z_i]
    r = float(env.zones.radii[z_i])
    return Predicate(
        f"pointmass_joint_agent{agent_i}_inside_zone_{z_i}",
        lambda tr, c=c, r=r, a=agent_i: _inside_circle_trace_joint(tr, c, r, margin, a),
    )


def pointmass_joint_outside_zone_predicate(
    env: MultiAgentPointMass2DEnv,
    zone_id: int,
    agent_id: int,
    margin: float = 0.0,
):
    z_i = _validate_zone_id(env, zone_id)
    agent_i = _validate_joint_agent_id(env, agent_id)
    c = env.zones.centers[z_i]
    r = float(env.zones.radii[z_i])
    return Predicate(
        f"pointmass_joint_agent{agent_i}_outside_zone_{z_i}",
        lambda tr, c=c, r=r, a=agent_i: _outside_circle_trace_joint(tr, c, r, margin, a),
    )


def pointmass_joint_all_agents_always_outside_obstacles(
    env: MultiAgentPointMass2DEnv,
    horizon: int,
    margin: float = 0.0,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    safe_formulas = []
    for agent_id in range(env.num_agents):
        for obs_id in range(env.obstacles.centers.shape[0]):
            pred = pointmass_joint_outside_obstacle_predicate(
                env,
                obstacle_id=obs_id,
                agent_id=agent_id,
                margin=margin,
            )
            safe_formulas.append(Always(pred > 0.0, interval=[0, horizon - 1]))
    return _conjunction(safe_formulas)


def pointmass_two_agent_button_task_spec(
    env: MultiAgentPointMass2DEnv,
    horizon: int,
    stay_steps: int = 1,
    obstacle_margin: float = 0.0,
    zone_margin: float = 0.0,
    agent_a_id: int = 0,
    agent_b_id: int = 1,
    goal1_zone_id: int = 0,
    goal2_zone_id: int = 1,
    button_b_zone_id: int = 2,
):
    """Two-agent task in joint state space.

    - All agents avoid all obstacles always.
    - Agent A eventually visits goal zone 1.
    - Agent B eventually visits goal zone 2.
    - Agent A cannot enter its goal zone until agent B has pressed button B.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stay_steps <= 0:
        raise ValueError("stay_steps must be positive.")
    if env.num_agents < 2:
        raise ValueError("Two-agent button task requires env.num_agents >= 2.")

    agent_a = _validate_joint_agent_id(env, agent_a_id)
    agent_b = _validate_joint_agent_id(env, agent_b_id)
    goal1 = _validate_zone_id(env, goal1_zone_id)
    goal2 = _validate_zone_id(env, goal2_zone_id)
    button_b = _validate_zone_id(env, button_b_zone_id)

    safe = pointmass_joint_all_agents_always_outside_obstacles(
        env,
        horizon=horizon,
        margin=obstacle_margin,
    )

    latest_start = horizon - stay_steps
    if latest_start < 0:
        raise ValueError(f"Horizon {horizon} too short for stay_steps={stay_steps}.")

    in_goal1_a = pointmass_joint_inside_zone_predicate(
        env,
        zone_id=goal1,
        agent_id=agent_a,
        margin=zone_margin,
    )
    in_goal2_b = pointmass_joint_inside_zone_predicate(
        env,
        zone_id=goal2,
        agent_id=agent_b,
        margin=zone_margin,
    )

    dwell_goal1_a = Always(in_goal1_a > 0.0, interval=[0, stay_steps - 1])
    dwell_goal2_b = Always(in_goal2_b > 0.0, interval=[0, stay_steps - 1])
    goal_a_eventually = Eventually(dwell_goal1_a, interval=[0, latest_start])
    goal_b_eventually = Eventually(dwell_goal2_b, interval=[0, latest_start])

    in_button_b = pointmass_joint_inside_zone_predicate(
        env,
        zone_id=button_b,
        agent_id=agent_b,
        margin=zone_margin,
    )
    button_pressed_eventually = Eventually(in_button_b > 0.0, interval=[0, horizon - 1])

    c_button = env.zones.centers[button_b]
    r_button = float(env.zones.radii[button_b])
    c_goal_a = env.zones.centers[goal1]
    r_goal_a = float(env.zones.radii[goal1])

    gate_pred = Predicate(
        f"pointmass_joint_agent{agent_a}_wait_for_button_agent{agent_b}",
        lambda tr, cb=c_button, rb=r_button, cg=c_goal_a, rg=r_goal_a, a=agent_a, b=agent_b: jnp.maximum(
            jnp.maximum.accumulate(_inside_circle_trace_joint(tr, cb, rb, zone_margin, b)),
            _outside_circle_trace_joint(tr, cg, rg, zone_margin, a),
        ),
    )
    gate_constraint = Always(gate_pred > 0.0, interval=[0, horizon - 1])

    return _conjunction(
        [
            safe,
            goal_a_eventually,
            goal_b_eventually,
            button_pressed_eventually,
            gate_constraint,
        ]
    )


def pointmass_multiagent_corridor_spec(
    env: MultiAgentPointMass2DEnv,
    horizon: int,
    corridor_center: jnp.ndarray,
    corridor_half_extent_x: float,
    corridor_half_extent_y: float,
    collision_radius: float,
):
    """Multi-agent corridor task.

    - Always no pairwise collisions.
    - Always avoid corridor wall regions.
    - Every agent eventually enters corridor rectangle.
    - At most one agent is inside corridor rectangle at any timestep.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if env.num_agents <= 0:
        raise ValueError("num_agents must be positive.")
    if collision_radius <= 0:
        raise ValueError("collision_radius must be positive.")
    if corridor_half_extent_x <= 0:
        raise ValueError("corridor_half_extent_x must be positive.")
    if corridor_half_extent_y <= 0:
        raise ValueError("corridor_half_extent_y must be positive.")

    formulas = []
    wall_half_thickness = float(corridor_half_extent_x)
    corridor_half_extents = jnp.asarray([corridor_half_extent_x, corridor_half_extent_y], dtype=jnp.float32)

    # Pairwise collision avoidance for all agents, all timesteps.
    for i in range(env.num_agents):
        for j in range(i + 1, env.num_agents):
            pred_ij = Predicate(
                f"pointmass_joint_agents_{i}_{j}_no_collision",
                lambda tr, a=i, b=j, r=collision_radius: _outside_agent_collision_trace(
                    tr,
                    agent_a_id=a,
                    agent_b_id=b,
                    collision_radius=r,
                ),
            )
            formulas.append(Always(pred_ij > 0.0, interval=[0, horizon - 1]))

    # Corridor wall avoidance for all agents, all timesteps.
    for i in range(env.num_agents):
        wall_pred_i = Predicate(
            f"pointmass_joint_agent_{i}_outside_corridor_walls",
            lambda tr, a=i, c=corridor_center, hy=corridor_half_extent_y, w=wall_half_thickness: _outside_corridor_walls_trace_joint(
                tr,
                corridor_center=c,
                corridor_half_height=hy,
                wall_half_thickness=w,
                agent_id=a,
            ),
        )
        formulas.append(Always(wall_pred_i > 0.0, interval=[0, horizon - 1]))

    # Every agent eventually reaches the corridor.
    for i in range(env.num_agents):
        in_corridor_i = Predicate(
            f"pointmass_joint_agent_{i}_in_corridor",
            lambda tr, a=i, c=corridor_center, hh=corridor_half_extents: _inside_rectangle_trace_joint(
                tr,
                center=c,
                half_extents=hh,
                agent_id=a,
            ),
        )
        formulas.append(Eventually(in_corridor_i > 0.0, interval=[0, horizon - 1]))
        # Each agent eventually moves to the right side (x > 0).
        x_positive_i = Predicate(
            f"pointmass_joint_agent_{i}_x_positive",
            lambda tr, a=i: _joint_agent_pos_trace(tr, agent_id=a)[:, 0],
        )
        formulas.append(Eventually(x_positive_i > 0.0, interval=[0, horizon - 1]))

    # Pairwise mutual-exclusion in corridor: no two agents may be inside together.
    # Robust encoding of (not in_i) OR (not in_j): max(-rho_i, -rho_j) > 0.
    for i in range(env.num_agents):
        for j in range(i + 1, env.num_agents):
            pair_exclusion_ij = Predicate(
                f"pointmass_joint_corridor_pair_exclusion_{i}_{j}",
                lambda tr, a=i, b=j, c=corridor_center, hh=corridor_half_extents: jnp.maximum(
                    -_inside_rectangle_trace_joint(tr, center=c, half_extents=hh, agent_id=a),
                    -_inside_rectangle_trace_joint(tr, center=c, half_extents=hh, agent_id=b),
                ),
            )
            formulas.append(Always(pair_exclusion_ij > 0.0, interval=[0, horizon - 1]))

    return _conjunction(formulas)


def pointmass_multiagent_synchronized_goals_spec(
    env: MultiAgentPointMass2DEnv,
    horizon: int,
    sync_delta_steps: int = 2,
    collision_radius: float = 0.12,
    goal_margin: float = 0.0,
):
    """Multi-agent synchronized goal-reaching task.

    Every agent i must reach goal zone i, and all arrivals must lie within
    one shared time window of width ``2 * sync_delta_steps``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if env.num_agents <= 0:
        raise ValueError("num_agents must be positive.")
    if sync_delta_steps < 0:
        raise ValueError("sync_delta_steps must be non-negative.")
    if collision_radius <= 0:
        raise ValueError("collision_radius must be positive.")
    if env.zones.centers.shape[0] < env.num_agents:
        raise ValueError(
            f"Need at least {env.num_agents} goal zones for synchronized-goals task, "
            f"got {env.zones.centers.shape[0]}."
        )

    window_len = 2 * sync_delta_steps
    latest_ref = horizon - 1 - window_len
    if latest_ref < 0:
        raise ValueError(
            f"Horizon {horizon} too short for sync_delta_steps={sync_delta_steps}. "
            f"Need at least {window_len + 1}."
        )

    formulas = []

    # Pairwise collision avoidance for all agents, all timesteps.
    for i in range(env.num_agents):
        for j in range(i + 1, env.num_agents):
            pred_ij = Predicate(
                f"pointmass_joint_agents_{i}_{j}_no_collision_sync",
                lambda tr, a=i, b=j, r=collision_radius: _outside_agent_collision_trace(
                    tr,
                    agent_a_id=a,
                    agent_b_id=b,
                    collision_radius=r,
                ),
            )
            formulas.append(Always(pred_ij > 0.0, interval=[0, horizon - 1]))

    per_agent_in_goal = []
    for agent_id in range(env.num_agents):
        in_goal_i = pointmass_joint_inside_zone_predicate(
            env,
            zone_id=agent_id,
            agent_id=agent_id,
            margin=goal_margin,
        )
        per_agent_in_goal.append(
            Eventually(
                in_goal_i > 0.0,
                interval=[0, window_len],
            )
        )

    # Existence of a shared reference time such that all agents hit their goals
    # within +/- sync_delta_steps around that reference.
    sync_window_formula = _conjunction(per_agent_in_goal)
    formulas.append(Eventually(sync_window_formula, interval=[0, latest_ref]))
    return _conjunction(formulas)


def pointmass_multiagent_leader_follow_spec(
    env: MultiAgentPointMass2DEnv,
    horizon: int,
    collision_radius: float = 0.12,
    desired_distance: float = 1.0,
    distance_epsilon: float = 0.1,
    leader_id: int = 0,
    first_goal_zone_id: int = 0,
    second_goal_zone_id: int = 1,
    goal_margin: float = 0.0,
):
    """Leader-follow multi-agent task in joint state space.

    - Leader agent eventually reaches first goal, then second goal.
    - In the last 100 steps of the horizon, each follower stays within
      ``desired_distance +/- distance_epsilon`` from the leader.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if env.num_agents <= 1:
        raise ValueError("Leader-follow task requires env.num_agents >= 2.")
    if collision_radius <= 0:
        raise ValueError("collision_radius must be positive.")
    if desired_distance <= 0:
        raise ValueError("desired_distance must be positive.")
    if distance_epsilon <= 0:
        raise ValueError("distance_epsilon must be positive.")
    if env.zones.centers.shape[0] < 2:
        raise ValueError("Need at least two goal zones for leader-follow task.")

    leader = _validate_joint_agent_id(env, leader_id)
    first_goal_zone = _validate_zone_id(env, first_goal_zone_id)
    second_goal_zone = _validate_zone_id(env, second_goal_zone_id)
    if first_goal_zone == second_goal_zone:
        raise ValueError("first_goal_zone_id and second_goal_zone_id must be different.")
    follow_window_start = max(0, horizon - 100)

    formulas = []

    # Leader must eventually reach first goal, and eventually reach second goal.
    leader_in_goal_1 = pointmass_joint_inside_zone_predicate(
        env,
        zone_id=first_goal_zone,
        agent_id=leader,
        margin=goal_margin,
    )
    leader_in_goal_2 = pointmass_joint_inside_zone_predicate(
        env,
        zone_id=second_goal_zone,
        agent_id=leader,
        margin=goal_margin,
    )
    formulas.append(Eventually(leader_in_goal_1 > 0.0, interval=[0, horizon - 1]))
    formulas.append(Eventually(leader_in_goal_2 > 0.0, interval=[0, horizon - 1]))

    # Followers maintain fixed distance band to leader in the last 100 steps.
    for i in range(env.num_agents):
        if i == leader:
            continue
        follow_band_i = Predicate(
            f"pointmass_joint_agent_{i}_distance_band_to_leader_{leader}",
            lambda tr, a=i, l=leader, d=desired_distance, e=distance_epsilon: e
            - jnp.abs(
                jnp.linalg.norm(
                    _joint_agent_pos_trace(tr, agent_id=a) - _joint_agent_pos_trace(tr, agent_id=l),
                    axis=-1,
                )
                - d
            ),
        )
        formulas.append(Always(follow_band_i > 0.0, interval=[follow_window_start, horizon - 1]))

    return _conjunction(formulas)


__all__ = [
    "crazyflie_backflip_spec",
    "quad_always_outside_obstacles",
    "quad_eventually_goal1_first_half",
    "quad_eventually_goal2_second_half",
    "quad_full_task_spec",
    "pointmass_always_outside_obstacles",
    "pointmass_eventually_zone1_first_half",
    "pointmass_eventually_zone2_second_half",
    "pointmass_full_task_spec",
    "quadruped_reach_avoid_periodic_gait_spec",
    "quadruped_teleop_tracking_spec",
    "quadruped_backflip_spec",
    "pointmass_visit_all_zones_task_spec",
    "pointmass_joint_outside_obstacle_predicate",
    "pointmass_joint_inside_zone_predicate",
    "pointmass_joint_outside_zone_predicate",
    "pointmass_joint_all_agents_always_outside_obstacles",
    "pointmass_two_agent_button_task_spec",
    "pointmass_multiagent_corridor_spec",
    "pointmass_multiagent_synchronized_goals_spec",
    "pointmass_multiagent_leader_follow_spec",
]
