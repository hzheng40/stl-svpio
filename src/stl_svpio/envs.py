from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=[2])
def quadrotor_step(state, control, dt=0.1):
    """Same as quadrotor() but with configurable dt for one HL step.
    state: (n, 6), control: (n, 3). Returns next_state (n, 6).
    """
    m = 0.03
    g = 9.81
    A = jnp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [dt, 0, 0, 1, 0, 0],
            [0, dt, 0, 0, 1, 0],
            [0, 0, dt, 0, 0, 1],
        ]
    )
    B = jnp.array(
        [
            [dt * g, 0, 0],
            [0, -dt * g, 0],
            [0, 0, dt / m],
            [0.5 * (dt**2) * g, 0, 0],
            [0, -0.5 * (dt**2) * g, 0],
            [0, 0, 0.5 * (dt**2) / m],
        ]
    )
    next_state = A @ state.T + B @ control.T
    return next_state.T


@dataclass
class BoxSet:
    centers: jnp.ndarray
    half_extents: jnp.ndarray

    @property
    def lower(self) -> jnp.ndarray:
        return self.centers - self.half_extents

    @property
    def upper(self) -> jnp.ndarray:
        return self.centers + self.half_extents


@dataclass
class CircleSet:
    centers: jnp.ndarray
    radii: jnp.ndarray


def _boxes_overlap(
    center_a: jnp.ndarray,
    half_a: jnp.ndarray,
    center_b: jnp.ndarray,
    half_b: jnp.ndarray,
    margin: float = 0.0,
) -> bool:
    sep = jnp.abs(center_a - center_b)
    limit = half_a + half_b + margin
    return bool(jnp.all(sep < limit))


def _circles_overlap(
    center_a: jnp.ndarray,
    radius_a: float,
    center_b: jnp.ndarray,
    radius_b: float,
    margin: float = 0.0,
) -> bool:
    dist = jnp.linalg.norm(center_a - center_b)
    return bool(dist < (radius_a + radius_b + margin))


def _point_in_box(point: jnp.ndarray, center: jnp.ndarray, half: jnp.ndarray) -> bool:
    return bool(jnp.all(jnp.abs(point - center) <= half))


def _point_in_circle(point: jnp.ndarray, center: jnp.ndarray, radius: float) -> bool:
    return bool(jnp.linalg.norm(point - center) <= radius)


class LinearizedQuadrotorEnv:
    """Linearized quadrotor environment with box-shaped obstacles and goals.

    State is [vx, vy, vz, x, y, z] and control is 3D.
    """

    def __init__(
        self,
        world_low: Sequence[float] = (-5.0, -5.0, 0.0),
        world_high: Sequence[float] = (5.0, 5.0, 5.0),
        num_obstacles: int = 4,
        num_goals: int = 2,
        obstacle_size_range: Tuple[float, float] = (0.4, 1.0),
        goal_size_range: Tuple[float, float] = (0.4, 0.8),
        dt: float = 0.1,
        seed: int = 0,
        landmark_margin: float = 0.05,
    ) -> None:
        self.world_low = jnp.asarray(world_low, dtype=jnp.float32)
        self.world_high = jnp.asarray(world_high, dtype=jnp.float32)
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals
        self.obstacle_size_range = obstacle_size_range
        self.goal_size_range = goal_size_range
        self.dt = dt
        self.landmark_margin = landmark_margin

        self._rng_key = jax.random.PRNGKey(seed)

        self.state = jnp.zeros((6,), dtype=jnp.float32)
        self.obstacles = BoxSet(
            centers=jnp.zeros((0, 3), dtype=jnp.float32),
            half_extents=jnp.zeros((0, 3), dtype=jnp.float32),
        )
        self.goals = BoxSet(
            centers=jnp.zeros((0, 3), dtype=jnp.float32),
            half_extents=jnp.zeros((0, 3), dtype=jnp.float32),
        )

    def _sample_non_overlapping_boxes(
        self,
        key: jax.Array,
        n: int,
        size_range: Tuple[float, float],
        existing: Optional[BoxSet] = None,
        max_tries: int = 5000,
    ) -> Tuple[BoxSet, jax.Array]:
        centers = []
        half_extents = []

        existing_centers = []
        existing_halves = []
        if existing is not None and existing.centers.shape[0] > 0:
            existing_centers = [c for c in existing.centers]
            existing_halves = [h for h in existing.half_extents]

        for _ in range(n):
            placed = False
            for _ in range(max_tries):
                key, k_half, k_center = jax.random.split(key, 3)
                half = jax.random.uniform(
                    k_half,
                    shape=(3,),
                    minval=size_range[0] / 2.0,
                    maxval=size_range[1] / 2.0,
                )
                lo = self.world_low + half
                hi = self.world_high - half
                if bool(jnp.any(lo >= hi)):
                    raise ValueError("World bounds are too small for sampled box sizes.")
                center = jax.random.uniform(k_center, shape=(3,), minval=lo, maxval=hi)

                overlap = False
                for c_e, h_e in zip(existing_centers, existing_halves):
                    if _boxes_overlap(center, half, c_e, h_e, margin=self.landmark_margin):
                        overlap = True
                        break
                if overlap:
                    continue

                for c_n, h_n in zip(centers, half_extents):
                    if _boxes_overlap(center, half, c_n, h_n, margin=self.landmark_margin):
                        overlap = True
                        break
                if overlap:
                    continue

                centers.append(center)
                half_extents.append(half)
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to sample non-overlapping boxes. Try fewer landmarks or a larger world."
                )

        if n == 0:
            box_set = BoxSet(
                centers=jnp.zeros((0, 3), dtype=jnp.float32),
                half_extents=jnp.zeros((0, 3), dtype=jnp.float32),
            )
        else:
            box_set = BoxSet(centers=jnp.stack(centers), half_extents=jnp.stack(half_extents))
        return box_set, key

    def _sample_initial_state(
        self,
        key: jax.Array,
        max_tries: int = 5000,
    ) -> Tuple[jnp.ndarray, jax.Array]:
        vel_lo = jnp.array([-0.5, -0.5, -0.5], dtype=jnp.float32)
        vel_hi = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)

        for _ in range(max_tries):
            key, k_vel, k_pos = jax.random.split(key, 3)
            vel = jax.random.uniform(k_vel, shape=(3,), minval=vel_lo, maxval=vel_hi)
            pos = jax.random.uniform(k_pos, shape=(3,), minval=self.world_low, maxval=self.world_high)

            inside_landmark = False
            for c, h in zip(self.obstacles.centers, self.obstacles.half_extents):
                if _point_in_box(pos, c, h):
                    inside_landmark = True
                    break
            if inside_landmark:
                continue
            for c, h in zip(self.goals.centers, self.goals.half_extents):
                if _point_in_box(pos, c, h):
                    inside_landmark = True
                    break
            if inside_landmark:
                continue

            return jnp.concatenate([vel, pos]), key

        raise RuntimeError("Failed to sample a valid initial quadrotor state.")

    def reset(
        self,
        key: Optional[jax.Array] = None,
        state: Optional[Sequence[float]] = None,
    ) -> jnp.ndarray:
        if key is None:
            self._rng_key, key = jax.random.split(self._rng_key)

        self.obstacles, key = self._sample_non_overlapping_boxes(
            key,
            n=self.num_obstacles,
            size_range=self.obstacle_size_range,
            existing=None,
        )
        self.goals, key = self._sample_non_overlapping_boxes(
            key,
            n=self.num_goals,
            size_range=self.goal_size_range,
            existing=self.obstacles,
        )

        if state is None:
            self.state, key = self._sample_initial_state(key)
        else:
            state_arr = jnp.asarray(state, dtype=jnp.float32)
            if state_arr.shape != (6,):
                raise ValueError(f"Expected state shape (6,), got {state_arr.shape}")
            self.state = state_arr

        self._rng_key = key
        return self.state

    def step(self, control: Sequence[float]) -> jnp.ndarray:
        control_arr = jnp.asarray(control, dtype=jnp.float32)
        if control_arr.shape != (3,):
            raise ValueError(f"Expected control shape (3,), got {control_arr.shape}")

        self.state = quadrotor_step(
            state=self.state[jnp.newaxis, :],
            control=control_arr[jnp.newaxis, :],
            dt=self.dt,
        )[0]
        return self.state

    def get_obstacle_boxes(self) -> Dict[str, jnp.ndarray]:
        return {
            "centers": self.obstacles.centers,
            "half_extents": self.obstacles.half_extents,
            "lower": self.obstacles.lower,
            "upper": self.obstacles.upper,
        }

    def get_goal_boxes(self) -> Dict[str, jnp.ndarray]:
        return {
            "centers": self.goals.centers,
            "half_extents": self.goals.half_extents,
            "lower": self.goals.lower,
            "upper": self.goals.upper,
        }

    def get_scene_bounding_boxes(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {
            "obstacles": self.get_obstacle_boxes(),
            "goals": self.get_goal_boxes(),
        }


class PointMass2DEnv:
    """2D point-mass environment with circular obstacles and zones.

    State is [x, y, vx, vy] and control is 2D acceleration.
    """

    def __init__(
        self,
        world_low: Sequence[float] = (-5.0, -5.0),
        world_high: Sequence[float] = (5.0, 5.0),
        num_obstacles: int = 4,
        num_zones: int = 2,
        obstacle_radius_range: Tuple[float, float] = (0.4, 1.0),
        zone_radius_range: Tuple[float, float] = (0.4, 0.9),
        dt: float = 0.1,
        seed: int = 0,
        landmark_margin: float = 0.05,
    ) -> None:
        self.world_low = jnp.asarray(world_low, dtype=jnp.float32)
        self.world_high = jnp.asarray(world_high, dtype=jnp.float32)
        self.num_obstacles = num_obstacles
        self.num_zones = num_zones
        self.obstacle_radius_range = obstacle_radius_range
        self.zone_radius_range = zone_radius_range
        self.dt = dt
        self.landmark_margin = landmark_margin

        self._rng_key = jax.random.PRNGKey(seed)

        self.state = jnp.zeros((4,), dtype=jnp.float32)
        self.obstacles = CircleSet(
            centers=jnp.zeros((0, 2), dtype=jnp.float32),
            radii=jnp.zeros((0,), dtype=jnp.float32),
        )
        self.zones = CircleSet(
            centers=jnp.zeros((0, 2), dtype=jnp.float32),
            radii=jnp.zeros((0,), dtype=jnp.float32),
        )

    def _sample_non_overlapping_circles(
        self,
        key: jax.Array,
        n: int,
        radius_range: Tuple[float, float],
        existing: Optional[CircleSet] = None,
        max_tries: int = 5000,
    ) -> Tuple[CircleSet, jax.Array]:
        centers = []
        radii = []

        existing_centers = []
        existing_radii = []
        if existing is not None and existing.centers.shape[0] > 0:
            existing_centers = [c for c in existing.centers]
            existing_radii = [float(r) for r in existing.radii]

        for _ in range(n):
            placed = False
            for _ in range(max_tries):
                key, k_r, k_c = jax.random.split(key, 3)
                radius = float(
                    jax.random.uniform(k_r, shape=(), minval=radius_range[0], maxval=radius_range[1])
                )
                lo = self.world_low + radius
                hi = self.world_high - radius
                if bool(jnp.any(lo >= hi)):
                    raise ValueError("World bounds are too small for sampled circle radii.")
                center = jax.random.uniform(k_c, shape=(2,), minval=lo, maxval=hi)

                overlap = False
                for c_e, r_e in zip(existing_centers, existing_radii):
                    if _circles_overlap(center, radius, c_e, r_e, margin=self.landmark_margin):
                        overlap = True
                        break
                if overlap:
                    continue

                for c_n, r_n in zip(centers, radii):
                    if _circles_overlap(center, radius, c_n, r_n, margin=self.landmark_margin):
                        overlap = True
                        break
                if overlap:
                    continue

                centers.append(center)
                radii.append(radius)
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to sample non-overlapping circles. Try fewer landmarks or a larger world."
                )

        if n == 0:
            circle_set = CircleSet(
                centers=jnp.zeros((0, 2), dtype=jnp.float32),
                radii=jnp.zeros((0,), dtype=jnp.float32),
            )
        else:
            circle_set = CircleSet(
                centers=jnp.stack(centers),
                radii=jnp.asarray(radii, dtype=jnp.float32),
            )
        return circle_set, key

    def _sample_joint_landmarks(
        self,
        key: jax.Array,
        max_tries: int = 5000,
    ) -> Tuple[CircleSet, CircleSet, jax.Array]:
        """Sample obstacles and zones jointly in one mixed placement pass."""
        num_obs = int(self.num_obstacles)
        num_zones = int(self.num_zones)
        total = num_obs + num_zones
        if total == 0:
            empty = CircleSet(
                centers=jnp.zeros((0, 2), dtype=jnp.float32),
                radii=jnp.zeros((0,), dtype=jnp.float32),
            )
            return empty, empty, key

        # Mix obstacle and zone placements to avoid obstacle-first packing bias.
        types = jnp.concatenate(
            [
                jnp.zeros((num_obs,), dtype=jnp.int32),  # 0 -> obstacle
                jnp.ones((num_zones,), dtype=jnp.int32),  # 1 -> zone
            ],
            axis=0,
        )
        key, perm_key = jax.random.split(key)
        types = types[jax.random.permutation(perm_key, total)]

        centers = []
        radii = []
        placed_types = []
        for landmark_type in types:
            is_zone = bool(int(landmark_type) == 1)
            r_lo, r_hi = self.zone_radius_range if is_zone else self.obstacle_radius_range

            placed = False
            for _ in range(max_tries):
                key, k_r, k_c = jax.random.split(key, 3)
                radius = float(jax.random.uniform(k_r, shape=(), minval=r_lo, maxval=r_hi))
                lo = self.world_low + radius
                hi = self.world_high - radius
                if bool(jnp.any(lo >= hi)):
                    raise ValueError("World bounds are too small for sampled circle radii.")
                center = jax.random.uniform(k_c, shape=(2,), minval=lo, maxval=hi)

                overlap = False
                for c_n, r_n in zip(centers, radii):
                    if _circles_overlap(center, radius, c_n, r_n, margin=self.landmark_margin):
                        overlap = True
                        break
                if overlap:
                    continue

                centers.append(center)
                radii.append(radius)
                placed_types.append(1 if is_zone else 0)
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to sample non-overlapping circles. Try fewer landmarks or a larger world."
                )

        centers_arr = jnp.stack(centers)
        radii_arr = jnp.asarray(radii, dtype=jnp.float32)
        types_arr = jnp.asarray(placed_types, dtype=jnp.int32)

        obs_mask = types_arr == 0
        zone_mask = types_arr == 1
        obstacles = CircleSet(
            centers=centers_arr[obs_mask],
            radii=radii_arr[obs_mask],
        )
        zones = CircleSet(
            centers=centers_arr[zone_mask],
            radii=radii_arr[zone_mask],
        )
        return obstacles, zones, key

    def _sample_landmarks(
        self,
        key: jax.Array,
        mode: str = "sequential",
    ) -> Tuple[CircleSet, CircleSet, jax.Array]:
        if mode == "joint":
            return self._sample_joint_landmarks(key)
        if mode == "sequential":
            obstacles, key = self._sample_non_overlapping_circles(
                key,
                n=self.num_obstacles,
                radius_range=self.obstacle_radius_range,
                existing=None,
            )
            zones, key = self._sample_non_overlapping_circles(
                key,
                n=self.num_zones,
                radius_range=self.zone_radius_range,
                existing=obstacles,
            )
            return obstacles, zones, key
        raise ValueError(f"Unknown landmark sampling mode: {mode}")

    def _sample_initial_state(
        self,
        key: jax.Array,
        max_tries: int = 5000,
    ) -> Tuple[jnp.ndarray, jax.Array]:
        vel_lo = jnp.array([-0.5, -0.5], dtype=jnp.float32)
        vel_hi = jnp.array([0.5, 0.5], dtype=jnp.float32)

        for _ in range(max_tries):
            key, k_v, k_p = jax.random.split(key, 3)
            vel = jax.random.uniform(k_v, shape=(2,), minval=vel_lo, maxval=vel_hi)
            pos = jax.random.uniform(k_p, shape=(2,), minval=self.world_low, maxval=self.world_high)

            inside_landmark = False
            for c, r in zip(self.obstacles.centers, self.obstacles.radii):
                if _point_in_circle(pos, c, float(r)):
                    inside_landmark = True
                    break
            if inside_landmark:
                continue
            for c, r in zip(self.zones.centers, self.zones.radii):
                if _point_in_circle(pos, c, float(r)):
                    inside_landmark = True
                    break
            if inside_landmark:
                continue

            return jnp.concatenate([pos, vel]), key

        raise RuntimeError("Failed to sample a valid initial point-mass state.")

    def reset(
        self,
        key: Optional[jax.Array] = None,
        state: Optional[Sequence[float]] = None,
        landmark_sampling_mode: str = "sequential",
    ) -> jnp.ndarray:
        if key is None:
            self._rng_key, key = jax.random.split(self._rng_key)

        self.obstacles, self.zones, key = self._sample_landmarks(
            key,
            mode=landmark_sampling_mode,
        )

        if state is None:
            self.state, key = self._sample_initial_state(key)
        else:
            state_arr = jnp.asarray(state, dtype=jnp.float32)
            if state_arr.shape != (4,):
                raise ValueError(f"Expected state shape (4,), got {state_arr.shape}")
            self.state = state_arr

        self._rng_key = key
        return self.state

    def step(self, control: Sequence[float]) -> jnp.ndarray:
        control_arr = jnp.asarray(control, dtype=jnp.float32)
        if control_arr.shape != (2,):
            raise ValueError(f"Expected control shape (2,), got {control_arr.shape}")

        pos = self.state[:2]
        vel = self.state[2:]

        pos_next = pos + self.dt * vel + 0.5 * (self.dt**2) * control_arr
        vel_next = vel + self.dt * control_arr
        self.state = jnp.concatenate([pos_next, vel_next])
        return self.state

    def get_obstacle_circles(self) -> Dict[str, jnp.ndarray]:
        return {"centers": self.obstacles.centers, "radii": self.obstacles.radii}

    def get_zone_circles(self) -> Dict[str, jnp.ndarray]:
        return {"centers": self.zones.centers, "radii": self.zones.radii}

    def get_scene_circles(self) -> Dict[str, Dict[str, jnp.ndarray]]:
        return {
            "obstacles": self.get_obstacle_circles(),
            "zones": self.get_zone_circles(),
        }


class MultiAgentPointMass2DEnv(PointMass2DEnv):
    """2D multi-agent point-mass environment with shared circular landmarks.

    Joint state is ``[x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]`` and
    joint control is ``[ax1, ay1, ax2, ay2, ...]``.
    """

    def __init__(
        self,
        num_agents: int,
        world_low: Sequence[float] = (-5.0, -5.0),
        world_high: Sequence[float] = (5.0, 5.0),
        num_obstacles: int = 4,
        num_zones: int = 2,
        obstacle_radius_range: Tuple[float, float] = (0.4, 1.0),
        zone_radius_range: Tuple[float, float] = (0.4, 0.9),
        dt: float = 0.1,
        seed: int = 0,
        landmark_margin: float = 0.05,
    ) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        super().__init__(
            world_low=world_low,
            world_high=world_high,
            num_obstacles=num_obstacles,
            num_zones=num_zones,
            obstacle_radius_range=obstacle_radius_range,
            zone_radius_range=zone_radius_range,
            dt=dt,
            seed=seed,
            landmark_margin=landmark_margin,
        )
        self.num_agents = int(num_agents)
        self.state = jnp.zeros((4 * self.num_agents,), dtype=jnp.float32)

    def _sample_initial_state(
        self,
        key: jax.Array,
        max_tries: int = 5000,
    ) -> Tuple[jnp.ndarray, jax.Array]:
        vel_lo = jnp.array([-0.5, -0.5], dtype=jnp.float32)
        vel_hi = jnp.array([0.5, 0.5], dtype=jnp.float32)

        agent_states = []
        placed_positions = []
        for _agent_id in range(self.num_agents):
            placed = False
            for _ in range(max_tries):
                key, k_v, k_p = jax.random.split(key, 3)
                vel = jax.random.uniform(k_v, shape=(2,), minval=vel_lo, maxval=vel_hi)
                pos = jax.random.uniform(k_p, shape=(2,), minval=self.world_low, maxval=self.world_high)

                inside_landmark = False
                for c, r in zip(self.obstacles.centers, self.obstacles.radii):
                    if _point_in_circle(pos, c, float(r)):
                        inside_landmark = True
                        break
                if inside_landmark:
                    continue
                for c, r in zip(self.zones.centers, self.zones.radii):
                    if _point_in_circle(pos, c, float(r)):
                        inside_landmark = True
                        break
                if inside_landmark:
                    continue

                # Keep initial positions distinct to avoid exact overlap at reset.
                conflict = False
                for p in placed_positions:
                    if float(jnp.linalg.norm(pos - p)) < 2.0 * self.landmark_margin:
                        conflict = True
                        break
                if conflict:
                    continue

                agent_states.append(jnp.concatenate([pos, vel]))
                placed_positions.append(pos)
                placed = True
                break

            if not placed:
                raise RuntimeError("Failed to sample a valid initial state for all point-mass agents.")

        return jnp.concatenate(agent_states, axis=0), key

    def reset(
        self,
        key: Optional[jax.Array] = None,
        state: Optional[Sequence[float]] = None,
        landmark_sampling_mode: str = "sequential",
    ) -> jnp.ndarray:
        if key is None:
            self._rng_key, key = jax.random.split(self._rng_key)

        self.obstacles, self.zones, key = self._sample_landmarks(
            key,
            mode=landmark_sampling_mode,
        )

        if state is None:
            self.state, key = self._sample_initial_state(key)
        else:
            state_arr = jnp.asarray(state, dtype=jnp.float32)
            expected_shape = (4 * self.num_agents,)
            if state_arr.shape != expected_shape:
                raise ValueError(f"Expected state shape {expected_shape}, got {state_arr.shape}")
            self.state = state_arr

        self._rng_key = key
        return self.state

    def step(self, control: Sequence[float]) -> jnp.ndarray:
        control_arr = jnp.asarray(control, dtype=jnp.float32)
        expected_shape = (2 * self.num_agents,)
        if control_arr.shape != expected_shape:
            raise ValueError(f"Expected control shape {expected_shape}, got {control_arr.shape}")

        state_agents = self.state.reshape((self.num_agents, 4))
        control_agents = control_arr.reshape((self.num_agents, 2))

        pos = state_agents[:, :2]
        vel = state_agents[:, 2:]
        pos_next = pos + self.dt * vel + 0.5 * (self.dt**2) * control_agents
        vel_next = vel + self.dt * control_agents
        self.state = jnp.concatenate([pos_next, vel_next], axis=-1).reshape((-1,))
        return self.state

    def get_agent_state(self, agent_id: int) -> jnp.ndarray:
        if not 0 <= int(agent_id) < self.num_agents:
            raise ValueError(f"agent_id must be in [0, {self.num_agents - 1}], got {agent_id}.")
        start = 4 * int(agent_id)
        return self.state[start : start + 4]


__all__ = [
    "quadrotor_step",
    "LinearizedQuadrotorEnv",
    "PointMass2DEnv",
    "MultiAgentPointMass2DEnv",
]
