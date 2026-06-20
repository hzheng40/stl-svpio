from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
try:
    import yaml
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Missing dependency `pyyaml`. Install it in your solver venv, e.g. `pip install pyyaml`."
    ) from exc

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "This solver script requires `jax` to reconstruct pointmass scenes. "
        "Install it in the solver environment (e.g. `pip install jax`)."
    ) from exc

from stl_svpio.envs import MultiAgentPointMass2DEnv, PointMass2DEnv


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    task: str
    dt: float
    horizon: int
    episode_steps: int
    num_obstacles: int
    num_zones: int
    num_agents: int
    scene: str
    layout_mode: str
    landmark_sampling_mode: str
    seed: int
    stay_steps: int
    agent_collision_radius: float
    corridor_half_extent: float
    sync_delta_steps: int


@dataclass
class SceneData:
    num_agents: int
    world_low: np.ndarray
    world_high: np.ndarray
    x0: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    zone_centers: np.ndarray
    zone_radii: np.ndarray
    goal_square_center: Optional[np.ndarray] = None
    goal_square_side: Optional[float] = None
    corridor_center: Optional[np.ndarray] = None
    corridor_half_extent_x: Optional[float] = None
    corridor_half_extent_y: Optional[float] = None


@dataclass(frozen=True)
class FormulaNode:
    pass


@dataclass(frozen=True)
class Atom(FormulaNode):
    name: str


@dataclass(frozen=True)
class AndNode(FormulaNode):
    children: tuple[FormulaNode, ...]


@dataclass(frozen=True)
class OrNode(FormulaNode):
    children: tuple[FormulaNode, ...]


@dataclass(frozen=True)
class AlwaysNode(FormulaNode):
    child: FormulaNode
    a: int
    b: int


@dataclass(frozen=True)
class EventuallyNode(FormulaNode):
    child: FormulaNode
    a: int
    b: int


def _and_all(nodes: Iterable[FormulaNode]) -> FormulaNode:
    items = tuple(nodes)
    if not items:
        return Atom("true")
    if len(items) == 1:
        return items[0]
    return AndNode(items)


def _or_all(nodes: Iterable[FormulaNode]) -> FormulaNode:
    items = tuple(nodes)
    if not items:
        return Atom("false")
    if len(items) == 1:
        return items[0]
    return OrNode(items)


def node_to_string(node: FormulaNode) -> str:
    if isinstance(node, Atom):
        if node.name in {"true", "false", "True", "False"}:
            return node.name
        # Local PyTeLo STL grammar expects atomic comparisons, not bare propositions.
        return f"({node.name} > 0)"
    if isinstance(node, AndNode):
        return "(" + " && ".join(node_to_string(c) for c in node.children) + ")"
    if isinstance(node, OrNode):
        return "(" + " || ".join(node_to_string(c) for c in node.children) + ")"
    if isinstance(node, AlwaysNode):
        return f"G[{node.a},{node.b}]({node_to_string(node.child)})"
    if isinstance(node, EventuallyNode):
        return f"F[{node.a},{node.b}]({node_to_string(node.child)})"
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct PyTeLo+Gurobi STL solver for pointmass tasks.")
    parser.add_argument(
        "--tasks-config",
        type=Path,
        default=Path("configs/paper/milp_pytelo_pointmass.yaml"),
        help="YAML containing task presets.",
    )
    parser.add_argument("--task-id", required=True, help="Task key from tasks config.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--time-limit-sec", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--polygon-sides", type=int, default=16)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/pytelo_milp"))
    parser.add_argument("--write-lp", action="store_true", help="Write .lp model for debugging.")
    parser.add_argument("--verify-trace", action="store_true", help="Run post-solve trace verifier.")
    parser.add_argument(
        "--verify-fail-as-warning",
        action="store_true",
        help="If --verify-trace fails, keep run non-fatal and save artifacts with verified_sat=false.",
    )
    parser.add_argument("--gurobi-log-file", type=Path, default=None)
    parser.add_argument(
        "--pytelo-root",
        type=Path,
        default=Path(os.environ.get("PYTELO_ROOT", str(Path.home() / "pytelo"))),
        help="Path to a local PyTeLo clone (default: ~/pytelo or $PYTELO_ROOT).",
    )
    return parser.parse_args()


def _default_num_agents(task: str) -> int:
    if task == "multiagent_corridor":
        return 10
    if task == "multiagent_sync_goals":
        return 9
    if task == "multiagent_leader_follow":
        return 5
    if task == "multiagent_button":
        return 2
    return 1


def load_task_config(path: Path, task_id: str, seed_override: Optional[int]) -> TaskInstance:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", {})
    if task_id not in tasks:
        raise KeyError(f"task-id `{task_id}` not found in {path}")

    raw = dict(tasks[task_id])
    task = str(raw["task"])
    num_agents = int(raw.get("num_agents", _default_num_agents(task)))
    seed = int(seed_override if seed_override is not None else raw.get("seed", 0))

    return TaskInstance(
        task_id=task_id,
        task=task,
        dt=float(raw["dt"]),
        horizon=int(raw["horizon"]),
        episode_steps=int(raw.get("episode_steps", raw["horizon"])),
        num_obstacles=int(raw.get("num_obstacles", 0)),
        num_zones=int(raw.get("num_zones", 0)),
        num_agents=num_agents,
        scene=str(raw.get("scene", "random")),
        layout_mode=str(raw.get("layout_mode", "random")),
        landmark_sampling_mode=str(raw.get("landmark_sampling_mode", "sequential")),
        seed=seed,
        stay_steps=int(raw.get("stay_steps", 1)),
        agent_collision_radius=float(raw.get("agent_collision_radius", 0.12)),
        corridor_half_extent=float(raw.get("corridor_half_extent", 0.6)),
        sync_delta_steps=int(raw.get("sync_delta_steps", 2)),
    )


def _as_np(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def apply_constructed_scene(env: PointMass2DEnv) -> None:
    env.obstacles.centers = jnp.array([[2.0, 2.0]], dtype=jnp.float32)
    env.obstacles.radii = jnp.array([1.0], dtype=jnp.float32)
    env.zones.centers = jnp.array([[4.0, 4.0]], dtype=jnp.float32)
    env.zones.radii = jnp.array([0.5], dtype=jnp.float32)
    env.goal_square_center = jnp.array([4.0, 4.0], dtype=jnp.float32)  # noqa: B010
    env.goal_square_side = 1.0  # noqa: B010
    env.state = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)


def apply_multiagent_corridor_scene(
    env: MultiAgentPointMass2DEnv,
    key: jax.Array,
    collision_radius: float,
    corridor_half_extent: float,
) -> tuple[jnp.ndarray, jax.Array]:
    corridor_half_extent_x = float(corridor_half_extent)
    corridor_half_extent_y = 3.0 * corridor_half_extent_x
    center = 0.5 * (env.world_low + env.world_high)

    env.obstacles.centers = jnp.zeros((0, 2), dtype=jnp.float32)
    env.obstacles.radii = jnp.zeros((0,), dtype=jnp.float32)
    env.zones.centers = center[None, :]
    env.zones.radii = jnp.array([corridor_half_extent_x], dtype=jnp.float32)

    min_sep = 2.2 * collision_radius

    def _sample_side(
        n: int,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        key_in: jax.Array,
    ) -> tuple[list[jnp.ndarray], jax.Array]:
        pts: list[jnp.ndarray] = []
        key_local = key_in
        for _ in range(n):
            placed = False
            for _ in range(6000):
                key_local, kp = jax.random.split(key_local)
                cand = jax.random.uniform(
                    kp,
                    shape=(2,),
                    minval=jnp.array([x_low, y_low], dtype=jnp.float32),
                    maxval=jnp.array([x_high, y_high], dtype=jnp.float32),
                )
                if bool(
                    (jnp.abs(cand[0] - center[0]) < corridor_half_extent_x)
                    & (jnp.abs(cand[1] - center[1]) < corridor_half_extent_y)
                ):
                    continue
                if pts:
                    d = jnp.linalg.norm(cand[None, :] - jnp.stack(pts), axis=-1)
                    if bool(jnp.any(d < min_sep)):
                        continue
                pts.append(cand)
                placed = True
                break
            if not placed:
                raise RuntimeError("Failed to sample separated initial states for corridor scene.")
        return pts, key_local

    world_pad = 0.2
    y_low = float(env.world_low[1] + world_pad)
    y_high = float(env.world_high[1] - world_pad)
    left_x_low = float(env.world_low[0] + world_pad)
    left_x_high = float(min(-world_pad, float(center[0] - corridor_half_extent_x - world_pad - 3.0)))
    if left_x_low >= left_x_high:
        raise ValueError("World bounds too small for corridor setup.")

    all_pts, key = _sample_side(env.num_agents, left_x_low, left_x_high, y_low, y_high, key)
    pos = jnp.stack(all_pts, axis=0) if all_pts else jnp.zeros((0, 2), dtype=jnp.float32)
    vel = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    env.state = jnp.concatenate([pos, vel], axis=-1).reshape((-1,))
    return env.state, key


def apply_multiagent_sync_goals_scene(
    env: MultiAgentPointMass2DEnv,
    key: jax.Array,
    collision_radius: float,
) -> tuple[jnp.ndarray, jax.Array]:
    center = 0.5 * (env.world_low + env.world_high)
    env.obstacles.centers = jnp.zeros((0, 2), dtype=jnp.float32)
    env.obstacles.radii = jnp.zeros((0,), dtype=jnp.float32)

    world_pad = 0.2
    goal_radius = 0.6
    min_goal_sep = 2.0 * goal_radius + 0.1
    cols = int(jnp.ceil(jnp.sqrt(env.num_agents)))
    rows = int(jnp.ceil(env.num_agents / cols))
    goal_spacing = 1.2 * min_goal_sep
    req_w = goal_spacing * max(cols - 1, 0)
    req_h = goal_spacing * max(rows - 1, 0)

    half_w = 0.5 * req_w
    half_h = 0.5 * req_h
    x_min = float(center[0] - half_w - goal_radius)
    x_max = float(center[0] + half_w + goal_radius)
    y_min = float(center[1] - half_h - goal_radius)
    y_max = float(center[1] + half_h + goal_radius)
    if (
        x_min < float(env.world_low[0] + world_pad)
        or x_max > float(env.world_high[0] - world_pad)
        or y_min < float(env.world_low[1] + world_pad)
        or y_max > float(env.world_high[1] - world_pad)
    ):
        raise ValueError("World bounds too small for centered sync-goals grid.")

    xs = jnp.linspace(center[0] - half_w, center[0] + half_w, cols)
    ys = jnp.linspace(center[1] - half_h, center[1] + half_h, rows)
    grid = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape((-1, 2))
    goal_centers = grid[: env.num_agents]
    env.zones.centers = goal_centers
    env.zones.radii = jnp.full((env.num_agents,), goal_radius, dtype=jnp.float32)

    spawn_radius = 5.0
    max_ring = float(
        jnp.min(
            jnp.array(
                [
                    center[0] - env.world_low[0] - world_pad,
                    env.world_high[0] - center[0] - world_pad,
                    center[1] - env.world_low[1] - world_pad,
                    env.world_high[1] - center[1] - world_pad,
                ],
                dtype=jnp.float32,
            )
        )
    )
    if spawn_radius >= max_ring:
        raise ValueError("World bounds too small for fixed sync-goals spawn radius=5.0.")

    min_sep = 2.2 * collision_radius
    if env.num_agents > 1:
        chord = 2.0 * spawn_radius * jnp.sin(jnp.pi / env.num_agents)
        if float(chord) < float(min_sep):
            raise ValueError("Sync-goals ring spacing is too tight for current parameters.")

    angles = jnp.linspace(0.0, 2.0 * jnp.pi, env.num_agents, endpoint=False, dtype=jnp.float32)
    pos = center[None, :] + spawn_radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    key_local, k_perm = jax.random.split(key)
    perm = jax.random.permutation(k_perm, env.num_agents)
    pos = pos[perm]

    vel = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    env.state = jnp.concatenate([pos, vel], axis=-1).reshape((-1,))
    return env.state, key_local


def build_scene(task_cfg: TaskInstance) -> SceneData:
    is_multiagent = task_cfg.task.startswith("multiagent_")

    if is_multiagent:
        if task_cfg.task in {"multiagent_corridor", "multiagent_sync_goals"}:
            world_low = (-10.0, -10.0)
            world_high = (10.0, 10.0)
        else:
            world_low = (-5.0, -5.0)
            world_high = (5.0, 5.0)

        env = MultiAgentPointMass2DEnv(
            num_agents=task_cfg.num_agents,
            world_low=world_low,
            world_high=world_high,
            dt=task_cfg.dt,
            seed=task_cfg.seed,
            num_obstacles=task_cfg.num_obstacles,
            num_zones=task_cfg.num_zones,
        )
    else:
        env = PointMass2DEnv(
            dt=task_cfg.dt,
            seed=task_cfg.seed,
            num_obstacles=task_cfg.num_obstacles,
            num_zones=task_cfg.num_zones,
        )

    reset_key = jax.random.PRNGKey(task_cfg.seed + 123)
    x0 = env.reset(key=reset_key, landmark_sampling_mode=task_cfg.landmark_sampling_mode)

    corridor_center = None
    corridor_half_extent_x = None
    corridor_half_extent_y = None

    if is_multiagent:
        if task_cfg.task == "multiagent_corridor":
            x0, _ = apply_multiagent_corridor_scene(
                env,
                key=reset_key,
                collision_radius=task_cfg.agent_collision_radius,
                corridor_half_extent=task_cfg.corridor_half_extent,
            )
            corridor_center = _as_np(0.5 * (env.world_low + env.world_high))
            corridor_half_extent_x = task_cfg.corridor_half_extent
            corridor_half_extent_y = 3.0 * task_cfg.corridor_half_extent
        elif task_cfg.task == "multiagent_sync_goals":
            x0, _ = apply_multiagent_sync_goals_scene(
                env,
                key=reset_key,
                collision_radius=task_cfg.agent_collision_radius,
            )
        elif task_cfg.task == "multiagent_button":
            # Match runner behavior: keep sampled positions, zero all velocities.
            x0_agents = x0.reshape((task_cfg.num_agents, 4))
            env.state = jnp.concatenate(
                [x0_agents[:, :2], jnp.zeros((task_cfg.num_agents, 2), dtype=x0.dtype)],
                axis=-1,
            ).reshape((-1,))
            x0 = env.state
    else:
        if task_cfg.scene == "constructed_pointmass_diag":
            apply_constructed_scene(env)
            x0 = env.state

        env.state = jnp.concatenate([x0[:2], jnp.zeros((2,), dtype=x0.dtype)])
        x0 = env.state

    return SceneData(
        num_agents=task_cfg.num_agents,
        world_low=_as_np(env.world_low),
        world_high=_as_np(env.world_high),
        x0=_as_np(x0),
        obstacle_centers=_as_np(env.obstacles.centers),
        obstacle_radii=_as_np(env.obstacles.radii),
        zone_centers=_as_np(env.zones.centers),
        zone_radii=_as_np(env.zones.radii),
        goal_square_center=(
            _as_np(getattr(env, "goal_square_center"))
            if getattr(env, "goal_square_center", None) is not None
            else None
        ),
        goal_square_side=(
            float(getattr(env, "goal_square_side"))
            if getattr(env, "goal_square_side", None) is not None
            else None
        ),
        corridor_center=corridor_center,
        corridor_half_extent_x=corridor_half_extent_x,
        corridor_half_extent_y=corridor_half_extent_y,
    )


def build_task_formula(task_cfg: TaskInstance, scene: SceneData) -> FormulaNode:
    H = task_cfg.horizon
    S = task_cfg.stay_steps

    if task_cfg.task == "single_default":
        safe = [AlwaysNode(Atom(f"outside_obstacle_0_{k}"), 0, H - 1) for k in range(scene.obstacle_centers.shape[0])]
        latest = H - S
        if scene.goal_square_center is not None and scene.goal_square_side is not None:
            goal = EventuallyNode(AlwaysNode(Atom("in_goal_square_0"), 0, S - 1), 0, latest)
        else:
            goal = EventuallyNode(AlwaysNode(Atom("inside_zone_0_0"), 0, S - 1), 0, latest)
        return _and_all([*safe, goal])

    if task_cfg.task == "single_visit_goals":
        terms: list[FormulaNode] = [
            AlwaysNode(Atom(f"outside_obstacle_0_{k}"), 0, H - 1)
            for k in range(scene.obstacle_centers.shape[0])
        ]
        latest = H - S
        for zone_id in range(scene.zone_centers.shape[0]):
            terms.append(EventuallyNode(AlwaysNode(Atom(f"inside_zone_0_{zone_id}"), 0, S - 1), 0, latest))
        return _and_all(terms)

    if task_cfg.task == "multiagent_button":
        safe = [
            AlwaysNode(Atom(f"outside_obstacle_{agent}_{obs}"), 0, H - 1)
            for agent in range(scene.num_agents)
            for obs in range(scene.obstacle_centers.shape[0])
        ]
        latest = H - S
        terms: list[FormulaNode] = [
            *safe,
            EventuallyNode(AlwaysNode(Atom("inside_zone_0_0"), 0, S - 1), 0, latest),
            EventuallyNode(AlwaysNode(Atom("inside_zone_1_1"), 0, S - 1), 0, latest),
            EventuallyNode(Atom("inside_zone_1_2"), 0, H - 1),
            AlwaysNode(Atom("gate_ok_0_1_0_2"), 0, H - 1),
        ]
        return _and_all(terms)

    if task_cfg.task == "multiagent_sync_goals":
        collision_terms = [
            AlwaysNode(Atom(f"no_collision_{i}_{j}"), 0, H - 1)
            for i in range(scene.num_agents)
            for j in range(i + 1, scene.num_agents)
        ]
        window = 2 * task_cfg.sync_delta_steps
        latest_ref = H - 1 - window
        sync_inner = _and_all(
            EventuallyNode(Atom(f"inside_zone_{i}_{i}"), 0, window)
            for i in range(scene.num_agents)
        )
        sync = EventuallyNode(sync_inner, 0, latest_ref)
        return _and_all([*collision_terms, sync])

    if task_cfg.task == "multiagent_corridor":
        terms: list[FormulaNode] = []
        for i in range(scene.num_agents):
            for j in range(i + 1, scene.num_agents):
                terms.append(AlwaysNode(Atom(f"no_collision_{i}_{j}"), 0, H - 1))
        for i in range(scene.num_agents):
            terms.append(AlwaysNode(Atom(f"corridor_wall_safe_{i}"), 0, H - 1))
            terms.append(EventuallyNode(Atom(f"in_corridor_{i}"), 0, H - 1))
            terms.append(EventuallyNode(Atom(f"x_positive_{i}"), 0, H - 1))
        for i in range(scene.num_agents):
            for j in range(i + 1, scene.num_agents):
                terms.append(AlwaysNode(Atom(f"corridor_pair_exclusion_{i}_{j}"), 0, H - 1))
        return _and_all(terms)

    raise ValueError(f"Unsupported task for formula generation: {task_cfg.task}")


def _append_pytelo_paths(pytelo_root: Path) -> None:
    """Put local PyTeLo folders on sys.path for legacy import layout."""
    candidate_paths = [
        pytelo_root / "stl",
        pytelo_root / "mtl",
        pytelo_root / "wstl",
        pytelo_root / "wmtl",
    ]
    for p in candidate_paths:
        ps = str(p)
        if p.exists() and ps not in sys.path:
            sys.path.insert(0, ps)


def try_pytelo_parse(formula_str: str, pytelo_root: Path) -> Any:
    """Attempt parsing with PyTeLo across common entry points.

    The solver uses the internal expanded AST for MILP compilation, but this
    validation step enforces that the generated textual STL is consumable by PyTeLo.
    """

    attempts: list[tuple[str, Callable[[str], Any]]] = []

    # Local clone mode (README flow): parser generated into ~/pytelo/stl.
    if pytelo_root.exists():
        _append_pytelo_paths(pytelo_root)
        try:
            import stl as stl_module  # type: ignore

            fn = getattr(stl_module, "to_ast", None)
            if callable(fn):
                attempts.append(("stl.to_ast", fn))
        except Exception:
            pass

    # Compatibility fallback for packaged layouts.
    try:
        from stl.api import parse as stl_parse

        attempts.append(("stl.api.parse", stl_parse))
    except Exception:
        pass

    if not attempts:
        raise RuntimeError(
            "PyTeLo parser not found. For local clone usage, ensure parser files are generated:\n"
            "  cd ~/pytelo/stl && antlr4 -Dlanguage=Python3 stl.g4\n"
            "and run with --pytelo-root ~/pytelo."
        )

    errors = []
    for name, fn in attempts:
        try:
            return fn(formula_str)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    hint = (
        "\nIf using a local PyTeLo clone, ensure generated parser modules exist "
        "(stlLexer.py, stlParser.py, stlVisitor.py)."
    )
    raise RuntimeError("PyTeLo parse failed. Attempts:\n" + "\n".join(errors) + hint)


def _polygon_normals(m: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _inside_apothem(radius: float, m: int) -> float:
    return float(radius) * math.cos(math.pi / m)


def _outside_circ_radius(radius: float, m: int) -> float:
    return float(radius) / math.cos(math.pi / m)


class MilpBuilder:
    def __init__(
        self,
        task_cfg: TaskInstance,
        scene: SceneData,
        polygon_sides: int,
        time_limit_sec: float,
        mip_gap: float,
        gurobi_log_file: Optional[Path],
    ) -> None:
        if polygon_sides < 4:
            raise ValueError("polygon_sides must be >= 4")

        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "gurobipy is required. Install Gurobi and run `pip install gurobipy`."
            ) from exc

        self.gp = gp
        self.GRB = GRB
        self.task_cfg = task_cfg
        self.scene = scene
        self.H = task_cfg.horizon
        self.num_agents = scene.num_agents
        self.m = polygon_sides
        self.normals = _polygon_normals(self.m)
        self.big_m = 1e3

        self.model = gp.Model(f"pointmass_{task_cfg.task_id}")
        self.model.Params.OutputFlag = 1
        self.model.Params.TimeLimit = float(time_limit_sec)
        self.model.Params.MIPGap = float(mip_gap)
        self.model.Params.MIPFocus = 1
        self.model.Params.SolutionLimit = 1
        if gurobi_log_file is not None:
            self.model.Params.LogFile = str(gurobi_log_file)

        self.gamma = self.model.addVar(lb=0.0, ub=5.0, name="safety_margin")

        self.x: dict[tuple[int, int], Any] = {}
        self.y: dict[tuple[int, int], Any] = {}
        self.vx: dict[tuple[int, int], Any] = {}
        self.vy: dict[tuple[int, int], Any] = {}
        self.ax: dict[tuple[int, int], Any] = {}
        self.ay: dict[tuple[int, int], Any] = {}
        self.abs_ax: dict[tuple[int, int], Any] = {}
        self.abs_ay: dict[tuple[int, int], Any] = {}

        self.atom_cache: dict[tuple[str, int], Any] = {}
        self.node_cache: dict[tuple[FormulaNode, int], Any] = {}
        self.button_prefix_cache: dict[int, Any] = {}

        self._add_state_and_control_vars()
        self._add_dynamics_constraints()

    def _add_state_and_control_vars(self) -> None:
        wx_lo, wy_lo = float(self.scene.world_low[0]), float(self.scene.world_low[1])
        wx_hi, wy_hi = float(self.scene.world_high[0]), float(self.scene.world_high[1])
        vel_lim = 20.0

        for t in range(self.H):
            for agent in range(self.num_agents):
                self.x[(t, agent)] = self.model.addVar(lb=wx_lo, ub=wx_hi, name=f"x_{t}_{agent}")
                self.y[(t, agent)] = self.model.addVar(lb=wy_lo, ub=wy_hi, name=f"y_{t}_{agent}")
                self.vx[(t, agent)] = self.model.addVar(lb=-vel_lim, ub=vel_lim, name=f"vx_{t}_{agent}")
                self.vy[(t, agent)] = self.model.addVar(lb=-vel_lim, ub=vel_lim, name=f"vy_{t}_{agent}")

        for t in range(self.H - 1):
            for agent in range(self.num_agents):
                self.ax[(t, agent)] = self.model.addVar(lb=-5.0, ub=5.0, name=f"ax_{t}_{agent}")
                self.ay[(t, agent)] = self.model.addVar(lb=-5.0, ub=5.0, name=f"ay_{t}_{agent}")
                self.abs_ax[(t, agent)] = self.model.addVar(lb=0.0, name=f"abs_ax_{t}_{agent}")
                self.abs_ay[(t, agent)] = self.model.addVar(lb=0.0, name=f"abs_ay_{t}_{agent}")

        self.model.update()

        x0 = self.scene.x0.reshape((self.num_agents, 4))
        for agent in range(self.num_agents):
            self.model.addConstr(self.x[(0, agent)] == float(x0[agent, 0]), name=f"init_x_{agent}")
            self.model.addConstr(self.y[(0, agent)] == float(x0[agent, 1]), name=f"init_y_{agent}")
            self.model.addConstr(self.vx[(0, agent)] == float(x0[agent, 2]), name=f"init_vx_{agent}")
            self.model.addConstr(self.vy[(0, agent)] == float(x0[agent, 3]), name=f"init_vy_{agent}")

    def _add_dynamics_constraints(self) -> None:
        dt = self.task_cfg.dt
        half_dt2 = 0.5 * dt * dt
        for t in range(self.H - 1):
            for agent in range(self.num_agents):
                self.model.addConstr(
                    self.x[(t + 1, agent)]
                    == self.x[(t, agent)] + dt * self.vx[(t, agent)] + half_dt2 * self.ax[(t, agent)],
                    name=f"dyn_x_{t}_{agent}",
                )
                self.model.addConstr(
                    self.y[(t + 1, agent)]
                    == self.y[(t, agent)] + dt * self.vy[(t, agent)] + half_dt2 * self.ay[(t, agent)],
                    name=f"dyn_y_{t}_{agent}",
                )
                self.model.addConstr(
                    self.vx[(t + 1, agent)] == self.vx[(t, agent)] + dt * self.ax[(t, agent)],
                    name=f"dyn_vx_{t}_{agent}",
                )
                self.model.addConstr(
                    self.vy[(t + 1, agent)] == self.vy[(t, agent)] + dt * self.ay[(t, agent)],
                    name=f"dyn_vy_{t}_{agent}",
                )

                self.model.addConstr(self.abs_ax[(t, agent)] >= self.ax[(t, agent)])
                self.model.addConstr(self.abs_ax[(t, agent)] >= -self.ax[(t, agent)])
                self.model.addConstr(self.abs_ay[(t, agent)] >= self.ay[(t, agent)])
                self.model.addConstr(self.abs_ay[(t, agent)] >= -self.ay[(t, agent)])

    def _outside_circle_sat(self, name: str, t: int, x_expr: Any, y_expr: Any, cx: float, cy: float, radius: float) -> Any:
        sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
        r_out = _outside_circ_radius(radius, self.m)

        z = [self.model.addVar(vtype=self.GRB.BINARY, name=f"{name}_disj_{t}_{k}") for k in range(self.m)]
        self.model.addConstr(self.gp.quicksum(z) >= sat)
        for k, n in enumerate(self.normals):
            lhs = n[0] * (x_expr - cx) + n[1] * (y_expr - cy)
            self.model.addConstr(lhs >= (r_out + self.gamma) - self.big_m * (1 - z[k]))
        return sat

    def _inside_circle_sat(self, name: str, t: int, x_expr: Any, y_expr: Any, cx: float, cy: float, radius: float) -> Any:
        sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
        r_in = _inside_apothem(radius, self.m)

        for k, n in enumerate(self.normals):
            lhs = n[0] * (x_expr - cx) + n[1] * (y_expr - cy)
            self.model.addConstr(lhs <= (r_in - self.gamma) + self.big_m * (1 - sat), name=f"{name}_in_{t}_{k}")
        return sat

    def _inside_rectangle_sat(self, name: str, t: int, x_expr: Any, y_expr: Any, cx: float, cy: float, hx: float, hy: float) -> Any:
        sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
        self.model.addConstr(x_expr - cx <= (hx - self.gamma) + self.big_m * (1 - sat))
        self.model.addConstr(cx - x_expr <= (hx - self.gamma) + self.big_m * (1 - sat))
        self.model.addConstr(y_expr - cy <= (hy - self.gamma) + self.big_m * (1 - sat))
        self.model.addConstr(cy - y_expr <= (hy - self.gamma) + self.big_m * (1 - sat))
        return sat

    def _outside_rectangle_sat(self, name: str, t: int, x_expr: Any, y_expr: Any, cx: float, cy: float, hx: float, hy: float) -> Any:
        sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
        z = [self.model.addVar(vtype=self.GRB.BINARY, name=f"{name}_or_{t}_{i}") for i in range(4)]
        self.model.addConstr(self.gp.quicksum(z) >= sat)

        self.model.addConstr(x_expr <= (cx - hx - self.gamma) + self.big_m * (1 - z[0]))
        self.model.addConstr(x_expr >= (cx + hx + self.gamma) - self.big_m * (1 - z[1]))
        self.model.addConstr(y_expr <= (cy - hy - self.gamma) + self.big_m * (1 - z[2]))
        self.model.addConstr(y_expr >= (cy + hy + self.gamma) - self.big_m * (1 - z[3]))
        return sat

    def _atom_sat(self, name: str, t: int) -> Any:
        key = (name, t)
        if key in self.atom_cache:
            return self.atom_cache[key]

        if name == "true":
            var = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_true_{t}")
            self.model.addConstr(var == 1)
            self.atom_cache[key] = var
            return var
        if name == "false":
            var = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_false_{t}")
            self.model.addConstr(var == 0)
            self.atom_cache[key] = var
            return var

        parts = name.split("_")

        if name.startswith("outside_obstacle_"):
            agent = int(parts[2])
            obs = int(parts[3])
            c = self.scene.obstacle_centers[obs]
            r = float(self.scene.obstacle_radii[obs])
            var = self._outside_circle_sat(name, t, self.x[(t, agent)], self.y[(t, agent)], c[0], c[1], r)

        elif name.startswith("inside_zone_"):
            agent = int(parts[2])
            zone = int(parts[3])
            c = self.scene.zone_centers[zone]
            r = float(self.scene.zone_radii[zone])
            var = self._inside_circle_sat(name, t, self.x[(t, agent)], self.y[(t, agent)], c[0], c[1], r)

        elif name.startswith("in_goal_square_"):
            if self.scene.goal_square_center is None or self.scene.goal_square_side is None:
                raise ValueError("goal square metadata missing for in_goal_square atom")
            c = self.scene.goal_square_center
            half = 0.5 * float(self.scene.goal_square_side)
            var = self._inside_rectangle_sat(
                name,
                t,
                self.x[(t, 0)],
                self.y[(t, 0)],
                float(c[0]),
                float(c[1]),
                half,
                half,
            )

        elif name.startswith("outside_zone_"):
            agent = int(parts[2])
            zone = int(parts[3])
            c = self.scene.zone_centers[zone]
            r = float(self.scene.zone_radii[zone])
            var = self._outside_circle_sat(name, t, self.x[(t, agent)], self.y[(t, agent)], c[0], c[1], r)

        elif name.startswith("no_collision_"):
            i = int(parts[2])
            j = int(parts[3])
            # Conservative outside condition for relative position circle.
            sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
            r_out = _outside_circ_radius(2.0 * self.task_cfg.agent_collision_radius, self.m)
            z = [self.model.addVar(vtype=self.GRB.BINARY, name=f"{name}_or_{t}_{k}") for k in range(self.m)]
            self.model.addConstr(self.gp.quicksum(z) >= sat)
            for k, n in enumerate(self.normals):
                lhs = n[0] * (self.x[(t, i)] - self.x[(t, j)]) + n[1] * (self.y[(t, i)] - self.y[(t, j)])
                self.model.addConstr(lhs >= (r_out + self.gamma) - self.big_m * (1 - z[k]))
            var = sat

        elif name.startswith("in_corridor_"):
            i = int(parts[2])
            if self.scene.corridor_center is None:
                raise ValueError("corridor_center missing for corridor task")
            var = self._inside_rectangle_sat(
                name,
                t,
                self.x[(t, i)],
                self.y[(t, i)],
                float(self.scene.corridor_center[0]),
                float(self.scene.corridor_center[1]),
                float(self.scene.corridor_half_extent_x),
                float(self.scene.corridor_half_extent_y),
            )

        elif name.startswith("corridor_wall_safe_"):
            i = int(parts[3])
            if self.scene.corridor_center is None:
                raise ValueError("corridor_center missing for corridor task")

            cx, cy = float(self.scene.corridor_center[0]), float(self.scene.corridor_center[1])
            hx = float(self.scene.corridor_half_extent_x)
            hy = float(self.scene.corridor_half_extent_y)

            # outside_x OR inside_opening_y
            sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
            z = [self.model.addVar(vtype=self.GRB.BINARY, name=f"{name}_or_{t}_{k}") for k in range(4)]
            self.model.addConstr(self.gp.quicksum(z) >= sat)

            # outside_x: x<=cx-hx or x>=cx+hx
            self.model.addConstr(self.x[(t, i)] <= (cx - hx - self.gamma) + self.big_m * (1 - z[0]))
            self.model.addConstr(self.x[(t, i)] >= (cx + hx + self.gamma) - self.big_m * (1 - z[1]))
            # inside_opening_y: y in [cy-hy, cy+hy]
            self.model.addConstr(self.y[(t, i)] <= (cy + hy - self.gamma) + self.big_m * (1 - z[2]))
            self.model.addConstr(self.y[(t, i)] >= (cy - hy + self.gamma) - self.big_m * (1 - z[3]))
            var = sat

        elif name.startswith("x_positive_"):
            i = int(parts[2])
            sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
            self.model.addConstr(self.x[(t, i)] >= self.gamma - self.big_m * (1 - sat))
            var = sat

        elif name.startswith("corridor_pair_exclusion_"):
            i = int(parts[3])
            j = int(parts[4])
            if self.scene.corridor_center is None:
                raise ValueError("corridor_center missing for corridor task")
            cx, cy = float(self.scene.corridor_center[0]), float(self.scene.corridor_center[1])
            hx = float(self.scene.corridor_half_extent_x)
            hy = float(self.scene.corridor_half_extent_y)

            outside_i = self._outside_rectangle_sat(f"outside_corridor_{i}", t, self.x[(t, i)], self.y[(t, i)], cx, cy, hx, hy)
            outside_j = self._outside_rectangle_sat(f"outside_corridor_{j}", t, self.x[(t, j)], self.y[(t, j)], cx, cy, hx, hy)

            sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
            self.model.addConstr(sat >= outside_i)
            self.model.addConstr(sat >= outside_j)
            self.model.addConstr(sat <= outside_i + outside_j)
            var = sat

        elif name.startswith("gate_ok_"):
            # gate_ok_A_B_goal_button
            a = int(parts[2])
            b = int(parts[3])
            goal_zone = int(parts[4])
            button_zone = int(parts[5])

            inside_button = self._atom_sat(f"inside_zone_{b}_{button_zone}", t)
            outside_goal = self._atom_sat(f"outside_zone_{a}_{goal_zone}", t)

            if t not in self.button_prefix_cache:
                g = self.model.addVar(vtype=self.GRB.BINARY, name=f"button_prefix_{t}")
                if t == 0:
                    self.model.addConstr(g == inside_button)
                else:
                    prev = self.button_prefix_cache[t - 1]
                    self.model.addConstr(g >= prev)
                    self.model.addConstr(g >= inside_button)
                    self.model.addConstr(g <= prev + inside_button)
                self.button_prefix_cache[t] = g

            prefix = self.button_prefix_cache[t]
            sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{name}_{t}")
            self.model.addConstr(sat >= prefix)
            self.model.addConstr(sat >= outside_goal)
            self.model.addConstr(sat <= prefix + outside_goal)
            var = sat

        else:
            raise ValueError(f"Unknown atom name: {name}")

        self.atom_cache[key] = var
        return var

    def compile_node(self, node: FormulaNode, t: int) -> Any:
        key = (node, t)
        if key in self.node_cache:
            return self.node_cache[key]

        if t < 0 or t >= self.H:
            v = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_oob_{abs(hash(node))}_{t}")
            self.model.addConstr(v == 0)
            self.node_cache[key] = v
            return v

        if isinstance(node, Atom):
            sat = self._atom_sat(node.name, t)
            self.node_cache[key] = sat
            return sat

        sat = self.model.addVar(vtype=self.GRB.BINARY, name=f"sat_{abs(hash(node))}_{t}")

        if isinstance(node, AndNode):
            children = [self.compile_node(c, t) for c in node.children]
            for c in children:
                self.model.addConstr(sat <= c)
            self.model.addConstr(sat >= self.gp.quicksum(children) - (len(children) - 1))

        elif isinstance(node, OrNode):
            children = [self.compile_node(c, t) for c in node.children]
            for c in children:
                self.model.addConstr(sat >= c)
            self.model.addConstr(sat <= self.gp.quicksum(children))

        elif isinstance(node, AlwaysNode):
            idxs = [t + tau for tau in range(node.a, node.b + 1) if 0 <= (t + tau) < self.H]
            if not idxs:
                self.model.addConstr(sat == 0)
            else:
                children = [self.compile_node(node.child, idx) for idx in idxs]
                for c in children:
                    self.model.addConstr(sat <= c)
                self.model.addConstr(sat >= self.gp.quicksum(children) - (len(children) - 1))

        elif isinstance(node, EventuallyNode):
            idxs = [t + tau for tau in range(node.a, node.b + 1) if 0 <= (t + tau) < self.H]
            if not idxs:
                self.model.addConstr(sat == 0)
            else:
                children = [self.compile_node(node.child, idx) for idx in idxs]
                for c in children:
                    self.model.addConstr(sat >= c)
                self.model.addConstr(sat <= self.gp.quicksum(children))

        else:
            raise TypeError(f"Unsupported formula node {type(node)!r}")

        self.node_cache[key] = sat
        return sat

    def finalize_objective(self) -> None:
        control_l1 = self.gp.quicksum(self.abs_ax.values()) + self.gp.quicksum(self.abs_ay.values())
        self.model.setObjectiveN(-self.gamma, index=0, priority=2, name="maximize_margin")
        self.model.setObjectiveN(control_l1, index=1, priority=1, name="minimize_control_l1")

    def extract_solution(self) -> tuple[np.ndarray, np.ndarray]:
        states = np.zeros((self.H, 4 * self.num_agents), dtype=np.float64)
        controls = np.zeros((self.H - 1, 2 * self.num_agents), dtype=np.float64)

        for t in range(self.H):
            for a in range(self.num_agents):
                states[t, 4 * a + 0] = self.x[(t, a)].X
                states[t, 4 * a + 1] = self.y[(t, a)].X
                states[t, 4 * a + 2] = self.vx[(t, a)].X
                states[t, 4 * a + 3] = self.vy[(t, a)].X

        for t in range(self.H - 1):
            for a in range(self.num_agents):
                controls[t, 2 * a + 0] = self.ax[(t, a)].X
                controls[t, 2 * a + 1] = self.ay[(t, a)].X

        return states, controls


def eval_atom(name: str, t: int, states: np.ndarray, task_cfg: TaskInstance, scene: SceneData, polygon_sides: int) -> bool:
    m = polygon_sides
    normals = _polygon_normals(m)

    def _inside_circle(x: float, y: float, cx: float, cy: float, r: float) -> bool:
        apothem = _inside_apothem(r, m)
        for n in normals:
            if n[0] * (x - cx) + n[1] * (y - cy) > apothem + 1e-9:
                return False
        return True

    def _outside_circle(x: float, y: float, cx: float, cy: float, r: float) -> bool:
        r_out = _outside_circ_radius(r, m)
        for n in normals:
            if n[0] * (x - cx) + n[1] * (y - cy) >= r_out - 1e-9:
                return True
        return False

    if name == "true":
        return True
    if name == "false":
        return False

    parts = name.split("_")

    if name.startswith("outside_obstacle_"):
        agent = int(parts[2])
        obs = int(parts[3])
        x, y = states[t, 4 * agent + 0], states[t, 4 * agent + 1]
        c = scene.obstacle_centers[obs]
        r = float(scene.obstacle_radii[obs])
        return _outside_circle(x, y, float(c[0]), float(c[1]), r)

    if name.startswith("inside_zone_"):
        agent = int(parts[2])
        zone = int(parts[3])
        x, y = states[t, 4 * agent + 0], states[t, 4 * agent + 1]
        c = scene.zone_centers[zone]
        r = float(scene.zone_radii[zone])
        return _inside_circle(x, y, float(c[0]), float(c[1]), r)

    if name.startswith("in_goal_square_"):
        if scene.goal_square_center is None or scene.goal_square_side is None:
            raise ValueError("goal square metadata missing for in_goal_square atom")
        x, y = states[t, 0], states[t, 1]
        half = 0.5 * float(scene.goal_square_side)
        cx, cy = float(scene.goal_square_center[0]), float(scene.goal_square_center[1])
        return abs(x - cx) <= half and abs(y - cy) <= half

    if name.startswith("outside_zone_"):
        agent = int(parts[2])
        zone = int(parts[3])
        x, y = states[t, 4 * agent + 0], states[t, 4 * agent + 1]
        c = scene.zone_centers[zone]
        r = float(scene.zone_radii[zone])
        return _outside_circle(x, y, float(c[0]), float(c[1]), r)

    if name.startswith("no_collision_"):
        i = int(parts[2])
        j = int(parts[3])
        xi, yi = states[t, 4 * i + 0], states[t, 4 * i + 1]
        xj, yj = states[t, 4 * j + 0], states[t, 4 * j + 1]
        return _outside_circle(xi - xj, yi - yj, 0.0, 0.0, 2.0 * task_cfg.agent_collision_radius)

    if name.startswith("in_corridor_"):
        i = int(parts[2])
        x, y = states[t, 4 * i + 0], states[t, 4 * i + 1]
        cx, cy = float(scene.corridor_center[0]), float(scene.corridor_center[1])
        hx, hy = float(scene.corridor_half_extent_x), float(scene.corridor_half_extent_y)
        return abs(x - cx) <= hx and abs(y - cy) <= hy

    if name.startswith("corridor_wall_safe_"):
        i = int(parts[3])
        x, y = states[t, 4 * i + 0], states[t, 4 * i + 1]
        cx, cy = float(scene.corridor_center[0]), float(scene.corridor_center[1])
        hx, hy = float(scene.corridor_half_extent_x), float(scene.corridor_half_extent_y)
        return (abs(x - cx) >= hx) or (abs(y - cy) <= hy)

    if name.startswith("x_positive_"):
        i = int(parts[2])
        return states[t, 4 * i + 0] >= 0.0

    if name.startswith("corridor_pair_exclusion_"):
        i = int(parts[3])
        j = int(parts[4])
        cx, cy = float(scene.corridor_center[0]), float(scene.corridor_center[1])
        hx, hy = float(scene.corridor_half_extent_x), float(scene.corridor_half_extent_y)
        in_i = abs(states[t, 4 * i + 0] - cx) <= hx and abs(states[t, 4 * i + 1] - cy) <= hy
        in_j = abs(states[t, 4 * j + 0] - cx) <= hx and abs(states[t, 4 * j + 1] - cy) <= hy
        return (not in_i) or (not in_j)

    if name.startswith("gate_ok_"):
        a = int(parts[2])
        b = int(parts[3])
        goal_zone = int(parts[4])
        button_zone = int(parts[5])

        pressed = False
        for tau in range(t + 1):
            if eval_atom(f"inside_zone_{b}_{button_zone}", tau, states, task_cfg, scene, polygon_sides):
                pressed = True
                break
        outside_goal = eval_atom(f"outside_zone_{a}_{goal_zone}", t, states, task_cfg, scene, polygon_sides)
        return pressed or outside_goal

    raise ValueError(f"Unknown atom in verifier: {name}")


def eval_formula(node: FormulaNode, t: int, horizon: int, atom_eval: Callable[[str, int], bool]) -> bool:
    if t < 0 or t >= horizon:
        return False
    if isinstance(node, Atom):
        return atom_eval(node.name, t)
    if isinstance(node, AndNode):
        return all(eval_formula(c, t, horizon, atom_eval) for c in node.children)
    if isinstance(node, OrNode):
        return any(eval_formula(c, t, horizon, atom_eval) for c in node.children)
    if isinstance(node, AlwaysNode):
        idxs = [t + tau for tau in range(node.a, node.b + 1) if 0 <= (t + tau) < horizon]
        return bool(idxs) and all(eval_formula(node.child, idx, horizon, atom_eval) for idx in idxs)
    if isinstance(node, EventuallyNode):
        idxs = [t + tau for tau in range(node.a, node.b + 1) if 0 <= (t + tau) < horizon]
        return bool(idxs) and any(eval_formula(node.child, idx, horizon, atom_eval) for idx in idxs)
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def pointmass_dynamics_step(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    pos = state[..., :2]
    vel = state[..., 2:]
    pos_next = pos + dt * vel + 0.5 * (dt**2) * control
    vel_next = vel + dt * control
    return np.concatenate([pos_next, vel_next], axis=-1)


def rollout_open_loop(x0: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x0, dtype=np.float64)
    out = [x.copy()]
    for u in controls:
        x = pointmass_dynamics_step(x, np.asarray(u, dtype=np.float64), dt=dt)
        out.append(x.copy())
    return np.stack(out, axis=0)


def save_artifacts(
    artifact_dir: Path,
    task_cfg: TaskInstance,
    scene: SceneData,
    states: np.ndarray,
    controls: np.ndarray,
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(artifact_dir / f"{task_cfg.task_id}_states.csv", states, delimiter=",")
    np.savetxt(artifact_dir / f"{task_cfg.task_id}_controls.csv", controls, delimiter=",")
    (artifact_dir / f"{task_cfg.task_id}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        # Keep solver usable even when plotting deps are missing.
        return

    matplotlib.use("Agg", force=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"{task_cfg.task_id} trajectory")
    ax.set_xlim(float(scene.world_low[0]), float(scene.world_high[0]))
    ax.set_ylim(float(scene.world_low[1]), float(scene.world_high[1]))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    for c, r in zip(scene.obstacle_centers, scene.obstacle_radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="tab:red", alpha=0.25))
    for idx, (c, r) in enumerate(zip(scene.zone_centers, scene.zone_radii)):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="tab:green", alpha=0.2))
        ax.text(float(c[0]), float(c[1]), f"Z{idx}", ha="center", va="center", fontsize=8)
    if scene.goal_square_center is not None and scene.goal_square_side is not None:
        half = 0.5 * float(scene.goal_square_side)
        cx, cy = float(scene.goal_square_center[0]), float(scene.goal_square_center[1])
        ax.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, color="tab:green", linewidth=1.3))

    for a in range(scene.num_agents):
        xs = states[:, 4 * a + 0]
        ys = states[:, 4 * a + 1]
        ax.plot(xs, ys, linewidth=1.8, label=f"agent {a}")
        ax.scatter([xs[0]], [ys[0]], marker="o", s=30)
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=40)

    if scene.corridor_center is not None:
        cx, cy = float(scene.corridor_center[0]), float(scene.corridor_center[1])
        hx, hy = float(scene.corridor_half_extent_x), float(scene.corridor_half_extent_y)
        rect = plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, linestyle="--", linewidth=1.0)
        ax.add_patch(rect)

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(artifact_dir / f"{task_cfg.task_id}_trajectory.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    task_cfg = load_task_config(args.tasks_config, args.task_id, args.seed)
    scene = build_scene(task_cfg)

    formula = build_task_formula(task_cfg, scene)
    formula_str = node_to_string(formula)
    _ = try_pytelo_parse(formula_str, pytelo_root=args.pytelo_root)

    builder = MilpBuilder(
        task_cfg=task_cfg,
        scene=scene,
        polygon_sides=args.polygon_sides,
        time_limit_sec=args.time_limit_sec,
        mip_gap=args.mip_gap,
        gurobi_log_file=args.gurobi_log_file,
    )

    root_sat = builder.compile_node(formula, t=0)
    builder.model.addConstr(root_sat == 1, name="root_sat")
    builder.finalize_objective()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.write_lp:
        builder.model.write(str(args.artifact_dir / f"{task_cfg.task_id}.lp"))

    builder.model.optimize()

    status = int(builder.model.Status)
    status_name = {
        builder.GRB.OPTIMAL: "OPTIMAL",
        builder.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        builder.GRB.TIME_LIMIT: "TIME_LIMIT",
        builder.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        builder.GRB.INFEASIBLE: "INFEASIBLE",
        builder.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(status, f"STATUS_{status}")

    feasible = status in {
        builder.GRB.OPTIMAL,
        builder.GRB.SUBOPTIMAL,
        builder.GRB.TIME_LIMIT,
        builder.GRB.SOLUTION_LIMIT,
    } and builder.model.SolCount > 0

    def _safe_attr(name: str) -> Optional[float]:
        try:
            return float(getattr(builder.model, name))
        except Exception:
            return None

    obj_bound = _safe_attr("ObjBound")
    if obj_bound is None:
        obj_bound = _safe_attr("ObjBoundC")

    summary: dict[str, Any] = {
        "task_id": task_cfg.task_id,
        "task": task_cfg.task,
        "seed": task_cfg.seed,
        "horizon": task_cfg.horizon,
        "polygon_sides": args.polygon_sides,
        "status": status_name,
        "status_code": status,
        "feasible": feasible,
        "sol_count": int(builder.model.SolCount),
        "runtime_sec": float(builder.model.Runtime),
        "node_count": float(builder.model.NodeCount),
        "obj_val": _safe_attr("ObjVal") if feasible else None,
        "obj_bound": obj_bound if feasible else None,
        "mip_gap": _safe_attr("MIPGap") if feasible else None,
        "formula": formula_str,
    }

    if not feasible:
        print("Run complete")
        print(f"task: {task_cfg.task_id}")
        print(f"status: {status_name}")
        print("feasible: False")
        save_artifacts(args.artifact_dir, task_cfg, scene, np.zeros((0, 0)), np.zeros((0, 0)), summary)
        return

    states, controls = builder.extract_solution()

    sat_ok = None
    if args.verify_trace:
        atom_eval = lambda atom_name, tt: eval_atom(
            atom_name,
            tt,
            states,
            task_cfg,
            scene,
            args.polygon_sides,
        )
        sat_ok = bool(eval_formula(formula, 0, task_cfg.horizon, atom_eval))
        summary["verified_sat"] = sat_ok
        if not sat_ok:
            summary["verify_failure"] = "solver_feasible_but_trace_unsatisfied"
            summary["safety_margin"] = float(builder.gamma.X)
            save_artifacts(args.artifact_dir, task_cfg, scene, states, controls, summary)
            if args.verify_fail_as_warning:
                print(
                    "WARNING: Post-check failed: solver feasible but verifier marked STL unsatisfied. "
                    "Artifacts saved with verified_sat=false."
                )
                return
            raise RuntimeError("Post-check failed: solver feasible but verifier marked STL unsatisfied.")

    summary["safety_margin"] = float(builder.gamma.X)

    save_artifacts(args.artifact_dir, task_cfg, scene, states, controls, summary)

    print("Run complete")
    print(f"task: {task_cfg.task_id}")
    print(f"status: {status_name}")
    print(f"feasible: {feasible}")
    print(f"runtime_sec: {summary['runtime_sec']:.3f}")
    print(f"node_count: {summary['node_count']:.1f}")
    print(f"mip_gap: {summary['mip_gap']}")
    print(f"safety_margin: {summary['safety_margin']:.6f}")
    if sat_ok is not None:
        print(f"verified_sat: {sat_ok}")
    print(f"artifact_dir: {args.artifact_dir}")


if __name__ == "__main__":
    main()
