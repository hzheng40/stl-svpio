from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import jax

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    try:
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
    except PermissionError:
        pass

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print("Setting environment variable to use GPU rendering:")
os.environ["MUJOCO_GL"] = "egl"

try:
    print("Checking that the installation succeeded:")
    import mujoco

    mujoco.MjModel.from_xml_string("<mujoco/>")
except Exception as e:
    raise e from RuntimeError(
        "Something went wrong during installation. Check the shell output above "
        "for more information.\n"
        "If using a hosted Colab runtime, make sure you enable GPU acceleration "
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )

print("Installation successful.")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from stljax.formula import And, Eventually, Predicate

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController, make_stl_cost_fn


class MJCFPandaEnv(PipelineEnv):
    """Minimal Panda env wrapper that runs directly on a provided MJCF system."""

    def __init__(
        self,
        sys,
        backend: str = "generalized",
        n_frames: int = 1,
        reset_noise_scale: float = 0.0,
    ):
        super().__init__(sys=sys, backend=backend, n_frames=n_frames)
        self._reset_noise_scale = float(reset_noise_scale)

    def _get_obs(self, pipeline_state) -> jnp.ndarray:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=0)

    def reset(self, rng: jax.Array) -> State:
        if self._reset_noise_scale > 0.0:
            rng, rng1, rng2 = jax.random.split(rng, 3)
            low, hi = -self._reset_noise_scale, self._reset_noise_scale
            q = self.sys.init_q + jax.random.uniform(
                rng1, (self.sys.q_size(),), minval=low, maxval=hi
            )
            qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        else:
            q = self.sys.init_q
            qd = jnp.zeros((self.sys.qd_size(),), dtype=self.sys.init_q.dtype)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {"reward": jnp.zeros(())}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = state.metrics
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )


def panda_reach_two_furthest_boxes_spec(
    horizon: int,
    reach_radius: float,
    goal_indices: tuple[int, int],
    num_goals: int,
):
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if reach_radius <= 0:
        raise ValueError("reach_radius must be positive.")
    if len(goal_indices) != 2:
        raise ValueError("goal_indices must contain exactly two goals.")
    if goal_indices[0] == goal_indices[1]:
        raise ValueError("goal_indices must be unique.")
    if num_goals < 2:
        raise ValueError("num_goals must be at least 2.")
    if any(idx < 0 or idx >= num_goals for idx in goal_indices):
        raise ValueError(f"goal_indices entries must be in [0, {num_goals - 1}].")

    def _reach_predicate(goal_idx: int) -> Predicate:
        start = 3 + 3 * goal_idx
        stop = start + 3
        return Predicate(
            f"panda_ee_reaches_box{goal_idx + 1}",
            lambda tr, s=start, e=stop, r=reach_radius: r
            - jnp.linalg.norm(tr[:, 0:3] - tr[:, s:e], axis=-1),
        )

    ee_reaches_goal_a = _reach_predicate(goal_indices[0])
    ee_reaches_goal_b = _reach_predicate(goal_indices[1])
    return And(
        Eventually(ee_reaches_goal_a > 0.0, interval=[0, horizon - 1]),
        Eventually(ee_reaches_goal_b > 0.0, interval=[0, horizon - 1]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global-plan SVGD-MPPI on Panda reach task with two furthest cubes as goals."
    )
    parser.add_argument("--horizon", type=int, default=300, help="Planning horizon (timesteps).")
    parser.add_argument(
        "--backend",
        choices=["mjx", "generalized", "spring", "positional"],
        default="mjx",
        help="Brax physics backend.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=Path("src/stl_svpio/assets/franka_panda/mjx_three_cubes.xml"),
        help="Path to Panda scene MJCF XML file.",
    )
    parser.add_argument("--n-frames", type=int, default=1, help="Physics substeps per environment step.")
    parser.add_argument(
        "--mj-solver",
        choices=["cg", "newton"],
        default="cg",
        help="MuJoCo solver used when loading the custom MJCF.",
    )
    parser.add_argument("--mj-iterations", type=int, default=4, help="MuJoCo solver iterations.")
    parser.add_argument("--mj-ls-iterations", type=int, default=2, help="MuJoCo line-search iterations.")
    parser.add_argument("--reset-noise-scale", type=float, default=0.0, help="Uniform reset noise added to q and qd.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of MPPI control-sequence samples.")
    parser.add_argument("--svgd-iters", type=int, default=300, help="Number of SVGD refinement iterations.")
    parser.add_argument("--svgd-step-size", type=float, default=10.0, help="SVGD update step size.")
    parser.add_argument(
        "--svgd-step-anneal",
        choices=["none", "linear", "exp", "cosine"],
        default="exp",
        help="Annealing schedule for SVGD step size across iterations.",
    )
    parser.add_argument("--svgd-step-final", type=float, default=0.01, help="Final SVGD step size for annealing.")
    parser.add_argument("--temperature", type=float, default=0.8, help="MPPI temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--sampling-mode",
        choices=["gaussian", "uniform", "truncated_gaussian", "squashed_gaussian"],
        default="uniform",
        help="Control-sequence sampling distribution.",
    )
    parser.add_argument(
        "--update-mode",
        choices=["svgd", "importance_sampling"],
        default="svgd",
        help="MPPI update mode.",
    )
    parser.add_argument(
        "--ctrl-limit-scale",
        type=float,
        default=1.0,
        help="Scale factor for actuator control limits used by MPPI sampling (0<scale<=1).",
    )
    parser.add_argument("--control-noise-sigma", type=float, default=0.30, help="Control-noise sigma per actuator.")
    parser.add_argument("--reach-radius", type=float, default=0.06, help="Distance threshold for end-effector to reach box.")
    parser.add_argument("--stl-approx-method", type=str, default="logsumexp", help="STL robustness approximation method.")
    parser.add_argument("--stl-temperature", type=float, default=100.0, help="STL robustness temperature.")
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("artifacts/panda_reach_svgd_mppi_global_plan_topdown.png"),
        help="Where to save top-down global-plan plot.",
    )
    parser.add_argument("--no-artifacts", action="store_true", help="Disable artifact saving.")
    parser.add_argument(
        "--save-brax-video",
        type=Path,
        default=Path("artifacts/panda_reach_svgd_mppi_brax.mp4"),
        help="Where to save Brax renderer rollout video (.mp4 or .gif).",
    )
    parser.add_argument("--no-brax-video", action="store_true", help="Disable Brax-rendered rollout video export.")
    parser.add_argument("--animation-fps", type=int, default=100, help="Animation/video frames per second.")
    parser.add_argument("--render-width", type=int, default=640, help="Brax render frame width.")
    parser.add_argument("--render-height", type=int, default=480, help="Brax render frame height.")
    parser.add_argument("--render-camera", type=str, default=None, help="Optional camera name passed to Brax renderer.")
    parser.add_argument(
        "--snapshot-steps",
        type=str,
        default=None,
        help="Comma-separated rollout timesteps for Brax snapshots (e.g. '0,100,200,300'). Default: 4 equally spaced steps.",
    )
    parser.add_argument(
        "--save-brax-snapshots-dir",
        type=Path,
        default=Path("artifacts/panda_reach_svgd_mppi_brax_snapshots"),
        help="Directory where Brax snapshot PNGs are saved.",
    )
    parser.add_argument(
        "--save-xy-samples-animation",
        type=Path,
        default=Path("artifacts/panda_reach_svgd_mppi_sampling_xy.gif"),
        help="Where to save XY animation of SVGD sampled trajectories (.gif or .mp4).",
    )
    parser.add_argument(
        "--no-xy-samples-animation",
        action="store_true",
        help="Disable XY animation export of SVGD sampled trajectories.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("artifacts/panda_reach_svgd_mppi_results.json"),
        help="Where to save run metrics (runtime, final robustness).",
    )
    parser.add_argument("--svgd-progress", action="store_true", help="Show SVGD progress bar.")
    parser.add_argument(
        "--jit-command",
        action="store_true",
        help="JIT-compile MPPI command. Disabled by default for execution safety.",
    )
    parser.add_argument(
        "--svgd-grad-mode",
        choices=["reverse", "forward"],
        default="reverse",
        help="Autodiff mode for SVGD particle gradients.",
    )
    return parser.parse_args()


def _resolve_ids(env) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    ee_site_id = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    if ee_site_id < 0:
        raise ValueError("Could not find Panda end-effector site `gripper` in MJCF.")

    hand_body_id = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_body_id < 0:
        raise ValueError("Could not find Panda hand body `hand` in MJCF.")

    box_body_ids: list[int] = []
    box_geom_ids: list[int] = []
    for box_name in ("box1", "box2"):
        box_body_id = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, box_name)
        if box_body_id < 0:
            raise ValueError(f"Could not find target body `{box_name}` in MJCF.")
        box_geom_id = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, box_name)
        if box_geom_id < 0:
            raise ValueError(f"Could not find target geom `{box_name}` in MJCF.")
        box_body_ids.append(int(box_body_id))
        box_geom_ids.append(int(box_geom_id))

    return int(ee_site_id), int(hand_body_id), tuple(box_body_ids), tuple(box_geom_ids)


def _resolve_link_index(env, body_id: int) -> int:
    body_name = mujoco.mj_id2name(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    link_names = list(getattr(env.sys, "link_names", ()))
    if body_name in link_names:
        return int(link_names.index(body_name))
    return max(int(body_id) - 1, 0)


def _planner_trace_from_data(
    data,
    ee_site_id: int,
    hand_link_id: int,
    box_link_ids: tuple[int, ...],
) -> jnp.ndarray:
    if hasattr(data, "site_xpos"):
        ee_pos = data.site_xpos[ee_site_id]
    elif hasattr(data, "x") and hasattr(data.x, "pos"):
        ee_pos = data.x.pos[hand_link_id]
    else:
        raise AttributeError("Cannot extract end-effector position from pipeline state.")

    box_positions = []
    for box_link_id in box_link_ids:
        if hasattr(data, "x") and hasattr(data.x, "pos"):
            box_pos = data.x.pos[box_link_id]
        elif hasattr(data, "xpos"):
            box_pos = data.xpos[box_link_id]
        else:
            raise AttributeError("Cannot extract box position from pipeline state.")
        box_positions.append(box_pos)

    return jnp.concatenate([ee_pos, *box_positions], axis=0)


def _tile_tree(tree, batch_size: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None, ...], (batch_size,) + x.shape), tree
    )


def _rollout_open_loop(
    env,
    data0,
    control_seq: jnp.ndarray,
    ee_site_id: int,
    hand_link_id: int,
    box_link_ids: tuple[int, ...],
):
    ctrl_dtype = data0.ctrl.dtype if hasattr(data0, "ctrl") else control_seq.dtype
    control_seq = control_seq.astype(ctrl_dtype)

    def step_fn(carry, u):
        data_next = env.pipeline_step(carry, u)
        trace_next = _planner_trace_from_data(data_next, ee_site_id, hand_link_id, box_link_ids)
        return data_next, (trace_next, data_next)

    _, (trace_hist, state_hist) = jax.lax.scan(step_fn, data0, control_seq)
    return trace_hist, state_hist


def _tree_time_to_list(tree_time_major: Any) -> list[Any]:
    leaves = jax.tree_util.tree_leaves(tree_time_major)
    if not leaves:
        return []
    n = int(leaves[0].shape[0])
    return [jax.tree_util.tree_map(lambda x, i=i: x[i], tree_time_major) for i in range(n)]


def _render_brax_rollout_frames(
    env,
    initial_state,
    state_hist_time_major,
    width: int,
    height: int,
    camera: str | None,
) -> list[Any]:
    trajectory = [initial_state] + _tree_time_to_list(state_hist_time_major)
    return env.render(trajectory, height=height, width=width, camera=camera)


def _save_brax_rollout_video(
    frames,
    save_path: Path,
    fps: int,
) -> None:
    import numpy as np
    from PIL import Image

    if len(frames) == 0:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix == ".gif":
        duration_ms = max(1, int(round(1000 / max(fps, 1))))
        pil_frames = [Image.fromarray(np.asarray(frame)) for frame in frames]
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return

    try:
        import mediapy as media
    except ImportError as e:
        raise ImportError(
            "mediapy is required to write non-gif video outputs; install mediapy or use --save-brax-video *.gif."
        ) from e

    media.write_video(str(save_path), frames, fps=fps)


def _resolve_snapshot_steps(snapshot_steps_arg: str | None, max_timestep: int) -> list[int]:
    if max_timestep < 0:
        return []

    if snapshot_steps_arg is not None and snapshot_steps_arg.strip() != "":
        out: list[int] = []
        seen: set[int] = set()
        for token in snapshot_steps_arg.split(","):
            tok = token.strip()
            if tok == "":
                continue
            try:
                step = int(tok)
            except ValueError as e:
                raise ValueError(f"Invalid snapshot step `{tok}`; expected integer.") from e
            if step < 0 or step > max_timestep:
                raise ValueError(
                    f"Snapshot step {step} out of range [0, {max_timestep}]."
                )
            if step not in seen:
                seen.add(step)
                out.append(step)
        if not out:
            raise ValueError("--snapshot-steps did not contain any valid timestep.")
        return out

    # Default: 4 equally spaced timesteps over [0, max_timestep].
    if max_timestep == 0:
        return [0]
    steps = jnp.linspace(0, max_timestep, 4)
    out = [int(round(float(s))) for s in steps]
    deduped: list[int] = []
    seen: set[int] = set()
    for step in out:
        if step not in seen:
            seen.add(step)
            deduped.append(step)
    return deduped


def _save_brax_snapshots(
    frames,
    timesteps: list[int],
    save_dir: Path,
) -> list[Path]:
    import numpy as np
    from PIL import Image

    if len(frames) == 0:
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for t in timesteps:
        img = Image.fromarray(np.asarray(frames[t]))
        out_path = save_dir / f"snapshot_t{t:04d}.png"
        img.save(out_path)
        saved_paths.append(out_path)
    return saved_paths


def _save_run_results(
    results_path: Path,
    runtime_seconds: float,
    planning_runtime_seconds: float,
    final_robustness: float,
    stl_satisfied: bool,
) -> None:
    payload = {
        "runtime_seconds": float(runtime_seconds),
        "planning_runtime_seconds": float(planning_runtime_seconds),
        "final_stl_robustness": float(final_robustness),
        "stl_satisfied": bool(stl_satisfied),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _plot_topdown_global_plan(
    trace_hist: jnp.ndarray,
    selected_goal_indices: tuple[int, int],
    reach_radius: float,
    save_path: Path,
    show: bool,
) -> None:
    import numpy as np

    def _finite_xy_rows(xy: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(xy), axis=1)

    ee_xy = trace_hist[:, 0:2]
    goal_a_idx, goal_b_idx = selected_goal_indices
    goal_a_xy = trace_hist[0, 3 + 3 * goal_a_idx : 5 + 3 * goal_a_idx]
    goal_b_xy = trace_hist[0, 3 + 3 * goal_b_idx : 5 + 3 * goal_b_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Panda final global plan (top-down x-y)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ee_xy_np = np.asarray(ee_xy, dtype=np.float32)
    ee_mask = _finite_xy_rows(ee_xy_np)
    if np.any(ee_mask):
        ee_xy_plot = ee_xy_np[ee_mask]
        ax.plot(
            ee_xy_plot[:, 0],
            ee_xy_plot[:, 1],
            "-",
            color="tab:blue",
            lw=2.2,
            label="EE trajectory",
        )
        ax.scatter(
            [ee_xy_plot[0, 0]],
            [ee_xy_plot[0, 1]],
            color="tab:blue",
            marker="o",
            s=42,
            label="EE start",
        )
        ax.scatter(
            [ee_xy_plot[-1, 0]],
            [ee_xy_plot[-1, 1]],
            color="tab:blue",
            marker="x",
            s=60,
            label="EE end",
        )

    goal_colors = ("tab:orange", "tab:red")
    goal_points = (goal_a_xy, goal_b_xy)
    goal_labels = (f"goal box{goal_a_idx + 1}", f"goal box{goal_b_idx + 1}")
    for gxy, gcolor, glabel in zip(goal_points, goal_colors, goal_labels):
        gx = float(gxy[0])
        gy = float(gxy[1])
        if not (np.isfinite(gx) and np.isfinite(gy)):
            continue
        ax.scatter([gx], [gy], color=gcolor, marker="*", s=130, label=f"{glabel} (xy proj)")
        reach_circle = plt.Circle(
            (gx, gy),
            float(reach_radius),
            color=gcolor,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.add_patch(reach_circle)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _animate_svgd_samples_xy(
    sampled_trajectories: jnp.ndarray,
    sampled_costs: jnp.ndarray,
    selected_goal_indices: tuple[int, int],
    reach_radius: float,
    save_path: Path,
    fps: int,
) -> None:
    import numpy as np
    from PIL import Image

    def _finite_xy_rows(xy: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(xy), axis=1)

    n_iters = int(sampled_trajectories.shape[0])
    if n_iters == 0:
        return

    all_xy = np.asarray(sampled_trajectories[..., 0:2], dtype=np.float32).reshape((-1, 2))
    finite_all = _finite_xy_rows(all_xy)
    if np.any(finite_all):
        finite_xy = all_xy[finite_all]
        min_xy = np.min(finite_xy, axis=0)
        max_xy = np.max(finite_xy, axis=0)
    else:
        min_xy = np.array([-1.0, -1.0], dtype=np.float32)
        max_xy = np.array([1.0, 1.0], dtype=np.float32)

    goal_a_idx, goal_b_idx = selected_goal_indices
    goal_a_xy = np.asarray(sampled_trajectories[0, 0, 0, 3 + 3 * goal_a_idx : 5 + 3 * goal_a_idx], dtype=np.float32)
    goal_b_xy = np.asarray(sampled_trajectories[0, 0, 0, 3 + 3 * goal_b_idx : 5 + 3 * goal_b_idx], dtype=np.float32)
    goal_points = [goal_a_xy, goal_b_xy]
    for gxy in goal_points:
        if np.all(np.isfinite(gxy)):
            min_xy = np.minimum(min_xy, gxy)
            max_xy = np.maximum(max_xy, gxy)

    center = 0.5 * (min_xy + max_xy)
    span = float(np.max(max_xy - min_xy))
    half_span = 0.55 * max(span, 1e-3)
    xlim = (float(center[0] - half_span), float(center[0] + half_span))
    ylim = (float(center[1] - half_span), float(center[1] + half_span))

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    frames: list[Image.Image] = []

    for it in range(n_iters):
        ax.cla()
        ax.set_title(f"Panda SVGD samples (x-y) {it + 1}/{n_iters}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        costs = np.asarray(sampled_costs[it], dtype=np.float32)
        finite_costs = np.where(np.isfinite(costs), costs, np.nan)
        if np.all(np.isnan(finite_costs)):
            q = np.zeros_like(costs, dtype=np.float32)
        else:
            cmin = float(np.nanmin(finite_costs))
            cmax = float(np.nanmax(finite_costs))
            safe_costs = np.where(np.isfinite(costs), costs, cmax)
            q = (cmax - safe_costs) / (cmax - cmin + 1e-8)
        q = np.clip(q, 0.0, 1.0)

        ee_xy = np.asarray(sampled_trajectories[it, :, :, 0:2], dtype=np.float32)
        for pidx in range(ee_xy.shape[0]):
            traj = ee_xy[pidx]
            mask = _finite_xy_rows(traj)
            if not np.any(mask):
                continue
            traj_plot = traj[mask]
            color = cmap(float(q[pidx]))
            ax.plot(
                traj_plot[:, 0],
                traj_plot[:, 1],
                color=color,
                alpha=0.35,
                linewidth=1.0,
            )
            ax.scatter(
                [traj_plot[-1, 0]],
                [traj_plot[-1, 1]],
                color=color,
                s=12,
                alpha=0.75,
            )

        goal_colors = ("tab:orange", "tab:red")
        goal_labels = (f"goal box{goal_a_idx + 1}", f"goal box{goal_b_idx + 1}")
        for gxy, gcolor, glabel in zip(goal_points, goal_colors, goal_labels):
            if not np.all(np.isfinite(gxy)):
                continue
            gx = float(gxy[0])
            gy = float(gxy[1])
            ax.scatter([gx], [gy], color=gcolor, marker="*", s=120, label=f"{glabel} (xy proj)")
            ax.add_patch(
                plt.Circle(
                    (gx, gy),
                    float(reach_radius),
                    color=gcolor,
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                )
            )
        if it == 0:
            ax.legend(loc="best")

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(Image.fromarray(img[..., :3]))

    plt.close(fig)
    if len(frames) == 0:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix == ".gif":
        duration_ms = max(1, int(round(1000 / max(fps, 1))))
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return

    try:
        import mediapy as media
    except ImportError as e:
        raise ImportError(
            "mediapy is required to write non-gif animation outputs; install mediapy or use *.gif."
        ) from e
    media.write_video(str(save_path), [np.asarray(f) for f in frames], fps=fps)


def _make_panda_env(args: argparse.Namespace):
    mjcf_path = args.mjcf_path.expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    # `mjcf.load` can fail on this include stack with a schema error; loading via
    # MuJoCo first is robust and still yields a Brax system.
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path.as_posix())
    mj_model.opt.solver = (
        mujoco.mjtSolver.mjSOL_CG
        if args.mj_solver == "cg"
        else mujoco.mjtSolver.mjSOL_NEWTON
    )
    # Force full solver iterations (no early stopping) for stable reverse-mode
    # differentiation through contact solver iterations.
    mj_model.opt.tolerance = 0.0
    mj_model.opt.ls_tolerance = 0.0
    mj_model.opt.iterations = int(args.mj_iterations)
    mj_model.opt.ls_iterations = int(args.mj_ls_iterations)
    sys = mjcf.load_model(mj_model)

    return MJCFPandaEnv(
        sys=sys,
        backend=args.backend,
        n_frames=args.n_frames,
        reset_noise_scale=args.reset_noise_scale,
    )


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    if args.ctrl_limit_scale <= 0.0 or args.ctrl_limit_scale > 1.0:
        raise ValueError("--ctrl-limit-scale must be in (0, 1].")
    if args.backend == "mjx" and args.svgd_grad_mode == "reverse" and args.mj_iterations != 1:
        print(
            "Reverse-mode with MJX requires --mj-iterations=1 "
            "(MJX solver uses lax.while_loop for iterations>1). "
            f"Overriding mj_iterations from {args.mj_iterations} to 1."
        )
        args.mj_iterations = 1
    if not args.show:
        matplotlib.use("Agg", force=True)

    env = _make_panda_env(args)
    reset_state = env.reset(jax.random.PRNGKey(args.seed))
    data0 = reset_state.pipeline_state

    ee_site_id, hand_body_id, box_body_ids, _ = _resolve_ids(env)
    hand_link_id = _resolve_link_index(env, hand_body_id)
    box_link_ids = tuple(_resolve_link_index(env, bid) for bid in box_body_ids)
    if len(box_link_ids) < 2:
        raise ValueError("Panda scene must define at least two goals (box1, box2).")
    initial_trace = _planner_trace_from_data(data0, ee_site_id, hand_link_id, box_link_ids)
    num_goals = len(box_link_ids)
    goal_positions0 = initial_trace[3:].reshape((num_goals, 3))
    initial_dists = jnp.linalg.norm(goal_positions0 - initial_trace[0:3][None, :], axis=-1)
    sorted_desc = jnp.argsort(initial_dists)[::-1]
    selected_goal_indices = (int(sorted_desc[0]), int(sorted_desc[1]))

    spec = panda_reach_two_furthest_boxes_spec(
        horizon=args.horizon,
        reach_radius=args.reach_radius,
        goal_indices=selected_goal_indices,
        num_goals=num_goals,
    )
    mppi_cost = make_stl_cost_fn(
        spec,
        approx_method=args.stl_approx_method,
        temperature=args.stl_temperature,
    )

    control_low = (args.ctrl_limit_scale * env.sys.actuator.ctrl_range[:, 0]).astype(jnp.float32)
    control_high = (args.ctrl_limit_scale * env.sys.actuator.ctrl_range[:, 1]).astype(jnp.float32)
    control_noise_sigma = args.control_noise_sigma * jnp.ones((env.sys.nu,), dtype=jnp.float32)

    def rollout_batch_pipeline(x0_data, controls: jnp.ndarray) -> jnp.ndarray:
        num_particles = controls.shape[0]
        ctrl_dtype = x0_data.ctrl.dtype if hasattr(x0_data, "ctrl") else controls.dtype
        controls = controls.astype(ctrl_dtype)
        data_batch = _tile_tree(x0_data, num_particles)
        u_time_major = jnp.swapaxes(controls, 0, 1)

        def scan_step(carry_data_batch, u_t_batch):
            next_data_batch = jax.vmap(env.pipeline_step)(carry_data_batch, u_t_batch)
            traces = jax.vmap(_planner_trace_from_data, in_axes=(0, None, None, None))(
                next_data_batch, ee_site_id, hand_link_id, box_link_ids
            )
            return next_data_batch, traces

        _, traces_time_major = jax.lax.scan(scan_step, data_batch, u_time_major)
        return jnp.swapaxes(traces_time_major, 0, 1)

    controller_cfg = MPPIConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        control_dim=env.sys.nu,
        sampling_mode=args.sampling_mode,
        update_mode=args.update_mode,
        control_low=control_low,
        control_high=control_high,
        control_noise_sigma=control_noise_sigma,
        temperature=args.temperature,
        svgd_iters=args.svgd_iters,
        svgd_step_size=args.svgd_step_size,
        svgd_step_size_anneal=args.svgd_step_anneal,
        svgd_step_size_final=args.svgd_step_final,
        svgd_grad_mode=args.svgd_grad_mode,
        svgd_selection_mode="best",
        record_svgd_history=(args.update_mode == "svgd"),
        show_svgd_progress=args.svgd_progress,
    )
    controller = MPPIController(
        config=controller_cfg,
        dynamics_fn=lambda x, _u: x,
        cost_fn=mppi_cost,
        rollout_batch_fn=rollout_batch_pipeline,
    )
    mppi_state = controller.init_state(jax.random.PRNGKey(args.seed + 1000))
    command_fn = jax.jit(controller.command) if args.jit_command else controller.command
    planning_start = time.perf_counter()
    _, _, info = command_fn(mppi_state, data0)
    planning_runtime_seconds = time.perf_counter() - planning_start

    planned_controls = info["selected_controls"]
    trace_hist, state_hist = _rollout_open_loop(
        env, data0, planned_controls, ee_site_id, hand_link_id, box_link_ids
    )
    goal_hist = trace_hist[:, 3:].reshape((trace_hist.shape[0], num_goals, 3))
    ee_hist = trace_hist[:, 0:3]
    all_dists = jnp.linalg.norm(goal_hist - ee_hist[:, None, :], axis=-1)  # [T, G]
    final_dists = [float(all_dists[-1, i]) for i in range(num_goals)]
    min_dists = [float(jnp.min(all_dists[:, i])) for i in range(num_goals)]
    reached_selected = all(min_dists[idx] <= args.reach_radius for idx in selected_goal_indices)
    final_rob = float(
        spec.robustness(
            trace_hist,
            approx_method=args.stl_approx_method,
            temperature=args.stl_temperature,
        )
    )
    final_sat = bool(spec.eval(trace_hist))

    print("Run complete")
    print(f"horizon: {args.horizon}")
    print(f"num_samples: {args.num_samples}")
    print(f"svgd_iters: {args.svgd_iters}")
    print(f"control_dim: {env.sys.nu}")
    print(
        "selected furthest goals at t=0: "
        f"box{selected_goal_indices[0] + 1} ({float(initial_dists[selected_goal_indices[0]]):.6f} m), "
        f"box{selected_goal_indices[1] + 1} ({float(initial_dists[selected_goal_indices[1]]):.6f} m)"
    )
    for i in range(num_goals):
        print(f"terminal distance to box{i + 1}: {final_dists[i]:.6f}")
    for i in range(num_goals):
        print(f"minimum distance to box{i + 1}: {min_dists[i]:.6f}")
    print(f"selected two goals reached (order-free check): {reached_selected}")
    print(f"final STL robustness: {final_rob:.6f}")
    print(f"STL satisfied: {final_sat}")
    print(f"planning runtime (s): {planning_runtime_seconds:.6f}")
    runtime_seconds = time.perf_counter() - run_start
    print(f"runtime (s): {runtime_seconds:.6f}")
    _save_run_results(
        results_path=args.results_path,
        runtime_seconds=runtime_seconds,
        planning_runtime_seconds=planning_runtime_seconds,
        final_robustness=final_rob,
        stl_satisfied=final_sat,
    )
    print(f"saved run results: {args.results_path}")

    artifacts_enabled = not args.no_artifacts
    if artifacts_enabled:
        _plot_topdown_global_plan(
            trace_hist=trace_hist,
            selected_goal_indices=selected_goal_indices,
            reach_radius=args.reach_radius,
            save_path=args.save_path,
            show=args.show,
        )
        print(f"saved top-down global-plan plot: {args.save_path}")

        if not args.no_brax_video:
            frames = _render_brax_rollout_frames(
                env=env,
                initial_state=data0,
                state_hist_time_major=state_hist,
                width=args.render_width,
                height=args.render_height,
                camera=args.render_camera,
            )
            _save_brax_rollout_video(
                frames=frames,
                save_path=args.save_brax_video,
                fps=args.animation_fps,
            )
            print(f"saved brax-rendered rollout video: {args.save_brax_video}")
            snapshot_steps = _resolve_snapshot_steps(
                snapshot_steps_arg=args.snapshot_steps,
                max_timestep=max(len(frames) - 1, 0),
            )
            _save_brax_snapshots(
                frames=frames,
                timesteps=snapshot_steps,
                save_dir=args.save_brax_snapshots_dir,
            )
            print(
                f"saved brax snapshots at timesteps {snapshot_steps}: "
                f"{args.save_brax_snapshots_dir}"
            )
        if (
            not args.no_xy_samples_animation
            and args.update_mode == "svgd"
            and "svgd_iter_trajectories" in info
            and "svgd_iter_costs" in info
        ):
            _animate_svgd_samples_xy(
                sampled_trajectories=info["svgd_iter_trajectories"],
                sampled_costs=info["svgd_iter_costs"],
                selected_goal_indices=selected_goal_indices,
                reach_radius=args.reach_radius,
                save_path=args.save_xy_samples_animation,
                fps=args.animation_fps,
            )
            print(f"saved xy samples animation: {args.save_xy_samples_animation}")


if __name__ == "__main__":
    main()
