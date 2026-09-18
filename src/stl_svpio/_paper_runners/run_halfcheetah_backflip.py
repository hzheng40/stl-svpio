from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import time
# Rendering defaults must not override the caller's GPU or driver setup.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
from typing import Any

import brax.envs
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_default_matmul_precision", "high")
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mujoco
from brax.io import mjcf

try:
    from brax.envs.half_cheetah import Halfcheetah
except ImportError:
    from brax.envs.halfcheetah import Halfcheetah  # type: ignore

from stl_svpio._legacy_mppi import MPPIConfig, MPPIController, make_stl_cost_fn
from stl_svpio.specifications import halfcheetah_backflip_spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global-plan SVGD-MPPI on Brax HalfCheetah with STL backflip specification."
    )
    parser.add_argument("--horizon", type=int, default=200, help="Planning horizon (timesteps).")
    parser.add_argument(
        "--backend",
        choices=["mjx", "generalized", "spring", "positional"],
        default="mjx",
        help="Brax physics backend.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets/half_cheetah/half_cheetah.xml",
        help="Path to HalfCheetah MJCF XML file.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="Physics substeps per environment step. If unset, uses backend default.",
    )
    parser.add_argument(
        "--mj-solver",
        choices=["cg", "newton"],
        default="cg",
        help="MuJoCo solver used when loading the custom MJCF.",
    )
    parser.add_argument(
        "--mj-iterations",
        type=int,
        default=1,
        help="MuJoCo solver iterations (lower is faster, less accurate).",
    )
    parser.add_argument(
        "--mj-ls-iterations",
        type=int,
        default=1,
        help="MuJoCo line-search iterations (lower is faster).",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of MPPI control-sequence samples.")
    parser.add_argument("--svgd-iters", type=int, default=300, help="Number of SVGD refinement iterations.")
    parser.add_argument("--svgd-step-size", type=float, default=0.001, help="SVGD update step size.")
    parser.add_argument(
        "--svgd-step-anneal",
        choices=["none", "linear", "exp", "cosine"],
        default="exp",
        help="Annealing schedule for SVGD step size across iterations.",
    )
    parser.add_argument(
        "--svgd-step-final",
        type=float,
        default=1e-5,
        help="Final SVGD step size for annealing schedules; defaults to initial step if unset.",
    )
    parser.add_argument(
        "--svgd-repulsion-coef",
        type=float,
        default=0.0,
        help="SVGD repulsion coefficient.",
    )
    parser.add_argument(
        "--svgd-repulsion-anneal",
        choices=["none", "linear", "exp", "cosine"],
        default="none",
        help="Annealing schedule for SVGD repulsion coefficient across iterations.",
    )
    parser.add_argument(
        "--svgd-repulsion-final",
        type=float,
        default=None,
        help="Final SVGD repulsion coefficient for annealing schedules.",
    )
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
    parser.add_argument("--control-noise-sigma", type=float, default=0.35, help="Control-noise sigma per actuator.")
    parser.add_argument(
        "--save-controls-path",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_controls.npy"),
        help="Where to save planned control sequence (.npy).",
    )
    parser.add_argument(
        "--load-controls",
        action="store_true",
        help="Load a saved control sequence and skip MPPI optimization.",
    )
    parser.add_argument(
        "--load-controls-path",
        type=Path,
        default=None,
        help="Path to load control sequence (.npy). Defaults to --save-controls-path.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_rollout.png"),
        help="Where to save rollout plot.",
    )
    parser.add_argument("--no-artifacts", action="store_true", help="Disable artifact saving.")
    parser.add_argument(
        "--save-animation",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_sampling.gif"),
        help="Where to save MPPI sampling animation (.gif).",
    )
    parser.add_argument(
        "--save-global-plan-animation",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_global_plan.gif"),
        help="Where to save global-plan animation (.gif).",
    )
    parser.add_argument(
        "--save-control-animation",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_controls.gif"),
        help="Where to save control-space animation (.gif or .mp4).",
    )
    parser.add_argument(
        "--save-brax-video",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_brax.mp4"),
        help="Where to save Brax renderer rollout video (.mp4 or .gif).",
    )
    parser.add_argument("--animation-fps", type=int, default=100, help="Animation frames per second.")
    parser.add_argument("--no-animation", action="store_true", help="Disable animation export.")
    parser.add_argument("--no-brax-video", action="store_true", help="Disable Brax-rendered rollout video export.")
    parser.add_argument("--render-width", type=int, default=640, help="Brax render frame width.")
    parser.add_argument("--render-height", type=int, default=480, help="Brax render frame height.")
    parser.add_argument("--render-camera", type=str, default=None, help="Optional camera name passed to Brax renderer.")
    parser.add_argument(
        "--snapshot-steps",
        type=str,
        default=None,
        help="Comma-separated rollout timesteps for Brax snapshots (e.g. '0,100,200'). Default: 4 equally spaced steps.",
    )
    parser.add_argument(
        "--save-brax-snapshots-dir",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_brax_snapshots"),
        help="Directory where Brax snapshot PNGs are saved.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("artifacts/halfcheetah_svgd_mppi_results.json"),
        help="Where to save run metrics (runtime, final robustness).",
    )
    parser.add_argument("--stl-approx-method", type=str, default="logsumexp", help="STL robustness approximation method.")
    parser.add_argument("--stl-temperature", type=float, default=500.0, help="STL robustness temperature.")
    parser.add_argument(
        "--completion-style",
        choices=["terminal", "staged", "stabilized", "feet_head_order"],
        default="terminal",
        help="Backflip STL completion style.",
    )
    parser.add_argument("--target-rotation", type=float, default=float(2.0 * jnp.pi), help="Target backward rotation (rad).")
    parser.add_argument("--rotation-tolerance", type=float, default=0.40, help="Tolerance around target rotation (rad).")
    parser.add_argument("--final-window-steps", type=int, default=8, help="Require completion in this many final steps.")
    parser.add_argument("--min-torso-height", type=float, default=-0.6, help="Minimum allowed torso root-z.")
    parser.add_argument("--max-abs-pitch-rate", type=float, default=40.0, help="Global pitch-rate bound |dtheta/dt|.")
    parser.add_argument("--stabilization-steps", type=int, default=4, help="Hold-length used by stabilized spec.")
    parser.add_argument("--stabilization-pitch-rate", type=float, default=6.0, help="Max |pitch_rate| during stabilized hold.")
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
    args = parser.parse_args(argv)
    if args.backend == "mjx" and args.svgd_grad_mode == "reverse" and args.mj_iterations != 1:
        parser.error("MJX reverse-mode gradients require --mj-iterations 1; larger values use a nondifferentiable solver loop.")
    return args


def _planner_trace_from_data(data, bfoot_body_id: int, ffoot_body_id: int) -> jnp.ndarray:
    # [root_x, root_z, root_pitch, root_xd, root_zd, root_pitchd, joint_0..joint_5, bfoot_z, ffoot_z]
    bfoot_z = data.x.pos[bfoot_body_id, 2:3]
    ffoot_z = data.x.pos[ffoot_body_id, 2:3]
    return jnp.concatenate([data.q[:3], data.qd[:3], data.q[3:], bfoot_z, ffoot_z], axis=0)


def _tile_tree(tree, batch_size: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None, ...], (batch_size,) + x.shape), tree
    )


def _rollout_open_loop(env, data0, control_seq: jnp.ndarray, bfoot_body_id: int, ffoot_body_id: int):
    ctrl_dtype = data0.ctrl.dtype if hasattr(data0, "ctrl") else control_seq.dtype
    control_seq = control_seq.astype(ctrl_dtype)

    def step_fn(carry, u):
        data_next = env.pipeline_step(carry, u)
        trace_next = _planner_trace_from_data(data_next, bfoot_body_id, ffoot_body_id)
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


def _save_control_sequence(control_seq: jnp.ndarray, save_path: Path) -> None:
    import numpy as np

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.asarray(control_seq, dtype=np.float32))


def _save_run_results(
    results_path: Path,
    runtime_seconds: float,
    final_robustness: float,
    stl_satisfied: bool,
    planning_executed: bool,
    args: argparse.Namespace,
) -> None:
    payload = {
        "runtime_seconds": float(runtime_seconds),
        "final_stl_robustness": float(final_robustness),
        "stl_satisfied": bool(stl_satisfied),
        "planning_executed": bool(planning_executed),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sampling_seed": args.seed + 1000,
        "backend": jax.default_backend(),
        "devices": [device.device_kind for device in jax.devices()],
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "stljax")},
        "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "MUJOCO_GL")},
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_control_sequence(load_path: Path, control_dim: int) -> jnp.ndarray:
    import numpy as np

    if not load_path.exists():
        raise FileNotFoundError(f"Control sequence file not found: {load_path}")
    controls = np.load(load_path)
    if controls.ndim != 2:
        raise ValueError(
            f"Expected loaded controls to have shape [T, U], got {controls.shape}."
        )
    if int(controls.shape[1]) != int(control_dim):
        raise ValueError(
            f"Loaded controls have control_dim={controls.shape[1]}, expected {control_dim}."
        )
    return jnp.asarray(controls, dtype=jnp.float32)


def _make_halfcheetah_env(args: argparse.Namespace):
    mjcf_path = args.mjcf_path.expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    mj_model = mujoco.MjModel.from_xml_path(mjcf_path.as_posix())
    mj_model.opt.solver = (
        mujoco.mjtSolver.mjSOL_CG
        if args.mj_solver == "cg"
        else mujoco.mjtSolver.mjSOL_NEWTON
    )
    mj_model.opt.iterations = int(args.mj_iterations)
    mj_model.opt.ls_iterations = int(args.mj_ls_iterations)
    sys = mjcf.load_model(mj_model)

    env_kwargs = {"backend": args.backend}
    if args.n_frames is not None:
        env_kwargs["n_frames"] = args.n_frames

    try:
        return Halfcheetah(sys=sys, **env_kwargs)
    except TypeError:
        # Older Brax versions don't accept `sys` in Halfcheetah constructor.
        # Build the stock env, then swap in our custom system.
        env = brax.envs.get_environment("halfcheetah", **env_kwargs)
        env.sys = sys
        return env


def _plot_rollout(trace_hist: jnp.ndarray, save_path: Path, show: bool) -> None:
    x = trace_hist[:, 0]
    z = trace_hist[:, 1]
    pitch = trace_hist[:, 2]
    pitch_rate = trace_hist[:, 5]
    t = jnp.arange(trace_hist.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0, ax1 = axes
    ax0.set_title("HalfCheetah Root X-Z Path")
    ax0.grid(True, alpha=0.3)
    ax0.plot(x, z, "k-", lw=2.0, label="root path")
    ax0.plot(x[0], z[0], "go", ms=7, label="start")
    ax0.plot(x[-1], z[-1], "mo", ms=7, label="end")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")
    ax0.legend(loc="best")

    ax1.set_title("Pitch Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.plot(t, pitch, color="tab:blue", lw=2.0, label="pitch (rad)")
    ax1.plot(t, pitch_rate, color="tab:orange", lw=1.5, alpha=0.8, label="pitch_rate")
    ax1.set_xlabel("timestep")
    ax1.legend(loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _animate_sampling_iterations_xz(
    sampled_trajectories: jnp.ndarray,
    sampled_costs: jnp.ndarray,
    selected_trajectories: jnp.ndarray,
    executed_trace: jnp.ndarray,
    save_path: Path,
    fps: int = 10,
) -> None:
    import numpy as np
    from PIL import Image

    n_frames = int(sampled_trajectories.shape[0])
    if n_frames == 0:
        return

    all_pts = jnp.concatenate(
        [
            sampled_trajectories[..., :2].reshape((-1, 2)),
            selected_trajectories[..., :2].reshape((-1, 2)),
            executed_trace[:, :2],
        ],
        axis=0,
    )
    mins = jnp.min(all_pts, axis=0)
    maxs = jnp.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(jnp.max(maxs - mins))
    half_span = 0.60 * max(span, 1e-3)
    x_low, x_high = float(center[0] - half_span), float(center[0] + half_span)
    z_low, z_high = float(center[1] - half_span), float(center[1] + half_span)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    frames = []
    for t in range(n_frames):
        ax.clear()
        ax.set_title(f"HalfCheetah MPPI sampling {t + 1}/{n_frames}")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(z_low, z_high)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("root x")
        ax.set_ylabel("root z")

        costs = sampled_costs[t]
        cmin = jnp.min(costs)
        cmax = jnp.max(costs)
        q = (cmax - costs) / (cmax - cmin + 1e-8)
        trajs = sampled_trajectories[t, :, :, :2]
        for k in range(trajs.shape[0]):
            color = cmap(float(q[k]))
            ax.plot(trajs[k, :, 0], trajs[k, :, 1], color=color, lw=1.0, alpha=0.35)
        sel = selected_trajectories[t, :, :2]
        ax.plot(sel[:, 0], sel[:, 1], color="black", lw=2.2, label="selected")
        exec_xz = executed_trace[: min(t + 2, executed_trace.shape[0]), :2]
        ax.plot(exec_xz[:, 0], exec_xz[:, 1], color="magenta", lw=2.0, label="executed")
        ax.legend(loc="upper right")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(Image.fromarray(rgba[..., :3]))

    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    plt.close(fig)


def _animate_global_plan_xz(
    trace_hist: jnp.ndarray,
    save_path: Path,
    fps: int = 10,
) -> None:
    import numpy as np
    from PIL import Image

    pos_xz = trace_hist[:, :2]
    mins = jnp.min(pos_xz, axis=0)
    maxs = jnp.max(pos_xz, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(jnp.max(maxs - mins))
    half_span = 0.60 * max(span, 1e-3)
    x_low, x_high = float(center[0] - half_span), float(center[0] + half_span)
    z_low, z_high = float(center[1] - half_span), float(center[1] + half_span)

    fig, ax = plt.subplots(figsize=(7, 6))
    frames = []
    for t in range(pos_xz.shape[0]):
        ax.clear()
        ax.set_title(f"HalfCheetah global plan rollout {t + 1}/{pos_xz.shape[0]}")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(z_low, z_high)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("root x")
        ax.set_ylabel("root z")
        tr = pos_xz[: t + 1]
        ax.plot(tr[:, 0], tr[:, 1], color="black", lw=2.2, label="path")
        ax.plot(tr[0, 0], tr[0, 1], "go", ms=7, label="start")
        ax.plot(tr[-1, 0], tr[-1, 1], "mo", ms=7, label="current")
        ax.legend(loc="upper right")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(Image.fromarray(rgba[..., :3]))

    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    plt.close(fig)


def _animate_sampling_controls(
    sampled_controls: jnp.ndarray,
    save_path: Path,
    fps: int = 10,
) -> None:
    import numpy as np
    from PIL import Image

    # [iters, particles, horizon, control_dim]
    n_iters = int(sampled_controls.shape[0])
    if n_iters == 0:
        return
    num_particles = int(sampled_controls.shape[1])
    horizon = int(sampled_controls.shape[2])
    control_dim = int(sampled_controls.shape[3])
    if horizon <= 0 or control_dim <= 0:
        return

    controls_np = np.asarray(sampled_controls, dtype=np.float32)
    finite_controls = controls_np[np.isfinite(controls_np)]
    if finite_controls.size > 0:
        y_min = float(np.min(finite_controls))
        y_max = float(np.max(finite_controls))
    else:
        y_min, y_max = -1.0, 1.0
    if y_max - y_min < 1e-6:
        y_pad = 1.0
    else:
        y_pad = 0.05 * (y_max - y_min)
    y_low = y_min - y_pad
    y_high = y_max + y_pad

    cols = min(3, control_dim)
    rows = int(math.ceil(control_dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.2 * rows), squeeze=False)
    x = np.arange(horizon)
    frames = []

    for it in range(n_iters):
        for ax in axes.flat:
            ax.clear()

        for u_idx in range(control_dim):
            r = u_idx // cols
            c = u_idx % cols
            ax = axes[r][c]
            ax.set_title(f"u[{u_idx}]")
            ax.set_xlim(0, max(horizon - 1, 1))
            ax.set_ylim(y_low, y_high)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("timestep")
            ax.set_ylabel("control value")

            vals = controls_np[it, :, :, u_idx]  # [particles, horizon]
            for p in range(num_particles):
                y = vals[p]
                mask = np.isfinite(y)
                if not np.any(mask):
                    continue
                ax.plot(x[mask], y[mask], lw=0.9, alpha=0.45, color="tab:blue")

        # Hide any unused subplot slots.
        for u_idx in range(control_dim, rows * cols):
            r = u_idx // cols
            c = u_idx % cols
            axes[r][c].axis("off")

        fig.suptitle(f"HalfCheetah sampled controls {it + 1}/{n_iters} (particles={num_particles})")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        frames.append(Image.fromarray(rgba[..., :3]))

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
            "mediapy is required to write non-gif control animations; install mediapy or use --save-control-animation *.gif."
        ) from e

    media.write_video(str(save_path), [np.asarray(f) for f in frames], fps=fps)


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    if not args.show:
        matplotlib.use("Agg", force=True)

    env = _make_halfcheetah_env(args)
    bfoot_body_id = mujoco.mj_name2id(
        env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "bfoot"
    )
    ffoot_body_id = mujoco.mj_name2id(
        env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "ffoot"
    )
    reset_state = env.reset(jax.random.PRNGKey(args.seed))
    data0 = reset_state.pipeline_state

    info: dict[str, Any] = {}
    sampled_trajectories_arr = None
    sampled_costs_arr = None
    selected_trajectory_arr = None
    final_rob = float("nan")
    final_sat = False
    achieved_back_rotation = float("nan")
    completed_turns = float("nan")
    planning_executed = not args.load_controls

    if args.load_controls:
        load_controls_path = (
            args.load_controls_path if args.load_controls_path is not None else args.save_controls_path
        ).expanduser()
        planned_controls = _load_control_sequence(
            load_path=load_controls_path,
            control_dim=env.sys.nu,
        )
        print(f"loaded control sequence: {load_controls_path} (shape={tuple(planned_controls.shape)})")
    else:
        spec = halfcheetah_backflip_spec(
            horizon=args.horizon,
            completion_style=args.completion_style,
            target_rotation=args.target_rotation,
            rotation_tolerance=args.rotation_tolerance,
            final_window_steps=args.final_window_steps,
            min_torso_height=args.min_torso_height,
            max_abs_pitch_rate=args.max_abs_pitch_rate,
            stabilization_steps=args.stabilization_steps,
            stabilization_pitch_rate=args.stabilization_pitch_rate,
        )
        mppi_cost = make_stl_cost_fn(
            spec,
            approx_method=args.stl_approx_method,
            temperature=args.stl_temperature,
        )

        control_low = env.sys.actuator.ctrl_range[:, 0].astype(jnp.float32)
        control_high = env.sys.actuator.ctrl_range[:, 1].astype(jnp.float32)
        control_noise_sigma = args.control_noise_sigma * jnp.ones((env.sys.nu,), dtype=jnp.float32)

        def rollout_batch_pipeline(x0_data, controls: jnp.ndarray) -> jnp.ndarray:
            num_particles = controls.shape[0]
            ctrl_dtype = x0_data.ctrl.dtype if hasattr(x0_data, "ctrl") else controls.dtype
            controls = controls.astype(ctrl_dtype)
            data_batch = _tile_tree(x0_data, num_particles)
            u_time_major = jnp.swapaxes(controls, 0, 1)

            def scan_step(carry_data_batch, u_t_batch):
                next_data_batch = jax.vmap(env.pipeline_step)(carry_data_batch, u_t_batch)
                traces = jax.vmap(
                    lambda d: _planner_trace_from_data(d, bfoot_body_id, ffoot_body_id)
                )(next_data_batch)
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
            svgd_repulsion_coef=args.svgd_repulsion_coef,
            svgd_repulsion_anneal=args.svgd_repulsion_anneal,
            svgd_repulsion_final=args.svgd_repulsion_final,
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
        _, _, info = command_fn(mppi_state, data0)

        planned_controls = info["selected_controls"]
        _save_control_sequence(planned_controls, args.save_controls_path)
        print(f"saved control sequence: {args.save_controls_path}")
        if "svgd_iter_trajectories" in info:
            sampled_trajectories_arr = info["svgd_iter_trajectories"]
            sampled_costs_arr = info["svgd_iter_costs"]
            selected_trajectory_arr = info["svgd_iter_selected_trajectories"]
        else:
            sampled_trajectories_arr = info["trajectories"][None, ...]
            sampled_costs_arr = info["costs"][None, ...]
            selected_trajectory_arr = info["selected_trajectory"][None, ...]

    trace_hist, state_hist = _rollout_open_loop(env, data0, planned_controls, bfoot_body_id, ffoot_body_id)
    rollout_steps = int(planned_controls.shape[0])

    if planning_executed:
        if trace_hist.shape[0] >= args.horizon:
            final_trace = trace_hist[: args.horizon]
        else:
            pad_n = args.horizon - trace_hist.shape[0]
            pad = jnp.repeat(trace_hist[-1][None, :], pad_n, axis=0)
            final_trace = jnp.concatenate([trace_hist, pad], axis=0)
        final_rob = float(
            spec.robustness(
                final_trace,
                approx_method=args.stl_approx_method,
                temperature=args.stl_temperature,
            )
        )
        final_sat = bool(spec.eval(final_trace))
        net_pitch_delta = float(final_trace[-1, 2] - final_trace[0, 2])
        achieved_back_rotation = -net_pitch_delta
        completed_turns = achieved_back_rotation / (2.0 * jnp.pi)

    print("Run complete")
    print(f"control sequence steps: {rollout_steps}")
    print(f"control_dim: {env.sys.nu}")
    if planning_executed:
        print(f"horizon: {args.horizon}")
        print(f"num_samples: {args.num_samples}")
        print(f"svgd_iters: {args.svgd_iters}")
        print(f"completion_style: {args.completion_style}")
        print(f"achieved backward rotation (rad): {achieved_back_rotation:.6f}")
        print(f"achieved backward turns: {float(completed_turns):.6f}")
        print(f"final STL robustness: {final_rob:.6f}")
        print(f"STL satisfied: {final_sat}")
    else:
        print("MPPI optimization skipped (--load-controls).")
    runtime_seconds = time.perf_counter() - run_start
    print(f"runtime (s): {runtime_seconds:.6f}")
    _save_run_results(
        results_path=args.results_path,
        runtime_seconds=runtime_seconds,
        final_robustness=final_rob,
        stl_satisfied=final_sat,
        planning_executed=planning_executed,
        args=args,
    )
    print(f"saved run results: {args.results_path}")

    artifacts_enabled = not args.no_artifacts
    animation_enabled = artifacts_enabled and (not args.no_animation)
    if artifacts_enabled:
        _plot_rollout(
            trace_hist=trace_hist,
            save_path=args.save_path,
            show=args.show,
        )
        print(f"saved rollout plot: {args.save_path}")
    if animation_enabled:
        _animate_global_plan_xz(
            trace_hist=trace_hist,
            save_path=args.save_global_plan_animation,
            fps=args.animation_fps,
        )
        print(f"saved global-plan animation: {args.save_global_plan_animation}")
        if sampled_trajectories_arr is not None and sampled_costs_arr is not None and selected_trajectory_arr is not None:
            _animate_sampling_iterations_xz(
                sampled_trajectories=sampled_trajectories_arr,
                sampled_costs=sampled_costs_arr,
                selected_trajectories=selected_trajectory_arr,
                executed_trace=trace_hist,
                save_path=args.save_animation,
                fps=args.animation_fps,
            )
            print(f"saved animation: {args.save_animation}")
        else:
            print("skipped sampling animation (no MPPI sampling history loaded).")
        if "svgd_iter_sampled_controls" in info:
            _animate_sampling_controls(
                sampled_controls=info["svgd_iter_sampled_controls"],
                save_path=args.save_control_animation,
                fps=args.animation_fps,
            )
            print(f"saved control-space animation: {args.save_control_animation}")
        if not args.no_brax_video:
            frames = _render_brax_rollout_frames(
                env=env,
                initial_state=data0,
                state_hist_time_major=state_hist,
                width=args.render_width,
                height=args.render_height,
                camera=args.render_camera,
            )
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
            _save_brax_rollout_video(
                frames=frames,
                save_path=args.save_brax_video,
                fps=args.animation_fps,
            )
            print(f"saved brax-rendered rollout video: {args.save_brax_video}")


if __name__ == "__main__":
    main()
