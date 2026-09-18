from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import jax
import yaml

from stl_svpio.tasks.pointmass import prepare_pointmass_trial, summarize_trials, write_summary_csv

METHOD_CONFIGS = {
    "stl_svpio": "stl_svpio_pointmass.yaml",
    "svmpc": "svmpc_pointmass.yaml",
    "dpi": "dpi_pointmass.yaml",
    "stlcg_gradient_descent": "stlcg_gradient_descent_pointmass.yaml",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_presets(path: Path) -> dict[str, dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        str(item["id"]): {"scene_seed": int(item["scene_seed"]), "args": dict(item["args"])}
        for item in payload["presets"]
    }


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Reproduce Figure 3 point-mass benchmark.")
    parser.add_argument("--configs-dir", type=Path, default=root / "configs/paper")
    parser.add_argument(
        "--tasks",
        default="single_visit_goals_long_horizon,multiagent_button,multiagent_sync_goals,multiagent_corridor_6_agents",
    )
    parser.add_argument("--methods", default="stl_svpio,svmpc,dpi,stlcg_gradient_descent")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0, help="First sampling seed; scene seeds come only from presets.")
    parser.add_argument("--out", type=Path, default=root / "artifacts/figure3_pointmass_summary.csv")
    parser.add_argument("--json-out", type=Path, default=root / "artifacts/figure3_pointmass_trials.json")
    parser.add_argument("--quick", action="store_true", help="Run one seed and reduce iterations for a smoke test.")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args(argv)
    if args.num_seeds < 1 or args.seed_offset < 0:
        parser.error("--num-seeds must be positive and --seed-offset must be nonnegative")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not methods or not tasks:
        parser.error("Select at least one task and method")
    num_seeds = 1 if args.quick else args.num_seeds
    all_results = []
    loaded = {method: _load_presets(args.configs_dir / METHOD_CONFIGS[method]) for method in methods}
    metadata = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "stljax")},
        "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS")},
        "sampling_seeds": list(range(args.seed_offset, args.seed_offset + num_seeds)),
        "jit": not args.no_jit,
        "quick": args.quick,
        "warmup": "first sampling seed per fixed scene and method",
        "config_hashes": {method: hashlib.sha256((args.configs_dir / METHOD_CONFIGS[method]).read_bytes()).hexdigest() for method in methods},
        "presets": {method: {task: loaded[method][task] for task in tasks} for method in methods},
        "effective_configs": {},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.json_out.with_suffix(".metadata.json")
    print(f"backend={metadata['backend']} devices={metadata['devices']}", flush=True)

    for task_id in tasks:
        for method in methods:
            preset = loaded[method][task_id]
            scene_seed = preset["scene_seed"]
            cfg = dict(preset["args"])
            cfg["record_svgd_history"] = method == "stl_svpio"
            if args.quick:
                cfg["svgd_iters"] = min(int(cfg.get("svgd_iters", 10)), 2)
                cfg["dpi_iters"] = min(int(cfg.get("dpi_iters", cfg.get("svgd_iters", 2))), 2)
                cfg["stl_gd_iters"] = min(int(cfg.get("stl_gd_iters", cfg.get("svgd_iters", 2))), 2)
                cfg["num_samples"] = min(int(cfg.get("num_samples", 10)), 4)
            metadata["effective_configs"][f"{task_id}/{method}"] = cfg
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            run = prepare_pointmass_trial(task_id, method, cfg, seed=scene_seed, jit=(not args.no_jit))
            for i in range(num_seeds):
                sampling_seed = args.seed_offset + i
                result = run(sampling_seed, warmup=(i == 0))
                all_results.append(result)
                # Preserve completed trials during long multi-task GPU runs.
                write_summary_csv(summarize_trials(all_results), args.out)
                args.json_out.write_text(json.dumps([r.__dict__ for r in all_results], indent=2), encoding="utf-8")
                print(
                    f"{task_id}/{method}/scene_seed={scene_seed}/sampling_seed={sampling_seed}: "
                    f"robustness={result.robustness:.6f} sat={int(result.satisfied)} "
                    f"runtime_ms={result.runtime_ms:.3f}", flush=True,
                )
            del run
            jax.clear_caches()

    rows = summarize_trials(all_results)
    write_summary_csv(rows, args.out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps([r.__dict__ for r in all_results], indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
