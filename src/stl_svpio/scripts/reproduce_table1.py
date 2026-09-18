from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import jax
import yaml

from stl_svpio.tasks.pointmass import run_pointmass_trial


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Reproduce Table I / Figure 2 reach-avoid benchmark.")
    parser.add_argument("--config", type=Path, default=root / "configs/paper/table1_reach_avoid.yaml")
    parser.add_argument("--methods", default="stl_svpio,mppi,svmpc,dpi,stlcg_gradient_descent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=1000)
    parser.add_argument("--stl-large-number", type=float, help="Explicit temporal-mask override; omitted preserves stljax defaults.")
    parser.add_argument("--out", type=Path, default=root / "artifacts/table1_reach_avoid_summary.csv")
    parser.add_argument("--json-out", type=Path, default=root / "artifacts/table1_reach_avoid_trials.json")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args(argv)
    if args.stl_large_number is not None and (not math.isfinite(args.stl_large_number) or args.stl_large_number <= 0):
        parser.error("--stl-large-number must be finite and positive")

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = dict(payload["base"])
    task_id = str(payload["task_id"])
    results = []
    configs = {}
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "stljax")},
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "backend": jax.default_backend(),
        "jax_enable_x64": jax.config.x64_enabled,
        "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS")},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scene_seed": args.seed,
        "sampling_seed": args.sampling_seed,
        "jit": not args.no_jit,
        "warmup": True,
        "effective_configs": configs,
    }
    print(f"backend={metadata['backend']} devices={metadata['devices']}")
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        cfg = {**base, **dict(payload["methods"][method])}
        if args.stl_large_number is not None:
            cfg["stl_large_number"] = args.stl_large_number
        metric = cfg.get("report_robustness", "true_robustness")
        if metric not in {"true_robustness", "evaluation_robustness"}:
            raise ValueError(f"Unknown report_robustness: {metric}")
        configs[method] = cfg
        result = run_pointmass_trial(task_id, method, cfg, seed=args.seed,
                                     sampling_seed=args.sampling_seed, jit=(not args.no_jit), warmup=True)
        row = {
            "method": method,
            "num_particles": result.num_particles,
            "num_iterations": result.num_iterations,
            "runtime_ms": result.runtime_ms,
            "robustness": result.true_robustness if metric == "true_robustness" else result.robustness,
            "true_robustness": result.true_robustness,
            "evaluation_robustness": result.robustness,
            "reported_metric": metric,
            "satisfied": result.satisfied,
        }
        results.append(row)
        print(
            f"{method}: reported={row['robustness']:.6f} true={result.true_robustness:.6f} "
            f"evaluation_robustness={result.robustness:.6f} "
            f"satisfied={int(result.satisfied)} runtime_ms={result.runtime_ms:.3f}"
        )

    if not results:
        parser.error("Select at least one method")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    metadata_out = args.json_out.with_suffix(".metadata.json")
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {metadata_out}")


if __name__ == "__main__":
    main()
