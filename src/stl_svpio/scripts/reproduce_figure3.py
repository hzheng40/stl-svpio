from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stl_svpio.tasks.pointmass import run_pointmass_trial, summarize_trials, write_summary_csv

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
    return {str(item["id"]): dict(item["args"]) for item in payload["presets"]}


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
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=root / "artifacts/figure3_pointmass_summary.csv")
    parser.add_argument("--json-out", type=Path, default=root / "artifacts/figure3_pointmass_trials.json")
    parser.add_argument("--quick", action="store_true", help="Run one seed and reduce iterations for a smoke test.")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    num_seeds = 1 if args.quick else args.num_seeds
    all_results = []
    loaded = {method: _load_presets(args.configs_dir / METHOD_CONFIGS[method]) for method in methods}

    for task_id in tasks:
        for method in methods:
            cfg = dict(loaded[method][task_id])
            if method == "stl_svpio":
                public_method = "stl_svpio"
            elif method == "svmpc":
                public_method = "svmpc"
            elif method == "dpi":
                public_method = "dpi"
            else:
                public_method = "stlcg_gradient_descent"
            if args.quick:
                cfg["svgd_iters"] = min(int(cfg.get("svgd_iters", 10)), 2)
                cfg["dpi_iters"] = min(int(cfg.get("dpi_iters", cfg.get("svgd_iters", 2))), 2)
                cfg["stl_gd_iters"] = min(int(cfg.get("stl_gd_iters", cfg.get("svgd_iters", 2))), 2)
                cfg["num_samples"] = min(int(cfg.get("num_samples", 10)), 4)
            for i in range(num_seeds):
                seed = args.seed_offset + i
                result = run_pointmass_trial(task_id, public_method, cfg, seed=seed, jit=(not args.no_jit))
                all_results.append(result)
                print(
                    f"{task_id}/{public_method}/seed={seed}: "
                    f"robustness={result.robustness:.6f} sat={int(result.satisfied)} "
                    f"runtime_ms={result.runtime_ms:.3f}"
                )

    rows = summarize_trials(all_results)
    write_summary_csv(rows, args.out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps([r.__dict__ for r in all_results], indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

