from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stl_svpio.tasks.pointmass import run_pointmass_trial, summarize_trials, write_summary_csv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Reproduce Table I / Figure 2 reach-avoid benchmark.")
    parser.add_argument("--config", type=Path, default=root / "configs/paper/table1_reach_avoid.yaml")
    parser.add_argument("--methods", default="stl_svpio,mppi,svmpc,dpi,stlcg_gradient_descent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=root / "artifacts/table1_reach_avoid_summary.csv")
    parser.add_argument("--json-out", type=Path, default=root / "artifacts/table1_reach_avoid_trials.json")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args(argv)

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = dict(payload["base"])
    task_id = str(payload["task_id"])
    results = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        cfg = {**base, **dict(payload["methods"][method])}
        result = run_pointmass_trial(task_id, method, cfg, seed=args.seed, jit=(not args.no_jit))
        results.append(result)
        print(
            f"{method}: robustness={result.robustness:.6f} "
            f"satisfied={int(result.satisfied)} runtime_ms={result.runtime_ms:.3f}"
        )

    rows = summarize_trials(results)
    write_summary_csv(rows, args.out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps([r.__dict__ for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

