from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _commands(root: Path) -> dict[str, list[str]]:
    return {
        "panda_goal_reach": [
            sys.executable,
            "-m",
            "stl_svpio._paper_runners.run_panda_goal_reach",
        ],
        "halfcheetah_backflip": [
            sys.executable,
            "-m",
            "stl_svpio._paper_runners.run_halfcheetah_backflip",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Show or launch nonlinear MJX paper experiments.")
    parser.add_argument("--config", type=Path, default=root / "configs/paper/nonlinear.yaml")
    parser.add_argument("--experiment", choices=["panda_goal_reach", "halfcheetah_backflip", "all"], default="all")
    parser.add_argument("--run", action="store_true", help="Actually launch the long-running MJX jobs.")
    args = parser.parse_args(argv)

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected = list(payload["experiments"]) if args.experiment == "all" else [args.experiment]
    commands = _commands(root)
    for name in selected:
        cfg = payload["experiments"][name]
        print(f"{name}: reference robustness={cfg['reference_robustness']} runtime_s={cfg['reference_runtime_seconds']}")
        print("command:", " ".join(commands[name]))
        if args.run:
            subprocess.run(commands[name], cwd=root, check=True)


if __name__ == "__main__":
    main()

