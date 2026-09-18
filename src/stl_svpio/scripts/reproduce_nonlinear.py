from __future__ import annotations

import argparse
import shlex
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


def _halfcheetah_arguments(config: dict) -> list[str]:
    renamed = {
        "num_particles": "num_samples",
        "num_stein_steps": "svgd_iters",
        "stein_step_size": "svgd_step_size",
        "stein_step_anneal": "svgd_step_anneal",
        "stein_step_final": "svgd_step_final",
        "stein_repulsion_coef": "svgd_repulsion_coef",
        "stein_repulsion_anneal": "svgd_repulsion_anneal",
        "stein_gradient_mode": "svgd_grad_mode",
        "path_integral_temperature": "temperature",
        "sampling_distribution": "sampling_mode",
    }
    arguments = ["--update-mode", "svgd"]
    for name, value in config.items():
        if name.startswith("reference_"):
            continue
        flag = "--" + renamed.get(name, name).replace("_", "-")
        if name == "jit_command":
            if value:
                arguments.append(flag)
        else:
            arguments.extend([flag, str(value)])
    return arguments


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Show or launch nonlinear paper experiments.")
    parser.add_argument("--config", type=Path, default=root / "configs/paper/nonlinear.yaml")
    parser.add_argument("--experiment", choices=["panda_goal_reach", "halfcheetah_backflip", "all"], default="all")
    parser.add_argument("--run", action="store_true", help="Actually launch the long-running MJX jobs.")
    args = parser.parse_args(argv)

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected = list(payload["experiments"]) if args.experiment == "all" else [args.experiment]
    commands = _commands(root)
    for name in selected:
        cfg = payload["experiments"][name]
        command = commands[name]
        if name == "halfcheetah_backflip":
            command = command + _halfcheetah_arguments(cfg)
        print(f"{name}: reference robustness={cfg['reference_robustness']} runtime_s={cfg['reference_runtime_seconds']}")
        print("command:", shlex.join(command), flush=True)
        if args.run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
