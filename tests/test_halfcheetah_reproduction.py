import ast
from pathlib import Path
import argparse
import math

import yaml
import pytest

from stl_svpio.scripts import reproduce_nonlinear


ROOT = Path(__file__).resolve().parents[1]


def _parser():
    # Read defaults without importing the GPU/rendering stack.
    path = ROOT / "src/stl_svpio/_paper_runners/run_halfcheetah_backflip.py"
    module = ast.parse(path.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    namespace = {"argparse": argparse, "Path": Path, "jnp": math, "__file__": str(path)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_args"]


def test_preset_matches_corrected_mjx_runner_defaults():
    config = yaml.safe_load((ROOT / "configs/paper/nonlinear.yaml").read_text())["experiments"]["halfcheetah_backflip"]
    parse = _parser()
    defaults = parse([])
    configured = parse(reproduce_nonlinear._halfcheetah_arguments(config))
    assert vars(configured) == vars(defaults)
    assert configured.backend == "mjx"
    assert configured.mj_iterations == 1
    assert configured.svgd_repulsion_coef == 0.0
    assert configured.mjcf_path.is_file()


def test_launcher_passes_halfcheetah_config(tmp_path, monkeypatch):
    payload = yaml.safe_load((ROOT / "configs/paper/nonlinear.yaml").read_text())
    payload["experiments"]["halfcheetah_backflip"]["seed"] = 7
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    calls = []
    monkeypatch.setattr(reproduce_nonlinear.subprocess, "run", lambda command, **kwargs: calls.append(command))
    reproduce_nonlinear.main(["--config", str(path), "--experiment", "halfcheetah_backflip", "--run"])
    args = _parser()(calls[0][3:])
    assert args.seed == 7
    assert args.backend == "mjx"
    assert args.mj_iterations == 1
    assert args.mj_ls_iterations == 1


def test_mjx_reverse_mode_rejects_multiple_solver_iterations():
    with pytest.raises(SystemExit):
        _parser()(["--mj-iterations", "4"])
