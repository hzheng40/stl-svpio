import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from stl_svpio.scripts import reproduce_figure3
from stl_svpio.tasks import pointmass

ROOT = Path(__file__).resolve().parents[1]


def test_presets_retain_scene_seed():
    presets = reproduce_figure3._load_presets(ROOT / "configs/paper/stl_svpio_pointmass.yaml")
    assert presets["multiagent_button"]["scene_seed"] == 32
    assert presets["single_visit_goals_long_horizon"]["scene_seed"] == 0
    assert presets["multiagent_button"]["args"]["task"] == "multiagent_button"


def test_seed_offset_changes_only_sampling(tmp_path, monkeypatch):
    prepared = []
    trials = []

    def prepare(task, method, cfg, seed, jit):
        prepared.append((task, method, seed))

        def run(sampling_seed, warmup=False):
            trials.append((seed, sampling_seed, warmup))
            return pointmass.PointMassTrialResult(task, method, seed, sampling_seed,
                                                  1., .1, True, 10, 300, .1)
        return run

    monkeypatch.setattr(reproduce_figure3, "prepare_pointmass_trial", prepare)
    out = tmp_path / "trials.json"
    reproduce_figure3.main([
        "--methods", "stl_svpio", "--tasks", "multiagent_button",
        "--num-seeds", "3", "--seed-offset", "7",
        "--out", str(tmp_path / "summary.csv"), "--json-out", str(out),
    ])
    assert prepared == [("multiagent_button", "stl_svpio", 32)]
    assert trials == [(32, 7, True), (32, 8, False), (32, 9, False)]
    saved = json.loads(out.read_text())
    assert [r["seed"] for r in saved] == [32, 32, 32]
    assert [r["sampling_seed"] for r in saved] == [7, 8, 9]
    metadata = json.loads(out.with_suffix(".metadata.json").read_text())
    assert metadata["sampling_seeds"] == [7, 8, 9]


def test_prepared_runner_keeps_scene_and_resets_optimizer(monkeypatch):
    original = pointmass.build_pointmass_problem
    scenes = []

    def build(cfg, seed):
        scenes.append(seed)
        return original(cfg, seed)

    monkeypatch.setattr(pointmass, "build_pointmass_problem", build)
    run = pointmass.prepare_pointmass_trial("single_default", "stl_svpio", {
        "horizon": 6, "episode_steps": 6, "num_samples": 4,
        "svgd_iters": 2, "sampling_mode": "uniform", "stl_approx_method": "true",
    }, seed=32)
    first = run(5, warmup=True)
    second = run(6)
    repeated = run(5)
    assert scenes == [32]
    assert first.seed == second.seed == repeated.seed == 32
    assert first.robustness == pytest.approx(repeated.robustness)
    assert second.sampling_seed == 6


def test_gradient_descent_respects_sampling_distribution(monkeypatch):
    initial = []

    def solve(config, initial_controls, **kwargs):
        initial.append(initial_controls)
        return initial_controls, jnp.zeros((1,)), jnp.zeros((1,))

    monkeypatch.setattr(pointmass, "run_stlcg_gradient_descent", solve)
    run = pointmass.prepare_pointmass_trial("single_default", "stlcg_gradient_descent", {
        "horizon": 6, "episode_steps": 6, "stl_gd_iters": 1,
        "sampling_mode": "truncated_gaussian", "noise_sigma": .1,
        "stl_approx_method": "true",
    }, seed=0, jit=False)
    run(0)
    assert float(jnp.max(jnp.abs(initial[0]))) < .5
