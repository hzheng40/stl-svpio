from pathlib import Path
import csv
import json

import jax
import jax.numpy as jnp
import pytest
import yaml

from stl_svpio._legacy_mppi import make_stl_cost_fn
from stl_svpio.baselines.mppi import make_reach_avoid_heuristic_cost
from stl_svpio.scripts import reproduce_table1
from stl_svpio.tasks.pointmass import PointMassTrialResult, build_pointmass_problem, run_pointmass_trial

ROOT = Path(__file__).resolve().parents[1]


def test_table1_matches_original_comparison_parameters():
    config = yaml.safe_load((ROOT / "configs/paper/table1_reach_avoid.yaml").read_text())
    original = json.loads((ROOT / "results/reference/table1_original_compare.json").read_text())
    ours = {**config["base"], **config["methods"]["stl_svpio"]}
    for key in ("dt", "horizon", "episode_steps", "noise_sigma", "num_samples",
                "sampling_mode", "stl_approx_method", "stl_temperature", "svgd_iters",
                "svgd_step_anneal", "svgd_step_final", "svgd_step_size", "temperature"):
        assert ours[key] == original["params"][key]
    for method, filename in [("svmpc", "svmpc"), ("dpi", "dpi"),
                             ("stlcg_gradient_descent", "stlcg_gradient_descent")]:
        presets = yaml.safe_load((ROOT / f"configs/paper/{filename}_pointmass.yaml").read_text())
        original_args = next(p["args"] for p in presets["presets"] if p["id"] == "single_default")
        merged = {**config["base"], **config["methods"][method]}
        for key, value in original_args.items():
            if key in merged:
                assert merged[key] == value, (method, key)


def test_heuristic_includes_stage_collision_penalty():
    cost = make_reach_avoid_heuristic_cost(
        jnp.array([2., 0.]), jnp.array([[0., 0.]]), jnp.array([1.]),
    )
    trajectory = jnp.array([[[0., 0., 0., 0.], [2., 0., 0., 0.]]])
    assert float(cost(trajectory)[0]) == pytest.approx(25.2)


def test_bounded_mask_has_finite_jitted_gradient():
    problem = build_pointmass_problem({"task": "single_default", "horizon": 20}, seed=0)
    cost = make_stl_cost_fn(problem.stl_specification, "logsumexp", 50., large_number=1e6)
    trace = jnp.zeros((20, 4)).at[:, :2].set(jnp.linspace(jnp.array([0., 0.]), jnp.array([4., 3.8]), 20))
    value, gradient = jax.jit(jax.value_and_grad(lambda tr: cost(tr[None])[0]))(trace)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0


def test_null_smoothing_temperature_fails_with_context():
    with pytest.raises(ValueError, match="svmpc: logsumexp requires"):
        run_pointmass_trial("single_default", "svmpc", {
            "horizon": 5, "stl_approx_method": "logsumexp", "stl_temperature": None,
        }, seed=0)


def test_table1_reports_both_metrics_and_explicit_seeds(tmp_path, monkeypatch):
    calls = []

    def fake_trial(task_id, method, config, **kwargs):
        calls.append(kwargs)
        return PointMassTrialResult(task_id, method, kwargs["seed"], kwargs["sampling_seed"],
                                    1.0, -0.1, True, 10, 20, true_robustness=0.2)

    monkeypatch.setattr(reproduce_table1, "run_pointmass_trial", fake_trial)
    out = tmp_path / "results.csv"
    json_out = tmp_path / "results.json"
    reproduce_table1.main(["--methods", "stl_svpio,mppi", "--seed", "3", "--sampling-seed", "7",
                           "--out", str(out), "--json-out", str(json_out)])
    with out.open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["robustness"]) == 0.2
    assert float(rows[1]["robustness"]) == -0.1
    assert all(c["seed"] == 3 and c["sampling_seed"] == 7 and c["warmup"] for c in calls)
    metadata = json.loads(json_out.with_suffix(".metadata.json").read_text())
    assert metadata["sampling_seed"] == 7
    assert metadata["effective_configs"]["mppi"]["cost_mode"] == "heuristic_reach_avoid"
