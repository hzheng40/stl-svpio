from __future__ import annotations

from stl_svpio.tasks.pointmass import PAPER_POINTMASS_TASKS, build_pointmass_problem, run_pointmass_trial


def test_paper_task_registry_uses_paper_names() -> None:
    assert "single_visit_goals" in PAPER_POINTMASS_TASKS
    assert "multiagent_corridor" in PAPER_POINTMASS_TASKS


def test_reach_avoid_problem_builds_constructed_scene() -> None:
    problem = build_pointmass_problem(
        {
            "task": "single_default",
            "horizon": 6,
            "episode_steps": 6,
            "scene": "constructed_pointmass_diag",
            "num_obstacles": 1,
            "num_zones": 1,
        },
        seed=0,
    )
    assert problem.control_dim == 2
    assert problem.initial_state.shape == (4,)


def test_quick_stl_svpio_trial_returns_metrics() -> None:
    result = run_pointmass_trial(
        "single_default",
        "stl_svpio",
        {
            "task": "single_default",
            "horizon": 5,
            "episode_steps": 5,
            "scene": "constructed_pointmass_diag",
            "num_samples": 4,
            "svgd_iters": 1,
            "svgd_step_size": 0.01,
            "sampling_mode": "gaussian",
            "stl_approx_method": "logsumexp",
            "stl_temperature": 10.0,
        },
        seed=0,
        jit=False,
    )
    assert result.method == "stl_svpio"
    assert isinstance(result.robustness, float)
    assert result.num_particles == 4

