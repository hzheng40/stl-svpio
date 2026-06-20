"""Lightweight metadata for the Franka Panda paper experiment."""

DEFAULT_CONFIG = {
    "horizon": 300,
    "num_particles": 10,
    "num_stein_steps": 300,
    "stein_step_size": 10.0,
    "stein_step_size_final": 0.01,
    "path_integral_temperature": 0.8,
    "sampling_distribution": "uniform",
    "reach_radius": 0.06,
    "stl_approx_method": "logsumexp",
    "stl_temperature": 100.0,
}

