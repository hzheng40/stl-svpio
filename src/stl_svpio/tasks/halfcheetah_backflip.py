"""Lightweight metadata for the Half-Cheetah backflip paper experiment."""

DEFAULT_CONFIG = {
    "horizon": 200,
    "num_particles": 10,
    "num_stein_steps": 300,
    "stein_step_size": 0.001,
    "path_integral_temperature": 0.8,
    "sampling_distribution": "uniform",
    "control_noise_sigma": 0.35,
    "stl_approx_method": "logsumexp",
    "stl_temperature": 500.0,
    "completion_style": "terminal",
    "rotation_tolerance": 0.40,
    "final_window_steps": 8,
}

