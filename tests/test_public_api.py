from __future__ import annotations

import jax
import jax.numpy as jnp

from stl_svpio import STLSVPIOConfig, STLSVPIOOptimizer


def test_stl_svpio_public_names_map_to_optimizer() -> None:
    cfg = STLSVPIOConfig(
        horizon=4,
        num_particles=4,
        control_dim=2,
        path_integral_temperature=1.0,
        sampling_distribution="gaussian",
        num_stein_steps=1,
        stein_step_size=0.01,
    )
    controller = STLSVPIOOptimizer(
        config=cfg,
        dynamics_fn=STLSVPIOOptimizer.pointmass2d_dynamics(dt=0.1),
        cost_fn=lambda tr: jnp.sum(tr**2, axis=(1, 2)),
    )
    state = controller.init_state(jax.random.PRNGKey(0))
    action, next_state, info = controller.optimize(state, jnp.zeros((4,), dtype=jnp.float32))

    assert action.shape == (2,)
    assert next_state.mean_controls.shape == (4, 2)
    assert info["selected_controls"].shape == (4, 2)

