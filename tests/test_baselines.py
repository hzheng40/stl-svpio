from __future__ import annotations

import jax
import jax.numpy as jnp

from stl_svpio.baselines import DPIConfig, DPIOptimizer, MPPIBaselineConfig, MPPIBaselineOptimizer, SVMPCConfig, SVMPCOptimizer
from stl_svpio._legacy_mppi import MPPIController


def _cost(tr):
    return jnp.sum(tr**2, axis=(1, 2))


def test_baseline_wrappers_return_finite_controls() -> None:
    dynamics = MPPIController.pointmass2d_dynamics(dt=0.1)
    x0 = jnp.zeros((4,), dtype=jnp.float32)
    configs_and_classes = [
        (MPPIBaselineConfig(horizon=4, num_particles=4, control_dim=2, num_update_steps=1), MPPIBaselineOptimizer),
        (SVMPCConfig(horizon=4, num_particles=4, control_dim=2, num_stein_steps=1, finite_difference_dirs=1), SVMPCOptimizer),
        (DPIConfig(horizon=4, num_particles=4, control_dim=2, num_update_steps=1), DPIOptimizer),
    ]
    for cfg, cls in configs_and_classes:
        optimizer = cls(cfg, dynamics, _cost)
        state = optimizer.init_state(jax.random.PRNGKey(0))
        _, _, info = optimizer.optimize(state, x0)
        assert jnp.all(jnp.isfinite(info["selected_controls"]))

