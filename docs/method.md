# Method Notes

STL-SVPIO solves the finite-horizon STL control problem:

```text
maximize_u rho_phi(x_0:H(u))
subject to x_{t+1} = f(x_t, u_t), u_t in [u_min, u_max].
```

The implementation minimizes the negative robustness cost:

```text
J_phi(u) = -rho_phi(x_0:H(u)).
```

The public API uses paper terminology:

- `STLSVPIOConfig`: optimizer configuration.
- `STLSVPIOOptimizer`: exact-gradient Stein optimizer.
- `num_particles`: number of control-sequence particles.
- `num_stein_steps`: number of SVGD transport steps.
- `path_integral_temperature`: Gibbs temperature lambda.
- `make_negative_robustness_cost`: creates `J_phi`.

Each particle is a full open-loop control sequence `u_i in R^{H x m}`. Particles are initialized from a bounded sampling distribution. Each Stein step computes:

```text
phi*(u_i) = (1/N) sum_j [K(u_j, u_i) grad_{u_j} log p*(u_j) + grad_{u_j} K(u_j, u_i)]
```

where:

```text
log p*(u) proportional to -J_phi(u) / lambda
```

The first term attracts particles toward high-STL-robustness regions. The second term is the Stein repulsion term that maintains population diversity and helps avoid mode collapse in non-convex STL landscapes.

The implementation uses a uniform bounded control prior. Controls are clipped to the admissible range after sampling and after particle updates.

