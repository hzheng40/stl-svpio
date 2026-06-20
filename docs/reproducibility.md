# Reproducibility Notes

This repository fixes random seeds where possible, but GPU bitwise reproducibility is not guaranteed.

## What Seeds Control

JAX uses explicit PRNG keys. Given the same code path and backend behavior, the random samples generated from `jax.random` keys are deterministic. See the official JAX random documentation:

- <https://docs.jax.dev/en/latest/jax.random.html>

## What Seeds Do Not Control

GPU execution can be nondeterministic even when pseudo-random streams are fixed. OpenXLA documents that GPU programs can be nondeterministic for operations including GEMMs/matrix multiplication, convolutions, scatter, select-and-scatter, and multi-headed attention:

- <https://openxla.org/xla/determinism>

OpenXLA also notes that compilation can vary because autotuning may select different kernels across runs unless persisted autotuning is reused.

## Practical Consequences

- Point-mass CPU runs are usually easier to reproduce tightly.
- GPU point-mass runs should match statistically and by satisfaction/robustness tolerance.
- MJX Panda and Half-Cheetah runs involve nonlinear dynamics and contact-sensitive optimization; exact final trajectories may differ across GPUs, drivers, XLA versions, and autotuning state.
- Report robustness, satisfaction, runtime, software versions, GPU model, and command/config used.

## Optional Mitigation

For stricter GPU determinism, try:

```bash
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops"
```

This can select deterministic implementations for some operations, slow down execution, or fail compilation when an operation has no deterministic implementation. It is not enabled by default because the paper experiments were run for performance on NVIDIA RTX A6000 GPUs.

