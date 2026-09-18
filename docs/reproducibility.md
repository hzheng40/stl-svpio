# Reproducibility Notes

This repository fixes random seeds where possible, but GPU bitwise reproducibility is not guaranteed.

## Table I: CUDA Required

Run Table I on a CUDA GPU. CPU execution does not reproduce the reported
STL-SVPIO result, even with the same configuration and random seeds.

For example, with JAX/JAXlib 0.10.1, stljax 1.1.3, float32, scene seed 0,
and sampling seed 1000, the same machine produced:

| Backend | STL-SVPIO exact robustness | Smoothed robustness | STL satisfied |
| --- | ---: | ---: | --- |
| NVIDIA GeForce RTX 3070 (CUDA) | 0.108034015 | 0.100979298 | Yes |
| CPU | -2.620258331 | -2.619108200 | No |

The RTX 3070 values match the original STL-SVPIO reference. These examples
illustrate backend-dependent numerical behavior; they do not imply identical
results on every CUDA GPU or establish nondeterminism across repeated GPU runs.

Run all five Table I optimizers on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen stl-svpio table1 --seed 0 --sampling-seed 1000
```

Add `--methods stl_svpio` to run only the paper method. Adjust
`CUDA_VISIBLE_DEVICES` to select another GPU. `JAX_PLATFORMS=cuda` requires CUDA
and prevents silent CPU fallback.

The output includes exact robustness (`true_robustness`), the configured STL
evaluation (`evaluation_robustness`), and Boolean satisfaction. The historical
Table I `robustness` column uses exact robustness for STL-SVPIO and smoothed
robustness for MPPI; `reported_metric` identifies the selected field. Use
`true_robustness` for consistent comparisons across methods. A sibling metadata
JSON records the effective settings, seeds, versions, and execution backend.

## Figure 3: Fixed Scenes and Sampling Seeds

Each task/method preset fixes the scene through its `scene_seed` field. The
100 trials vary only optimizer sampling seeds 0 through 99. `--seed-offset`
changes the first sampling seed; it does not change obstacle layouts, goals,
or initial states. The optimizer starts afresh for every sampling seed.

The scene seeds in the original presets are:

| Task | STL-SVPIO scene seed | SVMPC, DPI, STL-GD scene seed |
| --- | ---: | ---: |
| Long-horizon clutter | 0 | 0 |
| Button ordering | 32 | 0 |
| Synchronized goals | 0 | 0 |
| Corridor queuing | 0 | 0 |

These method-specific scene choices are preserved from the original benchmark;
the button presets do not all use the same arena.

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --frozen stl-svpio figure3 --num-seeds 100 --seed-offset 0
```

Use `--methods stl_svpio` for the paper method alone. Each fixed scene and
method is prepared once and warmed up before timing. Trial JSON records both
`seed` (scene) and `sampling_seed`; a sibling metadata JSON records the seed
range, presets, effective configs, backend, and package versions. Completed
trials are saved incrementally. `--quick` reduces the workload for a smoke run
and should not be compared with the 100-seed reference.

## What Seeds Control

JAX uses explicit PRNG keys. Given the same code path and backend behavior, the random samples generated from `jax.random` keys are deterministic. See the official JAX random documentation:

- <https://docs.jax.dev/en/latest/jax.random.html>

## What Seeds Do Not Control

GPU execution can be nondeterministic even when pseudo-random streams are fixed. OpenXLA documents that GPU programs can be nondeterministic for operations including GEMMs/matrix multiplication, convolutions, scatter, select-and-scatter, and multi-headed attention:

- <https://openxla.org/xla/determinism>

OpenXLA also notes that compilation can vary because autotuning may select different kernels across runs unless persisted autotuning is reused.

## Practical Consequences

- Use CUDA for Table I reproduction; CPU execution can produce substantially different robustness.
- Compare GPU point-mass results using satisfaction and robustness tolerances rather than assuming bitwise equality.
- MJX Panda and Half-Cheetah runs involve nonlinear dynamics and contact-sensitive optimization; exact final trajectories may differ across GPUs, drivers, XLA versions, and autotuning state.
- Report robustness, satisfaction, runtime, software versions, GPU model, and command/config used.

## Optional Mitigation

For stricter GPU determinism, try:

```bash
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops"
```

This can select deterministic implementations for some operations and slow down execution; operations without a deterministic implementation may be unsupported. It is not enabled by default because the paper experiments were run for performance on NVIDIA RTX A6000 GPUs.
