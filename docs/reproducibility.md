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

### STL-SVPIO Reference Check

A full run on an NVIDIA GeForce RTX 3070 with JAX/JAXlib 0.10.1 and
stljax 1.1.3 reproduced the original benchmark outcomes for all four tasks
using sampling seeds 0 through 99 and the fixed scenes above:

| Task | Reference mean robustness | RTX 3070 mean robustness | Satisfaction, reference / RTX 3070 |
| --- | ---: | ---: | ---: |
| Long-horizon clutter | 0.020127016795 | 0.020127016795 | 97% / 97% |
| Button ordering | 0.172627966404 | 0.172627966255 | 100% / 100% |
| Synchronized goals | 0.233219534755 | 0.233219534755 | 100% / 100% |
| Corridor queuing | 0.145587200411 | 0.145587200411 | 100% / 100% |

These are the configured robustness metrics used by the original Figure 3
benchmark, compared with `results/reference/figure3_pointmass_summary.csv`.
Against the original per-trial records, every satisfaction outcome matched.
Per-seed robustness matched exactly for long-horizon clutter, synchronized
goals, and corridor queuing; the maximum absolute difference for button
ordering was `1.49e-8`. This is an observed reproduction on this GPU/software
combination, not a guarantee of bitwise equality on other systems.

The full 100-seed check here covers STL-SVPIO only, not the baseline aggregates
or MILP. Runtime depends on hardware and is not an exact-match criterion.

## Half-Cheetah: MJX Command

Use MJX with one solver iteration. The old runner's `generalized` default
was inconsistent with the paper preset; the working-config note "Default
works" alone is not sufficient to reconstruct this experiment. The launcher
now passes the YAML settings explicitly, and the direct runner defaults to
`--backend mjx --mj-iterations 1`. This also addresses the backend and
reverse-mode startup issue identified in the
[external reproduction update](https://github.com/wow-rao/stl-svpio-results#update--halfcheetah-backflip-reproduces-once-the-mjx-backend-is-used).

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run --frozen stl-svpio nonlinear \
  --experiment halfcheetah_backflip --run
```

Omit `--run` to inspect the expanded command. The settings are horizon 200,
one physics substep, 10 particles, 300 Stein steps, exponential step-size
decay from `0.001` to `1e-5`, repulsion coefficient 0, uniform sampling,
noise sigma 0.35, and path-integral temperature 0.8. Environment reset uses
seed 0 and optimizer sampling uses seed 1000. The STL specification uses
`logsumexp` temperature 500, terminal completion, rotation tolerance 0.40,
and a final window of 8 steps. Gradients use reverse mode; the optional
outer `--jit-command` is off, as in the original defaults.

The MuJoCo solver is CG with 1 solver iteration and 1 line-search iteration.
MJX with reverse-mode gradients requires a single solver iteration: the
multi-iteration solver uses a dynamic loop that cannot be reverse-mode
differentiated. The runner rejects incompatible settings before planning.

The runner respects the caller's `CUDA_VISIBLE_DEVICES` and `MUJOCO_GL`
(default rendering backend: EGL). It no longer forces GPU 1, writes an EGL
driver configuration under `/usr/share`, or injects legacy XLA tuning/cache
options. NVIDIA/EGL drivers must be installed separately for video rendering.
The MJCF model is resolved relative to the installed package.

The historical result reports robustness `0.167646` and runtime `593.215199`
seconds, but does not include the execution configuration or software
versions. Using the corrected MJX settings is not proof of reproducing that
value on a new stack; compare robustness and satisfaction with tolerances.
Repeated executions with identical settings and seeds on the same GPU can
produce different optimized controls and robustness, not just differences
between GPU models. A fixed seed does not guarantee a deterministic
Half-Cheetah optimization. Record multiple runs rather than selecting only
one favorable result. Result JSONs include settings, seeds, package versions,
device information, and relevant environment variables. OpenXLA distinguishes
[compilation-time autotuning variability from execution-time nondeterminism](https://openxla.org/xla/determinism);
these repeated runs do not isolate which operation or stage causes a difference.

### RTX 3070 Repeated Runs

Three full-preset runs in separate processes on the same NVIDIA GeForce
RTX 3070 used reset seed 0 and sampling seed 1000, with identical optimization
settings and environment. The stack was JAX/JAXlib 0.10.1, Brax 0.14.2,
MuJoCo/MJX 3.9.0, and stljax 1.1.3. Plot/video export was disabled; horizon,
particle count, and optimization iterations were not reduced.

| Repeat | STL robustness | STL satisfied | Runtime (seconds) |
| --- | ---: | --- | ---: |
| 1 | 0.437948823 | Yes | 199.47 |
| 2 | -26.139793396 | No | 121.39 |
| 3 | NaN (non-finite) | No | 121.11 |

All three saved control arrays were finite, but the third rollout produced
non-finite robustness. The first two control arrays differed, with a maximum
absolute element difference of 2.0. Settings, seeds, software versions, and
recorded environment variables matched, apart from output filenames.

The corrected command executes the MJX optimization and can produce a
satisfying result, but these repeats do not reproduce the reference robustness
`0.167646` reliably. Neither exact scalar equality nor satisfaction on every
execution is guaranteed. These three observations are not an estimate of a
general success rate. The checks validate planning, not video rendering.

## What Seeds Control

JAX uses explicit PRNG keys. Given the same code path and backend behavior, the random samples generated from `jax.random` keys are deterministic. See the official JAX random documentation:

- <https://docs.jax.dev/en/latest/jax.random.html>

## What Seeds Do Not Control

GPU execution can be nondeterministic even when pseudo-random streams are fixed. OpenXLA discusses nondeterministic GPU algorithms and lowerings for GEMMs/matrix multiplication, convolutions, scatter, and select-and-scatter:

- <https://openxla.org/xla/determinism>

OpenXLA also notes that compilation can vary because autotuning may select different kernels across runs unless persisted autotuning is reused.

## Practical Consequences

- Use CUDA for Table I reproduction; CPU execution can produce substantially different robustness.
- Compare GPU point-mass results using satisfaction and robustness tolerances rather than assuming bitwise equality.
- Panda and Half-Cheetah MJX runs involve nonlinear dynamics and contact-sensitive optimization; exact final trajectories may differ even between repeated runs on the same GPU, as well as across GPUs, drivers, XLA versions, and autotuning state.
- Report robustness, satisfaction, runtime, software versions, GPU model, and command/config used.

## Optional Mitigation

For stricter GPU determinism, try:

```bash
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops"
```

This can select deterministic implementations for some operations and slow down execution; operations without a deterministic implementation may be unsupported. It is not enabled by default because the paper experiments were run for performance on NVIDIA RTX A6000 GPUs.
