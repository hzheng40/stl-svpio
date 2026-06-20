# STL-SVPIO Paper Reproduction

This repository contains the accompanying code for:

**STL-SVPIO: Signal Temporal Logic guided Stein Variational Path Integral Optimization**  
Hongrui Zheng, Zirui Zang, Ahmad Amine, Cristian Ioan Vasile, Rahul Mangharam  
[arXiv:2603.13333](https://arxiv.org/pdf/2603.13333)

The code is organized for reproducing the paper results rather than for preserving every exploratory script from the development repository.

## Install

```bash
git clone <repo-url> stl-svpio
cd stl-svpio
uv sync
```

The main package is `stl_svpio`; the console command is `stl-svpio`.

## Quick Smoke Runs

```bash
uv run stl-svpio table1 --methods stl_svpio --no-jit
uv run stl-svpio figure3 --quick --methods stl_svpio --tasks multiagent_sync_goals --no-jit
uv run stl-svpio nonlinear
```

## Paper Reproduction Commands

Reach-avoid Table I / Figure 2:

```bash
uv run stl-svpio table1
```

Figure 3 point-mass benchmark over 100 seeds:

```bash
uv run stl-svpio figure3
```

Nonlinear MJX tasks:

```bash
uv run stl-svpio nonlinear
uv run stl-svpio nonlinear --experiment panda_goal_reach --run
uv run stl-svpio nonlinear --experiment halfcheetah_backflip --run
```

The MJX jobs are long-running GPU experiments. Reference outputs are stored under `results/reference/`.

Optional MILP/PyTeLo/Gurobi baselines:

```bash
python -m stl_svpio.baselines.milp_pytelo \
  --task-id single_default \
  --time-limit-sec 36000 \
  --verify-trace \
  --pytelo-root ~/pytelo
```

The paper MILP runs use a 10 hour wall-clock budget per task. They are configured to stop at the first feasible solution (`MIPFocus=1`, `SolutionLimit=1`), not to prove global optimality. See `docs/external_dependencies.md` for setup details.

## Reproducibility

Seeds control JAX PRNG streams, but exact bitwise reproducibility on GPU is not guaranteed. OpenXLA documents nondeterminism in GPU execution for operations including GEMMs/matrix multiplication, convolutions, scatter, select-and-scatter, and attention. See `docs/reproducibility.md` for the exact limitations and mitigation flags.

## Repository Map

- `src/stl_svpio/algorithms`: paper-facing STL-SVPIO API.
- `src/stl_svpio/baselines`: MPPI, SVMPC, DPI, STLCG-style GD, and optional MILP baseline code.
- `src/stl_svpio/tasks`: paper-named task modules.
- `configs/paper`: tuned experiment configurations.
- `scripts`: thin paper reproduction entrypoints.
- `docs`: method, experiment, dependency, and reproducibility notes.
- `results/reference`: small reference CSV/JSON outputs.
