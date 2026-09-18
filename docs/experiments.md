# Paper Experiments

| Paper result | Task/config | Command | Reference output |
| --- | --- | --- | --- |
| Table I / Figure 2 | Single-agent reach-avoid | `JAX_PLATFORMS=cuda uv run stl-svpio table1` | `results/reference/table1_reach_avoid_summary.csv` |
| Figure 3 Long Horizon | `single_visit_goals_long_horizon` | `uv run stl-svpio figure3` | `results/reference/figure3_pointmass_summary.csv` |
| Figure 3 Button | `multiagent_button` | `uv run stl-svpio figure3` | `results/reference/figure3_pointmass_summary.csv` |
| Figure 3 Sync Goals | `multiagent_sync_goals` | `uv run stl-svpio figure3` | `results/reference/figure3_pointmass_summary.csv` |
| Figure 3 Corridor | `multiagent_corridor_6_agents` | `uv run stl-svpio figure3` | `results/reference/figure3_pointmass_summary.csv` |
| Figure 8 | Panda goal reach | `uv run stl-svpio nonlinear --experiment panda_goal_reach --run` | `results/reference/panda_goal_reach_results.json` |
| Figure 9 | Half-Cheetah backflip | `uv run stl-svpio nonlinear --experiment halfcheetah_backflip --run` | `results/reference/halfcheetah_backflip_results.json` |

## Point-Mass Baselines

Table I reproduction requires CUDA; CPU execution does not reproduce the
reported STL-SVPIO result. The runner reports both exact and smoothed robustness.
See [reproducibility notes](reproducibility.md) for the RTX 3070/CPU example
and the metric definitions.

Figure 3 compares:

- `stl_svpio`: exact-gradient STL-SVPIO.
- `svmpc`: finite-difference Stein MPC.
- `dpi`: deterministic path integral optimization.
- `stlcg_gradient_descent`: direct gradient descent on negative STL robustness.
- `milp`: optional PyTeLo/Gurobi baseline from saved or independently solved MILP runs.

MILP runs use PyTeLo for STL parsing/validation and Gurobi for the MILP solve. The paper setting gives each task a 10 hour budget (`--time-limit-sec 36000`) and stops at the first feasible solution (`MIPFocus=1`, `SolutionLimit=1`) rather than proving global optimality.

The paper reports 100 random sampling seeds for stochastic point-mass methods.
The default `figure3` command uses sampling seeds 0-99 and holds each preset's
scene seed fixed. `--seed-offset` affects sampling only. See
[reproducibility notes](reproducibility.md#figure-3-fixed-scenes-and-sampling-seeds)
for the scene-seed table and CUDA command.

## Nonlinear Tasks

The nonlinear tasks use MuJoCo MJX through JAX. They are intentionally separated from normal imports because they require GPU/MuJoCo setup and are long-running.

Reference values from the paper run:

- Panda: robustness `0.0288931243121624`, planning runtime `634.8450644160621` seconds.
- Half-Cheetah: robustness `0.167646`, runtime `593.215199` seconds.

These tasks should be checked with tolerance-based robustness and satisfaction criteria, not exact trajectory equality.
