# PyTeLo + Gurobi Solver Setup

## 1) Create a dedicated Python 3.13 environment

```bash
python3.13 -m venv .venv-pytelo-gurobi
source .venv-pytelo-gurobi/bin/activate
python -m pip install --upgrade pip
```

## 2) Install project package and solver dependencies

```bash
pip install -e . --no-deps
pip install gurobipy pyyaml numpy matplotlib jax antlr4-python3-runtime==4.13.0 scipy
```

`jax` is required because scene reconstruction uses the repo's pointmass environment utilities.

## 3) Prepare local PyTeLo clone (README-aligned)

Assume your clone is at `~/pytelo`.

PyTeLo in this repo layout is not a pip package; it expects generated ANTLR parser files in each logic folder.

1. Ensure Java is installed (`java -version`).
   - ANTLR 4.13.0 requires Java 11 or newer.
   - Java binary architecture must match your machine (e.g. x86_64 vs aarch64).
   - If you use Java 8, either install Java 11+ or use an older ANTLR jar compatible with Java 8.
2. Ensure `~/pytelo/lib/antlr-4.13.0-complete.jar` exists.
3. Generate parser files:

```bash
cd ~/pytelo
export CLASSPATH=".:$PWD/lib/antlr-4.13.0-complete.jar:$CLASSPATH"
alias antlr4="java -jar $PWD/lib/antlr-4.13.0-complete.jar -visitor"

cd ~/pytelo/stl && antlr4 -Dlanguage=Python3 stl.g4
cd ~/pytelo/mtl && antlr4 -Dlanguage=Python3 mtl.g4
cd ~/pytelo/wmtl && antlr4 -Dlanguage=Python3 wmtl.g4
cd ~/pytelo/wstl && antlr4 -Dlanguage=Python3 wstl.g4
```

For this solver, STL is required at minimum (`~/pytelo/stl/stlLexer.py`, `stlParser.py`, `stlVisitor.py`).

## 4) Install Gurobi optimizer binaries

Install Gurobi Optimizer from the official Gurobi distribution for your OS.

## 5) Activate a Gurobi license

```bash
grbgetkey <YOUR_KEY>
```

Alternative: set `GRB_LICENSE_FILE` to your local license file path.

## 6) Verify dependencies from Python

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python -c "import yaml, antlr4, jax; print('yaml+antlr4+jax ok')"
```

Expected first command output (example):

```text
(11, 0, 0)
```

## 7) Run the direct STL solver

The paper MILP baseline uses a **10 hour time budget per task**:

```text
10 hours = 36000 seconds
```

It is configured as a feasibility search rather than an optimality proof:

- `MIPFocus = 1` asks Gurobi to prioritize feasible solutions.
- `SolutionLimit = 1` stops after the first feasible solution.
- The model still defines objectives for safety margin and control effort, but the paper setting does not wait for global optimality certification.

```bash
pip install -e . --no-deps
python -m stl_svpio.baselines.milp_pytelo \
  --task-id single_default \
  --time-limit-sec 36000 \
  --verify-trace \
  --artifact-dir artifacts/pytelo_milp \
  --pytelo-root ~/pytelo
```

If you do not install the repo as an editable package, run with:

```bash
PYTHONPATH=src python -m stl_svpio.baselines.milp_pytelo ...
```

## 8) Run other curated tasks

Task ids are defined in `configs/paper/milp_pytelo_pointmass.yaml`:

- `single_default`
- `single_visit_goals`
- `multiagent_button`
- `multiagent_sync_goals`
- `multiagent_corridor`

Example:

```bash
python -m stl_svpio.baselines.milp_pytelo \
  --task-id multiagent_corridor \
  --time-limit-sec 36000 \
  --mip-gap 0.02 \
  --polygon-sides 20 \
  --verify-trace \
  --pytelo-root ~/pytelo
```
