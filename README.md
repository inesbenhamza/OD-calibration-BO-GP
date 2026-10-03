# SUMO OD Calibration with Bayesian Optimization

Bayesian Optimization (BO) for calibrating Origin-Destination (OD) demand
matrices against link (edge) flow observations from
[SUMO](https://eclipse.dev/sumo/) traffic simulations. The project compares
several surrogate-model families for the BO loop:

- **Vanilla BO** — single GP on an aggregated scalar error.
- **Independent GP (IndGP)** — one GP per output edge, no cross-edge
  correlation.
- **Multi-Output GP (MOGP / ICM)** — Intrinsic Coregionalization Model with a
  learned task-covariance matrix over edges.
- **Graph-structured kernels** — task covariance derived from route-overlap /
  network-adjacency graphs (`MOGP/graph.py`,
  `MOGP/graph_shared_private.py`), used by the "GraphGP" and
  "shared+private" model variants.

Everything is evaluated on small synthetic/benchmark SUMO networks under
`network/` (grid networks, a quickstart corridor, and cropped InTAS /
MoSTScenario benchmark scenarios vendored in `InTAS/` and `most_scenario/`).

## 1. Requirements

### Python

Experiments were run with **Python 3.13.7**
(`/opt/homebrew/opt/python@3.13/bin/python3.13`).

### SUMO

[SUMO](https://eclipse.dev/sumo/) is a system dependency, not a pip package.
Experiments were run against **SUMO 1.24.0**. Install it (e.g. via the macOS
`.dmg`/Homebrew, or from source) and note the install path — you'll need it
below.

### Python libraries

All Python dependencies are pinned in [`requirements.txt`](requirements.txt)
(generated with `python -m pip freeze` from the environment used to run the
experiments; core libraries are `torch==2.6.0`, `botorch==0.17.2`,
`gpytorch==1.15.2`, `numpy`, `pandas`, `scipy`, `networkx`, `matplotlib`).

```bash
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. One-time setup

The run scripts locate the repo and the SUMO install via two things you must
point at your own machine:

1. **Repo path** — most scripts under `scripts/` hardcode the absolute
   project path near the top:
   ```python
   base_path = Path("/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn")
   ```
   Change this to wherever you cloned the repo (or `sed` it across the
   scripts you plan to run).

2. **SUMO install** — set `SUMO_HOME` to your SUMO install's `share/sumo`
   directory. Scripts set it themselves at import time, e.g.:
   ```python
   os.environ["SUMO_HOME"] = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo"
   os.environ["LIBSUMO_AS_TRACI"] = "1"
   ```
   Update this to your SUMO version/path, and update
   [`config/config.json`](config/config.json) (`"SUMO": "<path>"`) to match.
   `LIBSUMO_AS_TRACI=1` uses the faster `libsumo` bindings instead of
   `traci`/`sumo-gui`; unset it if you want to watch simulations in the GUI.

Both of the above are per-script edits today (no single config controls
them) — grep for `base_path = Path(` and `SUMO_HOME` if you rename the repo
directory or move to a different machine.

## 3. Project layout

```
config/                   # per-network sim_setup_*.json (BO iters, OD bounds, sim duration, ...)
network/                  # SUMO network/route/OD files for each test scenario
  quickstart_grid_3/        3x3 grid corridor (main benchmark network)
  grid_2d_3x3/               2D grid
  quickstart_grid_3_dense/  denser demand variant of quickstart_grid_3
  intas_crop/, most_crop/    cropped InTAS / MoSTScenario benchmark networks
InTAS/, most_scenario/    vendored upstream benchmark scenario repos
MOGP/                     graph-kernel implementations (task-covariance kernels)
  graph.py                  route-overlap graph kernel (GraphGP)
  graph_shared_private.py   shared+private graph kernel
  helpers_MOGP.py           ICM helper utilities
src/
  simulation/               SUMO invocation + XML/CSV parsing (sumo_runner.py, data_loader.py)
  models/                   GP model construction (gp_models.py)
  utils/                    edge grouping, error metrics, seeding
scripts/
  models/<family>/          one run_*.py BO-loop script per model family, argparse CLI
  setup/                    generate ground truth + initial Sobol samples for a scenario
  batch/                    shell scripts that launch several run_*.py in parallel (by seed)
  plotting/                 convergence plots from result CSVs
  validation/               diagnostics (e.g. permutation/null test for rho^2)
  archive/                  superseded/one-off scripts, kept for reference
notebook_for_toy/, notebook_for_generated_network/   exploratory notebooks
output/                   BO run outputs (git-ignored)
```

## 4. Reproducing an experiment

Each model family under `scripts/models/<family>/run_*.py` is a standalone
CLI script: it loads a network's config, runs SUMO once to get ground-truth
edge flows, draws Sobol initial points, then runs a BO loop
(`qLogExpectedImprovement` on a signed-residual objective, fit via BoTorch/
GPyTorch) for `--max_iters` iterations. Results land under
`output/<network>_<variant>/.../restart_1_seed-<seed>/`.

### Single run example

Independent-GP BO on the `quickstart_grid_3` network, seed 42, 20 Sobol init
points, 50 BO iterations:

```bash
python3 scripts/models/indgp/run_indgp_mc_ei_bo.py \
    --seed 42 --n_init 20 --max_iters 50
```

Graph-structured ICM (route-overlap kernel, rank 4) on the same network:

```bash
python3 scripts/models/icm_kronecker/run_graphicm_bo_grid3.py \
    --seed 42 --rank 4 --n_init 20 --max_iters 50
```

Shared+private graph kernel under a scaled demand level (see setup step
below for `--demand_scale`):

```bash
python3 scripts/models/shared_private/run_congestion_shared_private_bo.py \
    --demand_scale 2.0 --seed 42
```

Each script prints its own `--help` for the exact flags it accepts (seed,
rank, `n_init`, `max_iters`, `demand_scale`, etc. — not all flags are
supported by all scripts).

### Congestion-level scenarios

Some experiments compare model performance under scaled demand (e.g. 0.5x /
2x nominal OD). Generate the ground-truth simulation + initial Sobol points
for a given scale first:

```bash
python3 scripts/setup/setup_congestion_level.py --demand_scale 0.5
```

then point the corresponding `run_congestion_*_bo.py` script at the same
`--demand_scale`.

### Batch / multi-seed runs

`scripts/batch/*.sh` reproduce full seed sweeps used in the writeup (they
launch several `run_*.py` invocations in parallel and log to
`scripts/logs/`), e.g.:

```bash
bash scripts/batch/run_icm_graphicm_comparison.sh
```

Open the script first — it hardcodes seeds/ranks and the `cd
<repo-path>` at the top needs to match your clone location (see §2).

### Plotting

```bash
python3 scripts/plotting/plot_grid3_ninit5_comparison.py
```

reads the relevant `output/.../convergence.csv` files and writes a
convergence comparison PNG (see `scripts/*.png` for examples already
committed).

## 5. Configuration reference

`config/sim_setup_<network>.json` controls, per network: SUMO simulation
window/interval, OD sampling bounds, number of Sobol initial points
(`n_init_search`), and default BO iteration/restart counts. Scripts load this
via `src.simulation.data_loader.load_kwargs_config(base_path, ..., 
sim_setup_filename="sim_setup_<network>.json")` and typically override
`n_init`/`max_iters` from the CLI rather than editing the JSON.
