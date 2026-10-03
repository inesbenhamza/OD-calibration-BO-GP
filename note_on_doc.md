# Codebase Documentation

> Rewritten 2026-10-03 to match the current code. The project went through a
> refactor that removed the old centralized `bo_pipeline.py` /
> `src/optimization/` / `src/visualization/` layout — there is no shared
> `BayesianOptimizationLoop` class anymore. Each model/network combination is
> now a **self-contained script** under `scripts/models/*/`, following a
> common procedural pattern (see §5).

## Overview

SUMO Origin-Destination (OD) matrix calibration via Bayesian Optimization
(BO): propose an OD matrix, simulate it in SUMO, compare simulated link
flows against ground-truth flows, and use a Gaussian Process (GP) surrogate
+ BoTorch acquisition to propose the next OD matrix to try.

Six GP/kernel families are implemented, differing in how they model
correlation across the `L` monitored links (tasks):

| Model | Kernel idea | Implementation |
|---|---|---|
| **ScalarGP** ("vanilla") | One GP on a scalar aggregated error | `src/models/gp_models.py: initialize_vanillabo_model` |
| **IndGP** | `L` independent per-link GPs (no cross-link correlation) | `src/models/gp_models.py: initialize_independent_gp_models_with_modellist` |
| **ICM** | Kronecker task covariance `B·Bᵀ + diag(v)`, rank-`r`, fully learned | `MOGP/helpers_MOGP.py` |
| **Graph-diffusion GP** | Task kernel = diffusion/regularized-Laplacian/Matérn spectral filter over a route-overlap graph Laplacian | `MOGP/graph.py: build_graph_multitask_gp` |
| **Graph-Kronecker (ICM+graph)** | Same as ICM but the task covariance is seeded from a graph-derived adjacency instead of learned from scratch | `MOGP/graph.py: GraphKroneckerGP` |
| **SharedPrivate** | Additive: `K = private_l(x,x')·δ_lm + ρ²·k_shared(x,x')·B[l,m]`, nests IndGP as `ρ²→0` | `MOGP/graph.py: SharedPrivateGraphKernel` |

`B` (the fixed task-correlation matrix for the graph-based models) comes
from route overlap: `adjacency_route_overlap_weighted()` builds it from each
network's `routes.csv` (edge-incidence × OD split-ratio Gram matrix).

---

## Repo Layout

```
src/
  models/gp_models.py        # ScalarGP + IndGP model builders, GP fit/multistart helper
  simulation/
    data_loader.py           # config loading, od.xml <-> DataFrame
    sumo_runner.py           # build od.xml/taz.xml, run SUMO, parse loop/edge outputs
  utils/
    edge_flow_grouping.py    # per-network edge groups, get_representative_edges()
    link_flow_analysis.py    # nRMSE / per-edge error metrics
    seed.py                  # set_seed()

MOGP/
  graph.py                   # ALL THREE graph-based kernel families (see table above)
  helpers_MOGP.py             # ICM only (ICMMultiOutputGP, fit_icm_multistart, ...)

scripts/
  models/
    scalargp/                # ScalarGP BO run scripts
    indgp/                   # IndGP BO run scripts
    icm_kronecker/            # ICM + Graph-Kronecker BO run scripts
    route_overlap_graphgp/    # Graph-diffusion-GP BO run scripts
    shared_private/           # SharedPrivateGraphKernel BO run scripts
  setup/                      # setup_congestion_level*.py — generates GT sim + Sobol init points at a given demand scale
  validation/                  # e.g. permutation_null_test_rho2.py
  plotting/                    # convergence comparison plots
  batch/                       # .sh wrappers that loop a run script over seeds/scales

config/sim_setup_{network}.json   # per-network simulation + BO hyperparameters
network/{network_name}/           # net.xml, routes.csv, od.xml, additional.xml, taz.xml
output/{experiment}/              # all run outputs (see §6)
```

**Networks currently present**: `quickstart` (toy, 4 OD pairs), `quickstart_grid_3` / `quickstart_grid_3_dense`, `grid_2d_3x3` (3×3 grid, 12 OD pairs, 12 links — the main congestion-sweep network), `intas_crop` (real Ingolstadt OSM crop), `network_3junction`.

---

## 1 — Config System

```python
config = load_kwargs_config(base_path, model_name, sim_setup_filename="sim_setup_<network>.json")
```

Merges `config/config.json` (paths: `SUMO_PATH`, output format strings like
`EDGE_OUT_STR`/`TRIPS2ODS_OUT_STR`) with the network-specific
`sim_setup_<network>.json`:

```json
{
  "network_name": "grid_2d_3x3",
  "routes_file": "routes.csv",
  "sim_end_time": 3900,
  "od_duration_sec": 3600,
  "n_init_search": 20,
  "od_bound_start": 1, "od_bound_end": 1000,
  "BO_niter": 100, "BO_num_restarts": 16, "BO_raw_samples": 256,
  "dim_od": 12
}
```

`od_bound_end` and `dim_od` set the search box `[od_bound_start, od_bound_end]^dim_od`
BoTorch optimizes over. Congestion-sweep scripts scale `od_bound_end` by
`demand_scale` at runtime rather than using a separate network/config (see §4).

---

## 2 — Simulation Utilities (`src/simulation/`)

- `od_xml_to_df(path)` — parse an OD XML into a DataFrame (`fromTaz`, `toTaz`, `count`).
- `create_taz_xml(path, od_df, duration, base_dir)` — write an OD matrix as SUMO TAZ-relation XML.
- `simulate_od(od_xml, prefix, base_dir, net_xml, taz2edge_xml, additional_xml, routes_df, sim_end_time, trips_out_str, seed)` —
  the core simulate step: builds trips from the OD, assigns them to one of
  the network's pre-defined alternative routes per `routes.csv` (fixed
  multi-route assignment, not dynamic routing), and runs SUMO in mesoscopic
  mode (`--mesosim`).
- `parse_loop_data_xml_to_pandas(base_dir, sim_edge_file, prefix, sumo_path)` —
  parse SUMO's edge-data output into a per-edge flow DataFrame.

`src/utils/edge_flow_grouping.py`:
- `get_edge_groups(network_name)` — maps a representative edge label to the
  underlying SUMO edges that should be averaged together (serial duplicate
  edges with identical flow, e.g. detector chains). Hard-coded per network in
  `EDGE_GROUPS_*` dicts.
- `get_representative_edges(network_name)` — the `L` edge labels used as
  model outputs/tasks.
- `group_edge_dataframe(df, edge_groups)` — aggregate raw per-edge flows into
  one row per group.

---

## 3 — GP Model Builders

**ScalarGP / IndGP** (`src/models/gp_models.py`):
- `initialize_vanillabo_model(train_X, train_Y, kernel)` — single `SingleTaskGP` on a scalar aggregated error.
- `initialize_independent_gp_models_with_modellist(train_X, train_Y_multi, kernel)` — `L` independent `SingleTaskGP`s wrapped in a `ModelListGP`, each with its own lengthscale/outputscale/noise (fixed `ZeroMean`, matching the standardized-residual convention used across all models — see `MOGP/graph.py`'s module docstring for why this matters for the SharedPrivate-nests-IndGP claim).
- `fit_gpytorch_mll_multistart(mll, n_restarts)` — L-BFGS multistart with an Adam fallback if every restart fails to converge; used by (almost) every model type.

**ICM** (`MOGP/helpers_MOGP.py`):
- `initialize_icm_gp(...)` / `initialize_icm_gp_from_adjacency(...)` — builds `ICMMultiOutputGP` (`MultitaskKernel` = data kernel ⊗ `(B Bᵀ + diag(v))`, rank-`r` learned task covariance; the `_from_adjacency` variant seeds `B` from a graph adjacency instead of random init).
- `fit_icm_multistart(...)`.

**Graph-based models, all three in `MOGP/graph.py`**:
- `adjacency_route_overlap_weighted(routes_csv, edge_ids, edge_groups)` → `L×L` adjacency `A[i,j] = mean over OD pairs of P(edge i used)·P(edge j used)`, derived from each route's edge list and split ratio in `routes.csv`.
- `laplacian_eigh(A, normalized)` — graph Laplacian eigendecomposition, shared by the diffusion/regularized-Laplacian/Matérn-spectral kernels.
- `build_graph_multitask_gp(...)` → `GraphKernelAggregationModel` — product kernel `k_X(x,x')·K_G(l,m)` where `K_G` is a fixed or learnable graph-spectral task kernel.
- `GraphKroneckerGP` / `initialize_graph_kronecker_gp(...)` — ICM-style Kronecker GP whose task covariance starts from the graph adjacency.
- `build_route_overlap_correlation_matrix(routes_csv, edge_ids, kind="gram"|"matern_spectral")` → unit-diagonal `B` for the SharedPrivate kernel. `"gram"` (default) is the correlation-normalized route-incidence Gram matrix directly — zero extra hyperparameters. `"matern_spectral"` additionally smooths it with a fixed-hyperparameter `GraphMaternTaskKernel` spectral filter.
- `SharedPrivateGraphKernel` / `build_shared_private_graph_gp(...)` / `run_shared_private_graph_gp(...)` — the additive shared+private model (see table above). Uses `PerTaskGaussianLikelihood` (one noise scalar per task, like IndGP) and a fixed `ZeroMean`, so it nests IndGP's covariance, noise, *and* mean exactly as `ρ²→0` (not literally at `ρ²=0`: `rho2_constraint = Interval(1e-6, 20.0)` keeps it strictly positive for numerical reasons — negligible in practice, gradient is already ~1e-15 there).

**Cost note**: because every BO query evaluates the *same* OD point `x` on
all `L` links, the per-iteration GP fit for IndGP is `L` separate `O(n³)`
Choleskys (`n` = points so far), while every "joint" model above (ICM,
graph-diffusion, Kronecker, SharedPrivate) fits one `O((Ln)³)` Cholesky over
all `L·n` stacked rows — `O(L²)` more expensive than IndGP as the number of
monitored links grows. See `SharedPrivateGraphKernel`'s docstring and
`scripts/validation/permutation_null_test_rho2.py` for related diagnostics.

---

## 4 — Congestion-Sweep Workflow

Congestion is a **runtime demand multiplier** on the base network, not a
separate network. Two-step workflow, always against `NETWORK_NAME =
"grid_2d_3x3"` (or `quickstart_grid_3` / `intas_crop` for the `_qg3`/`_intas`
variants) and its base `sim_setup_<network>.json`:

1. **`scripts/setup/setup_congestion_level.py --demand_scale 2.0`** —
   scales the GT OD by `demand_scale`, runs the GT SUMO simulation at that
   scale, draws `N_INIT` Sobol points in the scaled bounds
   (`od_bound_end * demand_scale`), simulates each, and writes:
   ```
   output/grid_2d_3x3_congestion_{scale}x/
     ground_truth/sim_loopOutputs.csv
     initial_search/NRMSE/data_set_ods_{n_init}.csv
   ```
2. **`scripts/models/<family>/run_congestion_<family>_bo.py --demand_scale 2.0 --seed 42`** —
   loads that GT + init data, then runs the BO loop (§5) for the given model
   family, writing to `output/grid_2d_3x3_congestion_{scale}x/<Family>_mcEI/`.
   `scripts/batch/*.sh` wrap these over multiple seeds/scales.

---

## 5 — The BO Run-Script Pattern

Every `scripts/models/*/run_*.py` is self-contained and follows this shape
(illustrated by `run_congestion_indgp_bo.py`; other families swap only the
GP-build step and the posterior-wrapper class):

```
1. Load config, GT OD, routes.csv, edge_groups/edge_ids (L of them).
2. Load precomputed GT link flows + N_init Sobol points' residuals
   (from setup_congestion_level.py, or run fresh for non-congestion scripts).
3. Resume check: scan output/.../iter_{i}/ for already-completed BO
   iterations on disk and fast-forward train_X/train_R from them.
4. Wrap the per-model GP in a `Model` subclass exposing `.posterior()` that
   returns a `GPyTorchPosterior(MultitaskMultivariateNormal(...))` over the
   L tasks jointly (e.g. `IndGPJointPosteriorModel` in the IndGP script).
5. BO loop, each iteration:
     a. normalize(train_X, bounds)
     b. build + fit the GP (family-specific; fit_gpytorch_mll_multistart)
     c. qLogExpectedImprovement with a GenericMCObjective that aggregates
        the L task outputs into a scalar: -sum_l(C_sq * w_l * sample_l^2)
        (BoTorch maximizes, so negated; C_sq normalizes by GT flow scale)
     d. optimize_acqf -> x_next (normalized), unnormalize -> real OD vector
     e. simulate_od(x_next) in SUMO -> link flows -> residual r_next,
        S_next = sum(C_sq * r_next^2)
     f. append to train_X/train_R, update best_S, log convergence.csv /
        hyperparameters_per_iteration.csv / gp_diagnostics.csv
6. Print final best nRMSE = sqrt(best_S).
```

Key difference from the old (deleted) `BayesianOptimizationLoop`: there is
no shared loop class — each script duplicates this structure, with the GP
build (step 5b) and the joint-posterior wrapper (step 4) swapped per model
family. `src/models/gp_models.py`'s `IndGPAggregationModel` /
`extract_independent_gp_hyperparameters` are older shared-aggregation
helpers from before this per-script pattern consolidated; not all current
scripts use them (several, like the congestion scripts, define their own
local posterior-wrapper class instead, e.g. `IndGPJointPosteriorModel`).

---

## 6 — Output Layout

```
output/{experiment_name}/{Family}_mcEI/
  ground_truth/sim_loopOutputs.csv        (congestion scripts: shared across families, one level up)
  initial_search/NRMSE/data_set_ods_{n}.csv
  restart_{idx}_seed-{seed}/
    iter_0/od.xml, sim_loopOutputs.csv, ...
    iter_1/...
  results/restart_{idx}_seed-{seed}/
    convergence.csv                 # iteration, best_S, runtime_sec, total_runtime_sec
    hyperparameters_per_iteration.csv
    gp_diagnostics.csv              # acq_val per iteration
```

Seed → restart index: `restart_idx = (seed - 42) // 1000 + 1` (seed 42 →
restart 1, seed 1042 → restart 2, seed 2042 → restart 3).

---
