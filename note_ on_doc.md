# Codebase Documentation

## Overview

This codebase implements Bayesian Optimization (BO) for SUMO Origin-Destination (OD) matrix calibration. It supports three model types:
- **Vanilla BO**: Single GP on aggregated error
- **Independent GP**: Separate GP per edge
- **Multi-Output GP (ICM)**: Intrinsic Coregionalization Model
- **Graph-based kernel**: acts as the task correlation kernel 

The codebase is organized into modular packages (`src/optimization/`, `src/visualization/`, `src/simulation/`) with backward-compatible wrappers.

---

## File Structure

### `src/` - Main Source Code

#### `src/optimization/` - BO Core Logic
- **`bo_loop.py`**: `BayesianOptimizationLoop` class - Main BO iteration loop
- **`restarts.py`**: `run_multiple_bo_restarts()` - Orchestrates multiple BO runs
- **`results.py`**: `BOResults` dataclass - Stores BO results
- **`io.py`**: `save_bo_results()`, `load_bo_results()`, `load_all_restarts()` - Save/load functionality

#### `src/models/` - Gaussian Process Models
- **`gp_models.py`**: GP model initialization utilities (set_covar_module,initialize_vanillabo_model, initialize_independent_gp_models_with_modellist,make_linear_aggregation_model_from_error_gps, initialize_model,extract_independent_gp_hyperparameters)
- **`vanilla_bo_complete.py`**: Vanilla BO implementation
- **`ind_gp_model.py`**: Independent GP model implementation

#### `src/simulation/` - SUMO Simulation & Data Loading
- **`data_loader.py`**: Configuration loading (`load_kwargs_config()`)
- **`sumo_runner.py`**: SUMO simulation execution and XML creation
- **`run_parse.py`**: `run_simulation_for_od()`, `parse_simulation_outputs()` — run SUMO for an OD vector and parse outputs (used by BO loop and pipeline)
- **`evaluation.py`**: Single-run evaluation (legacy, unused in BO loop)

#### `src/utils/` - Utility Functions
- **`link_flow_analysis.py`**: Error metrics (squared, cubic, per-edge, aggregated)
- **`edge_flow_grouping.py`**: Edge groups (`EDGE_GROUPS`), `get_representative_edges()`, `group_edge_dataframe()`, `compute_grouped_loss()` — grouped scalar loss for vanilla BO
- **`seed.py`**: Random seed utilities

#### `src/visualization/` - Plotting Functions
- **`bo_results.py`**: Plotting functions that work with `BOResults` objects
- **`convergence.py`**: CSV-based convergence plotting (CLI script)

### Root Level Files
- **`bo_pipeline.py`**: Main CLI pipeline for OD calibration (uses `src.simulation.run_parse`, `src.utils.edge_flow_grouping`, `src.optimization`)
- **`helpers_od_calibration_BN.py`**: Backward-compatible wrapper (re-exports from `src/`)

### `scripts/` - Standalone Analysis Scripts
- **`summarize_final_s_across_restarts.py`**: Reads `results/restart_*_seed-*/convergence.csv`, computes summary stats (mean, std, median, etc.) of final S per method, writes `final_S_summary_across_restarts.csv` and optional boxplot. Run manually after BO runs; not imported elsewhere.
- **`export_per_gp_hyperparameters.py`**: Exports per-edge GP hyperparameters from saved results for comparison.



## Complete File Save Locations

### Base Directory Structure

All files are saved under:
```
output/{network_name}_{model_name}_{kernel}/
```

Where:
- `network_name`: e.g., `quickstart`, `berlin`
- `model_name`: `vanillabo`, `independent_gp`, `mogp`
- `kernel`: `matern-1p5`, `matern-2p5`, `rbf`

**Example**: `output/quickstart_vanillabo_matern-2p5/`

---

### 1. Ground Truth Simulation Files

**Location**: `{simulation_run_path}/ground_truth/`

**Files Created**:
```
output/{network_name}_{model_name}_{kernel}/ground_truth/
├── od.xml                          # Ground truth OD matrix XML
├── sim_edge_data.xml               # SUMO edge output (GT simulation)
├── sim_tripinfo.xml                 # SUMO trip info (if enabled)
└── sim_*.xml                        # Other SUMO output files
```

**When Created**: At the start of `bo_pipeline.py`, before initial search

**Purpose**: 
- Ground truth OD matrix (`od.xml`)
- Ground truth edge flows (`sim_edge_data.xml`) used as target for calibration
- Parsed into `gt_edge_data` DataFrame for error computation

---

### 2. Initial Search (Sobol Samples)

**Location**: `{simulation_run_path}/initial_search/sobol_{i}/`

**Files Created** (per initial sample `i`):
```
output/{network_name}_{model_name}_{kernel}/initial_search/
├── sobol_0/
│   ├── od.xml                       # OD matrix for sample 0
│   ├── sim_edge_data.xml            # SUMO edge output
│   ├── sim_tripinfo.xml             # SUMO trip info
│   └── sim_*.xml                    # Other SUMO outputs
├── sobol_1/
│   └── ...
└── sobol_{n_init_search-1}/
    └── ...
```

**When Created**: During initial search phase in `bo_pipeline.py`

**Purpose**:
- `n_init_search` Sobol quasi-random samples (default: 5)
- Each sample is evaluated via SUMO simulation
- Results used to initialize GP model before BO iterations

**Note**: If `CLEANUP_INTERMEDIATE_FILES` is enabled in config, these directories may be cleaned up after evaluation.

---

### 3. BO Iteration Files

**Location**: `{simulation_run_path}/bo_iterations/bo_iter_{iteration}/`

**Files Created** (per BO iteration):
```
output/{network_name}_{model_name}_{kernel}/bo_iterations/
├── bo_iter_0/
│   ├── od.xml                       # OD matrix proposed by acquisition function
│   ├── sim_edge_data.xml            # SUMO edge output
│   ├── sim_tripinfo.xml             # SUMO trip info
│   └── sim_*.xml                    # Other SUMO outputs
├── bo_iter_1/
│   └── ...
└── bo_iter_{n_iterations-1}/
    └── ...
```

**When Created**: During each BO iteration in `BayesianOptimizationLoop.simulate_and_evaluate()` (line 145)

**Purpose**:
- Each BO iteration proposes a new OD matrix via acquisition optimization
- OD matrix is saved as `od.xml`
- SUMO simulation is run, producing edge flow outputs
- Errors are computed and used to update the GP model

**Note**: Iteration numbering starts at 0 (first BO iteration after initial search).

---

### 4. Results Directory (Per Restart)

**Location**: `{simulation_run_path}/results/restart_{restart_idx}_seed-{seed}/`

**Files Created** (per restart):
```
output/{network_name}_{model_name}_{kernel}/results/
├── restart_1_seed-42/
│   ├── bo_results_restart_1.pkl    # Full BOResults object (pickle)
│   ├── convergence.csv              # Convergence data (iteration, best_S_model, acq_value, wall_time)
│   ├── edge_stats.csv               # Simulated flows per edge per iteration (if available)
│   ├── metadata.json                # Metadata (best_S, best_idx, seed, model_name, kernel, etc.)
│   ├── hyperparameters_last_iteration.json  # GP hyperparams at last iter (e.g. independent_gp)
│   ├── hyperparameters_per_edge.csv         # Per-edge hyperparams (if available)
│   └── convergence.png             # Optional convergence plot (if generated)
├── restart_2_seed-1042/
│   └── ...
└── restart_{n_restarts}_seed-{seed}/
    └── ...
```

**When Created**:
- Automatically by `restarts.py` (via `io.save_bo_results`) after each BO restart completes
- Overall best result and convergence CSV by `bo_pipeline.py` after all restarts

**File Details**:

**`bo_results_restart_{i}.pkl`**:
- Complete `BOResults` object saved as pickle (by `io.save_bo_results`)
- Contains: `all_X`, `all_Y_errors`, `all_S`, `best_X`, `best_S`, `convergence_curve`, `acq_values`, `wall_times`, `df_edge_stats`, `hyperparams_last_iteration` (if available)
- **Use**: Load with `load_bo_results()` to access all BO data without re-running

**`convergence.csv`**:
- Columns: `iteration`, `best_S_model`, `acq_value`, `wall_time`
- Format compatible with `src/visualization/convergence.py` and `scripts/summarize_final_s_across_restarts.py`
- **Use**: For plotting convergence curves, comparing restarts

**`edge_stats.csv`**:
- Contains simulated flows per edge per iteration
- Only saved if `df_edge_stats` is populated during BO loop
- **Use**: For flow coverage plots, edge-level analysis

**`metadata.json`**:
- Contains: `best_S`, `best_idx`, `iteration_start`, `n_evaluations`, `n_bo_iterations`, `restart_idx`, `seed`, `model_name`, `kernel`, `network_name`
- **Use**: Quick reference for experiment parameters and results

**`convergence.png`**:
- Optional plot of convergence curve (if generated by a script)
- Shows best S vs iteration

**Seed Calculation**:
- Restart 1: seed = 42
- Restart 2: seed = 1042 (42 + 1*1000)
- Restart 3: seed = 2042 (42 + 2*1000)
- Formula: `seed = 42 + (restart_idx - 1) * 1000`

---

### 5. Overall Best Result

**Location**: `{simulation_run_path}/results/`

**Files Created**:
```
output/{network_name}_{model_name}_{kernel}/results/
├── convergence.csv                  # Best restart's convergence curve
├── bo_results.pkl                   # Best restart's BOResults object
├── metadata.json                    # Best restart's metadata
└── (restart subdirectories...)
```

**When Created**: After all restarts complete in `bo_pipeline.py`

**Purpose**: 
- Stores the best result across all restarts
- `convergence.csv` contains the best restart's convergence curve
- `bo_results.pkl` contains the best restart's full results

---

### 6. Saved Results (Notebook Format)

**Location**: `{simulation_run_path}/saved_results/`

**Files Created** (if using notebook):
```
output/{network_name}_{model_name}_{kernel}/saved_results/
├── {model}_bo_histories.json        # All restart histories + metadata (JSON)
└── {model}_bo_histories.csv         # All restart histories (CSV format)
```

**When Created**: In notebook after BO runs complete

**Purpose**:
- JSON format: Complete restart histories with metadata
- CSV format: Easy-to-view table with columns `Restart_1`, `Restart_2`, ..., `Mean`

**Note**: This format is notebook-specific. The `bo_pipeline.py` script uses the `results/` directory structure instead.

---

### 7. Visualization Outputs

**Location**: `visualization/figures/` (from `src/visualization/convergence.py` or similar CLI)

**Files Created**:
```
visualization/figures/
├── convergence_{network}_{kernel}.png
├── fit_to_GT_{network}_{kernel}.png
└── (other plots...)
```

**When Created**: When running convergence/visualization CLI scripts

**Purpose**: Post-hoc analysis plots generated from saved `convergence.csv` files

---

### 8. Post-hoc Summary (Optional)

**Location**: Directory passed as `--out-dir` (default: same as `--base`)

**Files Created** (by `scripts/summarize_final_s_across_restarts.py`):
```
{out_dir}/
├── final_S_summary_across_restarts.csv   # Summary stats of final S per method
└── boxplot_final_S_by_method.pdf         # If --plot
```

**When Created**: When running `python scripts/summarize_final_s_across_restarts.py --base output/... [--plot] [--out-dir ...]`

**Purpose**: Compare final S across restarts for vanilla BO, IND-GP squared, IND-GP cubic (or custom method folders)

---

## Loading Saved Results

### Load Single Restart

```python
from src.optimization import load_bo_results
from pathlib import Path

# Load from restart directory
results = load_bo_results(
    Path("output/quickstart_vanillabo_matern-2p5/results/restart_1_seed-42")
)

# Or load from pickle file directly
results = load_bo_results(
    Path("output/quickstart_vanillabo_matern-2p5/results/restart_1_seed-42/bo_results_restart_1.pkl")
)

# Access data
print(f"Best S: {results.best_S}")
print(f"Best OD: {results.best_X}")
print(f"Convergence: {results.convergence_curve}")
```

### Load All Restarts

```python
from src.optimization import load_all_restarts
from pathlib import Path

# Load all restarts from results directory
all_results = load_all_restarts(
    Path("output/quickstart_vanillabo_matern-2p5/results")
)

# Iterate over restarts
for i, res in enumerate(all_results):
    print(f"Restart {i+1}: Best S = {res.best_S:.6f}")
```

### Use Loaded Results for Plotting

```python
from src.visualization import plot_best_iteration_like_prof
from src.optimization import load_bo_results

# Load results
results = load_bo_results(Path("output/.../results/restart_1_seed-42"))

# Plot (requires gt_edge_data, edge_ids, gt_od_vals, bounds)
plot_best_iteration_like_prof(
    results=results,
    gt_edge_data=gt_edge_data,
    edge_ids=edge_ids,
    gt_od_vals=gt_od_vals,
    bounds=bounds,
    save_path="plot.png"
)
```

---

## Configuration System

### How Config Works

1. **Load config**: `load_kwargs_config()` in `src/simulation/data_loader.py` creates a dictionary
2. **Access config**: Use `config["key"]` or `self.config["key"]` inside classes
3. **Config keys**: Both naming conventions work:
   - UPPERCASE: `config["NUM_RESTARTS"]`, `config["RAW_SAMPLES"]`
   - Lowercase: `config["BO_num_restarts"]`, `config["BO_raw_samples"]`

### Config Dictionary Contents

**Paths**:
- `config["simulation_run_path"]` = `f"output/{network_name}_{model_name}_{kernel}"`
- `config["net_xml"]`, `config["taz2edge_xml"]`, `config["file_gt_od"]`, etc.

**Simulation settings**:
- `config["sim_start_time"]`, `config["sim_end_time"]`, `config["od_duration_sec"]`
- `config["n_init_search"]` - Number of initial Sobol samples

**BO parameters**:
- `config["NUM_RESTARTS"]` / `config["BO_num_restarts"]` - Acquisition optimization restarts
- `config["RAW_SAMPLES"]` / `config["BO_raw_samples"]` - Raw samples for acquisition
- `config["BATCH_SIZE"]` / `config["BO_batch_size"]` - Batch size
- `config["NITER"]` / `config["BO_niter"]` - Number of BO iterations

**Early stopping**:
- `config["EARLY_STOP_PATIENCE"]`, `config["EARLY_STOP_DELTA"]`, `config["EARLY_STOP_MIN_ACQ"]`

---

## Model Types

### 1. Vanilla BO (`model_name="vanillabo"`)
- **Metric**: `compute_squared_metric_all_edge()` (or `compute_grouped_loss()` from `src.utils.edge_flow_grouping` when using `--use_grouped_edges`) — returns scalar (mean relative squared error)
- **Config**: Uses `config["NUM_RESTARTS"]`, `config["RAW_SAMPLES"]` for acquisition
- **Aggregation**: Linear aggregation with uniform weight before fitting the GP
- **GP fitting**: Single GP on aggregated error

### 2. Independent GP (`model_name="independent_gp"`)
- **Metric**: `compute_squared_metric_per_edge()` or `compute_cubic_metric_per_edge()` - Returns array
- **GP**: Separate GP for each edge
- **Aggregation**: Linear aggregation with uniform or flow-proportional weights using raw output

### 3. Multi-Output GP / ICM (`model_name="mogp"` or `model_name="icm"`)
- **Metric**: `compute_squared_metric_per_edge()` or `compute_cubic_metric_per_edge()` - Returns array
- **GP**: Intrinsic Coregionalization Model (ICM) for multi-output GP
- **Standardization**: Per-edge errors are standardized before training
- **Aggregation**: Linear aggregation over standardized outputs

---

## Quick Reference

### Run BO Optimization

```bash
python bo_pipeline.py \
  --network_name quickstart \
  --model_name vanillabo \
  --kernel matern-2p5 \
  --n_iterations 30 \
  --n_restarts 5 \
  --seed 42
```

### Access Config in Code

```python
# In bo_pipeline.py
config = load_kwargs_config(base_path, model_name, sim_setup_filename, kernel)
bounds = torch.tensor([[config.get("od_bound_start", 0.0)] * dim_od, 
                       [config.get("od_bound_end", 2000.0)] * dim_od])

# In BayesianOptimizationLoop class
num_restarts = self.config.get("NUM_RESTARTS", 5)  # or config["BO_num_restarts"]
raw_samples = self.config.get("RAW_SAMPLES", 32)    # or config["BO_raw_samples"]
```

### Error Metrics
- **Vanilla BO**: `compute_squared_metric_all_edge()` or `compute_grouped_loss()` (grouped edges) → scalar; both in `link_flow_analysis` / `edge_flow_grouping`
- **Independent GP**: `compute_squared_metric_per_edge()` → array
- **Independent GP (flow-weighted)**: `compute_cubic_metric_per_edge()` → array

### Save Results Manually (in Notebook)

```python
from src.optimization import save_bo_results
from pathlib import Path

save_bo_results(
    results=results,
    save_dir=Path(config['simulation_run_path']) / 'results' / 'restart_1_seed-42',
    restart_idx=1,
    seed=42,
    metadata={'model_name': 'vanillabo', 'kernel': 'matern-2p5'}
)
```

---

## Directory Structure Summary

```
output/{network_name}_{model_name}_{kernel}/
├── ground_truth/                    # GT simulation files
│   ├── od.xml
│   └── sim_*.xml
│
├── initial_search/                  # Sobol initial samples
│   ├── sobol_0/
│   │   ├── od.xml
│   │   └── sim_*.xml
│   └── sobol_{n_init-1}/
│
├── bo_iterations/                   # BO iteration simulations
│   ├── bo_iter_0/
│   │   ├── od.xml
│   │   └── sim_*.xml
│   └── bo_iter_{n_iter-1}/
│
├── results/                         # BO results and convergence
│   ├── convergence.csv              # Best restart convergence
│   ├── bo_results.pkl               # Best restart full results
│   ├── metadata.json                # Best restart metadata
│   ├── restart_1_seed-42/
│   │   ├── bo_results_restart_1.pkl
│   │   ├── convergence.csv
│   │   ├── edge_stats.csv
│   │   ├── metadata.json
│   │   ├── hyperparameters_last_iteration.json  # if available
│   │   ├── hyperparameters_per_edge.csv        # if available
│   │   └── convergence.png
│   └── restart_{n}_seed-{seed}/
│
└── saved_results/                   # Notebook format (optional)
    ├── {model}_bo_histories.json
    └── {model}_bo_histories.csv
```

---

## Notes

- **Consistent Seeds**: `bo_pipeline.py` and notebooks use the same seed calculation: `seed = 42 + (restart_idx - 1) * 1000`
- **Same Directory**: Notebooks and `bo_pipeline.py` save to the same directory structure, so results are interchangeable
- **Load Without Re-running**: Use `load_bo_results()` from `src.optimization` to load saved results and plot without re-running BO
- **Cleanup**: Set `CLEANUP_INTERMEDIATE_FILES=True` in config to remove intermediate simulation files after evaluation
- **Simulation/parsing**: `run_simulation_for_od()` and `parse_simulation_outputs()` live in `src.simulation.run_parse`; grouping and grouped loss in `src.utils.edge_flow_grouping`

---

## Pseudocode: GP Fitting & BO Pipeline (Graph Methods)

This section gives the full algorithm pseudocode for the graph-kernel and ICM GP models, covering standardization, kernel construction, GP fitting, acquisition, and the BO loop.

---

### 0 — Notation

| Symbol | Meaning |
|--------|---------|
| `x ∈ R^d` | OD parameter vector (d OD pairs) |
| `L` | Number of monitored edges (tasks) |
| `N` | Number of observed points so far |
| `e_l(x)` | Error on edge l for OD vector x |
| `z_l(x)` | Standardized error on edge l |
| `w_l` | Aggregation weight for edge l |
| `S(x)` | Scalar surrogate: weighted sum of edge errors |
| `K_G ∈ R^{L×L}` | Graph task-kernel matrix |
| `L_sym` | Normalized graph Laplacian |
| `τ, γ` | Diffusion / regularized Laplacian kernel parameters |

---

### 1 — Input Normalization

```
normalize_inputs(X_real, bounds):
    # Map each OD pair from [0, 2000] to [0, 1]
    X_norm = (X_real - bounds[0]) / (bounds[1] - bounds[0])   # [N, d]
    return X_norm

unnormalize_inputs(X_norm, bounds):
    return X_norm * (bounds[1] - bounds[0]) + bounds[0]
```

Initial design: `N_init` points via **scrambled Sobol** sequence in `[0,1]^d`.

---

### 2 — Output Standardization (per edge, done externally)

```
standardize_per_edge(Y_raw):
    # Y_raw: [N, L], per-edge errors; may contain NaN for missing edges
    y_mean[l] = mean(Y_raw[:, l],  ignoring NaN)   # [L]
    y_std[l]  = std (Y_raw[:, l],  ignoring NaN)   # [L]
    y_std      = clamp(y_std, min=1e-6)

    Y_std[:, l] = (Y_raw[:, l] - y_mean[l]) / y_std[l]
    return Y_std, y_mean, y_std                     # [N, L], [L], [L]
```

> **ICM/MOGP** trains on `Y_std`.
> **Graph-GP** also standardizes per-task (per-edge) before fitting (see §4).

---

### 3 — Graph Construction (MOGP/graph.py)

#### 3a — Build Adjacency Matrix A ∈ {0,1}^{L×L}

**Option A — `connections` (driving connectivity)**:
```
adjacency_from_connections(net_xml, edge_ids):
    A = zeros(L, L)
    for each <connection from=f to=t> in net_xml:
        if f in edge_ids and t in edge_ids and f != t:
            A[idx(f), idx(t)] = 1
            A[idx(t), idx(f)] = 1    # undirected
    A.fill_diagonal(0)
    return A                         # [L, L]
```

**Option B — `shared_node_weighted` (topological proximity)**:
```
adjacency_by_shared_node_weighted(net_xml, edge_ids, normalize="k-1"):
    A = zeros(L, L)
    node_to_edges = {}
    for each <edge id=e from=u to=v> in net_xml:
        node_to_edges[u].add(idx(e))
        node_to_edges[v].add(idx(e))

    for each node n with incident edges E_n (|E_n| = k > 1):
        w = 1 / (k - 1)             # weight per pair
        for each pair (i, j) in E_n × E_n, i != j:
            A[i, j] += w

    A = clamp(A, max=1.0)           # cap bidirectional bonus
    A.fill_diagonal(0)
    return A                         # [L, L], weighted ∈ [0, 1]
```

#### 3b — Compute Laplacian

```
laplacian(A, normalized=True):
    A = 0.5 * (A + A^T)             # symmetrize
    deg = row_sums(A).clamp_min(1e-12)

    if normalized:
        d = deg^{-1/2}
        A_norm = diag(d) @ A @ diag(d)
        L_sym = I - A_norm          # eigenvalues in [0, 2]
    else:
        L = diag(deg) - A           # unnormalized Laplacian

    return 0.5 * (L + L^T)         # enforce symmetry numerically
```

#### 3c — Eigendecomposition (shared by all graph kernels)

```
laplacian_eigh(A, normalized=True, spectrum_k=None):
    L = laplacian(A, normalized)
    evals, evecs = eigh(L)          # ascending order, evals >= 0
    evals = clamp(evals, min=0)

    if spectrum_k is not None:
        # Low-pass truncation: keep only k smallest eigenpairs
        evals = evals[:k]
        evecs = evecs[:, :k]        # [L, k]

    return evals, evecs             # [k], [L, k]
```

---

### 4 — Graph Task Kernels K_G ∈ R^{L×L}

Three kernel options, all built via eigendecomposition for numerical stability.

#### 4a — Diffusion Kernel (heat kernel)

```
diffusion_kernel(A, tau_mode="auto", tau_scale=1.0):
    evals, evecs = laplacian_eigh(A)

    if tau_mode == "auto":
        scale = median(evals[evals > 1e-12])    # or 1.0 if none positive
        tau = tau_scale / scale
    else:
        tau = tau_fixed

    # K[i,j] = sum_k evec[i,k] * exp(-tau*eval[k]) * evec[j,k]
    exp_evals = exp(-tau * evals)
    K_G = evecs @ diag(exp_evals) @ evecs^T
    K_G = 0.5 * (K_G + K_G^T)
    return K_G, tau                 # [L, L]
```

**Learnable variant**: `tau` is a trainable softplus-constrained parameter (updated by MLL).

#### 4b — Regularized Laplacian Kernel

```
regularized_laplacian_kernel(A, gamma_mode="auto", gamma_scale=1.0):
    evals, evecs = laplacian_eigh(A)

    if gamma_mode == "auto":
        scale = median(evals[evals > 1e-12])
        gamma = gamma_scale / scale

    # K = (I + gamma * L)^{-1}
    inv_evals = 1 / (1 + gamma * evals)
    K_G = evecs @ diag(inv_evals) @ evecs^T
    K_G = 0.5 * (K_G + K_G^T)
    return K_G, gamma               # [L, L]
```

#### 4c — Random Walk Kernel

```
random_walk_kernel(A, p=2, a=2.0):
    evals, evecs = laplacian_eigh(A)
    # require a >= max(evals) to ensure PSD
    rw_evals = (a - evals)^p
    K_G = evecs @ diag(rw_evals) @ evecs^T
    K_G = 0.5 * (K_G + K_G^T)
    K_G = K_G / mean(diag(K_G))    # normalize scale
    return K_G                      # [L, L], fixed (no learnable params)
```

---

### 5 — GP Models

#### 5a — Graph-Kernel GP (MOGP/graph.py: `build_graph_multitask_gp`)

Training data is **stacked**: each observation `(x, l)` is a row with task index appended.

```
build_graph_multitask_gp(net_xml, edge_ids, train_X_stacked, train_Y, input_dim, ...):
    # train_X_stacked: [N*L, d+1]  — last column = task integer index
    # train_Y:         [N*L, 1]

    # --- Step 1: Build adjacency and graph kernel
    A = adjacency_from_connections(net_xml, edge_ids)   # or shared_node_weighted
    K_G = diffusion_kernel(A)                           # [L, L]

    # --- Step 2: Per-task standardization (external, done before calling)
    for each task t in 0..L-1:
        mask_t = (task_col == t) & (~isnan(y))
        mu_t   = mean(y[mask_t])
        sig_t  = std (y[mask_t]).clamp_min(1e-6)
        y_std[mask_t] = (y[mask_t] - mu_t) / sig_t

    # --- Step 3: Product kernel k((x,t), (x',t')) = k_X(x,x') * K_G(t,t')
    x_kernel    = ScaleKernel(MaternKernel(nu=2.5, ARD))
    task_kernel = FixedGraphTaskKernel(K_G)             # or Learnable variant

    x_kernel.active_dims    = [0, ..., d-1]             # acts on x columns
    task_kernel.active_dims = [d]                       # acts on task index column

    covar = x_kernel * task_kernel                      # product kernel

    # --- Step 4: Fit SingleTaskGP (no MultiTaskGP — would add extra IndexKernel)
    model = SingleTaskGP(
        train_X = train_X_stacked,   # [N*L, d+1]
        train_Y = y_std,             # [N*L, 1]
        covar_module = covar,
        outcome_transform = None     # standardization already done externally
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll_with_adam_fallback(mll)            # see §6

    return model, mll, {tau_used, A_density, ...}
```

#### 5b — ICM GP (MOGP/helpers_MOGP.py: `ICMMultiOutputGP`)

```
ICMMultiOutputGP(train_X_norm, train_Y_std, rank=1):
    # train_X_norm: [N, d]   — already normalized to [0,1]^d
    # train_Y_std:  [N, L]   — already standardized per edge

    mean_module  = MultitaskMean(ConstantMean(), num_tasks=L)
    data_kernel  = MaternKernel(nu=2.5, ARD, lengthscale ∈ [0.005, 4.0])
    covar_module = MultitaskKernel(data_kernel, num_tasks=L, rank=rank)
    #  MultitaskKernel = data_kernel ⊗ (B B^T + diag(v))   (ICM factorization)
    #  B ∈ R^{L×rank}: inter-task correlation factor matrix (learnable)

    likelihood = MultitaskGaussianLikelihood(num_tasks=L, noise ∈ [1e-4, 1.0])

    forward(x):
        mean  = mean_module(x)    # [N, L]
        covar = covar_module(x)   # [NL, NL] block structure
        return MultitaskMultivariateNormal(mean, covar)
```

---

### 6 — GP Fitting

```
fit_gpytorch_mll_with_adam_fallback(mll):
    try:
        fit_gpytorch_mll(mll)      # L-BFGS-B (BoTorch default)
        return
    except Exception:
        # Fallback: Adam optimizer
        mll.train()
        optimizer = Adam(mll.parameters(), lr=0.1)
        for _ in range(200):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets).sum()
            loss.backward()
            optimizer.step()
```

---

### 7 — Scalar Aggregation for Acquisition

Both the Graph-GP and ICM paths expose a **scalar surrogate** `S(x)` to the EI acquisition:

```
LinearAggregationICM.posterior(X):
    # X: [b, d]
    for each batch point x_i:
        dist_i = base_icm_model(x_i)        # MultitaskMultivariateNormal, L outputs
        mu_i   = dist_i.mean               # [L]
        Cov_i  = dist_i.covariance_matrix  # [L, L]

        mu_S_i  = w^T @ mu_i               # scalar mean of S(x_i)
        var_S_i = w^T @ Cov_i @ w          # scalar variance of S(x_i)

    return GPyTorchPosterior(MVN(mu_S, diag(var_S)))   # [b, 1]
```

Aggregation weights:
- **Uniform**: `w_l = 1/L`
- **Flow-proportional**: `w_l = f_l / (L * Σ f_l)` where `f_l` = GT flow on edge l

For the **ICM** path the weights absorb the per-edge std: `w_l_eff = w_l * sigma_l`, so `S` lives in the standardized Z-space and `best_f` for EI is `min(Z_train)`.

---

### 8 — Acquisition Function: Expected Improvement

```
optimize_EI(aggregated_model, best_f, bounds_norm):
    acqf = ExpectedImprovement(
        model   = aggregated_model,
        best_f  = best_f,       # see note on spaces above
        maximize= False         # we minimize S
    )

    X_next, acq_val = optimize_acqf(
        acq_function = acqf,
        bounds       = [[0]*d, [1]*d],   # normalized space
        q            = 1,
        num_restarts = 5,
        raw_samples  = 32,
        options      = {"batch_limit": 5, "maxiter": 200}
    )
    return X_next   # [d]
```

---

### 9 — Full BO Loop

```
run_bo_loop(train_X_init_norm, train_Y_init, n_iterations, ...):

    all_X      = [train_X_init_norm]          # list of [d] tensors
    all_Y_err  = [train_Y_init]               # list of [L] per-edge errors
    all_S      = [aggregate(train_Y_init)]    # list of scalar S values
    best_S     = min(all_S)

    for t = 0 .. n_iterations - 1:

        # ---- 1. Stack observations ----
        X_train = stack(all_X)                # [N, d]
        Y_train = stack(all_Y_err)            # [N, L]

        # ---- 2. Standardize outputs (ICM / Graph-GP) ----
        Y_std, y_mean, y_std = standardize_per_edge(Y_train)   # [N, L]

        # ---- 3. Fit GP ----
        if model == "icm":
            model, mll = initialize_icm_gp(X_train, Y_std, rank=1)
            fit(mll)
            w_eff = w * y_std                 # absorb std into weights
            agg_model = LinearAggregationICM(model, w_eff)
            best_f = min(X_train @ w_eff)     # best in Z-space

        elif model == "graph_gp":
            X_stacked, Y_stacked = stack_with_task_ids(X_train, Y_std, L)
            model, mll, _ = build_graph_multitask_gp(
                net_xml, edge_ids, X_stacked, Y_stacked, input_dim=d
            )
            fit(mll)
            agg_model = LinearAggregationGraphGP(model, w)    # analogous wrapper
            best_f = min aggregate over training set

        elif model == "independent_gp":
            model_list = [SingleTaskGP(X_train, Y_train[:,l]) for l in 0..L]
            [fit(mll_l) for mll_l in mlls]
            agg_model = LinearAggregationModel(model_list, w)
            best_f = min(sum_l w_l * Y_train[:,l])

        elif model == "vanillabo":
            S_train = sum_l w_l * Y_train[:,l]
            model   = SingleTaskGP(X_train, S_train.unsqueeze(-1))
            fit(mll)
            agg_model = model
            best_f = min(S_train)

        # ---- 4. Optimize acquisition ----
        X_next_norm = optimize_EI(agg_model, best_f)     # [d]

        # ---- 5. Simulate & evaluate ----
        X_next_real = unnormalize(X_next_norm, bounds)
        run_sumo(X_next_real)                              # writes sim_edge_data.xml
        e_new = compute_per_edge_error(sim_output, gt_flows)  # [L]
        S_new = sum_l w_l * e_new[l]

        # ---- 6. Update dataset ----
        all_X.append(X_next_norm)
        all_Y_err.append(e_new)
        all_S.append(S_new)

        if S_new < best_S:
            best_S = S_new

        convergence_curve[t] = best_S

        # ---- 7. Early stopping ----
        if no improvement >= delta for last `patience` iterations:
            break

    return BOResults(all_X, all_Y_err, all_S, best_S, convergence_curve, ...)
```

---

### 10 — Multiple Restarts

```
run_multiple_bo_restarts(n_restarts=5, n_bo_iterations=30, ...):
    for r = 0 .. n_restarts - 1:
        seed_r = 42 + r * 1000
        set_seed(seed_r)

        results_r = run_bo_loop(
            train_X_init, train_Y_init,
            n_iterations = n_bo_iterations, ...
        )
        save_bo_results(results_r, path / f"restart_{r+1}_seed-{seed_r}")

    overall_best = argmin_r(results_r.best_S)
    return {all_results, convergence_curves, mean_curve, std_curve, overall_best}
```

---

### Summary: Space Conventions per Model

| Model | GP trains on | `best_f` space | Aggregation weights |
|-------|-------------|---------------|---------------------|
| VanillaBO | raw scalar S | raw (Standardize inside GP) | — |
| Independent GP | raw per-edge `e_l` | raw (Standardize inside GP) | `w_l` uniform or flow-prop |
| ICM (MultitaskKernel) | standardized `z_l = (e_l-μ_l)/σ_l` | Z-space: `min(Σ w_l σ_l z_l)` | `w_l σ_l` effective weights |
| Graph-GP (product kernel) | standardized `z_l` per task | Z-space | same as ICM |





tree from tree = ET.parse(net_xml_path)
root = tree.getroot()

<net>  (root)
├── <location .../>
├── <edge id=":1_0" function="internal">
│   ├── <lane id=":1_0_0" .../>
│   ├── <lane id=":1_0_1" .../>
│   └── <lane id=":1_0_2" .../>
├── <edge id=":1_3" ...>
│   └── ...
├── ...
├── <edge id="L1" from="..." to="...">   (normal edges have from/to nodes)
│   └── <lane .../>
├── ...
└── <connection from="D2" to="L2" .../>
    <connection from="L10" to="L7" .../>
    ...


So the tree is the whole XML document turned into a navigable object model: one root (<net>), and under it a flat or nested list of elements you can iterate over and read attributes from.