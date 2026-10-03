"""
Same as run_signed_residual_mc_ei_bo.py (signed-residual target r_l(x), graph
diffusion task kernel, qLogExpectedImprovement MC acquisition), but using
GraphKroneckerGP instead of the stacked single-task-GP-over-augmented-inputs
trick. Key model differences from the stacked version, for direct comparison:

  - PER-TASK noise (MultitaskGaussianLikelihood, has_task_noise=True, 34
    independent noise values) instead of stacked's single GLOBAL noise.
  - PER-TASK constant mean (MultitaskMean(ConstantMean(), num_tasks=L), 34
    independent learned constants) instead of stacked's single GLOBAL constant.
  - Native multi-output GP (MultitaskKernel, Kronecker-structured, O(N^3+L^3)
    fitting cost) instead of the stacked (x,edge_id)-row trick (O((N*L)^3)).
  - No (x, edge_id) row-stacking needed: train_X is [n,d], train_Y is [n,L]
    directly -- GraphKroneckerGP takes native multi-output data.
  - No custom joint-posterior wrapper needed: GraphKroneckerGP already inherits
    botorch's GPyTorchModel.posterior(), with its own un-standardization
    override, so it's passed straight to qLogExpectedImprovement.

Same 20 initial Sobol points (loaded from data_set_ods_20.csv's r_<edge_id>
columns), same SUMO simulation loop, same square-and-sum MC objective, same
seed -- the only intended difference from run_signed_residual_mc_ei_bo.py is
the GP model itself (global vs per-task noise/mean).
"""
import os
import sys
import time
import warnings
import xml.etree.ElementTree as _ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

base_path = Path("/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn")
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))
os.chdir(base_path)

os.environ["SUMO_HOME"] = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo"
os.environ["LIBSUMO_AS_TRACI"] = "1"
SUMO_HOME = Path(os.environ["SUMO_HOME"])
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
SUMO_ROOT = SUMO_HOME.parents[1]
SUMO_BIN = str(SUMO_ROOT / "bin")
os.environ["PATH"] = SUMO_BIN + os.pathsep + os.environ.get("PATH", "")

from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from src.utils.seed import set_seed
from MOGP.graph import run_graph_kronecker_gp, adjacency_route_overlap_weighted, laplacian_eigh

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

NETWORK_NAME = "quickstart_grid_3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

N_BO_ITERS = 100
N_GP_RESTARTS = 5
SEED = 42
OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 1000))

OUT_ROOT = Path("output") / "quickstart_grid_3_graph" / "graph_bo_diffusion_signed_residual_kronecker_pertask_qLogEI"
RESTART_DIR = OUT_ROOT / "results" / f"restart_1_seed-{SEED}"
SIM_BASE = OUT_ROOT / f"restart_1_seed-{SEED}"
RESTART_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)

VANILLA_BASE = Path("output") / "quickstart_grid_3_bo_vanilla_matern-2p5"


def load_grouped_counts(loop_csv_path):
    df = pd.read_csv(loop_csv_path, sep=";")
    df["interval_nVehContrib"] = df["edge_arrived"] + df["edge_left"]
    df_agg = df.groupby("edge_id", as_index=False).agg(interval_nVehContrib=("interval_nVehContrib", "sum"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(df_agg, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)


gt_counts = load_grouped_counts(VANILLA_BASE / "ground_truth" / "sim_loopOutputs.csv")
valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)

init_csv = pd.read_csv(VANILLA_BASE / "initial_search" / "NRMSE" / "data_set_ods_20.csv")
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
X_init = init_csv[x_cols].values.astype(float)

r_cols = [f"r_{eid}" for eid in edge_ids]
M_resid_init = init_csv[r_cols].values.astype(float)
M_sq_init = C_sq * M_resid_init ** 2
print(f"[INIT] loaded {len(X_init)} initial points (reused, not re-simulated). "
      f"best S so far = {M_sq_init.sum(axis=1).min():.6f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * X_init.shape[1], [OD_BOUND_END] * X_init.shape[1]], dtype=dtype, device=device)
d = X_init.shape[1]

train_X = torch.tensor(X_init, dtype=dtype, device=device)
train_R = torch.tensor(M_resid_init, dtype=dtype, device=device)   # signed residual -- GP training target, [n, L]
train_S_raw = M_sq_init.sum(axis=1)

best_S = float(train_S_raw.min())
S_history = [best_S]
runtime_history = [0.0]  # no BO-loop runtime for the initial Sobol points
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# Resume support: if iter_N folders from a previous (possibly stopped) run of
# THIS script already exist on disk, reconstruct train_X/train_R from them
# (od.xml has the simulated OD vector, sim_loopOutputs.csv has the raw SUMO
# counts -- same format load_grouped_counts already parses) instead of
# re-simulating, and continue the loop from the first missing iteration.
start_iter = 0
for it in range(N_BO_ITERS):
    iter_dir = SIM_BASE / f"iter_{it}"
    od_xml_path = iter_dir / "od.xml"
    loop_csv_path = iter_dir / "sim_loopOutputs.csv"
    if not od_xml_path.exists() or not loop_csv_path.exists():
        break
    tree = _ET.parse(str(od_xml_path))
    rows = [el.attrib for el in tree.getroot().iter("tazRelation")]
    x_resumed = pd.DataFrame(rows)["count"].astype(float).values
    sim_counts_resumed = load_grouped_counts(loop_csv_path)
    r_resumed = sim_counts_resumed - gt_counts
    e_resumed = C_sq * r_resumed ** 2
    S_resumed = float(np.nansum(e_resumed))

    train_X = torch.cat([train_X, torch.tensor(x_resumed, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    train_R = torch.cat([train_R, torch.tensor(r_resumed, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    best_S = min(best_S, S_resumed)
    S_history.append(best_S)
    runtime_history.append(float("nan"))  # wall time for resumed iterations is unknown
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] loaded {start_iter} completed iterations from disk, "
          f"resuming from iteration {start_iter+1}. best_S so far = {best_S:.6f}", flush=True)
else:
    if hp_log_path.exists():
        hp_log_path.unlink()
    if diag_path.exists():
        diag_path.unlink()

pd.DataFrame({
    "iteration": range(len(S_history)), "best_S": S_history,
    "runtime_sec": runtime_history,
    "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
}).to_csv(convergence_file, index=False)

# Adjacency / Laplacian eigendecomposition is fixed for the whole run -- precompute once.
A_fixed = adjacency_route_overlap_weighted(
    str(config["fixed_routes"]), edge_ids, edge_groups=edge_groups,
    device=device, dtype=torch.float64,
)
Lap_evals_fixed, Lap_evecs_fixed = laplacian_eigh(A_fixed, normalized=True)

w = torch.ones(L, dtype=dtype, device=device)


def square_sum_objective(samples, X=None):
    # qLogExpectedImprovement only maximizes -- negate to turn minimization
    # of S(x) into maximization (and negate best_f to match).
    return -(C_sq * w * samples ** 2).sum(dim=-1)




mc_objective = GenericMCObjective(square_sum_objective)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    # No (x, edge_id) row-stacking -- GraphKroneckerGP takes native [n,d]/[n,L].
    graph_model, mll = run_graph_kronecker_gp(
        train_X=X_norm,
        train_Y=train_R,
        edge_ids=edge_ids,
        adjacency_type="route_overlap_weighted",
        routes_csv_path=str(config["fixed_routes"]),
        edge_groups=edge_groups,
        graph_kernel_type="diffusion",
        standardize=True,
        device=device,
        Lap_evals=Lap_evals_fixed,
        Lap_evecs=Lap_evecs_fixed,
        n_gp_restarts=N_GP_RESTARTS,
    )
    graph_model.eval()

    data_kern = graph_model.covar_module.data_covar_module
    task_kern = graph_model.covar_module.task_covar_module
    tau = float(task_kern.tau.detach()) if hasattr(task_kern, "tau") else float("nan")
    ls = float(data_kern.lengthscale.detach().mean())
    task_noises = graph_model.likelihood.task_noises.detach().cpu().numpy().flatten()
    noise_mean = float(task_noises.mean())

    pd.DataFrame([{
        "iteration": it, "tau": tau, "noise_mean": noise_mean,
        "noise_min": float(task_noises.min()), "noise_max": float(task_noises.max()),
        "lengthscale": ls, "n_train": n, "best_S": best_S,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    # GraphKroneckerGP IS already a botorch Model (GPyTorchModel + ExactGP,
    # num_outputs=L) with its own posterior() un-standardization override, so
    # it can be passed straight to qLogEI -- no custom wrapper needed.
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    best_f = -float(best_S)
    qEI = qLogExpectedImprovement(model=graph_model, best_f=best_f, sampler=sampler, objective=mc_objective)

    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1, num_restarts=8, raw_samples=128)

    pd.DataFrame([{
        "iteration": it, "acq_val_qLogEI": acq_val.item(),
        "tau": tau, "noise_mean": noise_mean, "n_train": n,
    }]).to_csv(diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy()
    x_next_real = np.clip(x_next_real, OD_BOUND_START, OD_BOUND_END)

    sim_dir = SIM_BASE / f"iter_{it}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    od_xml = sim_dir / "od.xml"
    prefix_output = sim_dir / "sim"

    base_od = gt_od_df.copy()
    base_od["count"] = [round(v, 1) for v in x_next_real]
    base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
    create_taz_xml(str(od_xml), base_od, config["od_duration_sec"], str(base_path))

    simulate_od(
        str(od_xml), str(prefix_output), str(base_path),
        config["net_xml"], config["taz2edge_xml"], config["additional_xml"],
        routes_df, config["sim_end_time"], config["TRIPS2ODS_OUT_STR"],
        seed=SEED + it,
    )

    sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
    curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
        str(base_path), sim_edge_out, str(prefix_output), str(config["SUMO_PATH"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curr_grouped = group_edge_dataframe(curr_loop_stats, edge_groups=edge_groups)
    curr_grouped = curr_grouped.set_index("edge_id").loc[edge_ids]
    sim_counts_next = curr_grouped["interval_nVehContrib"].values.astype(float)

    r_next = sim_counts_next - gt_counts
    e_next = C_sq * r_next ** 2
    S_next = float(np.nansum(e_next))
    best_S = min(best_S, S_next)
    S_history.append(best_S)

    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    train_R = torch.cat([train_R, torch.tensor(r_next, dtype=dtype, device=device).unsqueeze(0)], dim=0)

    dt = time.time() - iter_t0
    runtime_history.append(dt)

    pd.DataFrame({
        "iteration": range(len(S_history)), "best_S": S_history,
        "runtime_sec": runtime_history,
        "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
    }).to_csv(convergence_file, index=False)

    print(f"[iter {it+1}/{N_BO_ITERS}] tau={tau:.4f} noise_mean={noise_mean:.4f} S_next={S_next:.6f} "
          f"best_S={best_S:.6f} nRMSE={np.sqrt(best_S):.6f} acqLogEI={acq_val.item():.4f} "
          f"({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE. Final best_S={best_S:.6f}  nRMSE={np.sqrt(best_S):.6f}")
print(f"Saved to {RESTART_DIR}")
