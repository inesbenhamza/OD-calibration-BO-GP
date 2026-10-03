"""
GraphKronecker GP (diffusion, per-task noise/mean) BO loop on grid_2d_3x3.
Reuses GT and Sobol init points from run_scalargp_mc_ei_bo_grid2d.py. Resume-safe.

Output:
  output/grid_2d_3x3_bo_vanilla_matern-2p5/GraphGP_kronecker_mcEI{_suffix}/
  results/restart_{idx}_seed-{seed}/
"""
import argparse
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
os.environ["PATH"] = str(SUMO_ROOT / "bin") + os.pathsep + os.environ.get("PATH", "")

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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=100)
args = parser.parse_args()
SEED = args.seed
N_INIT = args.n_init
N_BO_ITERS = args.max_iters

NETWORK_NAME = "grid_2d_3x3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_grid_2d_3x3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 1000))
d = int(config.get("dim_od", 12))
N_GP_RESTARTS = 5

RESTART_IDX = (SEED - 42) // 1000 + 1
VANILLA_BASE = Path("output") / "grid_2d_3x3_bo_vanilla_matern-2p5"
_suffix = "" if N_INIT == 20 else f"_ninit{N_INIT}"
OUT_ROOT = VANILLA_BASE / f"GraphGP_kronecker_mcEI{_suffix}"
RESTART_DIR = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
RESTART_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)


def load_grouped_counts(loop_csv_path):
    df = pd.read_csv(loop_csv_path, sep=";")
    df["interval_nVehContrib"] = df["edge_arrived"] + df["edge_left"]
    df_agg = df.groupby("edge_id", as_index=False).agg(
        interval_nVehContrib=("interval_nVehContrib", "sum")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(df_agg, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)


def run_sim(od_values, out_dir, seed_sim):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    od_xml = out_dir / "od.xml"
    prefix = out_dir / "sim"
    base_od = gt_od_df.copy()
    base_od["count"] = [round(float(v), 1) for v in od_values]
    base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
    create_taz_xml(str(od_xml), base_od, config["od_duration_sec"], str(base_path))
    simulate_od(str(od_xml), str(prefix), str(base_path),
                config["net_xml"], config["taz2edge_xml"], config["additional_xml"],
                routes_df, config["sim_end_time"], config["TRIPS2ODS_OUT_STR"], seed=seed_sim)
    sim_edge_out = f"{prefix}_{config['EDGE_OUT_STR']}"
    stats, _, _ = parse_loop_data_xml_to_pandas(str(base_path), sim_edge_out, str(prefix), str(config["SUMO_PATH"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(stats, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)


# ── ground truth + init points ────────────────────────────────────────────────
GT_CSV = VANILLA_BASE / "ground_truth" / "sim_loopOutputs.csv"
if not GT_CSV.exists():
    raise FileNotFoundError(f"GT not found at {GT_CSV}. Run run_scalargp_mc_ei_bo_grid2d.py first.")
gt_counts = load_grouped_counts(GT_CSV)
valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
print(f"[GT] L={L}, valid={valid_mask.sum()}, C_sq={C_sq:.6e}", flush=True)

INIT_CSV = VANILLA_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv"
if not INIT_CSV.exists():
    raise FileNotFoundError(f"Init CSV not found at {INIT_CSV}. Run run_scalargp_mc_ei_bo_grid2d.py first.")

init_csv = pd.read_csv(INIT_CSV, index_col=0)
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols = [f"r_{eid}" for eid in edge_ids]
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
M_sq_init = C_sq * M_resid_init ** 2
print(f"[INIT] seed={SEED}: {N_INIT} points, best S={M_sq_init.sum(axis=1).min():.6f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
train_X = torch.tensor(X_init, dtype=dtype)
train_R = torch.tensor(M_resid_init, dtype=dtype)  # [n, L]
best_S = float(M_sq_init.sum(axis=1).min())
S_history = [best_S]
runtime_history = [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# ── resume ────────────────────────────────────────────────────────────────────
start_iter = 0
for it in range(N_BO_ITERS):
    iter_dir = SIM_BASE / f"iter_{it}"
    if not (iter_dir / "od.xml").exists() or not (iter_dir / "sim_loopOutputs.csv").exists():
        break
    tree = _ET.parse(str(iter_dir / "od.xml"))
    x_resumed = pd.DataFrame([el.attrib for el in tree.getroot().iter("tazRelation")])["count"].astype(float).values
    r_resumed = load_grouped_counts(iter_dir / "sim_loopOutputs.csv") - gt_counts
    S_resumed = float(np.nansum(C_sq * r_resumed ** 2))
    train_X = torch.cat([train_X, torch.tensor(x_resumed, dtype=dtype).unsqueeze(0)])
    train_R = torch.cat([train_R, torch.tensor(r_resumed, dtype=dtype).unsqueeze(0)])
    best_S = min(best_S, S_resumed)
    S_history.append(best_S)
    runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iters loaded. best_S={best_S:.6f}", flush=True)
else:
    if hp_log_path.exists(): hp_log_path.unlink()
    if diag_path.exists(): diag_path.unlink()

pd.DataFrame({
    "iteration": range(len(S_history)), "best_S": S_history,
    "runtime_sec": runtime_history,
    "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
}).to_csv(convergence_file, index=False)

# ── precompute adjacency/Laplacian once ───────────────────────────────────────
A_fixed = adjacency_route_overlap_weighted(
    str(config["fixed_routes"]), edge_ids, edge_groups=edge_groups,
    device=device, dtype=torch.float64,
)
Lap_evals_fixed, Lap_evecs_fixed = laplacian_eigh(A_fixed, normalized=True)

w = torch.ones(L, dtype=dtype)

def square_sum_objective(samples, X=None):  # noqa: ARG001
    return -(C_sq * w * samples ** 2).sum(dim=-1)

mc_objective = GenericMCObjective(square_sum_objective)

# ── BO loop ───────────────────────────────────────────────────────────────────
t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

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

    pd.DataFrame([{
        "iteration": it, "tau": tau, "noise_mean": float(task_noises.mean()),
        "noise_min": float(task_noises.min()), "noise_max": float(task_noises.max()),
        "lengthscale": ls, "n_train": n, "best_S": best_S,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    qEI = qLogExpectedImprovement(model=graph_model, best_f=-float(best_S),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1, num_restarts=8, raw_samples=128)

    pd.DataFrame([{
        "iteration": it, "acq_val": acq_val.item(), "tau": tau,
        "noise_mean": float(task_noises.mean()), "n_train": n,
    }]).to_csv(diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = np.clip(unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy(),
                          OD_BOUND_START, OD_BOUND_END)

    iter_dir = SIM_BASE / f"iter_{it}"
    sim_counts_next = run_sim(x_next_real, iter_dir, seed_sim=SEED + it)
    r_next = sim_counts_next - gt_counts
    S_next = float(np.nansum(C_sq * r_next ** 2))
    best_S = min(best_S, S_next)
    S_history.append(best_S)
    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype).unsqueeze(0)])
    train_R = torch.cat([train_R, torch.tensor(r_next, dtype=dtype).unsqueeze(0)])

    dt = time.time() - iter_t0
    runtime_history.append(dt)
    pd.DataFrame({
        "iteration": range(len(S_history)), "best_S": S_history,
        "runtime_sec": runtime_history,
        "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
    }).to_csv(convergence_file, index=False)

    print(f"[Kronecker-grid2d seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"tau={tau:.4f}  S_next={S_next:.6f}  best_S={best_S:.6f}  "
          f"nRMSE={np.sqrt(best_S):.6f}  acq={acq_val.item():.4f}  "
          f"({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [Kronecker-grid2d seed={SEED}]. nRMSE={np.sqrt(best_S):.6f}")
print(f"Results: {RESTART_DIR}")
