"""
SharedPrivate Graph GP (MOGP.graph.SharedPrivateGraphKernel, real implementation
used in the original non-congested grid_2d_3x3 comparison) on grid_2d_3x3 at a
specified demand scale (congestion sweep).

Supports both B-matrix variants used in the original comparison:
  --shared_kernel_type gram            (default, recommended)
  --shared_kernel_type matern_spectral

Usage:
  python3 scripts/models/shared_private/run_congestion_sharedprivate_graph_bo.py \
      --demand_scale 2.0 --seed 42 --shared_kernel_type gram

Requires setup_congestion_level.py to have been run first for the given scale.

Output:
  output/grid_2d_3x3_congestion_{scale}x/SharedPrivateGraphGP_mcEI_{kernel_type}/results/restart_{idx}_seed-{seed}/
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
from MOGP.graph import run_shared_private_graph_gp, GraphKernelAggregationModel

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--demand_scale", type=float, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=50)
parser.add_argument("--n_gp_restarts", type=int, default=5)
parser.add_argument("--shared_kernel_type", type=str, default="gram", choices=["gram", "matern_spectral"])
args = parser.parse_args()

DEMAND_SCALE  = args.demand_scale
SCALE_LABEL   = f"{DEMAND_SCALE:g}x"
SEED          = args.seed
N_INIT        = args.n_init
N_BO_ITERS    = args.max_iters
N_GP_RESTARTS = args.n_gp_restarts
KERNEL_TYPE   = args.shared_kernel_type

NETWORK_NAME = "grid_2d_3x3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_grid_2d_3x3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

scaled_gt_od = gt_od_df.copy()
scaled_gt_od["count"] = (gt_od_df["count"] * DEMAND_SCALE).round(1)
scaled_gt_od_sim = scaled_gt_od.rename(columns={"fromTaz": "from", "toTaz": "to"}, errors="ignore")

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids    = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

OD_BOUND_START = 1.0
OD_BOUND_END   = round(float(config.get("od_bound_end", 1000)) * DEMAND_SCALE)
d = int(config.get("dim_od", 12))
ACQF_NUM_RESTARTS = int(config.get("BO_num_restarts", 16))
ACQF_RAW_SAMPLES  = int(config.get("BO_raw_samples", 256))

CONG_BASE   = Path("output") / f"grid_2d_3x3_congestion_{SCALE_LABEL}"
RESTART_IDX = (SEED - 42) // 1000 + 1
OUT_ROOT    = CONG_BASE / f"SharedPrivateGraphGP_mcEI_{KERNEL_TYPE}"
RESTART_DIR = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE    = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
RESTART_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)


def load_grouped_counts(loop_csv_path):
    df = pd.read_csv(loop_csv_path, sep=";")
    df["interval_nVehContrib"] = df["edge_arrived"] + df["edge_left"]
    df_agg = df.groupby("edge_id", as_index=False).agg(
        interval_nVehContrib=("interval_nVehContrib", "sum"))
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
    base_od = scaled_gt_od_sim.copy()
    base_od["count"] = [round(float(v), 1) for v in od_values]
    create_taz_xml(str(od_xml), base_od, config["od_duration_sec"], str(base_path))
    simulate_od(str(od_xml), str(prefix), str(base_path),
                config["net_xml"], config["taz2edge_xml"], config["additional_xml"],
                routes_df, config["sim_end_time"], config["TRIPS2ODS_OUT_STR"], seed=seed_sim)
    sim_edge_out = f"{prefix}_{config['EDGE_OUT_STR']}"
    stats, _, _ = parse_loop_data_xml_to_pandas(
        str(base_path), sim_edge_out, str(prefix), str(config["SUMO_PATH"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(stats, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)


GT_CSV = CONG_BASE / "ground_truth" / "sim_loopOutputs.csv"
gt_counts = load_grouped_counts(GT_CSV)
valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
edge_ids_valid = [e for e, v in zip(edge_ids, valid_mask) if v]
L_valid = len(edge_ids_valid)
print(f"[GT] scale={SCALE_LABEL} kernel={KERNEL_TYPE} L={L} valid={L_valid} C_sq={C_sq:.6e}", flush=True)

INIT_CSV = CONG_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv"
init_csv = pd.read_csv(INIT_CSV, index_col=0)
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols_valid = [f"r_{eid}" for eid in edge_ids_valid]
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
R_init = init_csv[r_cols_valid].values.astype(float)[:N_INIT]         # (N_INIT, L_valid) signed residuals
F_init = C_sq * R_init ** 2                                            # (N_INIT, L_valid) per-edge nRMSE^2 contributions
S_init = F_init.sum(axis=1)
print(f"[INIT] SharedPrivate({KERNEL_TYPE}) scale={SCALE_LABEL} seed={SEED}: "
      f"best nRMSE={np.sqrt(S_init.min()):.4f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
train_X = torch.tensor(X_init, dtype=dtype)
train_F = torch.tensor(F_init, dtype=dtype)   # (N, L_valid) — per-edge nRMSE^2 contributions, the GP target
best_S = float(S_init.min())
S_history, runtime_history = [best_S], [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"

start_iter = 0
for it in range(N_BO_ITERS):
    iter_dir = SIM_BASE / f"iter_{it}"
    if not (iter_dir / "od.xml").exists() or not (iter_dir / "sim_loopOutputs.csv").exists():
        break
    tree = _ET.parse(str(iter_dir / "od.xml"))
    x_res = pd.DataFrame([el.attrib for el in tree.getroot().iter("tazRelation")])["count"].astype(float).values
    counts_res = load_grouped_counts(iter_dir / "sim_loopOutputs.csv")
    r_res = counts_res[valid_mask] - gt_counts[valid_mask]
    f_res = C_sq * r_res ** 2
    S_res = float(np.nansum(f_res))
    train_X = torch.cat([train_X, torch.tensor(x_res, dtype=dtype).unsqueeze(0)])
    train_F = torch.cat([train_F, torch.tensor(f_res, dtype=dtype).unsqueeze(0)])
    best_S = min(best_S, S_res)
    S_history.append(best_S); runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iters. best_S={best_S:.6f}", flush=True)
else:
    if hp_log_path.exists(): hp_log_path.unlink()

pd.DataFrame({"iteration": range(len(S_history)), "best_S": S_history,
              "runtime_sec": runtime_history,
              "total_runtime_sec": pd.Series(runtime_history).fillna(0).cumsum(),
              }).to_csv(convergence_file, index=False)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    # stack to (n*L_valid, d+1): OD-major, task fastest — matches run_shared_private_graph_gp's convention
    task_col = torch.arange(L_valid, dtype=dtype).view(1, L_valid, 1).expand(n, L_valid, 1)
    X_exp    = X_norm.unsqueeze(1).expand(n, L_valid, d)
    X_stacked = torch.cat([X_exp, task_col], dim=-1).reshape(n * L_valid, d + 1)
    Y_stacked = train_F.reshape(-1, 1)

    sp_model, sp_mll, sp_info, fit_method = run_shared_private_graph_gp(
        edge_ids=edge_ids_valid, train_X=X_stacked, train_Y=Y_stacked,
        input_dim=d, routes_csv_path=str(config["fixed_routes"]),
        edge_groups=edge_groups, shared_kernel_type=KERNEL_TYPE,
        rho2_init=1.0, n_gp_restarts=N_GP_RESTARTS,
    )
    rho2 = float(sp_model.covar_module.rho2.item())
    pd.DataFrame([{"iteration": it, "rho2": rho2, "fit_method": fit_method, "n_train": n,
                   "best_S": best_S}]).to_csv(
        hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    # aggregation weights = 1 (train_F is already C_sq*(sim-gt)^2 per edge, sum = nRMSE^2 directly)
    weights = torch.ones(L_valid, dtype=dtype)
    agg_model = GraphKernelAggregationModel(
        sp_model, weights=weights,
        task_mean=getattr(sp_model, "task_mean", None), task_std=getattr(sp_model, "task_std", None),
    )
    agg_model.eval()

    # minimize S(x): objective returns -S, best_f = -best_S_so_far
    def neg_objective(samples, X=None):
        return -samples.squeeze(-1)

    mc_objective = GenericMCObjective(neg_objective)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    qEI = qLogExpectedImprovement(model=agg_model, best_f=-float(best_S),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1,
                                          num_restarts=ACQF_NUM_RESTARTS,
                                          raw_samples=ACQF_RAW_SAMPLES)

    x_next_real = np.clip(unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy(),
                          OD_BOUND_START, OD_BOUND_END)
    iter_dir = SIM_BASE / f"iter_{it}"
    sim_counts_next = run_sim(x_next_real, iter_dir, seed_sim=SEED + it)
    r_next = sim_counts_next[valid_mask] - gt_counts[valid_mask]
    f_next = C_sq * r_next ** 2
    S_next = float(np.nansum(f_next))
    best_S = min(best_S, S_next)
    S_history.append(best_S)
    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype).unsqueeze(0)])
    train_F = torch.cat([train_F, torch.tensor(f_next, dtype=dtype).unsqueeze(0)])

    dt = time.time() - iter_t0
    runtime_history.append(dt)
    pd.DataFrame({"iteration": range(len(S_history)), "best_S": S_history,
                  "runtime_sec": runtime_history,
                  "total_runtime_sec": pd.Series(runtime_history).fillna(0).cumsum(),
                  }).to_csv(convergence_file, index=False)

    print(f"[SharedPrivate({KERNEL_TYPE}) scale={SCALE_LABEL} seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"nRMSE={np.sqrt(best_S):.6f}  rho2={rho2:.4f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE. nRMSE={np.sqrt(best_S):.6f}  → {RESTART_DIR}")
