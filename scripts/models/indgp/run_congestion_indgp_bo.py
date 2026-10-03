"""
IndGP BO on grid_2d_3x3 at a specified demand scale (congestion sweep).

Usage:
  python3 scripts/models/indgp/run_congestion_indgp_bo.py --demand_scale 0.5 --seed 42
  python3 scripts/models/indgp/run_congestion_indgp_bo.py --demand_scale 2.0 --seed 42

Requires setup_congestion_level.py to have been run first for the given scale.

Output:
  output/grid_2d_3x3_congestion_{scale}x/IndGP_mcEI/results/restart_{idx}_seed-{seed}/
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
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from src.utils.seed import set_seed
from src.models.gp_models import initialize_independent_gp_models_with_modellist, fit_gpytorch_mll_multistart

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--demand_scale", type=float, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=50)
parser.add_argument("--n_gp_restarts", type=int, default=5)
args = parser.parse_args()

DEMAND_SCALE = args.demand_scale
SCALE_LABEL  = f"{DEMAND_SCALE:g}x"
SEED         = args.seed
N_INIT       = args.n_init
N_BO_ITERS   = args.max_iters
N_GP_RESTARTS = args.n_gp_restarts

NETWORK_NAME = "grid_2d_3x3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_grid_2d_3x3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

# Scale GT OD for simulation
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
OUT_ROOT    = CONG_BASE / "IndGP_mcEI"
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


# ── Load GT + init ─────────────────────────────────────────────────────────────
GT_CSV = CONG_BASE / "ground_truth" / "sim_loopOutputs.csv"
gt_counts = load_grouped_counts(GT_CSV)
valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
print(f"[GT] scale={SCALE_LABEL} L={L} valid={valid_mask.sum()} "
      f"mean_flow={gt_counts[valid_mask].mean():.1f} C_sq={C_sq:.6e}", flush=True)

INIT_CSV = CONG_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv"
init_csv = pd.read_csv(INIT_CSV, index_col=0)
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols = [f"r_{eid}" for eid in edge_ids]
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
M_sq_init = C_sq * M_resid_init ** 2
print(f"[INIT] IndGP scale={SCALE_LABEL} seed={SEED}: best nRMSE={np.sqrt(M_sq_init.sum(axis=1).min()):.4f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
train_X = torch.tensor(X_init, dtype=dtype)
train_R = torch.tensor(M_resid_init, dtype=dtype)
best_S = float(M_sq_init.sum(axis=1).min())
S_history, runtime_history = [best_S], [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

start_iter = 0
for it in range(N_BO_ITERS):
    iter_dir = SIM_BASE / f"iter_{it}"
    if not (iter_dir / "od.xml").exists() or not (iter_dir / "sim_loopOutputs.csv").exists():
        break
    tree = _ET.parse(str(iter_dir / "od.xml"))
    x_res = pd.DataFrame([el.attrib for el in tree.getroot().iter("tazRelation")])["count"].astype(float).values
    r_res = load_grouped_counts(iter_dir / "sim_loopOutputs.csv") - gt_counts
    S_res = float(np.nansum(C_sq * r_res ** 2))
    train_X = torch.cat([train_X, torch.tensor(x_res, dtype=dtype).unsqueeze(0)])
    train_R = torch.cat([train_R, torch.tensor(r_res, dtype=dtype).unsqueeze(0)])
    best_S = min(best_S, S_res)
    S_history.append(best_S); runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iters. best_S={best_S:.6f}", flush=True)
else:
    if hp_log_path.exists(): hp_log_path.unlink()
    if diag_path.exists(): diag_path.unlink()

pd.DataFrame({"iteration": range(len(S_history)), "best_S": S_history,
              "runtime_sec": runtime_history,
              "total_runtime_sec": pd.Series(runtime_history).fillna(0).cumsum(),
              }).to_csv(convergence_file, index=False)


class IndGPJointPosteriorModel(Model):
    def __init__(self, model_list_gp, num_tasks):
        super().__init__()
        self.model_list_gp = model_list_gp
        self.L = num_tasks

    @property
    def num_outputs(self): return self.L

    def posterior(self, X, observation_noise=False, **kwargs):
        if X.dim() == 3:
            X = X.view(X.shape[0], X.shape[2])
        self.model_list_gp.eval()
        post = self.model_list_gp.posterior(X, observation_noise=observation_noise)
        b = X.shape[0]
        mean = post.mean.view(b, self.L)
        var  = post.variance.view(b, self.L).clamp_min(1e-12)
        mvn  = MultitaskMultivariateNormal(mean.unsqueeze(1), torch.diag_embed(var), interleaved=True)
        return GPyTorchPosterior(mvn)


w = torch.ones(L, dtype=dtype)

def square_sum_objective(samples, X=None):
    return -(C_sq * w * samples ** 2).sum(dim=-1)

mc_objective = GenericMCObjective(square_sum_objective)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    X_norm = normalize(train_X, bounds=bounds)
    model_list_gp, mlls = initialize_independent_gp_models_with_modellist(
        X_norm, train_R, kernel="matern-2p5", constraint_mode="constrained")
    for sub_model, sub_mll in zip(model_list_gp.models, mlls):
        fit_gpytorch_mll_multistart(sub_mll, n_restarts=N_GP_RESTARTS)
    model_list_gp.eval()

    ls_mean = float(np.mean([float(m.covar_module.base_kernel.lengthscale.detach().mean())
                             for m in model_list_gp.models]))
    noise_mean = float(np.mean([float(m.likelihood.noise.detach()) for m in model_list_gp.models]))
    pd.DataFrame([{"iteration": it, "lengthscale_mean": ls_mean, "noise_mean": noise_mean,
                   "n_train": train_X.shape[0], "best_S": best_S}]).to_csv(
        hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    joint_model = IndGPJointPosteriorModel(model_list_gp, L)
    joint_model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    qEI = qLogExpectedImprovement(model=joint_model, best_f=-float(best_S),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1,
                                          num_restarts=ACQF_NUM_RESTARTS,
                                          raw_samples=ACQF_RAW_SAMPLES)
    pd.DataFrame([{"iteration": it, "acq_val": acq_val.item()}]).to_csv(
        diag_path, mode="a", header=not diag_path.exists(), index=False)

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
    pd.DataFrame({"iteration": range(len(S_history)), "best_S": S_history,
                  "runtime_sec": runtime_history,
                  "total_runtime_sec": pd.Series(runtime_history).fillna(0).cumsum(),
                  }).to_csv(convergence_file, index=False)

    print(f"[IndGP scale={SCALE_LABEL} seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"nRMSE={np.sqrt(best_S):.6f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE. nRMSE={np.sqrt(best_S):.6f}  → {RESTART_DIR}")
