"""
Shared+private graph GP (signed residuals) BO loop on quickstart_grid_3.
Same model as run_sharedprivate_mc_ei_bo_grid2d.py, ported to this network's
config/routes (d=16, L=34) and this network's output/init-point conventions
(matches run_signed_residual_mc_ei_bo.py, the route-overlap+matern baseline
this is meant to compare against).

Output:
  output/quickstart_grid_3_graph/graph_bo_sharedprivate_signed_residual_mcEI{_ninit_suffix}/
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
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from src.utils.seed import set_seed
from MOGP.graph import run_shared_private_graph_gp

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=100)
parser.add_argument("--shared_kernel_type", type=str, default="gram", choices=["gram", "matern_spectral"])
args = parser.parse_args()
SEED = args.seed
N_INIT = args.n_init
N_BO_ITERS = args.max_iters
SHARED_KERNEL_TYPE = args.shared_kernel_type

NETWORK_NAME = "quickstart_grid_3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

N_GP_RESTARTS = 5
OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 1000))

RESTART_IDX = (SEED - 42) // 1000 + 1
_ninit_suffix = "" if N_INIT == 20 else f"_ninit{N_INIT}"
_ninit_suffix += "" if SHARED_KERNEL_TYPE == "gram" else f"_{SHARED_KERNEL_TYPE}"
OUT_ROOT = Path("output") / "quickstart_grid_3_graph" / f"graph_bo_sharedprivate_signed_residual_mcEI{_ninit_suffix}"
RESTART_DIR = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
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
print(f"[GT] L={L}, valid={valid_mask.sum()}, C_sq={C_sq:.6e}", flush=True)

init_csv = pd.read_csv(VANILLA_BASE / "initial_search" / "NRMSE" / "data_set_ods_20.csv")
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]

r_cols = [f"r_{eid}" for eid in edge_ids]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
M_sq_init = C_sq * M_resid_init ** 2
print(f"[INIT] seed={SEED}: {N_INIT} points, best S={M_sq_init.sum(axis=1).min():.6f}", flush=True)

d = X_init.shape[1]
bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
train_X = torch.tensor(X_init, dtype=dtype)
train_R = torch.tensor(M_resid_init, dtype=dtype)
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


class GraphJointPosteriorModel(Model):
    def __init__(self, graph_model, num_tasks):
        super().__init__()
        self.graph_model = graph_model
        self.L = num_tasks

    @property
    def num_outputs(self):
        return self.L

    def posterior(self, X, observation_noise=False, **kwargs):
        if X.dim() == 3:
            b, q, dd = X.shape
            X = X.view(b, dd)
        else:
            b, dd = X.shape
        L_ = self.L
        self.graph_model.eval()
        task_col = torch.arange(L_, dtype=X.dtype).view(1, L_, 1).expand(b, L_, 1)
        X_exp = X.unsqueeze(1).expand(b, L_, dd)
        X_stack = torch.cat([X_exp, task_col], dim=-1).reshape(b * L_, dd + 1)
        post = self.graph_model.posterior(X_stack, observation_noise=observation_noise)
        mean_flat = post.mean.view(b * L_)
        cov_flat = post.covariance_matrix
        if hasattr(cov_flat, "evaluate"):
            cov_flat = cov_flat.evaluate()
        mean_tasks = mean_flat.view(b, L_)
        cov_blocks = cov_flat.view(b, L_, b, L_)
        idx = torch.arange(b)
        cov_diag = cov_blocks[idx, :, idx, :]
        if hasattr(self.graph_model, "task_mean") and hasattr(self.graph_model, "task_std"):
            tm, ts = self.graph_model.task_mean, self.graph_model.task_std
            mean_tasks = mean_tasks * ts.unsqueeze(0) + tm.unsqueeze(0)
            cov_diag = cov_diag * (ts[:, None] * ts[None, :]).unsqueeze(0)
        cov_diag = cov_diag + 1e-8 * torch.eye(L_, dtype=X.dtype).unsqueeze(0)
        mvn = MultitaskMultivariateNormal(mean_tasks.unsqueeze(1), cov_diag, interleaved=True)
        return GPyTorchPosterior(mvn)


w = torch.ones(L, dtype=dtype)

def square_sum_objective(samples, X=None):
    return -(C_sq * w * samples ** 2).sum(dim=-1)

mc_objective = GenericMCObjective(square_sum_objective)

# ── BO loop ───────────────────────────────────────────────────────────────────
t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    task_col = torch.arange(L, dtype=dtype).view(1, L, 1).expand(n, L, 1)
    X_exp = X_norm.unsqueeze(1).expand(n, L, d)
    X_stacked = torch.cat([X_exp, task_col], dim=-1).reshape(n * L, d + 1)
    Y_stacked = train_R.reshape(-1, 1)

    graph_model, mll, info, fit_method = run_shared_private_graph_gp(
        edge_ids=edge_ids, train_X=X_stacked, train_Y=Y_stacked, input_dim=d,
        routes_csv_path=str(config["fixed_routes"]), edge_groups=edge_groups,
        shared_kernel_type=SHARED_KERNEL_TYPE, device=device, n_gp_restarts=N_GP_RESTARTS,
    )

    kern = graph_model.covar_module
    rho2 = float(kern.rho2.detach())
    priv_ls = torch.tensor([float(pk.base_kernel.lengthscale.detach().mean()) for pk in kern.private_kernels])
    priv_os = torch.tensor([float(pk.outputscale.detach()) for pk in kern.private_kernels])
    phi = float(kern.shared_kernel.lengthscale.detach().mean())
    noise = graph_model.likelihood.noise.detach()
    priv_ls_cols = {f"priv_ls_{eid}": float(v) for eid, v in zip(edge_ids, priv_ls)}
    priv_os_cols = {f"priv_os_{eid}": float(v) for eid, v in zip(edge_ids, priv_os)}
    noise_cols = {f"noise_{eid}": float(v) for eid, v in zip(edge_ids, noise)}
    pd.DataFrame([{
        "iteration": it, "rho2": rho2,
        "priv_lengthscale_mean": float(priv_ls.mean()), "priv_lengthscale_min": float(priv_ls.min()),
        "priv_lengthscale_max": float(priv_ls.max()),
        "priv_outputscale_mean": float(priv_os.mean()),
        "shared_lengthscale_phi": phi,
        "noise_mean": float(noise.mean()), "noise_max": float(noise.max()),
        **priv_ls_cols, **priv_os_cols, **noise_cols,
        "fit_method": fit_method, "n_train": n, "best_S": best_S,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    joint_model = GraphJointPosteriorModel(graph_model, L)
    joint_model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    qEI = qLogExpectedImprovement(model=joint_model, best_f=-float(best_S),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1, num_restarts=8, raw_samples=128)
    pd.DataFrame([{"iteration": it, "acq_val": acq_val.item(), "rho2": rho2,
                   "fit_method": fit_method, "n_train": n}]).to_csv(
        diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = np.clip(unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy(),
                          OD_BOUND_START, OD_BOUND_END)

    iter_dir = SIM_BASE / f"iter_{it}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    od_xml = iter_dir / "od.xml"
    prefix_output = iter_dir / "sim"
    base_od = gt_od_df.copy()
    base_od["count"] = [round(float(v), 1) for v in x_next_real]
    base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
    create_taz_xml(str(od_xml), base_od, config["od_duration_sec"], str(base_path))
    simulate_od(str(od_xml), str(prefix_output), str(base_path),
                config["net_xml"], config["taz2edge_xml"], config["additional_xml"],
                routes_df, config["sim_end_time"], config["TRIPS2ODS_OUT_STR"], seed=SEED + it)
    sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
    curr_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
        str(base_path), sim_edge_out, str(prefix_output), str(config["SUMO_PATH"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curr_grouped = group_edge_dataframe(curr_loop_stats, edge_groups=edge_groups)
    curr_grouped = curr_grouped.set_index("edge_id").loc[edge_ids]
    sim_counts_next = curr_grouped["interval_nVehContrib"].values.astype(float)

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

    print(f"[SharedPrivateGraphGP-grid3 seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"rho2={rho2:.4f} phi={phi:.4f} fit={fit_method}  S_next={S_next:.6f}  best_S={best_S:.6f}  "
          f"nRMSE={np.sqrt(best_S):.6f}  acq={acq_val.item():.4f}  "
          f"({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [SharedPrivateGraphGP-grid3 seed={SEED}]. nRMSE={np.sqrt(best_S):.6f}")
print(f"Results: {RESTART_DIR}")
