"""
Stacked Graph GP (diffusion kernel) BO on grid_2d_3x3 with TRUE partial coverage.

8 edges touching B1 (central bottleneck) are fully unobserved — no training data for them.
The stacked GP trains on (x, obs_task_id) pairs only.
At acquisition time it queries ALL L=24 task indices: the graph diffusion kernel
propagates information from observed to unobserved edges through the route-overlap graph.

NO imputation. The GP naturally predicts unobserved edges via cross-task correlation.

BO objective: proxy S = C_sq * sum_l E[r_l(x)^2] over ALL 24 predicted tasks
Final eval  : nRMSE on ALL 24 edges (true SUMO counts, including unobserved).

Reuses GT + Sobol init from run_scalargp_mc_ei_bo_grid2d.py.
Output: output/grid_2d_3x3_partial_cov/StackedGraphGP_partial_mcEI/results/restart_{idx}_seed-{seed}/
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
from MOGP.graph import run_graph_multitask_gp

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
L = len(edge_ids)  # 24 total edges

# ── sensor coverage ────────────────────────────────────────────────────────────
# Unobserved: 8 edges touching the central bottleneck node B1
# GP receives ZERO training data for these tasks. It predicts them via graph kernel.
UNOBS_EDGE_IDS = {"B0B1", "B1B0", "A1B1", "B1A1", "B1B2", "B2B1", "B1C1", "C1B1"}
OBS_EDGE_IDS   = [e for e in edge_ids if e not in UNOBS_EDGE_IDS]
obs_indices    = np.array([edge_ids.index(e) for e in OBS_EDGE_IDS])   # [L_obs] indices into edge_ids
unobs_indices  = np.array([i for i, e in enumerate(edge_ids) if e in UNOBS_EDGE_IDS])
obs_mask       = np.array([e not in UNOBS_EDGE_IDS for e in edge_ids])
L_obs          = len(OBS_EDGE_IDS)
print(f"[COVERAGE] {L_obs}/{L} edges observed. Unobserved (no training data): {sorted(UNOBS_EDGE_IDS)}", flush=True)

OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END   = float(config.get("od_bound_end", 1000))
d              = int(config.get("dim_od", 12))
N_GP_RESTARTS  = 5

RESTART_IDX  = (SEED - 42) // 1000 + 1
VANILLA_BASE = Path("output") / "grid_2d_3x3_bo_vanilla_matern-2p5"
PART_BASE    = Path("output") / "grid_2d_3x3_partial_cov"
_suffix      = "" if N_INIT == 20 else f"_ninit{N_INIT}"
OUT_ROOT     = PART_BASE / f"StackedGraphGP_partial_mcEI{_suffix}"
RESTART_DIR  = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE     = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
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


# ── ground truth ──────────────────────────────────────────────────────────────
GT_CSV = VANILLA_BASE / "ground_truth" / "sim_loopOutputs.csv"
if not GT_CSV.exists():
    raise FileNotFoundError("GT not found. Run run_scalargp_mc_ei_bo_grid2d.py first.")
gt_counts = load_grouped_counts(GT_CSV)

valid_mask_obs = gt_counts[obs_mask] > 0
C_sq_obs = float(valid_mask_obs.sum()) / (gt_counts[obs_mask][valid_mask_obs].sum() ** 2)

valid_mask_all = gt_counts > 0
C_sq_all = float(valid_mask_all.sum()) / (gt_counts[valid_mask_all].sum() ** 2)

print(f"[GT] L={L}, L_obs={L_obs}, C_sq_obs={C_sq_obs:.4e}, C_sq_all={C_sq_all:.4e}", flush=True)

# ── init points ───────────────────────────────────────────────────────────────
INIT_CSV = VANILLA_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv"
if not INIT_CSV.exists():
    raise FileNotFoundError("Init CSV not found. Run run_scalargp_mc_ei_bo_grid2d.py first.")

init_csv     = pd.read_csv(INIT_CSV, index_col=0)
x_cols       = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols       = [f"r_{eid}" for eid in edge_ids]
X_init       = init_csv[x_cols].values.astype(float)[:N_INIT]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]   # [N_INIT, 24] all true residuals

# BO tracks observed-edge objective; evaluation tracks full objective
M_sq_obs_init = C_sq_obs * (M_resid_init[:, obs_mask] ** 2).sum(axis=1)
S_all_init    = C_sq_all * (M_resid_init ** 2).sum(axis=1)
best_S_obs    = float(M_sq_obs_init.min())
best_S_all    = float(S_all_init.min())
print(f"[INIT] best_S_obs={best_S_obs:.6f}  best_S_all={best_S_all:.6f}  nRMSE_all={np.sqrt(best_S_all):.4f}", flush=True)

bounds      = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

# train_R stores ONLY observed residuals [n, L_obs] — no imputed values
train_X   = torch.tensor(X_init, dtype=dtype)
train_R   = torch.tensor(M_resid_init[:, obs_mask], dtype=dtype)   # [n, L_obs]
# For full-eval tracking, keep true residuals for all 24 (used only to update best_S_all — not fed to GP)
train_R_all_true = M_resid_init.copy()

S_obs_history = [best_S_obs]
S_all_history = [best_S_all]
runtime_history = [0.0]

convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path      = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path        = RESTART_DIR / "gp_diagnostics.csv"

# ── resume ────────────────────────────────────────────────────────────────────
start_iter = 0
for it in range(N_BO_ITERS):
    iter_dir = SIM_BASE / f"iter_{it}"
    if not (iter_dir / "od.xml").exists() or not (iter_dir / "sim_loopOutputs.csv").exists():
        break
    tree = _ET.parse(str(iter_dir / "od.xml"))
    x_resumed     = pd.DataFrame([el.attrib for el in tree.getroot().iter("tazRelation")])["count"].astype(float).values
    r_all_resumed = load_grouped_counts(iter_dir / "sim_loopOutputs.csv") - gt_counts
    r_obs_resumed = r_all_resumed[obs_mask]
    S_obs_r = float(np.nansum(C_sq_obs * r_obs_resumed ** 2))
    S_all_r = float(np.nansum(C_sq_all * r_all_resumed ** 2))
    train_X   = torch.cat([train_X, torch.tensor(x_resumed, dtype=dtype).unsqueeze(0)])
    train_R   = torch.cat([train_R, torch.tensor(r_obs_resumed, dtype=dtype).unsqueeze(0)])
    train_R_all_true = np.vstack([train_R_all_true, r_all_resumed])
    best_S_obs = min(best_S_obs, S_obs_r)
    best_S_all = min(best_S_all, S_all_r)
    S_obs_history.append(best_S_obs)
    S_all_history.append(best_S_all)
    runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iters. best_S_obs={best_S_obs:.6f}  best_S_all={best_S_all:.6f}", flush=True)
else:
    if hp_log_path.exists(): hp_log_path.unlink()
    if diag_path.exists(): diag_path.unlink()

pd.DataFrame({
    "iteration": range(len(S_obs_history)),
    "best_S_obs": S_obs_history, "best_S_all": S_all_history,
    "nRMSE_obs": np.sqrt(S_obs_history), "nRMSE_all": np.sqrt(S_all_history),
    "runtime_sec": runtime_history,
}).to_csv(convergence_file, index=False)


class StackedGraphGPJointModel(Model):
    """
    Wraps a stacked SingleTaskGP (trained only on observed task indices).
    At test time, queries ALL L tasks — the graph diffusion kernel propagates
    predictions to unobserved tasks via cross-task correlation.
    """
    def __init__(self, graph_model, num_tasks, input_dim, task_mean=None, task_std=None):
        super().__init__()
        self.graph_model = graph_model
        self.L = num_tasks
        self.d = input_dim
        self.task_mean = task_mean  # [L] or None
        self.task_std  = task_std   # [L] or None

    @property
    def num_outputs(self):
        return self.L

    def posterior(self, X, observation_noise=False, **kwargs):
        if X.dim() == 3:
            b, q, dd = X.shape
            X = X.view(b, dd)
        else:
            b, dd = X.shape
        # Stack ALL L tasks for prediction — includes unobserved tasks
        task_col = torch.arange(self.L, dtype=X.dtype, device=X.device).view(1, self.L, 1).expand(b, self.L, 1)
        X_exp    = X.unsqueeze(1).expand(b, self.L, self.d)
        X_stack  = torch.cat([X_exp, task_col], dim=-1).reshape(b * self.L, self.d + 1)

        self.graph_model.eval()
        post      = self.graph_model.posterior(X_stack, observation_noise=observation_noise)
        mean_flat = post.mean.view(b * self.L)         # [b*L]
        cov_flat  = post.covariance_matrix
        if hasattr(cov_flat, "evaluate"):
            cov_flat = cov_flat.evaluate()             # [b*L, b*L]

        # un-standardize: GP was fit on standardized residuals; undo per-task scaling
        if self.task_mean is not None and self.task_std is not None:
            tm = self.task_mean.to(X.device)           # [L]
            ts = self.task_std.to(X.device)            # [L]
            # mean_flat indexed as [b*L]: row i = (x_i//L, task i%L)
            ts_expanded = ts.repeat(b)                 # [b*L]
            tm_expanded = tm.repeat(b)
            mean_flat   = mean_flat * ts_expanded + tm_expanded
            # cov: scale by outer product of task stds
            ts_col  = ts.repeat(b).unsqueeze(1)        # [b*L, 1]
            cov_flat = cov_flat * (ts_col * ts_col.T)

        mean_tasks = mean_flat.view(b, self.L)         # [b, L]
        cov_blocks = cov_flat.view(b, self.L, b, self.L)
        idx        = torch.arange(b, device=X.device)
        cov_diag   = cov_blocks[idx, :, idx, :]       # [b, L, L] — per-batch-element task cov

        cov_diag = cov_diag + 1e-8 * torch.eye(self.L, dtype=X.dtype, device=X.device).unsqueeze(0)
        mvn = MultitaskMultivariateNormal(mean_tasks.unsqueeze(1), cov_diag, interleaved=True)
        return GPyTorchPosterior(mvn)


w_all = torch.ones(L, dtype=dtype)

def square_sum_objective(samples, X=None):
    # samples: [num_samples, q, L] — includes GP predictions for unobserved tasks
    return -(C_sq_obs * w_all * samples ** 2).sum(dim=-1)

mc_objective = GenericMCObjective(square_sum_objective)

# ── BO loop ───────────────────────────────────────────────────────────────────
t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    # Stack training data for OBSERVED tasks only
    task_col_tr  = torch.tensor(obs_indices, dtype=dtype).view(1, L_obs).expand(n, L_obs)  # [n, L_obs]
    X_exp_tr     = X_norm.unsqueeze(1).expand(n, L_obs, d)                                 # [n, L_obs, d]
    X_stacked    = torch.cat([X_exp_tr, task_col_tr.unsqueeze(-1)], dim=-1).reshape(n * L_obs, d + 1)
    Y_stacked    = train_R.reshape(-1, 1)   # [n*L_obs, 1] — only observed residuals

    graph_model, mll, info, fit_method = run_graph_multitask_gp(
        net_xml_path=None, edge_ids=edge_ids,
        train_X=X_stacked, train_Y=Y_stacked, input_dim=d,
        adjacency_type="route_overlap_weighted",
        routes_csv_path=str(config["fixed_routes"]),
        edge_groups=edge_groups, graph_kernel_type="diffusion",
        standardize=True, device=device, n_gp_restarts=N_GP_RESTARTS,
    )
    graph_model.eval()

    # extract graph kernel hyperparameters for logging
    task_kernel = graph_model.covar_module.kernels[1]
    tau   = float(task_kernel.tau.detach())   if hasattr(task_kernel, "tau")   else float("nan")
    gamma = float(task_kernel.gamma.detach()) if hasattr(task_kernel, "gamma") else float("nan")
    nu    = float(task_kernel.nu.detach())    if hasattr(task_kernel, "nu")    else float("nan")
    kappa = float(task_kernel.kappa.detach()) if hasattr(task_kernel, "kappa") else float("nan")
    noise = float(graph_model.likelihood.noise.detach())
    ls    = float(graph_model.covar_module.kernels[0].lengthscale.detach().mean())

    # retrieve per-task standardization buffers (so wrapper can un-standardize)
    task_mean = graph_model.task_mean.detach().clone() if hasattr(graph_model, "task_mean") else None
    task_std  = graph_model.task_std.detach().clone()  if hasattr(graph_model, "task_std")  else None

    pd.DataFrame([{
        "iteration": it, "tau": tau, "gamma": gamma, "nu": nu, "kappa": kappa,
        "noise": noise, "lengthscale": ls,
        "fit_method": fit_method, "n_train": n,
        "best_S_obs": best_S_obs, "best_S_all": best_S_all,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    joint_model = StackedGraphGPJointModel(graph_model, L, d, task_mean=task_mean, task_std=task_std)
    joint_model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    # best_f based on observed S — EI reference point from what we can actually measure
    qEI = qLogExpectedImprovement(model=joint_model, best_f=-float(best_S_obs),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1, num_restarts=8, raw_samples=128)

    pd.DataFrame([{
        "iteration": it, "acq_val": acq_val.item(), "tau": tau, "gamma": gamma,
        "nu": nu, "kappa": kappa, "n_train": n,
    }]).to_csv(diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = np.clip(unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy(),
                          OD_BOUND_START, OD_BOUND_END)

    iter_dir = SIM_BASE / f"iter_{it}"
    sim_counts_next = run_sim(x_next_real, iter_dir, seed_sim=SEED + it)
    r_all_next  = sim_counts_next - gt_counts
    r_obs_next  = r_all_next[obs_mask]

    S_obs_next = float(np.nansum(C_sq_obs * r_obs_next ** 2))
    S_all_next = float(np.nansum(C_sq_all * r_all_next ** 2))
    best_S_obs = min(best_S_obs, S_obs_next)
    best_S_all = min(best_S_all, S_all_next)

    S_obs_history.append(best_S_obs)
    S_all_history.append(best_S_all)
    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype).unsqueeze(0)])
    train_R = torch.cat([train_R, torch.tensor(r_obs_next, dtype=dtype).unsqueeze(0)])
    train_R_all_true = np.vstack([train_R_all_true, r_all_next])

    dt = time.time() - iter_t0
    runtime_history.append(dt)
    pd.DataFrame({
        "iteration": range(len(S_obs_history)),
        "best_S_obs": S_obs_history, "best_S_all": S_all_history,
        "nRMSE_obs": np.sqrt(S_obs_history), "nRMSE_all": np.sqrt(S_all_history),
        "runtime_sec": runtime_history,
    }).to_csv(convergence_file, index=False)

    print(f"[StackedGraphGP-partial seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"tau={tau:.4f}  nRMSE_obs={np.sqrt(best_S_obs):.4f}  nRMSE_all={np.sqrt(best_S_all):.4f}  "
          f"acq={acq_val.item():.4f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [StackedGraphGP-partial seed={SEED}].")
print(f"  nRMSE_obs={np.sqrt(best_S_obs):.6f}  (16 observed edges)")
print(f"  nRMSE_all={np.sqrt(best_S_all):.6f}  (all 24 edges — fair eval metric)")
print(f"  Results: {RESTART_DIR}")
