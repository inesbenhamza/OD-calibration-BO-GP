"""
Shared+Private GraphGP on quickstart_grid_3 at a specified demand scale.

Kernel: K((x,l),(x',m)) = k_input(x,x') * (sigma_priv^2 * I + sigma_shared^2 * B_route)
B_route normalized so diagonal = 1.

rho = sigma_shared^2 / (sigma_priv^2 + sigma_shared^2)

Usage:
  python3 scripts/models/shared_private/run_congestion_shared_private_bo_qg3.py --demand_scale 0.5 --seed 42
  python3 scripts/models/shared_private/run_congestion_shared_private_bo_qg3.py --demand_scale 2.0 --seed 42
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
import gpytorch
import torch.nn.functional as F
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from linear_operator.operators import DenseLinearOperator

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

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from src.utils.seed import set_seed
from src.models.gp_models import fit_gpytorch_mll_multistart
from MOGP.graph import adjacency_route_overlap_weighted

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

DEMAND_SCALE  = args.demand_scale
SCALE_LABEL   = f"{DEMAND_SCALE:g}x"
SEED          = args.seed
N_INIT        = args.n_init
N_BO_ITERS    = args.max_iters
N_GP_RESTARTS = args.n_gp_restarts

NETWORK_NAME = "quickstart_grid_3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
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
d = int(config.get("dim_od", 16))
ACQF_NUM_RESTARTS = int(config.get("BO_num_restarts", 16))
ACQF_RAW_SAMPLES  = int(config.get("BO_raw_samples", 256))

CONG_BASE   = Path("output") / f"quickstart_grid_3_congestion_{SCALE_LABEL}"
RESTART_IDX = (SEED - 42) // 1000 + 1
OUT_ROOT    = CONG_BASE / "SharedPrivate_mcEI"
RESTART_DIR = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE    = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
RESTART_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)

# ── Build normalized route-overlap adjacency ───────────────────────────────────
B_route_raw = adjacency_route_overlap_weighted(
    edge_ids=edge_ids,
    routes_csv_path=str(config["fixed_routes"]),
    edge_groups=edge_groups,
)
row_sums = B_route_raw.sum(dim=1)
shift = row_sums.max().item() + 1e-4
B_psd   = B_route_raw + shift * torch.eye(L, dtype=B_route_raw.dtype)
diag    = torch.sqrt(torch.diag(B_psd))
B_route = B_psd / diag.unsqueeze(0) / diag.unsqueeze(1)
print(f"[ADJ] shift={shift:.4f}  B_route diag mean={torch.diag(B_route).mean():.4f}  "
      f"offdiag mean={B_route[~torch.eye(L, dtype=torch.bool)].mean():.4f}  "
      f"min_eig={torch.linalg.eigvalsh(B_route).min():.4f}", flush=True)


# ── Shared+Private GP ─────────────────────────────────────────────────────────
# K((x,l),(x',m)) = k_input(x,x') * (sigma_priv^2 * delta_lm + sigma_shared^2 * B_route[l,m])
class SharedPrivateGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, d_in, num_tasks, B_route_normalized):
        super().__init__(train_x, train_y, likelihood)
        self.d = d_in
        self.L = num_tasks
        self.register_buffer("B_route", B_route_normalized.to(dtype=dtype))
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d_in, dtype=dtype))
        self.raw_sigma_priv   = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.raw_sigma_shared = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def sigma_priv(self):
        return F.softplus(self.raw_sigma_priv) + 1e-6

    @property
    def sigma_shared(self):
        return F.softplus(self.raw_sigma_shared) + 1e-6

    def transform_inputs(self, X=None, **kwargs):
        return X

    @property
    def rho(self):
        sp2 = self.sigma_priv.detach() ** 2
        ss2 = self.sigma_shared.detach() ** 2
        return float(ss2 / (sp2 + ss2))

    def forward(self, x):
        d_in = self.d
        x_input = x[..., :d_in]
        t_idx   = x[..., d_in:d_in+1].squeeze(-1).long()  # (n,)

        k_xx = self.covar_module(x_input).to_dense()  # (n, n)

        eye = torch.eye(self.L, dtype=x.dtype, device=x.device)
        B   = self.sigma_priv**2 * eye + self.sigma_shared**2 * self.B_route
        k_tt = B[t_idx[:, None], t_idx[None, :]]       # (n, n) outer indexing

        K    = k_xx * k_tt
        K    = K + 1e-4 * torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
        mean = self.mean_module(x_input).expand(x.shape[-2])
        return MultivariateNormal(mean, DenseLinearOperator(K))


def fit_shared_private(train_X_stack, train_R_flat, n_restarts=5):
    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(dtype=dtype)
    model = SharedPrivateGP(train_X_stack, train_R_flat, likelihood, d, L, B_route).to(dtype=dtype)
    mll   = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll_multistart(mll, n_restarts=n_restarts)
    model.eval(); likelihood.eval()
    return model, likelihood


class SPJointPosteriorModel(Model):
    def __init__(self, sp_model, num_tasks):
        super().__init__()
        self.sp_model = sp_model
        self.L = num_tasks

    @property
    def num_outputs(self): return self.L

    def posterior(self, X, observation_noise=False, **kwargs):
        if X.dim() == 3:
            b, q, dd = X.shape
            X = X.view(b, dd)
        else:
            b, dd = X.shape
        L_ = self.L
        self.sp_model.eval()
        task_col  = torch.arange(L_, dtype=X.dtype).view(1, L_, 1).expand(b, L_, 1)
        X_exp     = X.unsqueeze(1).expand(b, L_, dd)
        X_stack   = torch.cat([X_exp, task_col], dim=-1).reshape(b * L_, dd + 1)
        with gpytorch.settings.fast_pred_var():
            post = self.sp_model.likelihood(self.sp_model(X_stack))
        mean_flat = post.mean
        var_flat  = post.variance.clamp_min(1e-12)
        mean_tasks = mean_flat.view(b, L_)
        var_tasks  = var_flat.view(b, L_)
        mvn = MultitaskMultivariateNormal(
            mean_tasks.unsqueeze(1), torch.diag_embed(var_tasks), interleaved=True)
        return GPyTorchPosterior(mvn)


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
print(f"[GT] scale={SCALE_LABEL} L={L} mean_flow={gt_counts[valid_mask].mean():.1f} "
      f"C_sq={C_sq:.6e}", flush=True)

INIT_CSV = CONG_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv"
init_csv = pd.read_csv(INIT_CSV, index_col=0)
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols = [f"r_{eid}" for eid in edge_ids]
X_init       = init_csv[x_cols].values.astype(float)[:N_INIT]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
M_sq_init    = C_sq * M_resid_init ** 2
print(f"[INIT] SharedPrivate-qg3 scale={SCALE_LABEL} seed={SEED}: "
      f"best nRMSE={np.sqrt(M_sq_init.sum(axis=1).min()):.4f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
train_X = torch.tensor(X_init, dtype=dtype)
train_R = torch.tensor(M_resid_init, dtype=dtype)
best_S = float(M_sq_init.sum(axis=1).min())
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

pd.DataFrame({"iteration": range(len(S_history)), "best_S": S_history,
              "runtime_sec": runtime_history,
              "total_runtime_sec": pd.Series(runtime_history).fillna(0).cumsum(),
              }).to_csv(convergence_file, index=False)

w = torch.ones(L, dtype=dtype)

def square_sum_objective(samples, X=None):
    return -(C_sq * w * samples ** 2).sum(dim=-1)

mc_objective = GenericMCObjective(square_sum_objective)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    task_col  = torch.arange(L, dtype=dtype).view(1, L, 1).expand(n, L, 1)
    X_exp     = X_norm.unsqueeze(1).expand(n, L, d)
    X_stacked = torch.cat([X_exp, task_col], dim=-1).reshape(n * L, d + 1)
    R_flat    = train_R.reshape(-1)

    sp_model, likelihood = fit_shared_private(X_stacked, R_flat, n_restarts=N_GP_RESTARTS)

    rho          = sp_model.rho
    sigma_priv   = float(sp_model.sigma_priv.detach())
    sigma_shared = float(sp_model.sigma_shared.detach())
    ls           = float(sp_model.covar_module.base_kernel.lengthscale.detach().mean())
    noise        = float(sp_model.likelihood.noise.detach())
    pd.DataFrame([{"iteration": it, "rho": rho, "sigma_priv": sigma_priv,
                   "sigma_shared": sigma_shared, "lengthscale": ls,
                   "noise": noise, "n_train": n, "best_S": best_S}]).to_csv(
        hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    joint_model = SPJointPosteriorModel(sp_model, L)
    joint_model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    qEI = qLogExpectedImprovement(model=joint_model, best_f=-float(best_S),
                                   sampler=sampler, objective=mc_objective)
    x_next_norm, _ = optimize_acqf(qEI, bounds=bounds_norm, q=1,
                                    num_restarts=ACQF_NUM_RESTARTS,
                                    raw_samples=ACQF_RAW_SAMPLES)

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

    print(f"[SP-qg3 scale={SCALE_LABEL} seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"nRMSE={np.sqrt(best_S):.6f}  rho={rho:.3f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE. nRMSE={np.sqrt(best_S):.6f}  → {RESTART_DIR}")
