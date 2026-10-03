"""
Graph-warm-started ICM BO for quickstart_grid_3.

Same as run_icm_mc_ei_bo.py, but the ICM task covariance B = W Wᵀ + diag(κ)
is initialised at each BO iteration from the route-overlap adjacency matrix:
  W  ← top-rank SVD of A_route_overlap
  κ  ← residual diagonal of A - W Wᵀ

MLL optimisation then refines B freely — the graph is a warm start, not a
constraint.  This combines the structural inductive bias of graph GPs with
ICM's ability to learn the true task correlations from data.

Output:
  output/quickstart_grid_3_graph/
    graph_icm_rank{RANK}_route_overlap_signed_residual_mcEI{_ninit_suffix}/
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
SUMO_BIN = str(SUMO_ROOT / "bin")
os.environ["PATH"] = SUMO_BIN + os.pathsep + os.environ.get("PATH", "")

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
from MOGP.helpers_MOGP import initialize_icm_gp_from_adjacency, fit_icm_multistart

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rank", type=int, default=1)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=100)
parser.add_argument("--n_gp_restarts", type=int, default=5)
args = parser.parse_args()
SEED = args.seed
RANK = args.rank
N_INIT = args.n_init
MAX_ITERS = args.max_iters
N_GP_RESTARTS = args.n_gp_restarts

NETWORK_NAME = "quickstart_grid_3"
ROUTE_OVERLAP_CSV = Path(__file__).parent / "adjacency_quickstart_grid_3_route_overlap.csv"

config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

# Load route-overlap adjacency once (reused every BO iter as warm-start)
_adj_df = pd.read_csv(ROUTE_OVERLAP_CSV, index_col=0)
ADJ_TENSOR = torch.tensor(
    _adj_df.loc[edge_ids, edge_ids].values, dtype=dtype, device=device
)

N_BO_ITERS = MAX_ITERS
OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 1000))
ACQF_NUM_RESTARTS = 8
ACQF_RAW_SAMPLES = 128

RESTART_IDX = (SEED - 42) // 1000 + 1
_ninit_suffix = "" if N_INIT == 20 else f"_ninit{N_INIT}"
OUT_ROOT = Path("output") / "quickstart_grid_3_graph" / f"graph_icm_rank{RANK}_route_overlap_signed_residual_mcEI{_ninit_suffix}"
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

init_csv = pd.read_csv(VANILLA_BASE / "initial_search" / "NRMSE" / f"data_set_ods_{N_INIT}.csv")
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]

r_cols = [f"r_{eid}" for eid in edge_ids]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
M_sq_init = C_sq * M_resid_init ** 2
print(f"[INIT] GraphICM rank={RANK} seed={SEED}: loaded {len(X_init)} initial points. "
      f"best S so far = {M_sq_init.sum(axis=1).min():.6f}", flush=True)

bounds = torch.tensor([[OD_BOUND_START] * X_init.shape[1], [OD_BOUND_END] * X_init.shape[1]], dtype=dtype, device=device)
d = X_init.shape[1]

train_X = torch.tensor(X_init, dtype=dtype, device=device)
train_R = torch.tensor(M_resid_init, dtype=dtype, device=device)
train_S_raw = M_sq_init.sum(axis=1)

best_S = float(train_S_raw.min())
S_history = [best_S]
runtime_history = [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# Resume support
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
    runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] loaded {start_iter} completed iterations from disk, resuming from "
          f"iteration {start_iter+1}. best_S so far = {best_S:.6f}", flush=True)
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


class IcmJointPosteriorModel(Model):
    """Wraps ICM multitask GP, exposes joint [b, L] posterior for MC acquisition."""

    def __init__(self, icm_model, num_tasks):
        super().__init__()
        self.icm_model = icm_model
        self.L = num_tasks

    @property
    def num_outputs(self) -> int:
        return self.L

    def posterior(self, X: torch.Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        if X.dim() == 3:
            b, q, dd = X.shape
            if q != 1:
                raise RuntimeError("q=1 only.")
            X = X.view(b, dd)
        else:
            b, dd = X.shape
        L_ = self.L
        self.icm_model.eval()

        X_batched = X.unsqueeze(1)
        dist = self.icm_model(X_batched)
        if observation_noise:
            dist = self.icm_model.likelihood(dist)
        mean_tasks = dist.mean.reshape(b, L_)
        cov = dist.covariance_matrix
        if hasattr(cov, "evaluate"):
            cov = cov.evaluate()
        cov_diag = cov.reshape(b, L_, L_)

        if hasattr(self.icm_model, "task_mean") and hasattr(self.icm_model, "task_std"):
            tm = self.icm_model.task_mean
            ts = self.icm_model.task_std
            mean_tasks = mean_tasks * ts.unsqueeze(0) + tm.unsqueeze(0)
            sigma_outer = ts[:, None] * ts[None, :]
            cov_diag = cov_diag * sigma_outer.unsqueeze(0)

        cov_diag = cov_diag + 1e-8 * torch.eye(L_, dtype=X.dtype, device=X.device).unsqueeze(0)

        from gpytorch.distributions import MultitaskMultivariateNormal
        mvn = MultitaskMultivariateNormal(mean_tasks.unsqueeze(1), cov_diag, interleaved=True)
        return GPyTorchPosterior(mvn)


w = torch.ones(L, dtype=dtype, device=device)


def square_sum_objective(samples, X=None):
    return -(C_sq * w * samples ** 2).sum(dim=-1)


mc_objective = GenericMCObjective(square_sum_objective)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    n = train_X.shape[0]
    X_norm = normalize(train_X, bounds=bounds)

    y_mean = train_R.mean(dim=0, keepdim=True)
    y_std = train_R.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_R_std = (train_R - y_mean) / y_std

    # Graph-warm-started ICM: initialise B from route-overlap adjacency
    icm_model, mll = initialize_icm_gp_from_adjacency(
        X_norm, train_R_std, adjacency_tensor=ADJ_TENSOR, rank=RANK,
    )
    icm_model = icm_model.to(device=device, dtype=dtype)
    icm_model.likelihood = icm_model.likelihood.to(device=device, dtype=dtype)
    mll = mll.to(device=device, dtype=dtype)
    icm_model.register_buffer("task_mean", y_mean.squeeze(0).to(device=device, dtype=dtype))
    icm_model.register_buffer("task_std", y_std.squeeze(0).to(device=device, dtype=dtype))

    icm_model.train()
    mll.train()
    fit_method = fit_icm_multistart(mll, n_restarts=N_GP_RESTARTS, verbose=False)
    icm_model.eval()

    ls = float(icm_model.covar_module.data_covar_module.lengthscale.detach().mean())
    noise_mean = float(icm_model.likelihood.task_noises.detach().mean())
    task_covar = icm_model.covar_module.task_covar_module.covar_matrix.to_dense().detach()
    off_diag_mean = float((task_covar - torch.diag(torch.diag(task_covar))).abs().mean())

    pd.DataFrame([{
        "iteration": it, "lengthscale": ls, "noise_mean": noise_mean,
        "task_covar_offdiag_mean": off_diag_mean, "n_train": n, "best_S": best_S,
        "fit_method": fit_method,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    joint_model = IcmJointPosteriorModel(icm_model, L).to(device=device, dtype=dtype)
    joint_model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    best_f = -float(best_S)
    qEI = qLogExpectedImprovement(model=joint_model, best_f=best_f, sampler=sampler, objective=mc_objective)

    x_next_norm, acq_val = optimize_acqf(
        qEI, bounds=bounds_norm, q=1,
        num_restarts=ACQF_NUM_RESTARTS, raw_samples=ACQF_RAW_SAMPLES,
    )

    pd.DataFrame([{
        "iteration": it, "acq_val_qLogEI": acq_val.item(), "n_train": n,
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

    print(f"[GraphICM rank={RANK} seed={SEED}] [iter {it+1}/{N_BO_ITERS}] "
          f"off_diag={off_diag_mean:.4f} S_next={S_next:.6f} best_S={best_S:.6f} "
          f"nRMSE={np.sqrt(best_S):.6f} acqLogEI={acq_val.item():.4f} "
          f"({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [GraphICM rank={RANK} seed={SEED}]. Final best_S={best_S:.6f}  nRMSE={np.sqrt(best_S):.6f}")
print(f"Saved to {RESTART_DIR}")
