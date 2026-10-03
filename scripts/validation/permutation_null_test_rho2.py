"""
Permutation/null test for rho^2: shuffle each edge's residuals independently
across training points (destroying true cross-edge correspondence while
preserving each edge's own marginal distribution), refit the shared+private
kernel with the HalfNormal prior active, and see what rho^2 comes out.

If the fitting procedure is behaving correctly, rho^2 on this shuffled null
data should drop close to 0 (no real cross-edge structure left to exploit).
If it stays meaningfully large, that's direct evidence of residual fitting
bias in the real (non-shuffled) rho^2 = 6.13 result.

Uses the exact same real N=100 grid_2d_3x3 trajectory as the earlier
verify_rho2_prior.py diagnostic, for a direct apples-to-apples comparison.
"""
import sys, warnings
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import pandas as pd
import numpy as np

base_path = Path("/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn")
sys.path.insert(0, str(base_path))
import os
os.chdir(base_path)
torch.set_default_dtype(torch.float64)
dtype = torch.double

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize

from src.simulation.data_loader import load_kwargs_config
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from MOGP.graph import (
    SharedPrivateGraphKernel, PerTaskGaussianLikelihood,
    build_route_overlap_correlation_matrix, fit_shared_private_mll_multistart,
)

NETWORK_NAME = "grid_2d_3x3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_grid_2d_3x3.json")
edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)
d = int(config.get("dim_od", 12))

B = build_route_overlap_correlation_matrix(
    str(config["fixed_routes"]), edge_ids, edge_groups=edge_groups, kind="gram",
)

VANILLA_BASE = Path("output") / "grid_2d_3x3_bo_vanilla_matern-2p5"
GT_CSV = VANILLA_BASE / "ground_truth" / "sim_loopOutputs.csv"

def load_grouped_counts(loop_csv_path):
    df = pd.read_csv(loop_csv_path, sep=";")
    df["interval_nVehContrib"] = df["edge_arrived"] + df["edge_left"]
    df_agg = df.groupby("edge_id", as_index=False).agg(interval_nVehContrib=("interval_nVehContrib", "sum"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(df_agg, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)

gt_counts = load_grouped_counts(GT_CSV)
RUN_DIR = VANILLA_BASE / "IndGP_mcEI" / "restart_1_seed-42"
INIT_CSV = VANILLA_BASE / "initial_search" / "NRMSE" / "data_set_ods_20.csv"
init_csv = pd.read_csv(INIT_CSV, index_col=0)
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols = [f"r_{eid}" for eid in edge_ids]
X_all = [init_csv[x_cols].values.astype(float)[:20]]
R_all = [init_csv[r_cols].values.astype(float)[:20]]
for it in range(100):
    iter_dir = RUN_DIR / f"iter_{it}"
    if not (iter_dir / "od.xml").exists():
        break
    tree = ET.parse(str(iter_dir / "od.xml"))
    x_it = pd.DataFrame([el.attrib for el in tree.getroot().iter("tazRelation")])["count"].astype(float).values
    r_it = load_grouped_counts(iter_dir / "sim_loopOutputs.csv") - gt_counts
    X_all.append(x_it[None, :])
    R_all.append(r_it[None, :])
X_full = np.concatenate(X_all, axis=0)
R_full = np.concatenate(R_all, axis=0)
N = min(100, X_full.shape[0])
print(f"Loaded N={N} real training points, L={L} edges, d={d}")

bounds = torch.tensor([[float(config.get("od_bound_start", 1))] * d,
                        [float(config.get("od_bound_end", 1000))] * d], dtype=dtype)
train_X = torch.tensor(X_full[:N], dtype=dtype)
train_R_real = torch.tensor(R_full[:N], dtype=dtype)
X_norm = normalize(train_X, bounds=bounds)


def fit_and_report(train_R, label, seed):
    task_col = torch.arange(L, dtype=dtype).view(1, L, 1).expand(N, L, 1)
    X_exp = X_norm.unsqueeze(1).expand(N, L, d)
    X_stacked = torch.cat([X_exp, task_col], dim=-1).reshape(N * L, d + 1)
    Y_stacked = train_R.reshape(-1, 1)

    kern = SharedPrivateGraphKernel(L=L, input_dim=d, B=B, rho2_init=0.3, dtype=torch.float64)
    lik = PerTaskGaussianLikelihood(num_tasks=L)
    task_ids = X_stacked[:, d].long()
    y = Y_stacked.squeeze(-1)
    task_mean = torch.zeros(L, dtype=y.dtype)
    task_std = torch.ones(L, dtype=y.dtype)
    for t in range(L):
        m = task_ids == t
        task_mean[t] = y[m].mean()
        task_std[t] = y[m].std(unbiased=False).clamp_min(1e-6)
    y_std = (y - task_mean[task_ids]) / task_std[task_ids]
    model = SingleTaskGP(X_stacked, y_std.unsqueeze(-1), covar_module=kern, likelihood=lik, outcome_transform=None)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    torch.manual_seed(seed)
    fit_method = fit_shared_private_mll_multistart(mll, n_restarts=5, verbose=False)

    rho2 = float(kern.rho2.detach())
    priv_os = torch.tensor([float(pk.outputscale.detach()) for pk in kern.private_kernels])
    print(f"[{label}] fit_method={fit_method}  rho2={rho2:.4f}  priv_os_mean={priv_os.mean():.4f}")
    return rho2


print("\n=== REAL (unshuffled) data, real cross-edge correspondence intact ===")
rho2_real = fit_and_report(train_R_real, "REAL", seed=0)

print("\n=== NULL test: each edge's residuals independently permuted across N training points ===")
rng = np.random.default_rng(42)
R_shuffled_np = R_full[:N].copy()
for ell in range(L):
    perm = rng.permutation(N)
    R_shuffled_np[:, ell] = R_shuffled_np[perm, ell]
train_R_shuffled = torch.tensor(R_shuffled_np, dtype=dtype)
rho2_null = fit_and_report(train_R_shuffled, "NULL (shuffled)", seed=1)

print("\n=== Repeat null test with 2 more independent shuffles, to check consistency ===")
rho2_null_list = [rho2_null]
for trial in range(2, 4):
    rng2 = np.random.default_rng(42 + trial)
    R_shuffled_np2 = R_full[:N].copy()
    for ell in range(L):
        perm = rng2.permutation(N)
        R_shuffled_np2[:, ell] = R_shuffled_np2[perm, ell]
    train_R_shuffled2 = torch.tensor(R_shuffled_np2, dtype=dtype)
    rho2_null_list.append(fit_and_report(train_R_shuffled2, f"NULL (shuffled, trial {trial})", seed=trial))

print("\n=== SUMMARY ===")
print(f"rho^2 on REAL data:              {rho2_real:.4f}")
print(f"rho^2 on NULL (shuffled) trials:  {[f'{v:.4f}' for v in rho2_null_list]}")
print(f"rho^2 NULL mean:                 {np.mean(rho2_null_list):.4f}")
