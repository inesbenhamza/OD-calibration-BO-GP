"""
ScalarGP BO loop on grid_2d_3x3 using qLogExpectedImprovement.

Self-contained: generates ground truth + Sobol initial points on first run,
then runs the BO loop. Resume-safe.

Output:
  output/grid_2d_3x3_bo_vanilla_matern-2p5/ScalarGP_mcEI{_suffix}/
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
from torch.quasirandom import SobolEngine

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
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.models.gp_models import initialize_vanillabo_model, fit_gpytorch_mll_multistart

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
OUT_ROOT = VANILLA_BASE / f"ScalarGP_mcEI{_suffix}"
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


# ── ground truth ──────────────────────────────────────────────────────────────
GT_DIR = VANILLA_BASE / "ground_truth"
GT_CSV = GT_DIR / "sim_loopOutputs.csv"

if GT_CSV.exists():
    print("[GT] Loading existing ground truth.", flush=True)
    gt_counts = load_grouped_counts(GT_CSV)
else:
    print("[GT] Running ground truth simulation...", flush=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    gt_counts = run_sim(gt_od_df["count"].values, GT_DIR, seed_sim=0)

valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
print(f"[GT] L={L}, valid={valid_mask.sum()}, C_sq={C_sq:.6e}", flush=True)

# ── initial Sobol points ──────────────────────────────────────────────────────
INIT_DIR = VANILLA_BASE / "initial_search" / "NRMSE"
INIT_CSV = INIT_DIR / f"data_set_ods_{N_INIT}.csv"
bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

if INIT_CSV.exists():
    print(f"[INIT] Loading {N_INIT} initial points from {INIT_CSV}", flush=True)
    init_csv = pd.read_csv(INIT_CSV, index_col=0)
    x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
    X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
    sim_counts_init = np.array([
        load_grouped_counts(INIT_DIR / f"sobol_{i}_rep0_loopOutputs.csv")
        for i in range(N_INIT)
    ])
else:
    print(f"[INIT] Generating {N_INIT} Sobol points (seed=42)...", flush=True)
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    sobol_eng = SobolEngine(dimension=d, scramble=True, seed=42)
    X_sobol_norm = sobol_eng.draw(N_INIT).to(dtype=dtype)
    X_init = unnormalize(X_sobol_norm, bounds).numpy()
    sim_counts_init = []
    for i, x in enumerate(X_init):
        counts = run_sim(x, INIT_DIR / f"sobol_{i}", seed_sim=100 + i)
        (INIT_DIR / f"sobol_{i}" / "sim_loopOutputs.csv").rename(
            INIT_DIR / f"sobol_{i}_rep0_loopOutputs.csv"
        )
        sim_counts_init.append(counts)
        print(f"  sobol {i+1}/{N_INIT} done", flush=True)
    sim_counts_init = np.array(sim_counts_init)

M_resid_init = sim_counts_init - gt_counts
M_sq_init = C_sq * M_resid_init ** 2

# Save init CSV (with r_ columns for IndGP/GraphGP reuse)
if not INIT_CSV.exists():
    rows = {}
    for j, eid in enumerate(edge_ids):
        rows[f"x_{j}"] = X_init[:, j] if j < d else [0]*N_INIT
    for j, eid in enumerate(edge_ids):
        rows[f"r_{eid}"] = M_resid_init[:, j]
    x_dict = {f"x_{j}": X_init[:, j] for j in range(d)}
    r_dict = {f"r_{eid}": M_resid_init[:, k] for k, eid in enumerate(edge_ids)}
    pd.DataFrame({**x_dict, **r_dict}).to_csv(INIT_CSV)
    print(f"[INIT] Saved to {INIT_CSV}", flush=True)

print(f"[INIT] best S = {M_sq_init.sum(axis=1).min():.6f}", flush=True)

train_X = torch.tensor(X_init, dtype=dtype, device=device)
train_S = torch.tensor(M_sq_init.sum(axis=1), dtype=dtype, device=device).unsqueeze(-1)
best_S = float(train_S.min())
S_history = [best_S]
runtime_history = [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# ── resume ────────────────────────────────────────────────────────────────────
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
    S_resumed = float(np.nansum(C_sq * r_resumed ** 2))
    train_X = torch.cat([train_X, torch.tensor(x_resumed, dtype=dtype).unsqueeze(0)], dim=0)
    train_S = torch.cat([train_S, torch.tensor([[S_resumed]], dtype=dtype)], dim=0)
    best_S = min(best_S, S_resumed)
    S_history.append(best_S)
    runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iters loaded, resuming from {start_iter+1}. best_S={best_S:.6f}", flush=True)
else:
    if hp_log_path.exists(): hp_log_path.unlink()
    if diag_path.exists(): diag_path.unlink()

pd.DataFrame({
    "iteration": range(len(S_history)), "best_S": S_history,
    "runtime_sec": runtime_history,
    "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
}).to_csv(convergence_file, index=False)

# ── BO loop ───────────────────────────────────────────────────────────────────
def scalar_objective(samples, X=None):  # noqa: ARG001
    return samples.squeeze(-1)

mc_objective = GenericMCObjective(scalar_objective)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    X_norm = normalize(train_X, bounds=bounds)

    gp = initialize_vanillabo_model(X_norm, -train_S, kernel="matern-2p5", constraint_mode="constrained")
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll_multistart(mll, n_restarts=N_GP_RESTARTS)
    gp.eval()

    ls = float(gp.covar_module.base_kernel.lengthscale.detach().mean())
    noise = float(gp.likelihood.noise.detach())
    pd.DataFrame([{"iteration": it, "lengthscale": ls, "noise": noise,
                   "n_train": train_X.shape[0], "best_S": best_S}]).to_csv(
        hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    best_f = float((-train_S).max())
    qEI = qLogExpectedImprovement(model=gp, best_f=best_f, sampler=sampler, objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1,
                                          num_restarts=8, raw_samples=128)
    pd.DataFrame([{"iteration": it, "acq_val": acq_val.item(), "n_train": train_X.shape[0]}]).to_csv(
        diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy()
    x_next_real = np.clip(x_next_real, OD_BOUND_START, OD_BOUND_END)

    iter_dir = SIM_BASE / f"iter_{it}"
    sim_counts_next = run_sim(x_next_real, iter_dir, seed_sim=SEED + it)

    r_next = sim_counts_next - gt_counts
    S_next = float(np.nansum(C_sq * r_next ** 2))
    best_S = min(best_S, S_next)
    S_history.append(best_S)

    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype).unsqueeze(0)], dim=0)
    train_S = torch.cat([train_S, torch.tensor([[S_next]], dtype=dtype)], dim=0)

    dt = time.time() - iter_t0
    runtime_history.append(dt)
    pd.DataFrame({
        "iteration": range(len(S_history)), "best_S": S_history,
        "runtime_sec": runtime_history,
        "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
    }).to_csv(convergence_file, index=False)

    print(f"[ScalarGP-grid2d seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"S_next={S_next:.6f}  best_S={best_S:.6f}  nRMSE={np.sqrt(best_S):.6f}  "
          f"acq={acq_val.item():.4f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [ScalarGP-grid2d seed={SEED}]. nRMSE={np.sqrt(best_S):.6f}")
print(f"Results: {RESTART_DIR}")
