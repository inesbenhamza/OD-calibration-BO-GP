"""
ScalarGP BO loop on quickstart_grid_3_dense using qLogExpectedImprovement.

Self-contained: generates ground truth and 20 Sobol initial points on first
run, then runs the BO loop.  Resume-safe: re-running skips already-done work.

Output root:
  output/quickstart_grid_3_dense_bo_vanilla_matern-2p5/ScalarGP_mcEI/
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
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe
from src.utils.seed import set_seed
from src.models.gp_models import initialize_vanillabo_model, fit_gpytorch_mll_multistart

torch.set_default_dtype(torch.float64)
dtype = torch.double
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_init", type=int, default=None)
parser.add_argument("--max_iters", type=int, default=None)
args = parser.parse_args()
SEED = args.seed

NETWORK_NAME = "quickstart_grid_3_dense"
config = load_kwargs_config(
    base_path, "bo_vanilla",
    sim_setup_filename="sim_setup_quickstart_grid_3_dense.json",
)
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

N_INIT = args.n_init if args.n_init is not None else int(config.get("n_init_search", 20))
N_BO_ITERS = args.max_iters if args.max_iters is not None else int(config.get("BO_niter", 20))
OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 2000))
d = int(config.get("dim_od", 8))

RESTART_IDX = (SEED - 42) // 1000 + 1
VANILLA_BASE = Path("output") / "quickstart_grid_3_dense_bo_vanilla_matern-2p5"
_suffix = "" if N_INIT == 20 else f"_ninit{N_INIT}"
OUT_ROOT = VANILLA_BASE / f"ScalarGP_mcEI{_suffix}"
RESTART_DIR = OUT_ROOT / "results" / f"restart_{RESTART_IDX}_seed-{SEED}"
SIM_BASE = OUT_ROOT / f"restart_{RESTART_IDX}_seed-{SEED}"
RESTART_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

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
    """Simulate a given OD vector, return edge counts array."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    od_xml = out_dir / "od.xml"
    prefix = out_dir / "sim"

    base_od = gt_od_df.copy()
    base_od["count"] = [round(float(v), 1) for v in od_values]
    base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
    create_taz_xml(str(od_xml), base_od, config["od_duration_sec"], str(base_path))

    simulate_od(
        str(od_xml), str(prefix), str(base_path),
        config["net_xml"], config["taz2edge_xml"], config["additional_xml"],
        routes_df, config["sim_end_time"], config["TRIPS2ODS_OUT_STR"],
        seed=seed_sim,
    )

    sim_edge_out = f"{prefix}_{config['EDGE_OUT_STR']}"
    curr_stats, _, _ = parse_loop_data_xml_to_pandas(
        str(base_path), sim_edge_out, str(prefix), str(config["SUMO_PATH"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = group_edge_dataframe(curr_stats, edge_groups=edge_groups)
    grouped = grouped.set_index("edge_id").loc[edge_ids]
    return grouped["interval_nVehContrib"].values.astype(float)


# ── 1. Ground truth ───────────────────────────────────────────────────────────

GT_DIR = VANILLA_BASE / "ground_truth"
GT_CSV = GT_DIR / "sim_loopOutputs.csv"

if GT_CSV.exists():
    print("[GT] Ground truth already exists, loading.", flush=True)
    gt_counts = load_grouped_counts(GT_CSV)
else:
    print("[GT] Running ground truth simulation...", flush=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    gt_od_values = gt_od_df["count"].values
    gt_counts = run_sim(gt_od_values, GT_DIR, seed_sim=0)
    print(f"[GT] Done. gt_counts sum={gt_counts.sum():.0f}", flush=True)

valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
print(f"[GT] L={L}, valid edges={valid_mask.sum()}, C_sq={C_sq:.6e}", flush=True)


# ── 2. Initial Sobol points ───────────────────────────────────────────────────

INIT_DIR = VANILLA_BASE / "initial_search" / "NRMSE"
INIT_CSV = INIT_DIR / f"data_set_ods_{N_INIT}.csv"

bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

if INIT_CSV.exists():
    print(f"[INIT] Loading {N_INIT} initial points from {INIT_CSV}", flush=True)
    init_csv = pd.read_csv(INIT_CSV, index_col=0)
    x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")],
                    key=lambda c: int(c.split("_")[1]))
    X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
    sim_counts_init = np.array([
        load_grouped_counts(INIT_DIR / f"sobol_{i}_rep0_loopOutputs.csv")
        for i in range(N_INIT)
    ])
else:
    print(f"[INIT] Generating {N_INIT} Sobol initial points (seed=42)...", flush=True)
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    sobol_eng = SobolEngine(dimension=d, scramble=True, seed=42)
    X_sobol_norm = sobol_eng.draw(N_INIT).to(dtype=dtype)
    X_init = unnormalize(X_sobol_norm, bounds).numpy()

    sim_counts_init = []
    for i, x in enumerate(X_init):
        sobol_dir = INIT_DIR / f"sobol_{i}"
        sobol_loop_csv = INIT_DIR / f"sobol_{i}_rep0_loopOutputs.csv"
        if sobol_loop_csv.exists():
            counts = load_grouped_counts(sobol_loop_csv)
            print(f"[INIT] sobol_{i}: already done, loading.", flush=True)
        else:
            print(f"[INIT] sobol_{i}: simulating...", flush=True)
            counts = run_sim(x, sobol_dir, seed_sim=42 + i)
            # copy the loopOutputs csv to the expected filename
            import shutil
            src = sobol_dir / "sim_loopOutputs.csv"
            shutil.copy(src, sobol_loop_csv)
        sim_counts_init.append(counts)
    sim_counts_init = np.array(sim_counts_init)

    # save data_set_ods_{N_INIT}.csv with x_ and r_ columns
    rows = {}
    for i in range(N_INIT):
        for j, eid in enumerate(edge_ids):
            rows.setdefault(f"r_{eid}", []).append(sim_counts_init[i, j] - gt_counts[j])
        for j in range(d):
            rows.setdefault(f"x_{j}", []).append(X_init[i, j])
    pd.DataFrame(rows).to_csv(INIT_CSV)
    print(f"[INIT] Saved initial dataset to {INIT_CSV}", flush=True)

M_resid_init = sim_counts_init - gt_counts[None, :]
S_init = (C_sq * M_resid_init ** 2).sum(axis=1)
print(f"[INIT] Loaded {len(X_init)} initial points. best S = {S_init.min():.6f}", flush=True)


# ── 3. ScalarGP BO loop ───────────────────────────────────────────────────────

set_seed(SEED)

train_X = torch.tensor(X_init, dtype=dtype, device=device)
train_S = torch.tensor(S_init, dtype=dtype, device=device)

best_S = float(train_S.min())
S_history = [best_S]
runtime_history = [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# Resume: reload already-completed BO iterations
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
    train_X = torch.cat([train_X,
                         torch.tensor(x_resumed, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    train_S = torch.cat([train_S, torch.tensor([S_resumed], dtype=dtype, device=device)], dim=0)
    best_S = min(best_S, S_resumed)
    S_history.append(best_S)
    runtime_history.append(float("nan"))
    start_iter = it + 1

if start_iter > 0:
    print(f"[RESUME] {start_iter} iterations already done, resuming from {start_iter+1}. "
          f"best_S={best_S:.6f}", flush=True)
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


def negate_objective(samples, X=None):
    return -samples.squeeze(-1)


mc_objective = GenericMCObjective(negate_objective)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    X_norm = normalize(train_X, bounds=bounds)
    S_col = train_S.view(-1, 1)

    model = initialize_vanillabo_model(X_norm, S_col, kernel="matern-2p5",
                                       constraint_mode="constrained")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll_multistart(mll, n_restarts=5)
    model.eval()

    ls_val = float(model.covar_module.base_kernel.lengthscale.detach().mean())
    os_val = float(model.covar_module.outputscale.detach())
    noise_val = float(model.likelihood.noise.detach().mean())
    pd.DataFrame([{
        "iteration": it, "lengthscale": ls_val, "outputscale": os_val,
        "noise": noise_val, "n_train": train_X.shape[0], "best_S": best_S,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    best_f = -float(best_S)
    qEI = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler,
                                   objective=mc_objective)
    x_next_norm, acq_val = optimize_acqf(
        qEI, bounds=bounds_norm, q=1,
        num_restarts=int(config.get("BO_num_restarts", 16)),
        raw_samples=int(config.get("BO_raw_samples", 256)),
    )

    pd.DataFrame([{
        "iteration": it, "acq_val_qLogEI": acq_val.item(), "n_train": train_X.shape[0],
    }]).to_csv(diag_path, mode="a", header=not diag_path.exists(), index=False)

    x_next_real = unnormalize(x_next_norm.detach(), bounds).squeeze(0).cpu().numpy()
    x_next_real = np.clip(x_next_real, OD_BOUND_START, OD_BOUND_END)

    iter_dir = SIM_BASE / f"iter_{it}"
    sim_counts_next = run_sim(x_next_real, iter_dir, seed_sim=SEED + it)

    r_next = sim_counts_next - gt_counts
    S_next = float(np.nansum(C_sq * r_next ** 2))
    best_S = min(best_S, S_next)
    S_history.append(best_S)

    train_X = torch.cat([train_X,
                         torch.tensor(x_next_real, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    train_S = torch.cat([train_S, torch.tensor([S_next], dtype=dtype, device=device)], dim=0)

    dt = time.time() - iter_t0
    runtime_history.append(dt)

    pd.DataFrame({
        "iteration": range(len(S_history)), "best_S": S_history,
        "runtime_sec": runtime_history,
        "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
    }).to_csv(convergence_file, index=False)

    print(f"[ScalarGP-dense seed={SEED}] iter {it+1}/{N_BO_ITERS}  "
          f"S_next={S_next:.6f}  best_S={best_S:.6f}  nRMSE={np.sqrt(best_S):.6f}  "
          f"acqLogEI={acq_val.item():.4f}  ({dt:.1f}s, total {time.time()-t_start:.0f}s)",
          flush=True)

print(f"\nDONE [ScalarGP-dense seed={SEED}]. Final best_S={best_S:.6f}  "
      f"nRMSE={np.sqrt(best_S):.6f}")
print(f"Results: {RESTART_DIR}")
