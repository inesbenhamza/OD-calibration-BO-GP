"""
ScalarGP BO loop using qLogExpectedImprovement (MC-based), starting from the
SAME initial points used by every other method in this project, for a
fair, acquisition-matched comparison against the graph/IndGP models.

ScalarGP's target is already the scalar S(x) = sum_l C_sq*r_l(x)^2 (no
per-task structure -- single-output GP over the OD input directly, no signed
residual distinction needed since there's no per-edge structure to apply it
to).

Parameterized over --seed, --n_init, --max_iters. Output path:
  output/quickstart_grid_3_graph/ScalarGP_mcEI{_ninit5 if n_init!=20}/
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
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--max_iters", type=int, default=100)
args = parser.parse_args()
SEED = args.seed
N_INIT = args.n_init

NETWORK_NAME = "quickstart_grid_3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

N_BO_ITERS = args.max_iters
OD_BOUND_START = float(config.get("od_bound_start", 1))
OD_BOUND_END = float(config.get("od_bound_end", 1000))

RESTART_IDX = (SEED - 42) // 1000 + 1  # matches this project's restart_seed = 42 + r*1000 convention
_suffix = "" if N_INIT == 20 else f"_ninit{N_INIT}"
OUT_ROOT = Path("output") / "quickstart_grid_3_graph" / f"ScalarGP_mcEI{_suffix}"
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

init_csv = pd.read_csv(VANILLA_BASE / "initial_search" / "NRMSE" / "data_set_ods_20.csv")
x_cols = sorted([c for c in init_csv.columns if c.startswith("x_")], key=lambda c: int(c.split("_")[1]))
r_cols = [f"r_{eid}" for eid in edge_ids]
X_init = init_csv[x_cols].values.astype(float)[:N_INIT]
M_resid_init = init_csv[r_cols].values.astype(float)[:N_INIT]
S_init = (C_sq * M_resid_init ** 2).sum(axis=1)
print(f"[INIT] ScalarGP seed={SEED}: loaded {len(X_init)} initial points (same cached points as "
      f"IndGP/GraphGP/SharedPrivate ninit5 runs). best S so far = {S_init.min():.6f}", flush=True)

d = X_init.shape[1]
bounds = torch.tensor([[OD_BOUND_START] * d, [OD_BOUND_END] * d], dtype=dtype, device=device)
bounds_norm = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])

train_X = torch.tensor(X_init, dtype=dtype, device=device)
train_S = torch.tensor(S_init, dtype=dtype, device=device)

best_S = float(train_S.min())
S_history = [best_S]
runtime_history = [0.0]
convergence_file = RESTART_DIR / "convergence.csv"
hp_log_path = RESTART_DIR / "hyperparameters_per_iteration.csv"
diag_path = RESTART_DIR / "gp_diagnostics.csv"

# Resume support (same convention as the graph/kronecker/indgp scripts).
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
    train_S = torch.cat([train_S, torch.tensor([S_resumed], dtype=dtype, device=device)], dim=0)
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


def negate_objective(samples, X=None):
    return -samples.squeeze(-1)


mc_objective = GenericMCObjective(negate_objective)

t_start = time.time()
for it in range(start_iter, N_BO_ITERS):
    iter_t0 = time.time()
    X_norm = normalize(train_X, bounds=bounds)
    S_col = train_S.view(-1, 1)

    model = initialize_vanillabo_model(X_norm, S_col, kernel="matern-2p5", constraint_mode="constrained")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll_multistart(mll, n_restarts=5)
    model.eval()

    ls_val = float(model.covar_module.base_kernel.lengthscale.detach().mean())
    os_val = float(model.covar_module.outputscale.detach())
    noise_val = float(model.likelihood.noise.detach().mean())
    pd.DataFrame([{
        "iteration": it, "lengthscale": ls_val, "outputscale": os_val, "noise": noise_val,
        "n_train": train_X.shape[0], "best_S": best_S,
    }]).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    best_f = -float(best_S)
    qEI = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, objective=mc_objective)

    x_next_norm, acq_val = optimize_acqf(qEI, bounds=bounds_norm, q=1, num_restarts=8, raw_samples=128)

    pd.DataFrame([{
        "iteration": it, "acq_val_qLogEI": acq_val.item(), "n_train": train_X.shape[0],
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
    S_next = float(np.nansum(C_sq * r_next ** 2))
    best_S = min(best_S, S_next)
    S_history.append(best_S)

    train_X = torch.cat([train_X, torch.tensor(x_next_real, dtype=dtype, device=device).unsqueeze(0)], dim=0)
    train_S = torch.cat([train_S, torch.tensor([S_next], dtype=dtype, device=device)], dim=0)

    dt = time.time() - iter_t0
    runtime_history.append(dt)

    pd.DataFrame({
        "iteration": range(len(S_history)), "best_S": S_history,
        "runtime_sec": runtime_history,
        "total_runtime_sec": pd.Series(runtime_history).fillna(0.0).cumsum(),
    }).to_csv(convergence_file, index=False)

    print(f"[ScalarGP seed={SEED}] [iter {it+1}/{N_BO_ITERS}] S_next={S_next:.6f} best_S={best_S:.6f} "
          f"nRMSE={np.sqrt(best_S):.6f} acqLogEI={acq_val.item():.4f} "
          f"({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)

print(f"\nDONE [ScalarGP seed={SEED}]. Final best_S={best_S:.6f}  nRMSE={np.sqrt(best_S):.6f}")
print(f"Saved to {RESTART_DIR}")
