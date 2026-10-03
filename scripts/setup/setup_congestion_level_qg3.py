"""
Generate GT simulation and 20 initial BO points for a scaled demand level
on the quickstart_grid_3 network.

Usage:
  python3 scripts/setup_congestion_level_qg3.py --demand_scale 0.5
  python3 scripts/setup_congestion_level_qg3.py --demand_scale 1.0
  python3 scripts/setup_congestion_level_qg3.py --demand_scale 2.0

Output:
  output/quickstart_grid_3_congestion_{scale}x/
    ground_truth/sim_loopOutputs.csv       — GT link flows at scaled demand
    initial_search/NRMSE/data_set_ods_20.csv — 20 Sobol init points + residuals
"""
import argparse
import os
import sys
import warnings
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

from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges, group_edge_dataframe

parser = argparse.ArgumentParser()
parser.add_argument("--demand_scale", type=float, required=True)
parser.add_argument("--n_init", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

DEMAND_SCALE = args.demand_scale
N_INIT = args.n_init
SEED = args.seed
SCALE_LABEL = f"{DEMAND_SCALE:g}x"

NETWORK_NAME = "quickstart_grid_3"
config = load_kwargs_config(base_path, "bo_vanilla", sim_setup_filename="sim_setup_quickstart_grid_3.json")
gt_od_df = od_xml_to_df(config["file_gt_od"])
routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids = get_representative_edges(network_name=NETWORK_NAME)
L = len(edge_ids)

OD_BOUND_START = 1.0
OD_BOUND_END = round(float(config.get("od_bound_end", 1000)) * DEMAND_SCALE)
d = int(config.get("dim_od", 16))

OUT_BASE = Path("output") / f"quickstart_grid_3_congestion_{SCALE_LABEL}"
GT_DIR   = OUT_BASE / "ground_truth"
INIT_DIR = OUT_BASE / "initial_search" / "NRMSE"
GT_DIR.mkdir(parents=True, exist_ok=True)
INIT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[SETUP] demand_scale={DEMAND_SCALE}  scale_label={SCALE_LABEL}  od_bound_end={OD_BOUND_END}  L={L}  d={d}", flush=True)


def run_sim(od_values, out_dir, seed_sim, scaled_gt_od_df):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    od_xml = out_dir / "od.xml"
    prefix = out_dir / "sim"
    base_od = scaled_gt_od_df.copy()
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


# ── 1. Scaled GT OD ───────────────────────────────────────────────────────────
scaled_gt_od = gt_od_df.copy()
scaled_gt_od["count"] = (gt_od_df["count"] * DEMAND_SCALE).round(1)
scaled_gt_od_for_sim = scaled_gt_od.rename(
    columns={"fromTaz": "from", "toTaz": "to"}, errors="ignore")
gt_od_counts = scaled_gt_od["count"].values.astype(float)

print(f"[GT OD] Scaled: min={gt_od_counts.min():.0f}  max={gt_od_counts.max():.0f}  "
      f"sum={gt_od_counts.sum():.0f}  bound_end={OD_BOUND_END}", flush=True)

# ── 2. GT simulation at scaled demand ────────────────────────────────────────
gt_sim_dir = GT_DIR / "sim_run"
print("[GT SIM] Running GT simulation...", flush=True)
gt_counts = run_sim(gt_od_counts, gt_sim_dir, seed_sim=999, scaled_gt_od_df=scaled_gt_od_for_sim)

import shutil
loop_src = gt_sim_dir / "sim_loopOutputs.csv"
loop_dst = GT_DIR / "sim_loopOutputs.csv"
shutil.copy(str(loop_src), str(loop_dst))

valid_mask = gt_counts > 0
C_sq = float(valid_mask.sum()) / (gt_counts[valid_mask].sum() ** 2)
mean_flow = gt_counts[valid_mask].mean()
print(f"[GT] L={L}, valid={valid_mask.sum()}, mean_flow={mean_flow:.1f}, "
      f"C_sq={C_sq:.6e}", flush=True)

# ── 3. Generate N_INIT Sobol init points in scaled bounds ─────────────────────
torch.manual_seed(SEED)
sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=SEED)
X_unit = sobol.draw(N_INIT).numpy().astype(float)
X_init = X_unit * (OD_BOUND_END - OD_BOUND_START) + OD_BOUND_START

rows = []
for i, x in enumerate(X_init):
    init_dir = OUT_BASE / "initial_search" / f"point_{i}"
    counts_i = run_sim(x, init_dir, seed_sim=1000 + i, scaled_gt_od_df=scaled_gt_od_for_sim)
    r_i = counts_i - gt_counts
    S_i = float(np.nansum(C_sq * r_i ** 2))
    row = {f"x_{k}": float(x[k]) for k in range(d)}
    row.update({f"r_{eid}": float(r_i[j]) for j, eid in enumerate(edge_ids)})
    row["S"] = S_i
    rows.append(row)
    print(f"  init {i+1}/{N_INIT}: nRMSE={np.sqrt(S_i):.4f}", flush=True)

init_df = pd.DataFrame(rows)
init_csv_path = INIT_DIR / f"data_set_ods_{N_INIT}.csv"
init_df.to_csv(init_csv_path, index=True)
print(f"\n[DONE] Saved {N_INIT} init points → {init_csv_path}")
print(f"       Best init nRMSE: {np.sqrt(init_df['S'].min()):.4f}")
print(f"       Mean flow at GT: {mean_flow:.1f}  (proxy for congestion level)")
