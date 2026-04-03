"""
GP Standardize Diagnostic Script — GROUPED EDGES, FULL RUN
============================================================
Reads all 200 iterations from restart_1_seed-42 of indgp_bo_squared_grouped.
Reconstructs exact train_X from od.xml and train_Y from link_measure_compare.csv.

Q1. Is train_targets standardized (mean~0, std~1)?
Q2. Are lengthscales hitting the upper bound (> 3.9)?  — per dim, all edges
Q3. Is mean relative error at training points above 5%?
"""
import sys, xml.etree.ElementTree as ET
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from botorch.fit import fit_gpytorch_mll

from src.models.gp_models import initialize_independent_gp_models_with_modellist

# ── Config ───────────────────────────────────────────────────────────────────
RUN_DIR = (Path(project_root) /
           "output/quickstart_bo_vanilla_matern-2p5"
           "/indgp_bo_squared_grouped/noise_1e-4_1/NEW"
           "/restart_1_seed-42")

OD_BOUND_LOW  = 1.0
OD_BOUND_HIGH = 2000.0

GROUP_IDS = ["L12", "L14", "L18", "L10", "L15", "L16"]

# OD route order in od.xml
OD_ROUTES = [
    ("taz91", "taz93"),
    ("taz91", "taz94"),
    ("taz92", "taz93"),
    ("taz92", "taz94"),
]

# ── Load all iterations ───────────────────────────────────────────────────────
iter_dirs = sorted(RUN_DIR.glob("iter_*"),
                   key=lambda p: int(p.name.split("_")[1]))
print(f"Found {len(iter_dirs)} iteration folders in {RUN_DIR.name}")

X_rows = []
Y_rows = []

for it_dir in iter_dirs:
    od_path  = it_dir / "od.xml"
    lmc_path = it_dir / "link_measure_compare.csv"
    if not od_path.exists() or not lmc_path.exists():
        print(f"  Skipping {it_dir.name} (missing file)")
        continue

    # --- X: parse od.xml ---
    root = ET.parse(od_path).getroot()
    interval = root.find("interval")
    counts = {}
    for rel in interval.findall("tazRelation"):
        counts[(rel.get("from"), rel.get("to"))] = float(rel.get("count"))
    x_raw = np.array([counts[r] for r in OD_ROUTES])
    x_norm = (x_raw - OD_BOUND_LOW) / (OD_BOUND_HIGH - OD_BOUND_LOW)
    X_rows.append(x_norm)

    # --- Y: read link_measure_compare.csv (already grouped) ---
    df = pd.read_csv(lmc_path)
    df = df.set_index("edge_id")
    y_row = []
    for gid in GROUP_IDS:
        gt  = df.loc[gid, "interval_nVehContrib_gt"]
        sim = df.loc[gid, "interval_nVehContrib_sim"]
        y_row.append(((sim - gt) / gt) ** 2)
    Y_rows.append(y_row)

train_X = torch.tensor(np.array(X_rows), dtype=torch.double)
train_Y = torch.tensor(np.array(Y_rows), dtype=torch.double)
N = train_X.shape[0]

print(f"train_X shape: {train_X.shape}  (N={N}, d=4)")
print(f"train_Y shape: {train_Y.shape}  (N={N}, L={len(GROUP_IDS)})")

# ── Raw Y stats ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("RAW Y STATISTICS — SQUARED METRIC (before GP normalization)")
print("="*70)
print(f"  {'Group':6s} | {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print("  " + "-"*50)
for j, gid in enumerate(GROUP_IDS):
    y = train_Y[:, j].numpy()
    print(f"  {gid:6s} | {y.mean():>8.4f} {y.std():>8.4f} {y.min():>8.4f} {y.max():>8.4f}")

# ── Fit models ────────────────────────────────────────────────────────────────
print(f"\nFitting {len(GROUP_IDS)} independent GPs on {N} points...")
model_list_gp, mlls = initialize_independent_gp_models_with_modellist(
    train_X, train_Y, kernel="matern-2p5"
)
for edge_idx, mll in enumerate(mlls):
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"  [{GROUP_IDS[edge_idx]}] fit failed: {e}")
print("Done.")

# ── Hyperparameter table ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HYPERPARAMETERS AFTER FITTING")
print("="*70)
print(f"  {'Group':6s} | {'tgt_mean':>9} {'tgt_std':>8} | "
      f"{'outputscale':>11} {'noise':>10} | "
      f"{'dim0':>7} {'dim1':>7} {'dim2':>7} {'dim3':>7} | {'@bound?':>7}")
print("  " + "-"*90)

all_results = []
for edge_idx, gp_model in enumerate(model_list_gp.models):
    gid = GROUP_IDS[edge_idx]
    gp_model.eval()

    y_int       = gp_model.train_targets
    tgt_mean    = y_int.mean().item()
    tgt_std     = y_int.std().item()
    outputscale = gp_model.covar_module.outputscale.item()
    noise       = gp_model.likelihood.noise.item()
    ls          = gp_model.covar_module.base_kernel.lengthscale.detach().squeeze().tolist()
    ls_max      = max(ls)
    at_bound    = ls_max > 3.9

    print(f"  {gid:6s} | {tgt_mean:>9.4f} {tgt_std:>8.4f} | "
          f"{outputscale:>11.4f} {noise:>10.6f} | "
          + "  ".join(f"{v:>7.4f}" for v in ls)
          + f" | {'YES ⚠' if at_bound else 'no':>7s}")

    all_results.append(dict(
        edge=gid, tgt_mean=tgt_mean, tgt_std=tgt_std,
        outputscale=outputscale, noise=noise,
        ls=ls, ls_max=ls_max, at_bound=at_bound,
    ))

# ── Q1 ────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("Q1 - Is train_targets standardized (GP sees mean~0, std~1)?")
q1_ok = all(abs(r["tgt_mean"]) < 0.1 and abs(r["tgt_std"] - 1.0) < 0.1
            for r in all_results)
if q1_ok:
    print("  PASS  All groups: |mean|<0.1, |std-1|<0.1. Standardize is working.")
    print("  -> outputscales are in standardized space. Any value >> 1 is pathological.")
else:
    bad = [r["edge"] for r in all_results
           if abs(r["tgt_mean"]) >= 0.1 or abs(r["tgt_std"] - 1.0) >= 0.1]
    print(f"  FAIL  Groups {bad} do NOT have standardized train_targets.")

# ── Q2 ────────────────────────────────────────────────────────────────────────
at_bound_edges = [r["edge"] for r in all_results if r["at_bound"]]
print(f"\nQ2 - Lengthscales at upper bound (>3.9)?")
if at_bound_edges:
    print(f"  YES  Groups: {at_bound_edges}")
    print("  -> Fix: Interval(0.005, 4.0) -> Interval(0.005, 2.0) + GammaPrior(3.0, 6.0)")
else:
    print("  NO  No lengthscales at upper bound.")

# ── Q3 ────────────────────────────────────────────────────────────────────────
print(f"\nQ3 - Mean relative error at training points:")
print(f"  {'Group':6s} | {'raw_Y_var':>10} | {'outputscale':>11} | {'mean_rel_err':>12} | verdict")
print("  " + "-"*65)
any_degen = False
for edge_idx, gp_model in enumerate(model_list_gp.models):
    gp_model.eval()
    gid    = GROUP_IDS[edge_idx]
    raw_y  = train_Y[:, edge_idx]
    with torch.no_grad():
        pred_mean = gp_model.posterior(train_X).mean.squeeze()
    rel_err = ((raw_y - pred_mean).abs() / raw_y.abs().clamp(min=1e-8)).mean().item()
    raw_var = raw_y.var().item()
    osc     = all_results[edge_idx]["outputscale"]
    degen   = rel_err > 0.05
    any_degen = any_degen or degen
    print(f"  {gid:6s} | {raw_var:>10.4f} | {osc:>11.4f} | {rel_err:>12.4f} | "
          f"{'DEGENERATE' if degen else 'ok'}")
print()
if any_degen:
    print("  VERDICT  Some GPs degenerate — kernel collapse or noise-floor collapse.")
else:
    print("  VERDICT  All GPs fit training data within 5% relative error.")
