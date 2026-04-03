"""
Build and save the graph adjacency matrix from real SUMO simulations.

Samples N diverse OD vectors with Sobol quasi-random sampling over [lb, ub]^d,
runs SUMO for each, parses the vehicle routes output (vehroutes XML), and
counts how many vehicles actually traversed each edge→edge pair.

The adjacency is joint/flow-based:
    W[i,j] = 0.5 * (total_count(i→j) + total_count(j→i))   symmetrized
    A[i,j] = W[i,j] / W.max()                                normalized to [0,1]

Usage
-----
    cd /path/to/Sumo_od_calibration_bn
    python scripts/build_graph_adjacency.py
    python scripts/build_graph_adjacency.py --n-samples 32 --output output/graph_adjacency.npz

Load in graph.py
----------------
    data     = np.load("output/graph_adjacency.npz", allow_pickle=True)
    A        = data["adjacency"]       # shape (6, 6), float64, values in [0,1]
    edge_ids = list(data["edge_ids"])
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Add SUMO bin to PATH so od2trips/sumo are found by subprocess
# ---------------------------------------------------------------------------
_cfg = json.loads((PROJECT_ROOT / "config" / "config.json").read_text())
# SUMO config value is .../EclipseSUMO/share/sumo  →  go up 2 levels to reach EclipseSUMO/bin
_SUMO_BIN = Path(_cfg["SUMO"]).parent.parent / "bin"
if _SUMO_BIN.exists():
    os.environ["PATH"] = str(_SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")

from src.simulation.sumo_runner import run_simulation_for_od
from src.utils.edge_transition_matrix import (
    QUICKSTART_EDGE_GROUPS,
    count_transitions_from_sequences,
    parse_rou_xml_sequences,
    symmetrize_transition_matrix,
)


# ---------------------------------------------------------------------------
# Sobol sampling
# ---------------------------------------------------------------------------

def sobol_od_samples(n: int, n_dims: int, lb: float, ub: float, seed: int) -> np.ndarray:
    """
    Return (n, n_dims) Sobol quasi-random samples in [lb, ub].
    n is rounded up to the next power of 2 for Sobol balance.
    """
    from scipy.stats.qmc import Sobol
    m = int(np.ceil(np.log2(max(n, 2))))   # next power of 2 exponent
    engine = Sobol(d=n_dims, scramble=True, seed=seed)
    unit = engine.random_base2(m)           # shape (2^m, n_dims), values in [0,1)
    samples = lb + unit * (ub - lb)
    return samples[:n]                      # trim to requested n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_configs(base_path: Path) -> tuple[dict, dict]:
    cfg = json.loads((base_path / "config" / "config.json").read_text())
    sim = json.loads((base_path / "config" / "sim_setup.json").read_text())
    return cfg, sim


def find_vehroutes(base_path: Path, prefix_output: str) -> Path | None:
    """SUMO writes vehroutes to {base_path}/{prefix_output}_routes.vehroutes.xml."""
    candidate = Path(base_path) / f"{prefix_output}_routes.vehroutes.xml"
    if candidate.exists():
        return candidate
    # fallback: search inside the sim dir
    sim_dir = Path(base_path) / Path(prefix_output).parent
    hits = list(sim_dir.glob("*routes.vehroutes.xml"))
    return hits[0] if hits else None


def _print_matrix(M: np.ndarray, labels: list[str], fmt: str = ".3f") -> None:
    w = max(len(r) for r in labels) + 2
    print(" " * (w + 2) + "  ".join(f"{r:>{w}}" for r in labels))
    for i, r in enumerate(labels):
        row = "  ".join(f"{M[i, j]:{w}{fmt}}" for j in range(len(labels)))
        print(f"  {r:>{w}}  {row}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graph adjacency from SUMO simulations with Sobol OD samples."
    )
    parser.add_argument(
        "--n-samples", type=int, default=16,
        help="Number of Sobol OD samples to simulate (default: 16, rounded up to power of 2).",
    )
    parser.add_argument(
        "--output", type=str, default="output/graph_adjacency.npz",
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--work-dir", type=str, default="output/adj_sims",
        help="Temporary directory for SUMO outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sobol scramble seed.")
    parser.add_argument(
        "--keep-sims", action="store_true",
        help="Keep intermediate simulation files after the script finishes.",
    )
    args = parser.parse_args()

    base_path = Path(PROJECT_ROOT)
    _, sim_setup = load_configs(base_path)

    lb       = float(sim_setup.get("od_bound_start", 1))
    ub       = float(sim_setup.get("od_bound_end", 2000))
    n_od     = int(sim_setup.get("dim_od", 4))
    net_name = sim_setup.get("network_name", "quickstart")
    routes_file = sim_setup.get("routes_file", "routes.csv")

    net_dir    = base_path / "network" / net_name
    routes_df  = pd.read_csv(net_dir / routes_file)
    base_od_df = routes_df[["fromTaz", "toTaz"]].copy()

    config = {
        "od_duration_sec"  : sim_setup["od_duration_sec"],
        "net_xml"          : net_dir / "net.xml",
        "taz2edge_xml"     : net_dir / "taz.xml",
        "additional_xml"   : net_dir / "additional.xml",
        "sim_end_time"     : sim_setup["sim_end_time"],
        "TRIPS2ODS_OUT_STR": "trips.xml",
        "EDGE_OUT_STR"     : f"edge_data_{net_name}.xml",
        "SUMO_PATH"        : _cfg["SUMO"],
    }

    # Sample diverse OD vectors with Sobol
    od_samples = sobol_od_samples(args.n_samples, n_od, lb, ub, args.seed)
    n_actual   = len(od_samples)
    print(f"Network      : {net_name}")
    print(f"OD bounds    : [{lb:.0f}, {ub:.0f}]^{n_od}")
    print(f"SUMO runs    : {n_actual}  (Sobol, seed={args.seed})")
    print(f"OD pairs     : {list(routes_df['fromTaz'] + '→' + routes_df['toTaz'])}")

    # Run SUMO for each OD sample and aggregate transition counts
    total_counts: np.ndarray | None = None
    rep_ids: list[str] | None = None
    n_successful = 0

    work_dir = Path(args.work_dir)

    for i, od_vec in enumerate(od_samples):
        sim_dir = work_dir / f"run_{i:04d}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Run {i + 1}/{n_actual}  OD={np.round(od_vec, 0).astype(int).tolist()} ---")

        try:
            prefix_output = run_simulation_for_od(
                od_vector=od_vec,
                base_od_df=base_od_df.copy(),
                sim_dir=sim_dir,
                config=config,
                base_path=base_path,
                routes_df=routes_df,
                seed=i,
            )
        except Exception as e:
            print(f"  SUMO failed: {e}")
            continue

        vehroutes = find_vehroutes(base_path, prefix_output)
        if vehroutes is None:
            print(f"  vehroutes file not found, skipping.")
            continue

        print(f"  Parsing: {vehroutes.name}")
        seqs = parse_rou_xml_sequences(vehroutes, ignore_internal=True)
        print(f"  Vehicles with routes: {len(seqs)}")

        eids, counts, _ = count_transitions_from_sequences(
            seqs, edge_groups=QUICKSTART_EDGE_GROUPS
        )

        if total_counts is None:
            rep_ids      = eids
            total_counts = counts.astype(np.float64)
        else:
            total_counts += counts

        n_successful += 1

    if total_counts is None or rep_ids is None:
        print("\nERROR: No successful simulations. Check SUMO installation and PATH.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Successful runs: {n_successful}/{n_actual}")

    # Joint/flow-based adjacency
    W         = symmetrize_transition_matrix(total_counts, how="mean")  # 0.5*(C + C.T)
    w_max     = W.max()
    adjacency = W / w_max if w_max > 0 else W

    L = len(rep_ids)

    print(f"\nAggregate raw counts (summed over {n_successful} runs):")
    _print_matrix(total_counts, rep_ids, fmt=".0f")

    print(f"\nAdjacency  (symmetrized counts, normalized to [0,1]):")
    _print_matrix(adjacency, rep_ids, fmt=".3f")

    # Also show conditional probabilities P(j|i) for comparison
    print(f"\nConditional P(j|i)  [for reference — NOT used as adjacency]:")
    P_ref = np.zeros_like(total_counts)
    for k in range(L):
        s = total_counts[k].sum()
        if s > 0:
            P_ref[k] = total_counts[k] / s
    _print_matrix(P_ref, rep_ids, fmt=".3f")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        adjacency=adjacency,
        counts=total_counts,
        edge_ids=np.array(rep_ids),
        n_runs=np.array(n_successful),
        od_lb=np.array(lb),
        od_ub=np.array(ub),
    )
    print(f"\nSaved → {out_path.resolve()}")
    print(f"\nTo load in graph.py:")
    print(f'  data     = np.load("{out_path}", allow_pickle=True)')
    print(f'  A        = data["adjacency"]        # shape ({L}, {L}), float64')
    print(f'  edge_ids = list(data["edge_ids"])')

    if not args.keep_sims:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Cleaned up {work_dir}  (use --keep-sims to keep simulation files)")


if __name__ == "__main__":
    main()
