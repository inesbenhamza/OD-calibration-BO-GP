"""
Generate initial Sobol samples for BO under different traffic regimes.

Two regimes are supported:
- freeflow:   OD demands sampled from [1, FREE_FLOW_MAX]  per pair → low flows
- congestion: OD demands sampled from [CONGESTION_MIN, 2000] per pair → high flows

Capacity assumption: 1800 veh/h/lane × 3 lanes = 5400 veh/h per edge.
V/C > 0.85 is flagged as congested.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas
from src.utils.edge_flow_grouping import group_edge_dataframe
from src.utils.link_flow_analysis import compute_squared_metric_per_edge
from src.utils.seed import set_seed


# Capacity per edge (veh per simulation period)
# 1800 veh/h/lane × 3 lanes, simulation is 1 hour
CAPACITY_PER_EDGE = 5400.0
CONGESTION_VC_THRESHOLD = 0.85

# Default OD bounds per regime
REGIME_BOUNDS = {
    "freeflow":   (1.0,    500.0),
    "congestion": (1500.0, 2000.0),
}


def generate_initial_points(
    regime: str,
    config: Dict,
    gt_od_df: pd.DataFrame,
    gt_edge_grouped: pd.DataFrame,
    edge_ids: List[str],
    weights_np: np.ndarray,
    routes_df: pd.DataFrame,
    base_path: Path,
    dim_od: int,
    device: torch.device,
    dtype: torch.dtype,
    n_init: int = 10,
    n_rep: int = 1,
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Generate initial Sobol points for a given traffic regime, simulate each,
    and return a DataFrame of OD vectors, per-edge errors, and aggregated loss.

    Saves results to:
        {config['simulation_run_path']}/initial_search/{regime}/data_set_ods_{regime}.csv

    If the CSV already exists and overwrite=False, loads and returns it directly.

    Parameters
    ----------
    regime : str
        Either 'freeflow' or 'congestion'. Controls the OD sampling bounds.
    config : dict
        Simulation configuration (from load_kwargs_config).
    gt_od_df : pd.DataFrame
        Ground truth OD DataFrame with fromTaz/toTaz/count columns.
    gt_edge_grouped : pd.DataFrame
        Grouped ground truth edge flows (edge_id, interval_nVehContrib).
    edge_ids : list of str
        Representative edge IDs to evaluate.
    weights_np : np.ndarray
        Aggregation weights per edge, shape [L].
    routes_df : pd.DataFrame
        Routes DataFrame required by simulate_od.
    base_path : Path
        Project root path.
    dim_od : int
        Number of OD pairs (input dimension).
    device : torch.device
    dtype : torch.dtype
    n_init : int
        Number of initial Sobol points to generate.
    n_rep : int
        Number of SUMO replications per point (results are averaged).
    seed : int, optional
        Sobol seed. Defaults to 42 for freeflow, 123 for congestion.
    overwrite : bool
        If True, re-simulate even if CSV already exists.

    Returns
    -------
    pd.DataFrame
        Columns: x_1..x_d, loss, regime, e_<edge_id> for each edge.
    """
    if regime not in REGIME_BOUNDS:
        raise ValueError(f"Unknown regime '{regime}'. Choose from {list(REGIME_BOUNDS.keys())}.")

    csv_path = Path(config["simulation_run_path"]) / "initial_search" / regime / f"data_set_ods_{regime}.csv"

    if csv_path.exists() and not overwrite:
        print(f"[{regime}] Loading existing data from {csv_path}")
        return pd.read_csv(csv_path)

    # Build regime-specific bounds
    lo, hi = REGIME_BOUNDS[regime]
    regime_bounds = torch.tensor(
        [[lo] * dim_od, [hi] * dim_od], dtype=dtype, device=device
    )

    # Same seed for all regimes so the only difference is the OD range
    if seed is None:
        seed = 42

    print(f"REGIME: {regime}  |  OD range: [{lo}, {hi}]  |  seed: {seed}")


    # Draw Sobol points in [0,1]^d then map to regime bounds
    sobol = SobolEngine(dim_od, scramble=True, seed=seed)
    x_norm = sobol.draw(n_init).to(dtype=dtype).to(device)
    train_X0 = unnormalize(x_norm, regime_bounds)

    simulation_run_path_init = Path(config["simulation_run_path"]) / "initial_search" / regime
    simulation_run_path_init.mkdir(parents=True, exist_ok=True)

    batch_rows = []

    for i, x in enumerate(train_X0.tolist()):
        print(f"\n[{regime}] Point {i+1}/{n_init}")
        curr_od = np.array(x, dtype=float)
        print(f"  OD: {curr_od}  |  total demand: {np.sum(curr_od):.1f}")

        new_od_xml = str(simulation_run_path_init / f"od_{regime}_{i}.xml")
        prefix_output_base = str(simulation_run_path_init / f"sobol_{i}")

        base_od = gt_od_df.copy()
        base_od["count"] = [round(v, 1) for v in curr_od]
        base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})
        create_taz_xml(new_od_xml, base_od, config["od_duration_sec"], base_path)

        per_edge_errors_reps = []

        for rep in range(n_rep):
            prefix_output_rep = f"{prefix_output_base}_rep{rep}"

            simulate_od(
                new_od_xml,
                prefix_output_rep,
                base_path,
                config["net_xml"],
                config["taz2edge_xml"],
                config["additional_xml"],
                routes_df,
                config["sim_end_time"],
                config["TRIPS2ODS_OUT_STR"],
                sim_start_time=0,
                seed=rep,
            )

            sim_edge_out_rep = f"{prefix_output_rep}_{config['EDGE_OUT_STR']}"
            curr_loop_stats_rep, _, _ = parse_loop_data_xml_to_pandas(
                base_path, sim_edge_out_rep, prefix_output_rep, config["SUMO_PATH"],
            )

            curr_loop_stats_rep_grouped = group_edge_dataframe(curr_loop_stats_rep)
            sim_counts = curr_loop_stats_rep_grouped.set_index("edge_id").loc[
                edge_ids, "interval_nVehContrib"
            ].values

            print(f"  Simulated flows (rep {rep}):")
            for eid, cnt in zip(edge_ids, sim_counts):
                vc = cnt / CAPACITY_PER_EDGE
                flag = "CONGESTED" if vc > CONGESTION_VC_THRESHOLD else "free flow"
                print(f"    {eid}: {cnt:.0f} veh  (V/C={vc:.2f}  {flag})")

            per_edge_errors_rep = compute_squared_metric_per_edge(
                df_true=gt_edge_grouped,
                df_simulated=curr_loop_stats_rep_grouped,
                edge_ids=edge_ids,
            )
            per_edge_errors_reps.append(per_edge_errors_rep)

        per_edge_errors = np.stack(per_edge_errors_reps, axis=0).mean(axis=0)
        S_x = float((per_edge_errors * weights_np).sum())
        print(f"  Aggregated loss S(x): {S_x:.6f}")

        row = {f"x_{k+1}": curr_od[k] for k in range(dim_od)}
        row["loss"] = S_x
        row["regime"] = regime
        for edge_idx, eid in enumerate(edge_ids):
            row[f"e_{eid}"] = per_edge_errors[edge_idx]
        batch_rows.append(row)

    df = pd.DataFrame(batch_rows)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved {regime} initial data to {csv_path}")
    return df
