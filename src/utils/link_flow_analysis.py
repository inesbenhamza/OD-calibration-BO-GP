# Standard library imports
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


def _ensure_edge_id(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Ensure the dataframe has an 'edge_id' column. If common alternatives exist,
    rename them; otherwise raise a clear error.
    """
    if "edge_id" in df.columns:
        return df
    # Common alternatives in SUMO outputs: link_id / linkID / edgeID, etc.
    for alt in ["edgeID", "edgeid", "edge", "link_id", "linkID", "linkid", "link"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "edge_id"})
            return df
    raise ValueError(
        f"[{context}] expected column 'edge_id' but found {list(df.columns)}"
    )

def parse_link_flow_xml_to_pandas(
    base_dir: Path,
    sim_link_file: Path,
    prefix_output: str,
    sensor_start_time: float,
    sensor_end_time: float,
    link_list: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:

    output_file = Path(base_dir) / f"{prefix_output}_link_measurements.csv"

    root = ET.parse(sim_link_file).getroot()
    data = []

    for interval in root.findall("interval"):
        interval_id = interval.get("id")

        interval_begin_str = interval.get("begin")
        interval_end_str = interval.get("end")
        interval_begin = float(interval_begin_str) if interval_begin_str is not None else 0.0
        interval_end = float(interval_end_str) if interval_end_str is not None else 0.0

        for link in interval.findall("edge"):
            link_id = link.get("id")
            speed_str = link.get("speed")
            arrived_str = link.get("arrived")
            left_str = link.get("left")

            link_data = {
                "interval_begin": interval_begin,
                "interval_end": interval_end,
                "interval_id": interval_id,
                "link_id": link_id,
                "link_speed": float(speed_str)*2.23694 if speed_str is not None else 0.0,
                "link_arrived": float(arrived_str) if arrived_str is not None else 0.0,
                "link_left": float(left_str) if left_str is not None else 0.0,
            }
            data.append(link_data)

    df_trips = pd.DataFrame(data)[
        [
            "interval_begin",
            "interval_end",
            "link_id",
            "link_speed",
            "link_arrived",
            "link_left",
        ]
    ]

    df_trips.to_csv(output_file, index=False)

    df_trips["interval_nVehContrib"] = df_trips["link_arrived"] + df_trips["link_left"]

    df_trips["interval_harmonicMeanSpeed"] = df_trips.loc[df_trips["interval_nVehContrib"] > 0, "link_speed"]

    df_trips = df_trips[
        (df_trips["interval_begin"] >= sensor_start_time) & (df_trips["interval_end"] <= sensor_end_time)
    ]

    df_agg = df_trips.groupby("link_id", as_index=False).agg(
        {"interval_nVehContrib": "sum", "interval_harmonicMeanSpeed": "mean"}
    )

    if link_list is not None:
        print(f"Filtering links to: {link_list}")
        df_agg = df_agg[df_agg["link_id"].isin(link_list)]

    return df_agg, df_trips, output_file



# METRICS

##########################
################### NRMSE with squared error 
##########################

def precompute_error_constant(df_true, edge_ids):
    """
    Computes C_sq = |S| / (sum_l gt_l)^2 once before the BO loop.
    Used to compute per-link nRMSE^2 contributions f_l = C_sq * (sim_l - gt_l)^2.

    L and total_gt are computed over valid edges only (gt > 0), so that
    sum_l f_l = nRMSE^2 exactly.
    """
    
    import warnings

    df_true = _ensure_edge_id(df_true, "precompute_error_constant")
    true_lookup = df_true.set_index("edge_id")["interval_nVehContrib"].to_dict()

    gt_values = np.array([true_lookup.get(e, 0.0) for e in edge_ids], dtype=float)
    valid_mask = gt_values > 0

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        invalid_ids = [e for e, v in zip(edge_ids, valid_mask) if not v]
        warnings.warn(
            f"precompute_error_constant: {n_invalid} edge(s) have zero or "
            f"missing GT counts and will be excluded from L and total_gt: {invalid_ids}",
            UserWarning,
            stacklevel=2,
        )

    L = int(valid_mask.sum())
    total_gt = gt_values[valid_mask].sum()

    if L == 0 or total_gt == 0:
        raise ValueError("No valid GT edges (gt > 0) found — check your edge_ids / df_true.")

    return L / (total_gt ** 2)  # C_sq = |S| / (sum_l gt_l)^2


def compute_error_per_edge(df_true, df_simulated, edge_ids, C):
    """
    Per-edge nRMSE^2 contributions:
        f_l = C_sq * (sim_l - gt_l)^2  >= 0
    where C_sq = |S| / (sum_l gt_l)^2 is precomputed and fixed.

    Properties:
        f_l >= 0               <- non-negative
        sum_l f_l = nRMSE^2   <- BO objective recovered exactly
        sqrt(sum_l f_l) = nRMSE <- interpretable calibration metric
    """
    df_true      = _ensure_edge_id(df_true,      "compute_error_per_edge/df_true")
    df_simulated = _ensure_edge_id(df_simulated, "compute_error_per_edge/df_simulated")

    true_lookup = df_true.set_index("edge_id")["interval_nVehContrib"].to_dict()
    sim_lookup  = df_simulated.set_index("edge_id")["interval_nVehContrib"].to_dict()

    f_l = np.empty(len(edge_ids))
    for i, edge_id in enumerate(edge_ids):
        true_count = true_lookup.get(edge_id, np.nan)
        sim_count  = sim_lookup.get(edge_id, 0.0)

        if np.isnan(true_count) or true_count <= 0:
            f_l[i] = np.nan
        else:
            f_l[i] = C * (sim_count - true_count) ** 2

    return f_l




#for scalar gp

def compute_nrmse_all_edge(df_true, df_simulated, edge_ids=None):
    """
    nRMSE = sqrt(sum_l f_l) where f_l = C * (sim_l - gt_l)^2, C = L / (sum_l gt_l)^2.
    """
    df_true      = _ensure_edge_id(df_true,      "compute_nrmse_all_edge/df_true")
    df_simulated = _ensure_edge_id(df_simulated, "compute_nrmse_all_edge/df_simulated")

    df = df_true.merge(df_simulated, on="edge_id", suffixes=("_GT", "_sim"), how="left")

    if edge_ids is not None:
        df = df[df["edge_id"].isin(edge_ids)]

    df["interval_nVehContrib_sim"] = df["interval_nVehContrib_sim"].fillna(0.0)
    df = df[df["interval_nVehContrib_GT"] > 0]

    if len(df) == 0:
        return float("nan")

    gt  = df["interval_nVehContrib_GT"].astype(float).to_numpy()
    sim = df["interval_nVehContrib_sim"].astype(float).to_numpy()

    C = len(gt) / (gt.sum() ** 2)
    nrmse_sq = float((C * (sim - gt) ** 2).sum())

    return float(np.sqrt(nrmse_sq))








##########################
################### NRMSE with signed residuals
##########################

def compute_signed_residual_per_edge(df_true, df_simulated, edge_ids):
    """
    Per-edge SIGNED residual r_l = sim_l - gt_l (no squaring, no C scaling).

    This is the GP training target for the signed-residual + MC-acquisition
    BO loop: S(x) = C * sum_l r_l(x)^2 is a nonlinear (quadratic) transform of
    jointly-Gaussian GP outputs, so it must be reconstructed by squaring and
    scaling AFTER sampling from the posterior (analytic EI does not apply) to 
    get back to the nrmse^2 --
    see run_signed_residual_mc_ei_bo.py.

    Properties:
        r_l can be negative or positive  <- unlike f_l = C*(sim_l-gt_l)^2 >= 0
        C * r_l^2 = f_l                  <- recovers compute_error_per_edge exactly
    """
    df_true      = _ensure_edge_id(df_true,      "compute_signed_residual_per_edge/df_true")
    df_simulated = _ensure_edge_id(df_simulated, "compute_signed_residual_per_edge/df_simulated")

    true_lookup = df_true.set_index("edge_id")["interval_nVehContrib"].to_dict() # to_dict() for fast lookup, create a dict with edge_id as key and interval_nVehContrib as value
    sim_lookup  = df_simulated.set_index("edge_id")["interval_nVehContrib"].to_dict()

    r_l = np.empty(len(edge_ids))
    for i, edge_id in enumerate(edge_ids):
        true_count = true_lookup.get(edge_id, np.nan)
        sim_count  = sim_lookup.get(edge_id, 0.0)

        if np.isnan(true_count) or true_count <= 0:
            r_l[i] = np.nan
        else:
            r_l[i] = sim_count - true_count

    return r_l