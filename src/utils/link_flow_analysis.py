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
#for scalar gp


def compute_squared_metric_all_edge(df_true, df_simulated, edge_ids=None):
    """
    Aggregated squared error across all edges used for scalar gp (Vanilla BO)
    it computes the relative squared error per edge, then averages them using uniform weights. (1/L)*sum_l (s_l^sim - s_l^GT)^2
    Gp is fitted on this scalar metric.
  


    Returning the mean so we assume the edges are equally weighted ( uniform weights)
    """
    df_true = _ensure_edge_id(df_true, "compute_squared_metric_all_edge/df_true")
    df_simulated = _ensure_edge_id(df_simulated, "compute_squared_metric_all_edge/df_simulated")

    df = df_true.merge(
        df_simulated,
        on="edge_id",
        suffixes=("_GT", "_sim"),
        how="left",
    )
    
    if edge_ids is not None:
        df = df[df["edge_id"].isin(edge_ids)]

    df["interval_nVehContrib_sim"] = df["interval_nVehContrib_sim"].fillna(0.0)
    
    df = df[df["interval_nVehContrib_GT"] > 0] #filter out edges with zero GT flow
    
    if len(df) == 0:
        return float('nan')

    true_counts = df["interval_nVehContrib_GT"].astype(float).to_numpy()
    sim_counts  = df["interval_nVehContrib_sim"].astype(float).to_numpy()

    rel_err_sq = ((sim_counts - true_counts) / true_counts) ** 2 # computes the relative squared error for all edges at once (element-wise)

    return float(rel_err_sq.mean()) # return the mean of the squared relative errors for all edges e.g equivalent to dividing by the number of edges/uniform weights





#for ind gp 

def compute_squared_metric_per_edge(df_true, df_simulated, edge_ids):
    """
    Per-edge squared relative error (IND-GP).
    for ind gp, each gp is fitted on the squared relative error of a single edge.
    """
    df_true = _ensure_edge_id(df_true, "compute_squared_metric_per_edge/df_true")
    df_simulated = _ensure_edge_id(df_simulated, "compute_squared_metric_per_edge/df_simulated")
    errors = []

    for edge_id in edge_ids:
        true_row = df_true[df_true["edge_id"] == edge_id]
        if len(true_row) == 0:
            errors.append(np.nan)
            continue

        true_count = float(true_row["interval_nVehContrib"].values[0])

        sim_row = df_simulated[df_simulated["edge_id"] == edge_id]
        sim_count = (
            float(sim_row["interval_nVehContrib"].values[0])
            if len(sim_row) > 0
            else 0.0
        )

        if true_count > 0:
            errors.append(((sim_count - true_count) / true_count) ** 2)
        else:
            errors.append(np.nan)

    return np.array(errors) # return an array of squared relative errors for each edge








#to evaluate the aggregated discrepancy e_agg(x)
def compute_eagg_from_sim(gt_edge_data, sim_edge_data, edge_ids):
    gt_edge_data = _ensure_edge_id(gt_edge_data, "compute_eagg_from_sim/gt_edge_data")
    sim_edge_data = _ensure_edge_id(sim_edge_data, "compute_eagg_from_sim/sim_edge_data")
    diffs = []
    for edge_id in edge_ids:
        gt_row = gt_edge_data[gt_edge_data["edge_id"] == edge_id]
        if len(gt_row) == 0:
            continue

        s_gt = float(gt_row["interval_nVehContrib"].values[0])

        sim_row = sim_edge_data[sim_edge_data["edge_id"] == edge_id]
        s_sim = float(sim_row["interval_nVehContrib"].values[0]) if len(sim_row) > 0 else 0.0

        if s_gt > 0:
            diffs.append(((s_sim - s_gt) / s_gt) ** 2)

    return float(np.mean(diffs))