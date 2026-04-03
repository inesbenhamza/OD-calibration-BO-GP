"""
Group serial duplicate edges into unique flow variables for SUMO calibration.
Edges in the same group (e.g. D2, L2, L12) always have identical flows; we aggregate by mean.
Also provides the scalar grouped loss (mean relative squared error over groups) used by BO.
"""

import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List

import numpy as np
import pandas as pd

from src.utils.link_flow_analysis import compute_squared_metric_all_edge

EDGE_GROUPS = {
    "L12": ["D2", "L2", "L12"],
    "L14": ["D4", "L4", "L14"],
    "L18": ["D5", "L5", "L18"],
    "L10": ["D7", "L7", "L10"],
    "L15": ["L15"],
    "L16": ["L16"],
}

__all__ = [
    "EDGE_GROUPS",
    "get_edge_groups",
    "get_representative_edges",
    "group_edge_flows",
    "group_edge_dataframe",
    "compute_grouped_loss",
]


def get_edge_groups(
    network_name: Optional[str] = "quickstart",
    net_xml: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Return edge groups for the given network.
    - quickstart: use hard-coded EDGE_GROUPS (serial duplicates merged).
    - other networks: if net_xml is given, one group per edge (all edges from net).
    - otherwise: fall back to EDGE_GROUPS (backward compatible).
    """
    if network_name == "quickstart":
        return dict(EDGE_GROUPS)
    if net_xml:
        try:
            import sumolib
            net = sumolib.net.readNet(str(net_xml))
            edge_ids = sorted(edge.getID() for edge in net.getEdges())
            return {eid: [eid] for eid in edge_ids}
        except Exception as e:
            warnings.warn(
                f"Could not load net_xml for automatic edge groups: {e}. Using EDGE_GROUPS."
            )
    return dict(EDGE_GROUPS)


def get_representative_edges(
    network_name: Optional[str] = "quickstart", 
    net_xml: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Return the list of grouped edge IDs used as model outputs."""
    groups = get_edge_groups(network_name=network_name, net_xml=net_xml)
    return list(groups.keys())


def group_edge_flows(
    flows: Union[dict, pd.Series],
    tolerance: float = 1e-6,
    check_consistency: bool = True,
    edge_groups: Optional[Dict[str, List[str]]] = None,
) -> Union[dict, pd.Series]:
    """
    Aggregate raw edge flows into unique flow variables by group.

    Parameters
    ----------
    flows : dict or pd.Series
        Raw edge flows (edge_id -> flow value).
    tolerance : float
        Max allowed difference between flows in the same group before warning.
    check_consistency : bool
        If True, warn when flows within a group differ by more than tolerance.
    edge_groups : dict, optional
        Map group_name -> list of edge IDs. If None, uses EDGE_GROUPS (quickstart).

    Returns
    -------
    dict or pd.Series
        Grouped flows with keys from get_representative_edges().
    """
    groups = edge_groups if edge_groups is not None else EDGE_GROUPS
    input_is_series = isinstance(flows, pd.Series)
    if input_is_series:
        flows = flows.to_dict() # converting the flows (df) to a dictionary

    grouped = {}
    for group_name, edge_ids in groups.items():
        values = []
        for eid in edge_ids:
            if eid not in flows: # checking if the edge id is in the flows
                raise KeyError(
                    f"Edge '{eid}' (group {group_name}) not found in flows."
                )
            values.append(float(flows[eid]))

        values = np.array(values)
        if check_consistency and len(values) > 1:
            if np.ptp(values) > tolerance: # peak to peak difference returns the difference between the maximum and minimum values in the array
                warnings.warn(
                    f"Flows in group {group_name} differ by more than {tolerance}: "
                    f"{dict(zip(edge_ids, values))}"
                )

        grouped[group_name] = float(values.mean())

    if input_is_series:
        return pd.Series(grouped)
    return grouped


def group_edge_dataframe(
    df: pd.DataFrame,
    flow_column: str = "interval_nVehContrib",
    check_consistency: bool = True,
    tolerance: float = 1e-6,
    edge_groups: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Aggregate a per-edge flow DataFrame into one row per group (representative edge).

    Expects columns 'edge_id' and flow_column (e.g. 'interval_nVehContrib').
    Returns a DataFrame with one row per group: edge_id = group name, flow_column = mean flow.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'edge_id' and flow_column. If 'edge_id' is missing, 'link_id' is accepted.
    flow_column : str
        Column name for flow values to aggregate.
    check_consistency : bool
        If True, warn when flows within a group differ by more than tolerance.
    tolerance : float
        Max allowed difference within a group before warning.
    edge_groups : dict, optional
        Map group_name -> list of edge IDs. If None, uses EDGE_GROUPS (quickstart).

    Returns
    -------
    pd.DataFrame
        Columns: edge_id, flow_column. One row per group.
    """
    df = df.copy()
    if "edge_id" not in df.columns and "link_id" in df.columns:
        df = df.rename(columns={"link_id": "edge_id"})  # if the dataframe does not have an edge_id column, rename the link_id column to edge_id
    if "edge_id" not in df.columns:
        raise ValueError(
            f"group_edge_dataframe expects 'edge_id' (or 'link_id'); got {list(df.columns)}" # checking what columns for the edge id is used 
        )
    if flow_column not in df.columns:
        raise ValueError(
            f"group_edge_dataframe expects column '{flow_column}'; got {list(df.columns)}"
        )

    flows = df.set_index("edge_id")[flow_column].to_dict()

    grouped_flows = group_edge_flows(
        flows,
        tolerance=tolerance,
        check_consistency=check_consistency,
        edge_groups=edge_groups,
    ) # cheking the consistency of the flows in the group

    out = pd.DataFrame(
        [{"edge_id": gid, flow_column: val} for gid, val in grouped_flows.items()]
    )
    return out


def compute_grouped_loss(
    gt_edge_data: pd.DataFrame,
    sim_edge_data: pd.DataFrame,
    edge_ids: Optional[List[str]] = None,
    network_name: Optional[str] = None,
    net_xml: Optional[Union[str, Path]] = None,
) -> float:
    """
    Compute scalar squared loss between grouped GT and simulated flows.

    Both dataframes are aggregated by edge group, then
    mean relative squared error is computed over the groups (uniform weights).

    Parameters
    ----------
    gt_edge_data : pd.DataFrame
        Ground truth edge flows (columns: edge_id, interval_nVehContrib).
    sim_edge_data : pd.DataFrame
        Simulated edge flows (same structure).
    edge_ids : list of str, optional
        Representative edge IDs to use. If None, uses get_representative_edges(network_name, net_xml).
    network_name : str, optional
        Network name (e.g. 'quickstart', '2corridor') for automatic edge groups.
    net_xml : str or Path, optional
        Path to net.xml for automatic edge groups when network is not quickstart.

    Returns
    -------
    float
        Scalar loss (mean relative squared error over groups). Used in scalar GP.
    """
    edge_groups = get_edge_groups(network_name=network_name, net_xml=net_xml)
    if edge_ids is None:
        edge_ids = list(edge_groups.keys())
    gt_grouped = group_edge_dataframe(gt_edge_data, edge_groups=edge_groups)
    sim_grouped = group_edge_dataframe(sim_edge_data, edge_groups=edge_groups)
    loss = compute_squared_metric_all_edge(
        gt_grouped,
        sim_grouped,
        edge_ids=edge_ids,
    )
    return float(np.asarray(loss).flat[0])
