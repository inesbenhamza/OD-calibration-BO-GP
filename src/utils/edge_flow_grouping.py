"""
Group serial duplicate edges into unique flow variables for SUMO calibration.
Edges in the same group always have identical flows; we aggregate by mean.
"""

import warnings
from pathlib import Path
from typing import Union, Optional, Dict, List

import numpy as np
import pandas as pd

from src.utils.link_flow_analysis import compute_nrmse_all_edge

EDGE_GROUPS = {
    "L12": ["D2", "L2", "L12"],
    "L14": ["D4", "L4", "L14"],
    "L18": ["D5", "L5", "L18"],
    "L10": ["D7", "L7", "L10"],
    "L15": ["L15"],
    "L16": ["L16"],
}


EDGE_GROUPS_QUICKSTART_GRID_3 = {
    # --- Copy 0 ---
    "c0_in_91": ["c0_D2", "c0_L2", "c0_L12"],   # taz_91 -> junction c0_6
    "c0_out_94": ["c0_L10", "c0_L7", "c0_D7"],  # junction c0_6 -> taz_94
    "c0_L15": ["c0_L15"],                        # c0_6 -> c0_3
    "c0_L16": ["c0_L16"],                        # c0_3 -> c0_6
    "c0_in_92": ["c0_D4", "c0_L4"],              # taz_92 -> junction c0_2
    "c0_L14": ["c0_L14"],                        # c0_2 -> c0_3
    "c0_L18": ["c0_L18"],                        # c0_3 -> c0_4
    "c0_out_93": ["c0_L5", "c0_D5"],             # junction c0_4 -> taz_93
    # --- Cross-copy 0<->1 ---
    "green_01": ["green_01"],                    # c0_2 -> c1_1
    "green_01_r": ["green_01_r"],                # c1_1 -> c0_2
    "purple_01": ["purple_01"],                  # c0_4 -> c1_5
    "purple_01_r": ["purple_01_r"],              # c1_5 -> c0_4
    # --- Copy 1 ---
    "c1_in_O3": ["c1_D2", "c1_L2"],              # taz_O3 -> junction c1_1
    "c1_L12": ["c1_L12"],                        # c1_1 -> c1_6
    "c1_L10": ["c1_L10"],                        # c1_6 -> c1_5
    "c1_L15": ["c1_L15"],                        # c1_6 -> c1_3
    "c1_L16": ["c1_L16"],                        # c1_3 -> c1_6
    "c1_out_D3": ["c1_L7", "c1_D7"],             # junction c1_5 -> taz_D3
    "c1_in_O4": ["c1_D4", "c1_L4"],              # taz_O4 -> junction c1_2
    "c1_L14": ["c1_L14"],                        # c1_2 -> c1_3
    "c1_L18": ["c1_L18"],                        # c1_3 -> c1_4
    "c1_out_D4": ["c1_L5", "c1_D5"],             # junction c1_4 -> taz_D4
    # --- Cross-copy 1<->2 ---
    "green_12": ["green_12"],                    # c1_2 -> c2_1
    "green_12_r": ["green_12_r"],                # c2_1 -> c1_2
    "purple_12": ["purple_12"],                  # c1_4 -> c2_5
    "purple_12_r": ["purple_12_r"],              # c2_5 -> c1_4
    # --- Copy 2 ---
    # c2_2 and c2_4 are pass-through (no copy 3): their chains absorb L14 and L18.
    "c2_in_O5": ["c2_D2", "c2_L2"],                    # taz_O5 -> junction c2_1
    "c2_L12": ["c2_L12"],                              # c2_1 -> c2_6
    "c2_L10": ["c2_L10"],                              # c2_6 -> c2_5
    "c2_L15": ["c2_L15"],                              # c2_6 -> c2_3
    "c2_L16": ["c2_L16"],                              # c2_3 -> c2_6
    "c2_out_D5": ["c2_L7", "c2_D7"],                   # junction c2_5 -> taz_D5
    "c2_in_O6": ["c2_D4", "c2_L4", "c2_L14"],         # taz_O6 -> junction c2_3
    "c2_out_D6": ["c2_L18", "c2_L5", "c2_D5"],        # c2_3 -> c2_4 -> taz_D6
}

# Dense variant: only the 31 edge groups actually traversed by routes in
# quickstart_grid_3_dense.  Seven groups from EDGE_GROUPS_QUICKSTART_GRID_3
# are omitted because no route ever uses them:
#   c0_in_91, c0_out_94, c0_L15, c0_L16  (copy-0 top lane + diagonals)
#   c1_L16                                (copy-1 reverse diagonal)
#   c2_in_O6, c2_L16                     (copy-2 unused O6 inlet + reverse diagonal)
# Confirmed by GT valid_mask = 31.  L=31.
# Use with network_name="quickstart_grid_3_dense" only.
EDGE_GROUPS_QUICKSTART_GRID_3_DENSE = {
    # --- Copy 0: bottom lane only (top lane 91/94 never traversed) ---
    "c0_in_92":    ["c0_D4", "c0_L4"],
    "c0_L14":      ["c0_L14"],
    "c0_L18":      ["c0_L18"],
    "c0_out_93":   ["c0_L5", "c0_D5"],
    # --- Cross-copy 0 <-> 1: all 6 connectors used ---
    "green_01":    ["green_01"],
    "green_01_r":  ["green_01_r"],
    "purple_01":   ["purple_01"],
    "purple_01_r": ["purple_01_r"],
    "yellow_01":   ["yellow_01"],
    "yellow_01_r": ["yellow_01_r"],
    # --- Copy 1: both lanes (c1_L16 reverse diagonal never used) ---
    "c1_in_O3":    ["c1_D2", "c1_L2"],
    "c1_L12":      ["c1_L12"],
    "c1_L10":      ["c1_L10"],
    "c1_L15":      ["c1_L15"],
    "c1_out_D3":   ["c1_L7", "c1_D7"],
    "c1_in_O4":    ["c1_D4", "c1_L4"],
    "c1_L14":      ["c1_L14"],
    "c1_L18":      ["c1_L18"],
    "c1_out_D4":   ["c1_L5", "c1_D5"],
    # --- Cross-copy 1 <-> 2: all 6 connectors used ---
    "green_12":    ["green_12"],
    "green_12_r":  ["green_12_r"],
    "purple_12":   ["purple_12"],
    "purple_12_r": ["purple_12_r"],
    "yellow_12":   ["yellow_12"],
    "yellow_12_r": ["yellow_12_r"],
    # --- Copy 2: both lanes (c2_in_O6 and c2_L16 never used) ---
    "c2_in_O5":    ["c2_D2", "c2_L2"],
    "c2_L12":      ["c2_L12"],
    "c2_L10":      ["c2_L10"],
    "c2_L15":      ["c2_L15"],
    "c2_out_D5":   ["c2_L7", "c2_D7"],
    "c2_out_D6":   ["c2_L18", "c2_L5", "c2_D5"],
}

# 3x3 grid: each edge is its own group (no serial duplicates in a grid)
EDGE_GROUPS_GRID_2D_3X3 = {e: [e] for e in [
    "A0A1", "A0B0", "A1A0", "A1A2", "A1B1", "A2A1", "A2B2",
    "B0A0", "B0B1", "B0C0", "B1A1", "B1B0", "B1B2", "B1C1",
    "B2A2", "B2B1", "B2C2", "C0B0", "C0C1", "C1B1", "C1C0", "C1C2", "C2B2", "C2C1",
]}


__all__ = [
    "EDGE_GROUPS",
    "EDGE_GROUPS_QUICKSTART_GRID_3",
    "EDGE_GROUPS_QUICKSTART_GRID_3_DENSE",
    "EDGE_GROUPS_GRID_2D_3X3",
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
    """Return edge groups for the given network."""
    if network_name == "quickstart":
        return dict(EDGE_GROUPS)
    if network_name == "quickstart_grid_3":
        return dict(EDGE_GROUPS_QUICKSTART_GRID_3)
    if network_name == "quickstart_grid_3_dense":
        return dict(EDGE_GROUPS_QUICKSTART_GRID_3_DENSE)
    if network_name == "grid_2d_3x3":
        return dict(EDGE_GROUPS_GRID_2D_3X3)
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
    return list(get_edge_groups(network_name=network_name, net_xml=net_xml).keys())


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
        Grouped flows keyed by group name.
    """
    groups = edge_groups if edge_groups is not None else EDGE_GROUPS
    input_is_series = isinstance(flows, pd.Series)
    if input_is_series:
        flows = flows.to_dict()

    grouped = {}
    for group_name, edge_ids in groups.items():
        values = []
        for eid in edge_ids:
            # missing from sim output == zero vehicles traversed it
            values.append(float(flows[eid]) if eid in flows else 0.0)

        values = np.array(values)
        if check_consistency and len(values) > 1 and np.ptp(values) > tolerance:
            warnings.warn(
                f"Flows in group {group_name} differ by more than {tolerance}: "
                f"{dict(zip(edge_ids, values))}"
            )

        grouped[group_name] = float(values.mean())

    return pd.Series(grouped) if input_is_series else grouped


def group_edge_dataframe(
    df: pd.DataFrame,
    flow_column: str = "interval_nVehContrib",
    check_consistency: bool = True,
    tolerance: float = 1e-6,
    edge_groups: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Aggregate a per-edge flow DataFrame into one row per group.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'edge_id' (or 'link_id') and flow_column.
    flow_column : str
        Column to aggregate.
    check_consistency : bool
        Warn when flows within a group differ by more than tolerance.
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
        df = df.rename(columns={"link_id": "edge_id"})
    if "edge_id" not in df.columns:
        raise ValueError(
            f"group_edge_dataframe expects 'edge_id' (or 'link_id'); got {list(df.columns)}"
        )
    if flow_column not in df.columns:
        raise ValueError(
            f"group_edge_dataframe expects column '{flow_column}'; got {list(df.columns)}"
        )

    grouped_flows = group_edge_flows(
        df.set_index("edge_id")[flow_column].to_dict(),
        tolerance=tolerance,
        check_consistency=check_consistency,
        edge_groups=edge_groups,
    )
    return pd.DataFrame(
        [{"edge_id": gid, flow_column: val} for gid, val in grouped_flows.items()]
    )


def compute_grouped_loss(
    gt_edge_data: pd.DataFrame,
    sim_edge_data: pd.DataFrame,
    edge_ids: Optional[List[str]] = None,
    network_name: Optional[str] = None,
    net_xml: Optional[Union[str, Path]] = None,
) -> float:
    """
    Compute nRMSE between grouped GT and simulated flows.

    Parameters
    ----------
    gt_edge_data : pd.DataFrame
        Ground truth edge flows (columns: edge_id, interval_nVehContrib).
    sim_edge_data : pd.DataFrame
        Simulated edge flows (same structure).
    edge_ids : list of str, optional
        Representative edge IDs to use. If None, derived from network_name/net_xml.
    network_name : str, optional
        Network name for automatic edge groups.
    net_xml : str or Path, optional
        Path to net.xml for automatic edge groups.

    Returns
    -------
    float
        nRMSE.
    """
    edge_groups = get_edge_groups(network_name=network_name, net_xml=net_xml)
    if edge_ids is None:
        edge_ids = list(edge_groups.keys())
    gt_grouped  = group_edge_dataframe(gt_edge_data,  edge_groups=edge_groups)
    sim_grouped = group_edge_dataframe(sim_edge_data, edge_groups=edge_groups)
    return float(compute_nrmse_all_edge(gt_grouped, sim_grouped, edge_ids=edge_ids))
