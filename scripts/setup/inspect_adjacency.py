"""
Inspect the L x L graph task-kernel adjacency matrix used by run_graph_multitask_gp.

Usage:
    python scripts/inspect_adjacency.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.edge_flow_grouping import get_edge_groups, get_representative_edges
from MOGP.graph import adjacency_route_overlap_weighted, adjacency_by_shared_node_weighted

NETWORK_NAME = "quickstart_grid_3"
ROUTES_CSV   = f"network/{NETWORK_NAME}/routes.csv"
NET_XML      = f"network/{NETWORK_NAME}/net.xml"

edge_groups = get_edge_groups(network_name=NETWORK_NAME)
edge_ids    = get_representative_edges(network_name=NETWORK_NAME)
L           = len(edge_ids)

A_route = adjacency_route_overlap_weighted(
    ROUTES_CSV, edge_ids, edge_groups=edge_groups, verbose=True,
)

A_shared = adjacency_by_shared_node_weighted(
    NET_XML, edge_ids, ignore_internal=True,
)

print(f"\nedge_ids (L={L}): {edge_ids}\n")

df_route = pd.DataFrame(A_route.numpy(), index=edge_ids, columns=edge_ids)
df_shared = pd.DataFrame(A_shared.numpy(), index=edge_ids, columns=edge_ids)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

print("=== route_overlap_weighted adjacency ===")
print(df_route)
print(f"\ndensity (nonzero frac): {(A_route > 0).float().mean().item():.4f}")
print(f"isolated tasks: {int((A_route.sum(dim=1) == 0).sum())}")

print("\n=== shared_node_weighted adjacency ===")
print(df_shared)
print(f"\ndensity (nonzero frac): {(A_shared > 0).float().mean().item():.4f}")
print(f"isolated tasks: {int((A_shared.sum(dim=1) == 0).sum())}")

df_route.to_csv(f"scripts/adjacency_{NETWORK_NAME}_route_overlap.csv")
df_shared.to_csv(f"scripts/adjacency_{NETWORK_NAME}_shared_node.csv")
print(f"\nSaved CSVs to scripts/adjacency_{NETWORK_NAME}_*.csv")
