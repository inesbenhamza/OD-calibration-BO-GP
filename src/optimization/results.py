"""
BOResults dataclass for storing Bayesian Optimization results.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
import pandas as pd


@dataclass
class BOResults:
    """Store BO optimization results"""
    all_X: torch.Tensor                 # All evaluated OD matrices (normalized)
    all_Y_errors: torch.Tensor          # All per-edge errors
    all_S: torch.Tensor                 # All aggregated errors (raw mean SRE)
    best_idx: int                       # Index of best solution
    best_X: torch.Tensor                # Best OD matrix (normalized)
    best_S: float                       # Best aggregated error (raw)
    convergence_curve: List[float]      # Best-so-far raw S per iteration
    acq_values: List[float]             # Acquisition values
    wall_times: List[float]             # Wall-clock time per iteration
    iteration_start: int                # Which iteration BO started from
    df_edge_stats: Optional[pd.DataFrame] = None  # All simulated flows per edge
    gp_optimizer_log: Optional[List[str]] = None  # 'lbfgs' or 'adam' per BO iteration (MOGP only)
