"""
Save and load BOResults objects to/from disk.

This module provides functions to save BOResults objects so they can be
loaded later for analysis and plotting without re-running BO.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import torch

from src.optimization.results import BOResults


def save_bo_results(
    results: BOResults,
    save_dir: Path,
    restart_idx: Optional[int] = None,
    seed: Optional[int] = None,
    metadata: Optional[dict] = None,
):
    """
    Save BOResults object to disk for later loading.
    
    Saves:
    - BOResults object as pickle (for full reload)
    - convergence.csv (for visualization scripts)
    - edge_stats.csv (if available)
    - metadata.json (config info)
    
    Parameters
    ----------
    results : BOResults
        Results object to save
    save_dir : Path
        Directory to save files (will be created if needed)
    restart_idx : int, optional
        Restart index (for organizing multiple restarts)
    seed : int, optional
        Seed used for this restart
    metadata : dict, optional
        Additional metadata to save (e.g., config, model_name, kernel)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save BOResults as pickle (for full reload)
    if restart_idx is not None:
        pickle_path = save_dir / f"bo_results_restart_{restart_idx}.pkl"
    else:
        pickle_path = save_dir / "bo_results.pkl"
    
    # Move tensors to CPU before saving
    results_cpu = BOResults(
        all_X=results.all_X.cpu(),
        all_Y_errors=results.all_Y_errors.cpu(),
        all_S=results.all_S.cpu(),
        best_idx=results.best_idx,
        best_X=results.best_X.cpu(),
        best_S=results.best_S,
        convergence_curve=results.convergence_curve,
        acq_values=results.acq_values,
        wall_times=results.wall_times,
        iteration_start=results.iteration_start,
        df_edge_stats=results.df_edge_stats,
    )

    with open(pickle_path, 'wb') as f:
        pickle.dump(results_cpu, f)
    print(f"  ✓ Saved BOResults: {pickle_path}")
    

    n_conv = len(results.convergence_curve)
    pad = n_conv - len(results.acq_values) 
    conv_df = pd.DataFrame({
        "iteration": range(n_conv),
        "best_S_model": results.convergence_curve,
        "acq_value": [0.0] * pad + list(results.acq_values),
        "wall_time": [0.0] * pad + list(results.wall_times),
    })
    conv_path = save_dir / "convergence.csv"
    conv_df.to_csv(conv_path, index=False)
    print(f"  ✓ Saved convergence.csv: {conv_path}")
    

    if results.df_edge_stats is not None and not results.df_edge_stats.empty:
        edge_stats_path = save_dir / "edge_stats.csv"
        results.df_edge_stats.to_csv(edge_stats_path, index=False)
        print(f"  ✓ Saved edge_stats.csv: {edge_stats_path}")
    

    meta = {
        "best_S": float(results.best_S),
        "best_idx": int(results.best_idx),
        "iteration_start": int(results.iteration_start),
        "n_evaluations": int(results.all_X.shape[0]),
        "n_bo_iterations": len(results.acq_values),
    }
    if restart_idx is not None:
        meta["restart_idx"] = restart_idx
    if seed is not None:
        meta["seed"] = seed
    if metadata:
        meta.update(metadata)
    
    meta_path = save_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata.json: {meta_path}")



def load_bo_results(
    load_path: Path,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> BOResults:
    """
    Load BOResults object from disk.
    
    Parameters
    ----------
    load_path : Path
        Path to .pkl file or directory containing bo_results.pkl
    device : torch.device, optional
        Device to move tensors to (default: CPU)
    dtype : torch.dtype, optional
        Data type for tensors (default: keep original)
    
    Returns
    -------
    BOResults
        Loaded results object
    """
    load_path = Path(load_path)
    
    # If directory, look for bo_results.pkl or bo_results_restart_*.pkl
    if load_path.is_dir():
        pickle_files = list(load_path.glob("bo_results*.pkl"))
        if not pickle_files:
            raise FileNotFoundError(f"No bo_results*.pkl found in {load_path}")
        if len(pickle_files) > 1:
            # If multiple, prefer bo_results.pkl, otherwise use first
            pickle_path = load_path / "bo_results.pkl"
            if pickle_path.exists():
                load_path = pickle_path
            else:
                load_path = pickle_files[0]
        else:
            load_path = pickle_files[0]
    elif not load_path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")
    
    with open(load_path, 'rb') as f:
        results = pickle.load(f)
    
    # Move tensors to specified device/dtype
    if device is not None or dtype is not None:
        results = BOResults(
            all_X=results.all_X.to(device=device, dtype=dtype) if device or dtype else results.all_X,
            all_Y_errors=results.all_Y_errors.to(device=device, dtype=dtype) if device or dtype else results.all_Y_errors,
            all_S=results.all_S.to(device=device, dtype=dtype) if device or dtype else results.all_S,
            best_idx=results.best_idx,
            best_X=results.best_X.to(device=device, dtype=dtype) if device or dtype else results.best_X,
            best_S=results.best_S,
            convergence_curve=results.convergence_curve,
            acq_values=results.acq_values,
            wall_times=results.wall_times,
            iteration_start=results.iteration_start,
            df_edge_stats=results.df_edge_stats,
        )
    
    print(f"✓ Loaded BOResults from: {load_path}")
    print(f"  Best S: {results.best_S:.6f}")
    print(f"  Evaluations: {results.all_X.shape[0]}")
    print(f"  BO iterations: {len(results.acq_values)}")
    
    return results


def load_all_restarts(
    results_dir: Path,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[BOResults]:
    """
    Load all restart results from a results directory.
    
    Looks for: results_dir/restart_*_seed-*/bo_results_restart_*.pkl
    
    Parameters
    ----------
    results_dir : Path
        Directory containing restart subdirectories
    device : torch.device, optional
        Device to move tensors to
    dtype : torch.dtype, optional
        Data type for tensors
    
    Returns
    -------
    List[BOResults]
        List of BOResults objects, one per restart (sorted by restart number)
    """
    results_dir = Path(results_dir)
    
    # Find all restart directories
    restart_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("restart_")],
        key=lambda x: int(x.name.split("_")[1].split("_")[0])  # Extract restart number
    )
    
    if not restart_dirs:
        raise FileNotFoundError(f"No restart directories found in {results_dir}")
    
    all_results = []
    for restart_dir in restart_dirs:
        # Look for bo_results_restart_*.pkl
        pickle_files = list(restart_dir.glob("bo_results_restart_*.pkl"))
        if pickle_files:
            result = load_bo_results(pickle_files[0], device=device, dtype=dtype)
            all_results.append(result)
        else:
            print(f"  Warning: No bo_results_restart_*.pkl found in {restart_dir}")
    
    print(f"\n✓ Loaded {len(all_results)} restart results from {results_dir}")
    return all_results
