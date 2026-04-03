"""
Multiple BO restarts wrapper.

This module contains the run_multiple_bo_restarts function which runs
multiple independent BO loops with different random seeds and aggregates results.
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.optimization.bo_loop import BayesianOptimizationLoop
from src.optimization.results import BOResults
from src.optimization.io import save_bo_results


def run_multiple_bo_restarts(
    config: Dict,
    gt_edge_data: pd.DataFrame,
    edge_ids: List[str],
    gt_od_vals: np.ndarray,
    routes_df: pd.DataFrame,
    base_path: str,
    bounds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    train_X_init: torch.Tensor,
    train_Y_errors_init: torch.Tensor,
    n_restarts: int = 5,
    n_bo_iterations: int = 10,
    error_metric_func: Optional[Callable[[pd.DataFrame, pd.DataFrame, List[str]], np.ndarray]] = None,
    early_stop_patience: int = 0,
    early_stop_delta: float = 0.0,
) -> Dict:

    print(f"RUNNING {n_restarts} BO RESTARTS")

    all_results: List[BOResults] = []

    for restart in range(n_restarts):
        print(f"RESTART {restart + 1}/{n_restarts}")

        restart_seed = 42 + restart * 1000
        try:
            from src.utils.seed import set_seed
            set_seed(restart_seed)
            print(f"  Reseeded all RNGs with seed {restart_seed}")
        except Exception:
            torch.manual_seed(restart_seed)
            print(f"  Reseeded torch with seed {restart_seed}")

        restart_dir = None
        if "simulation_run_path" in config:
            restart_dir = Path(config["simulation_run_path"]) / "results" / f"restart_{restart+1}_seed-{restart_seed}"
            restart_dir.mkdir(parents=True, exist_ok=True)

        bo_loop = BayesianOptimizationLoop(
            config,
            gt_edge_data,
            edge_ids,
            gt_od_vals,
            routes_df,
            base_path,
            bounds,
            device,
            dtype,
            error_metric_func=error_metric_func,
            restart_output_dir=restart_dir,
        )

        print(f"  Using SAME {train_X_init.shape[0]} initial samples for every restart.")

        # Run BO starting from same initial data
        results = bo_loop.run_bo_loop(
            train_X_init, 
            train_Y_errors_init, 
            n_iterations=n_bo_iterations,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
        )

        all_results.append(results)
        
        # Save individual restart results (if config provides save path)
        if "simulation_run_path" in config:
            restart_seed = 42 + restart * 1000
            save_dir = Path(config["simulation_run_path"]) / "results" / f"restart_{restart+1}_seed-{restart_seed}"
            save_bo_results(
                results=results,
                save_dir=save_dir,
                restart_idx=restart+1,
                seed=restart_seed,
                metadata={
                    "model_name": config.get("model_name", "unknown"),
                    "kernel": config.get("kernel", "unknown"),
                    "network_name": config.get("network_name", "unknown"),
                }
            )

    print("AGGREGATING RESULTS ACROSS RESTARTS")

    # Use model-specific metric for aggregation
    convergence_curves_list = [r.convergence_curve for r in all_results]
    best_S_all = [r.best_S for r in all_results]

    max_len = max(len(curve) for curve in convergence_curves_list)
    convergence_curves = np.full((n_restarts, max_len), np.nan)

    for i, curve in enumerate(convergence_curves_list):
        convergence_curves[i, :len(curve)] = curve

    mean_curve = np.nanmean(convergence_curves, axis=0)
    std_curve = np.nanstd(convergence_curves, axis=0)

    overall_best_idx = int(np.argmin(best_S_all))
    overall_best_result = all_results[overall_best_idx]

    metric_label = "S (raw)"
    print(f"\nResults across {n_restarts} restarts ({metric_label}):")
    for i, s in enumerate(best_S_all):
        print(f"  Restart {i+1}: {metric_label} = {s:.6f} (NRMSE = {np.sqrt(s):.4f})")

    print(
        f"\nMean best {metric_label}: {np.mean(best_S_all):.6f} ± {np.std(best_S_all):.6f}"
    )
    best_S_value = overall_best_result.best_S
    print(
        f"Overall best {metric_label}: {best_S_value:.6f} "
        f"(restart {overall_best_idx + 1})"
    )

    aggregated_results = {
        "all_results": all_results,
        "convergence_curves": convergence_curves,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "best_S_all": best_S_all,
        "overall_best_result": overall_best_result,
        "overall_best_restart": overall_best_idx,
    }

    return aggregated_results
