"""
Bayesian Optimization Loop for OD calibration.

This module contains the BayesianOptimizationLoop class which implements
the complete BO framework including GP model updates, acquisition optimization,
and simulation evaluation.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.simulation.sumo_runner import run_simulation_for_od, parse_simulation_outputs
from src.models.gp_models import (
    initialize_vanillabo_model,
    initialize_independent_gp_models_with_modellist,
    make_linear_aggregation_model_from_error_gps,
    extract_independent_gp_hyperparameters,
    extract_vanillabo_hyperparameters,
)
from MOGP.helpers_MOGP import (
    initialize_icm_gp,
    LinearAggregationICM,
    fit_gpytorch_mll_with_adam_fallback,
)
from src.optimization.results import BOResults


class BayesianOptimizationLoop:
    """Complete Bayesian Optimization framework for OD calibration"""

    def __init__(
        self,
        config: Dict,
        gt_edge_data: pd.DataFrame,
        edge_ids: List[str],
        gt_od_vals: np.ndarray,
        routes_df: pd.DataFrame,
        base_path: str,
        bounds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        error_metric_func: Optional[Callable[[pd.DataFrame, pd.DataFrame, List[str]], np.ndarray]] = None,
        restart_output_dir: Optional[Path] = None,
    ):
        self.config = config
        # When running multiple restarts, sim outputs (bo_iter_*) go inside this dir; else under simulation_run_path
        self.restart_output_dir = Path(restart_output_dir) if restart_output_dir is not None else None
        self.gt_edge_data = gt_edge_data
        self.edge_ids = edge_ids
        self.gt_od_vals = gt_od_vals
        self.routes_df = routes_df
        self.base_path = base_path
        self.bounds = bounds
        self.device = device
        self.dtype = dtype

        self.error_metric_func = error_metric_func

        # Uniform weights: w_l = 1/L for all edges
        num_edges = len(edge_ids)
        self.weights = torch.full(
            (num_edges,),
            1.0 / num_edges,
            dtype=dtype,
            device=device
        )
        print(f"\nUsing uniform weights (1/{num_edges} each), metric_func={getattr(error_metric_func, '__name__', None)}")

        # Storage for BO progress
        self.all_X_norm: List[torch.Tensor] = []   # Normalized OD matrices
        # For vanillabo: aggregated errors (scalars). For independent_gp: per-edge errors (arrays)
        self.all_Y_errors: List[torch.Tensor] = []
        self.all_S: List[float] = []               # Aggregated errors (raw S) - model-specific
        self.convergence_curve: List[float] = []   # Best-so-far model-specific S per iteration
        self.acq_values: List[float] = []
        self.wall_times: List[float] = []

        # Store simulated edge flows for coverage plots
        self.edge_stats_records: List[pd.DataFrame] = []

        # Current best (raw objective)
        self.best_S: float = float("inf")
        self.best_idx: int = -1

        print("BO Framework initialized")
        print(f"  Edges: {len(edge_ids)}")
        print(f"  OD dimension: {len(gt_od_vals)}")

    def simulate_and_evaluate(
        self,
        X_norm: torch.Tensor,
        iteration: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Simulate the flow for a given OD matrix and evaluate the error
        
        Equivalent to the notebook's simulation and evaluation steps:
        1. Unnormalize OD matrix
        2. Create OD XML file
        3. Run SUMO simulation
        4. Parse results and compute errors
        """
        # Unnormalize OD
        X_real = unnormalize(X_norm.unsqueeze(0), self.bounds).squeeze()
        curr_od = X_real.cpu().numpy()

        # Simulation directory (inside restart folder when running multiple restarts)
        base = self.restart_output_dir if self.restart_output_dir is not None else Path(self.config["simulation_run_path"])
        sim_dir = base / f"bo_iter_{iteration}"

        # Use actual TAZ names from routes_df (e.g. taz91, taz92, ...)
        od_pairs = (
            self.routes_df[["fromTaz", "toTaz"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(curr_od) != len(od_pairs):
            raise ValueError(
                f"OD dimension mismatch: curr_od has {len(curr_od)} values "
                f"but routes_df has {len(od_pairs)} unique OD pairs. "
                f"OD pairs: {od_pairs.values.tolist()}"
            )

        print(f"  Current OD matrix: {curr_od}")

        seed = self.config.get("seed", 0)
        prefix_output = run_simulation_for_od(
            curr_od,
            od_pairs,
            sim_dir,
            self.config,
            self.base_path,
            self.routes_df,
            seed=seed,
        )
        curr_loop_stats = parse_simulation_outputs(
            prefix_output,
            self.config,
            self.base_path,
        )

        edge_stats_iter = (
            curr_loop_stats[curr_loop_stats["edge_id"].isin(self.edge_ids)]
            .loc[:, ["edge_id", "interval_nVehContrib"]]
            .copy()
        )
        edge_stats_iter["bo_iteration"] = iteration
        edge_stats_iter["batch"] = 0  # since q = 1
        self.edge_stats_records.append(edge_stats_iter)

        # Debug: which metric function is used this iteration
        print(f"  [DEBUG] Using error metric func={self.error_metric_func.__name__}")

        errors = self.error_metric_func(
            self.gt_edge_data,
            curr_loop_stats,
            self.edge_ids,
        )

        if isinstance(errors, (int, float)):
            # Vanilla BO
            error_values = np.array(errors) 
            if np.isnan(errors): # if the error is nan, set the S_observed to inf
                S_observed = float('inf')
            else:
                S_observed = float(errors)
        else:
            # Independent GP
            error_values = np.asarray(errors)

            # Filter out NaN values before aggregation (handle edges with zero GT flow)
            valid_mask = ~np.isnan(error_values)
            if valid_mask.sum() == 0:
                # All edges have NaN - invalid evaluation
                S_observed = float('inf')
            else:
                # Only aggregate over valid (non-NaN) edges
                weights_np = self.weights.cpu().numpy()
                S_observed = float((error_values[valid_mask] * weights_np[valid_mask]).sum())
                # Renormalize if some edges were filtered (to maintain same scale)
                if valid_mask.sum() < len(weights_np):
                    weight_sum = weights_np[valid_mask].sum()
                    if weight_sum > 0:
                        # Scale to maintain the same objective scale
                        S_observed = S_observed / weight_sum * weights_np.sum()

        return error_values, S_observed

    def update_gp_model(
        self,
        train_X: torch.Tensor,
        train_Y_errors: torch.Tensor,
    ):
        """
        Refit GP models with updated training data.

        Returns
        -------
        aggregated_gp_model :
            For vanillabo: Single GP model for aggregated error S(x).
            For independent_gp: Model whose output is S_std(x) = sum_l w_l * e_l_std(x)
            (linear aggregation of standardized per-edge error GPs).
        """
        model_name = self.config.get("model_name") or "vanillabo"
        kernel = self.config.get("kernel", "matern-2p5")
        
        print(f"  Refitting GPs with {len(train_X)} samples...")
        print(f"  Using model: {model_name} with kernel: {kernel}")

        if model_name == "vanillabo":
            # Vanilla BO: Single GP models aggregated error directly
            # For vanilla BO, we should use compute_squared_metric_all_edge which returns aggregated values
            if train_Y_errors.dim() == 1:
                # Already aggregated (shape: n_samples) - use directly for single GP
                # This is the correct case for vanilla BO with compute_squared_metric_all_edge
                train_Y_S = train_Y_errors.unsqueeze(-1)  # (n_samples, 1)
            elif train_Y_errors.dim() == 2:
                # Per-edge errors (shape: n_samples, n_edges) - aggregate using weights
                # This case is for backward compatibility, but vanilla BO should use aggregated values
                train_Y_S = (train_Y_errors * self.weights).sum(dim=1, keepdim=True)  # (n_samples, 1)
            else:
                raise ValueError(f"train_Y_errors has unexpected shape: {train_Y_errors.shape}")
            
            # Filter out NaN values before GP training
            # For vanilla BO, if aggregated error is NaN, filter out that entire sample
            nan_mask = torch.isnan(train_Y_S).squeeze(-1)  # [N]
            if nan_mask.any():
                n_filtered = nan_mask.sum().item()
                print(f"    Filtering NaN samples: {n_filtered}/{len(train_Y_S)} samples removed")
                valid_samples = ~nan_mask
                train_X_filtered = train_X[valid_samples]
                train_Y_S_filtered = train_Y_S[valid_samples]
                
                if len(train_X_filtered) == 0:
                    raise ValueError("All samples have NaN aggregated error - cannot train Vanilla GP model")
            else:
                train_X_filtered = train_X
                train_Y_S_filtered = train_Y_S
            
            # Initialize and fit single GP model on aggregated errors
            # train_Y_S_filtered shape: (n_samples_filtered, 1) - ONE aggregated error value per sample
            model = initialize_vanillabo_model(train_X_filtered, train_Y_S_filtered, kernel)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"    WARNING: Vanilla GP fit warning: {str(e)[:80]}")
            
            # Set model to eval mode for inference
            model.eval()
            return model
            
        elif model_name in ["independent_gp"]:
            # Independent GP models: per-edge GPs + linear aggregation
            # Must have per-edge errors (2D tensor)
            if train_Y_errors.dim() != 2:
                raise ValueError(
                    f"independent_gp requires per-edge errors (2D tensor with shape n_samples, n_edges), "
                    f"got shape: {train_Y_errors.shape}"
                )

            # Filter NaN values per edge before GP training
            # Each edge GP will only train on samples where that edge has valid (non-NaN) error
            train_Y_errors_filtered = train_Y_errors.clone()
            nan_mask = torch.isnan(train_Y_errors_filtered)
            n_nan_total = nan_mask.sum().item()
            
            if n_nan_total > 0:
                print(f"    Filtering NaN values: {n_nan_total} NaN entries found across {nan_mask.shape[0]} samples")
                # Count NaN per edge
                nan_per_edge = nan_mask.sum(dim=0)
                for edge_idx in range(len(self.edge_ids)):
                    if nan_per_edge[edge_idx] > 0:
                        print(f"      Edge {self.edge_ids[edge_idx]}: {nan_per_edge[edge_idx].item()} NaN values "
                              f"({nan_per_edge[edge_idx].item()}/{len(train_Y_errors)} samples)")
            
            # Note: We pass NaN-filtered data to GP models
            # Each edge GP will train only on valid samples for that edge
            # GPyTorch's Standardize transform should handle NaN gracefully, but filtering is safer
            model_list_gp_errors, mlls = initialize_independent_gp_models_with_modellist(
                    train_X, train_Y_errors_filtered, kernel
            )

            for i, mll in enumerate(mlls): # for ind gp fitting each error independently
                try:
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"    WARNING: Edge {self.edge_ids[i]} fit warning: {str(e)[:80]}")

            # Use stored weights for GP aggregation across edges
            weights_np = self.weights.cpu().numpy()

            aggregated_gp_model = make_linear_aggregation_model_from_error_gps(
                model_list_gp_errors=model_list_gp_errors,
                weights=weights_np,
                ground_truth_flows=None,
                expect_flow_weights=False,
            )

            # Set models to eval mode for inference
            aggregated_gp_model.eval()

            return aggregated_gp_model
        
        elif model_name in ["mogp", "icm"]:
            # Multi-output GP with ICM + linear aggregation over tasks
            if train_Y_errors.dim() != 2:
                raise ValueError(
                    f"mogp/icm requires per-edge errors (2D tensor with shape n_samples, n_edges), "
                    f"got shape: {train_Y_errors.shape}"
                )
            
            # Filter NaN values before standardization
            # For ICM, we need consistent sample sizes, so we filter out samples where ANY edge has NaN
            nan_mask = torch.isnan(train_Y_errors)
            samples_with_nan = nan_mask.any(dim=1)  # [N] - True if sample has ANY NaN
            
            if samples_with_nan.any():
                n_filtered = samples_with_nan.sum().item()
                print(f"    Filtering samples with NaN: {n_filtered}/{len(train_Y_errors)} samples removed")
                valid_samples = ~samples_with_nan
                train_X_filtered = train_X[valid_samples]
                train_Y_errors_filtered = train_Y_errors[valid_samples]
                
                if len(train_X_filtered) == 0:
                    raise ValueError("All samples have NaN values - cannot train ICM model")
            else:
                train_X_filtered = train_X
                train_Y_errors_filtered = train_Y_errors
            
            # Standardize per-edge errors for GP stability (as done in ICM kernel notebook)
            # This is necessary because different edges can have very different error scales
            # Use nanmean/nanstd to handle any remaining edge-specific NaN (shouldn't happen after filtering)
            y_mean = torch.nanmean(train_Y_errors_filtered, dim=0, keepdim=True)  # [1, L]
            y_std = torch.nanstd(train_Y_errors_filtered, dim=0, keepdim=True).clamp_min(1e-6)  # [1, L]
            train_Y_errors_std = (train_Y_errors_filtered - y_mean) / y_std  # [N_filtered, L]
            
            print(f"    Standardized per-edge errors for ICM training")
            print(f"    Mean range per edge: [{y_mean.min().item():.6f}, {y_mean.max().item():.6f}]")
            print(f"    Std range per edge: [{y_std.min().item():.6f}, {y_std.max().item():.6f}]")
            
            # Store standardization parameters for computing best_f_for_ei in standardized space
            # (needed because model outputs are in standardized space)
            self.icm_y_mean = y_mean
            self.icm_y_std = y_std
            
            # Train ICM on standardized per-edge errors (using filtered data)
            icm_model, mll = initialize_icm_gp(train_X_filtered, train_Y_errors_std)
            fit_ok = fit_gpytorch_mll_with_adam_fallback(mll, verbose=True)
            self._last_gp_optimizer = fit_ok  # 'lbfgs' or 'adam'
            if fit_ok not in ("lbfgs", "adam"):
                print("    WARNING: ICM GP fit failed")

            # Scalar for EI: Z(x) = sum_l (w_l * sigma_l) * z_l(x) with z_l standardized edges.
            # Then S_raw(x) = sum_l w_l * mu_l + Z(x), so minimizing Z is equivalent to minimizing raw S.
            sigma = y_std.squeeze(0)  # [L]
            if self.weights is None:
                L = icm_model.train_targets.shape[-1]
                weights = torch.full(
                    (L,),
                    1.0 / L,
                    dtype=icm_model.train_targets.dtype,
                    device=icm_model.train_targets.device,
                )
            else:
                weights = self.weights.to(device=sigma.device, dtype=sigma.dtype)

            weights_eff = weights * sigma
            aggregated_gp_model = LinearAggregationICM(icm_model, weights_eff)
            aggregated_gp_model.eval()
            
            return aggregated_gp_model

    def optimize_acquisition(
        self,
        aggregated_gp_model,
        best_f_for_ei,         
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Optimize EI on S(x) to find the next candidate (in normalized space).

        best_f_for_ei must be in the same space as model.posterior() outputs:
        - For vanillabo/independent_gp: raw space (Standardize transform auto-untransforms)
        - For ICM/MOGP: Z = sum_l (w_l * sigma_l) * z_l (affine to raw S; see update_gp_model)
        
        Parameters are read from self.config if not provided:
        - num_restarts: from config["NUM_RESTARTS"] or config["BO_num_restarts"] (default: 5)
        - raw_samples: from config["RAW_SAMPLES"] or config["BO_raw_samples"] (default: 32)
        """
        # Read from config if not provided
        # Check both naming conventions: UPPERCASE (from load_kwargs_config) and BO_ prefix
        if num_restarts is None:
            num_restarts = self.config.get("NUM_RESTARTS") or self.config.get("BO_num_restarts", 5)
        if raw_samples is None:
            raw_samples = self.config.get("RAW_SAMPLES") or self.config.get("BO_raw_samples", 32)
        
        print(f"  Using num_restarts={num_restarts}, raw_samples={raw_samples} (from config)")
        
        acqf = ExpectedImprovement(
            model=aggregated_gp_model,
            best_f=best_f_for_ei,  # same space as aggregated_gp_model.posterior() mean
            maximize=False,   # we minimize S(x)
        )

        d = self.bounds.shape[1]
        acq_bounds = torch.stack(
            [
                torch.zeros(d, device=self.device, dtype=self.dtype),
                torch.ones(d, device=self.device, dtype=self.dtype),
            ]
        )

        candidates, acq_val = optimize_acqf(
            acq_function=acqf,
            bounds=acq_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )

        X_next_norm = candidates[0].detach()
        return X_next_norm, float(acq_val.item())

    def run_bo_loop(
        self,
        train_X_init: torch.Tensor,
        train_Y_errors_init: torch.Tensor,
        n_iterations: int,
        early_stop_patience: int = 0,
        early_stop_delta: float = 0.0,
    ) -> BOResults:

        print("STARTING BAYESIAN OPTIMIZATION LOOP")

        # Initialize lists with initial data
        self.all_X_norm = list(train_X_init)
        self.all_Y_errors = list(train_Y_errors_init)

        # Compute initial aggregated S_raw for each sample
        # train_Y_errors_init is a torch.Tensor:
        # - For vanillabo: shape [n_samples] (already aggregated)
        # - For independent_gp/mogp/icm: shape [n_samples, n_edges] (per-edge, need to aggregate)
        if train_Y_errors_init.dim() == 1:
            # Already aggregated (vanillabo case)
            self.all_S = train_Y_errors_init.cpu().tolist()
        elif train_Y_errors_init.dim() == 2:
            # Per-edge errors - aggregate using weights
            self.all_S = (train_Y_errors_init * self.weights).sum(dim=1).cpu().tolist()
        else:
            raise ValueError(
                f"train_Y_errors_init must be 1D [n_samples] or 2D [n_samples, n_edges], "
                f"got shape: {train_Y_errors_init.shape}"
            )

        self.best_idx = int(np.argmin(self.all_S))
        self.best_S = float(self.all_S[self.best_idx])   # raw S
        self.convergence_curve = [self.best_S]
        self.gp_optimizer_log = []   # 'lbfgs' or 'adam' per BO iteration (MOGP only)

        print(f"\nInitial training data: {len(train_X_init)} samples")
        print(f"Initial best S (raw): {self.best_S:.6f}")

        start_time = time.time()
        iteration_start = len(train_X_init)

        # Early stopping tracking
        best_metric_for_stop = self.best_S
        no_improve_steps = 0
        
        # Print early stopping configuration if enabled
        if early_stop_patience > 0:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING ENABLED")
            print(f"{'='*60}")
            print(f"  Metric: S (raw)")
            print(f"  Patience: {early_stop_patience} iterations")
            print(f"  Minimum improvement: {early_stop_delta:.6f}")
            print(f"  Initial best S: {best_metric_for_stop:.6f}")
            print(f"{'='*60}\n")

        for iteration in range(n_iterations):
            iter_start = time.time()
    
            print(f"BO ITERATION {iteration + 1}/{n_iterations}")

            # Build tensors from lists
            train_X = torch.stack(self.all_X_norm)
            train_Y_errors = torch.stack(
                [
                    torch.tensor(e, device=self.device, dtype=self.dtype)
                    if not isinstance(e, torch.Tensor)
                    else e.to(self.device, self.dtype)
                    for e in self.all_Y_errors
                ]
            )

            # 1) Update GP model (per-edge GPs + aggregated S_std model)
            print("\nStep 1: Update GP models")
            aggregated_gp_model = self.update_gp_model(train_X, train_Y_errors)
            # Ensure model is in eval mode for inference
            aggregated_gp_model.eval()
            # Track which optimizer was used this iteration (only set for MOGP)
            self.gp_optimizer_log.append(getattr(self, "_last_gp_optimizer", None))

            model_name = self.config.get("model_name") or "vanillabo"

            # Log GP hyperparameters every iteration to CSV
            _hp_dir = self.restart_output_dir if self.restart_output_dir is not None else Path(self.config["simulation_run_path"])
            _hp_path = _hp_dir / "hyperparameters_per_iteration.csv"
            if model_name == "vanillabo":
                _hp_row = {"iteration": iteration, **extract_vanillabo_hyperparameters(aggregated_gp_model)}
            elif model_name == "independent_gp":
                _hp_summary = extract_independent_gp_hyperparameters(aggregated_gp_model, edge_ids=self.edge_ids)["summary"]
                _hp_row = {"iteration": iteration, **_hp_summary}
            else:
                _hp_row = None
            if _hp_row is not None:
                pd.DataFrame([_hp_row]).to_csv(_hp_path, mode="a", header=not _hp_path.exists(), index=False)

            # 2) Compute best_f for EI
            # ExpectedImprovement expects best_f in the same space as model.posterior() outputs
            # For models with Standardize transform, posterior() auto-untransforms to raw space
            # For ICM: model is trained on standardized per-edge errors, so aggregated outputs are in standardized space
            # For independent_gp: Standardize transform auto-untransforms to raw space
            if model_name == "vanillabo":
                train_Y_raw = train_Y_errors.reshape(-1)  # Raw aggregated errors
                best_f_for_ei = train_Y_raw.min()
            elif model_name in ["independent_gp"]:
                # Aggregate raw per-edge errors (Standardize transform auto-untransforms to raw)
                train_Y_raw_agg = (train_Y_errors * self.weights).sum(dim=1) if train_Y_errors.dim() == 2 else train_Y_errors.reshape(-1)
                best_f_for_ei = train_Y_raw_agg.min()
            elif model_name in ["mogp", "icm"]:
                # Z = sum_l (w_l * sigma_l) * z_l matches LinearAggregationICM(..., w * sigma) posterior
                if hasattr(self, "icm_y_mean") and hasattr(self, "icm_y_std"):
                    train_Y_errors_std = (train_Y_errors - self.icm_y_mean) / self.icm_y_std
                    sigma = self.icm_y_std.squeeze(0)  # [L]
                    if self.weights is None:
                        w = torch.ones(
                            train_Y_errors.shape[1],
                            dtype=sigma.dtype,
                            device=sigma.device,
                        ) / train_Y_errors.shape[1]
                    else:
                        w = self.weights.to(device=sigma.device, dtype=sigma.dtype)
                    train_Z_agg = (train_Y_errors_std * (w * sigma)).sum(dim=1)
                    best_f_for_ei = train_Z_agg.min()
                else:
                    # Fallback: compute standardization on the fly (shouldn't happen if update_gp_model was called)
                    y_mean = train_Y_errors.mean(dim=0, keepdim=True)
                    y_std = train_Y_errors.std(dim=0, keepdim=True).clamp_min(1e-6)
                    train_Y_errors_std = (train_Y_errors - y_mean) / y_std
                    sigma = y_std.squeeze(0)
                    if self.weights is None:
                        w = torch.ones(
                            train_Y_errors.shape[1],
                            dtype=sigma.dtype,
                            device=sigma.device,
                        ) / train_Y_errors.shape[1]
                    else:
                        w = self.weights.to(device=sigma.device, dtype=sigma.dtype)
                    train_Z_agg = (train_Y_errors_std * (w * sigma)).sum(dim=1)
                    best_f_for_ei = train_Z_agg.min()
            else:
                raise ValueError(f"Unknown model_name: {model_name}")
            
            # Also get posterior predictions for comparison/debugging
            with torch.no_grad():
                post_train = aggregated_gp_model.posterior(train_X)
                S_train_from_posterior = post_train.mean.reshape(-1)
            
            # Determine space label based on model type
            if model_name in ["mogp", "icm"]:
                space_label = "Z"
                space_desc = "Z = sum_l (w_l*sigma_l*z_l), affine to raw S"
            else:
                space_label = "raw"
                space_desc = "raw (Standardize transform auto-untransforms)"
            
            print(f"  Best f ({space_label}, for EI): {best_f_for_ei:.6f} (from {space_desc})")
            _post_note = (
                " (Z on training pts; need not equal raw S)"
                if model_name in ["mogp", "icm"]
                else " (should align with raw S)"
            )
            print(f"  Best scalar from posterior: {S_train_from_posterior.min():.6f}{_post_note}")
            print(
                f"  Training scalar (posterior) range: "
                f"[{S_train_from_posterior.min():.6f}, {S_train_from_posterior.max():.6f}]"
            )

            # 3) Optimize acquisition function (EI minimizes same raw S via model scalar space)
            print("\nStep 2: Optimize acquisition function")
            print(f"  Current best S (raw): {self.best_S:.6f}")
            X_next_norm, acq_val = self.optimize_acquisition(
                aggregated_gp_model,
                best_f_for_ei=best_f_for_ei,
                # num_restarts and raw_samples will be read from self.config
            )
            self.acq_values.append(acq_val)
            print(f"  Acquisition value (EI): {acq_val:.6f}")

            with torch.no_grad():
                post = aggregated_gp_model.posterior(X_next_norm.unsqueeze(0))
                pred_mean_std = float(post.mean.item())
                pred_std_std = float(post.variance.sqrt().item())
            if model_name in ["mogp", "icm"]:
                print(f"  GP predicts Z (ICM agg): {pred_mean_std:.6f} ± {pred_std_std:.6f}")
            else:
                print(f"  GP predicts S: {pred_mean_std:.6f} ± {pred_std_std:.6f}")
            print(f"  Candidate X (norm): {X_next_norm.cpu().numpy()}")
            
            # Check if candidate is too close to existing points
            if len(self.all_X_norm) > 0:
                existing_X = torch.stack(self.all_X_norm)
                distances = torch.norm(existing_X - X_next_norm.unsqueeze(0), dim=1)
                min_dist = float(distances.min().item())
                print(f"  Min distance to existing points: {min_dist:.6f}")
                if min_dist < 1e-4:
                    print(f"  WARNING: Candidate is very close to existing point (distance: {min_dist:.6f})")

            # 4) Simulate & evaluate (raw objective)
            print("\nStep 3: Run SUMO simulation")
            X_next_real = unnormalize(
                X_next_norm.unsqueeze(0),
                self.bounds
            ).squeeze()
            print(f"  OD (real): {X_next_real.cpu().numpy()}")

            error_values, S_observed = self.simulate_and_evaluate(
                X_next_norm, iteration_start + iteration
            )

            print(f"  Observed S (raw, model-specific): {S_observed:.6f} ")

            # 5) Update training data
            # For vanillabo: error_values is aggregated (scalar). For independent_gp: per-edge (array)
            self.all_X_norm.append(X_next_norm)
            self.all_Y_errors.append(
                torch.tensor(error_values, device=self.device, dtype=self.dtype)
            )
            self.all_S.append(S_observed)

            # 6) Update best raw S (model-specific) and best common metric
            if S_observed < self.best_S:
                improvement = self.best_S - S_observed
                self.best_S = S_observed
                self.best_idx = len(self.all_S) - 1
                print(f"  NEW BEST FOUND! S = {self.best_S:.6f}")
                print(
                    f"      Improvement: {improvement:.6f} "
                    f"({improvement/self.convergence_curve[-1]*100:.1f}%)"
                )
            else:
                print(f"  Current best remains: {self.best_S:.6f}")

            self.convergence_curve.append(self.best_S)

            # Early stopping check
            current_metric = self.best_S
            if early_stop_patience > 0:
                # Calculate improvement from the best metric tracked so far
                improvement = best_metric_for_stop - current_metric
                
                # Check if we have significant improvement
                if improvement >= early_stop_delta:
                    # Significant improvement found - reset counter and update best
                    best_metric_for_stop = current_metric
                    no_improve_steps = 0
                    print(
                        f"  Early-stop: Improvement of {improvement:.6f} ≥ {early_stop_delta:.6f} "
                        f"on S (raw) - counter reset"
                    )
                else:
                    # No significant improvement
                    no_improve_steps += 1
                    print(
                        f"  Early-stop counter: {no_improve_steps}/{early_stop_patience} "
                        f"(improvement {improvement:.6f} < {early_stop_delta:.6f} on S (raw))"
                    )
                    if no_improve_steps >= early_stop_patience:
                        print(
                            f"\n{'='*60}\n"
                            f"EARLY STOPPING TRIGGERED\n"
                            f"{'='*60}\n"
                            f"No improvement ≥ {early_stop_delta:.6f} "
                            f"in S (raw) for {early_stop_patience} consecutive iterations.\n"
                            f"  Best S achieved: {best_metric_for_stop:.6f}\n"
                            f"  Final S: {current_metric:.6f}\n"
                            f"{'='*60}\n"
                        )
                        break
                
                # Always update best_metric_for_stop if current is better (for tracking)
                if current_metric < best_metric_for_stop:
                    best_metric_for_stop = current_metric

            iter_time = time.time() - iter_start
            self.wall_times.append(iter_time)
            elapsed = time.time() - start_time
            print(f"\nIteration time: {iter_time:.1f}s")
            print(f"Total elapsed: {elapsed/60:.1f} min")
            print(
                f"Best S so far (raw): {self.best_S:.6f} "
            )

        print("BAYESIAN OPTIMIZATION COMPLETE")
        print(f"Total evaluations: {len(self.all_S)}")
        print(f"Initial best S: {self.convergence_curve[0]:.6f}")
        print(f"Final best S: {self.best_S:.6f}")
        print(f"Total improvement: {self.convergence_curve[0] - self.best_S:.6f}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} min")

        # Concatenate edge stats for coverage plot
        if self.edge_stats_records:
            df_edge_stats = pd.concat(self.edge_stats_records, ignore_index=True)
        else:
            df_edge_stats = None

        results = BOResults(
            all_X=torch.stack(self.all_X_norm),
            all_Y_errors=torch.stack(
                [
                    torch.tensor(e, device=self.device, dtype=self.dtype)
                    if not isinstance(e, torch.Tensor)
                    else e.to(self.device, self.dtype)
                    for e in self.all_Y_errors
                ]
            ),
            all_S=torch.tensor(self.all_S, device=self.device, dtype=self.dtype),
            best_idx=self.best_idx,
            best_X=self.all_X_norm[self.best_idx],
            best_S=self.best_S,
            convergence_curve=self.convergence_curve,
            acq_values=self.acq_values,
            wall_times=self.wall_times,
            iteration_start=iteration_start,
            df_edge_stats=df_edge_stats,
            gp_optimizer_log=self.gp_optimizer_log if self.gp_optimizer_log else None,
        )
        return results
