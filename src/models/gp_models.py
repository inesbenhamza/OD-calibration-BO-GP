from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


def set_covar_module(kernel: str, dim: int):
    if kernel == "matern-1p5":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    elif kernel == "matern-2p5":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    elif kernel == "rbf":
        return ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")





#scalar gp
def initialize_vanillabo_model(train_X, train_Y, kernel: str):
    
    dim = train_X.size(-1)
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1))
    covar_module = set_covar_module(kernel, dim)
    
    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    
    return model




# INDEPENDENT GP MODELS
def initialize_independent_gp_models_with_modellist(train_X, train_Y_multi, kernel: str = "matern-2p5"):
    """
    Fit independent GPs for each edge/link output using ModelListGP.

    Parameters
    ----------
    train_X : torch.Tensor
        (n_observations, n_od_pairs) - OD configurations
    train_Y_multi : torch.Tensor
        (n_observations, n_edges) - Traffic errors for each edge
    kernel : str, optional
        Kernel type ("matern-1p5", "matern-2p5", "rbf"). Default: "matern-2p5"

    Returns
    -------
    tuple
        (model_list_gp, mlls) - ModelListGP containing all edge GPs and list of MLLs
    """
    n_edges = train_Y_multi.size(1)
    dim = train_X.size(1)  # n_od_pairs

    individual_models = []
    mlls = []

    for edge_idx in range(n_edges):
        
        train_Y_edge = train_Y_multi[:, edge_idx].unsqueeze(-1)  # [N, 1]
        
        # Filter out NaN values for this specific edge
        # Each edge GP trains only on samples where that edge has valid (non-NaN) error
        valid_mask = ~torch.isnan(train_Y_edge).squeeze(-1)  # [N]
        #boolean tensor where True means "this value is a valid number" and False means "this value is NaN".
        
        
        #mask.sum(): In PyTorch, summing a boolean tensor treats True as 1 and False as 0.
        if valid_mask.sum() == 0: #meaning all are nan as its all false 
            # All samples have NaN for this edge - this shouldn't happen in practice
            # but we'll use all samples and let GPyTorch handle it
            train_X_edge = train_X
            train_Y_edge_filtered = train_Y_edge
        else:
            train_X_edge = train_X[valid_mask] # nouveau tenseur train_X_edge qui ne contient que les éléments de train_X où la valeur correspondante dans valid_mask est True.
            train_Y_edge_filtered = train_Y_edge[valid_mask]

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1))
        covar_module = set_covar_module(kernel, dim)
        gp_model = SingleTaskGP(
            train_X_edge,
            train_Y_edge_filtered,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1)
        )

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        individual_models.append(gp_model)
        mlls.append(mll)

    # Wrap all individual GPs in a ModelListGP
    model_list_gp = ModelListGP(*individual_models)

    return model_list_gp, mlls




def predict_independent_gps(model_list_gp, test_X):
  
    predictions = []
    uncertainties = []


    for gp_model in model_list_gp.models: # Access individual GPs from ModelListGP
        gp_model.eval()

        with torch.no_grad():
            posterior = gp_model.posterior(test_X)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            predictions.append(mean)
            uncertainties.append(variance.sqrt())

    predictions = torch.stack(predictions, dim=-1)
    uncertainties = torch.stack(uncertainties, dim=-1)

    return predictions, uncertainties



# LINEAR AGGREGATION MODEL


def make_linear_aggregation_model_from_error_gps(model_list_gp_errors, weights=None, ground_truth_flows=None, expect_flow_weights: bool = False):
    
    L = len(model_list_gp_errors.models)  # number of edges
    
    # Infer dtype from the first GP model (for consistency with GP models)
    # Default to double precision for numerical stability
    if len(model_list_gp_errors.models) > 0:
        model_dtype = next(model_list_gp_errors.models[0].parameters()).dtype
    else:
        model_dtype = torch.double  

    if weights is None: 
        # No weights provided - calculate based on ground_truth_flows or use uniform
        if ground_truth_flows is not None:
            flows = torch.as_tensor(ground_truth_flows, dtype=model_dtype)
            w = flows / (L * flows.sum())
        else:
            w = torch.ones(L, dtype=model_dtype) / L
            print(f"Using uniform weights (1/{L} each)")
    else: # Use provided weights directly 
        if isinstance(weights, torch.Tensor):
            w = weights.to(dtype=model_dtype)
        else:
            w = torch.as_tensor(weights, dtype=model_dtype)
        is_uniform = torch.allclose(w, torch.full_like(w, 1.0/L), atol=1e-6)
        print(f"Using provided weights (sum={w.sum().item():.6f}, scale={w.sum().item() * L:.6f}, expect_flow_weights={expect_flow_weights})")
        print(f"  Weight range: [{w.min().item():.6f}, {w.max().item():.6f}]")
        print(f"  First 5 weights: {w[:5].numpy()}")
        if expect_flow_weights:
            if is_uniform:
                print("  WARNING: Expected flow-proportional weights, but got uniform.")
            else:
                print("  Weights are flow-proportional (non-uniform)")

    class LinearAggregationModel(ModelListGP):
        def __init__(self, *models, w): # call the model using model = linear_aggregation_model(gp1, gp2, gp3, w)
            super().__init__(*models)# call the parent class ModelListGP 
            # its equvalent to :
            #  ModelListGP.__init__(self, gp1, gp2, gp3, ..., gp51)
            # This creates: self.models = [gp1, gp2, gp3, ..., gp51]
            self.register_buffer("w", w)

        @property
        def num_outputs(self):
            """Override to indicate this is a single-output model."""
            return 1

        def posterior(self, X, observation_noise=False, **kwargs):
            """
            Compute posterior for S(x) = sum_l [w_l * e_l(x)]

            Returns
            -------
            GPyTorchPosterior with:
            - mean: (batch_size,) or (batch_size, 1)
            - covariance: (batch_size, 1, 1) diagonal
            """
            # collecting posteriors from each per-edge-error GP
            posts = [m.posterior(X, observation_noise=observation_noise) for m in self.models]
            means = torch.stack([p.mean.squeeze(-1) for p in posts], dim=-1)     # (b, L)
            vars_ = torch.stack([p.variance.squeeze(-1) for p in posts], dim=-1) # (b, L)

            # linear aggregation is allowed because it's a linear combination of gaussian random variables
            # Any linear combination of jointly Gaussian random variables is Gaussian.
            mu_S = (means * self.w).sum(dim=-1)  # (b,)
            var_S = (vars_ * (self.w ** 2)).sum(dim=-1).clamp_min(1e-12)  # (b,)

            # Create MultivariateNormal with proper shapes
            # MVN expects: mean (b,), covariance_matrix (b, 1, 1) (botorch expects 3D for covariance)
            cov_matrix = var_S.view(-1, 1, 1)

            mvn = MultivariateNormal(mu_S, cov_matrix)
            return GPyTorchPosterior(mvn)

        def forward(self, X):
            """For compatibility with BoTorch"""
            return self.posterior(X)

    return LinearAggregationModel(*model_list_gp_errors.models, w=w)




def initialize_model(model_name: str, train_X, train_Y, kernel: str):
    """
    Factory function to initialize GP models based on model name.
    
    Parameters
    ----------
    model_name : str
        Model type: "vanillabo" or "independent_gp"/"ind_gp"
    train_X : torch.Tensor
        Training inputs
    train_Y : torch.Tensor
        Training targets (for vanillabo: aggregated error, for ind_gp: per-edge errors)
    kernel : str
        Kernel type ("matern-1p5", "matern-2p5", "rbf")
    
    Returns
    -------
    Model instance (SingleTaskGP for vanillabo, ModelListGP for ind_gp)
    """
    if model_name == "vanillabo":
        return initialize_vanillabo_model(train_X, train_Y, kernel)
    elif model_name in ["ind_gp", "independent_gp"]:
        return initialize_independent_gp_models_with_modellist(train_X, train_Y, kernel)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'vanillabo', 'independent_gp', 'ind_gp'")























########################
################################
##########################################

def extract_independent_gp_hyperparameters(
    aggregated_gp_model,
    edge_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Extract kernel and likelihood hyperparameters from the last-fitted independent GP
    (ModelListGP / LinearAggregationModel with per-edge SingleTaskGPs).

    Parameters
    ----------
    aggregated_gp_model : ModelListGP-like
        Model with .models = list of SingleTaskGP (each with ScaleKernel + base_kernel, likelihood).
    edge_ids : list of str, optional
        Edge identifiers for labeling; if None, indices 0..L-1 are used.

    Returns
    -------
    dict
        Keys: "per_edge" (list of dicts with lengthscale, outputscale, noise per edge),
              "summary" (mean/std across edges for lengthscale, outputscale, noise).
    """
    L = len(aggregated_gp_model.models)
    if edge_ids is None:
        edge_ids = [str(i) for i in range(L)]
    if len(edge_ids) != L:
        edge_ids = [str(i) for i in range(L)]

    per_edge = []
    lengthscales_all = []
    outputscales_all = []
    noises_all = []

    for i, gp in enumerate(aggregated_gp_model.models):
        eid = edge_ids[i] if i < len(edge_ids) else str(i)
        cov = gp.covar_module
        # ScaleKernel: .outputscale, .base_kernel (Matern/RBF with .lengthscale)
        outputscale = cov.outputscale.detach().item()
        base = cov.base_kernel
        lengthscale = base.lengthscale.detach()
        if lengthscale.dim() >= 2:
            lengthscale = lengthscale.squeeze()
        lengthscale_np = lengthscale.cpu().numpy()
        if lengthscale_np.size == 1:
            lengthscale_np = float(lengthscale_np.flat[0])
        else:
            lengthscale_np = lengthscale_np.tolist()
        noise = gp.likelihood.noise.detach().item()

        if isinstance(lengthscale_np, (int, float)):
            ls_scalar = lengthscale_np
        elif hasattr(lengthscale_np, "__len__"):
            ls_scalar = float(np.mean(lengthscale_np))
        else:
            ls_scalar = lengthscale_np
        lengthscales_all.append(ls_scalar)
        outputscales_all.append(outputscale)
        noises_all.append(noise)

        per_edge.append({
            "edge_id": eid,
            "lengthscale": lengthscale_np,
            "outputscale": outputscale,
            "noise": noise,
        })

    lengthscales_arr = np.array(lengthscales_all)
    outputscales_arr = np.array(outputscales_all)
    noises_arr = np.array(noises_all)

    summary = {
        "lengthscale_mean": float(np.mean(lengthscales_arr)),
        "lengthscale_std": float(np.std(lengthscales_arr)),
        "lengthscale_min": float(np.min(lengthscales_arr)),
        "lengthscale_max": float(np.max(lengthscales_arr)),
        "outputscale_mean": float(np.mean(outputscales_arr)),
        "outputscale_std": float(np.std(outputscales_arr)),
        "outputscale_min": float(np.min(outputscales_arr)),
        "outputscale_max": float(np.max(outputscales_arr)),
        "noise_mean": float(np.mean(noises_arr)),
        "noise_std": float(np.std(noises_arr)),
        "noise_min": float(np.min(noises_arr)),
        "noise_max": float(np.max(noises_arr)),
    }

    return {"per_edge": per_edge, "summary": summary}


def extract_vanillabo_hyperparameters(model) -> Dict:
    """
    Extract kernel and likelihood hyperparameters from a fitted vanilla BO SingleTaskGP.

    Returns
    -------
    dict with keys:
        "lengthscale_mean", "lengthscale_min", "lengthscale_max"  — summary across ARD dims
        "lengthscale_dim_<i>"                                      — per-dimension lengthscale
        "outputscale"                                              — ScaleKernel outputscale
        "noise"                                                    — likelihood noise
    """
    cov = model.covar_module
    outputscale = float(cov.outputscale.detach().item())
    ls = cov.base_kernel.lengthscale.detach().squeeze().cpu().numpy()
    if ls.ndim == 0:
        ls = np.array([float(ls)])
    noise = float(model.likelihood.noise.detach().item())

    row = {
        "lengthscale_mean": float(ls.mean()),
        "lengthscale_min":  float(ls.min()),
        "lengthscale_max":  float(ls.max()),
        "outputscale":      outputscale,
        "noise":            noise,
    }
    for i, v in enumerate(ls):
        row[f"lengthscale_dim_{i}"] = float(v)
    return row




def load_all_hyperparameters(
    results_dir,
    bo_folder: str,
    noise_label: str,
    run_subfolder: str = "",
    n_restarts_max: int = 100,
    seed_base: int = 42,
    seed_step: int = 1000,
) -> pd.DataFrame:
    """
    Load GP hyperparameters from all iterations and all restarts.

    Reads the ``hyperparameters_per_iteration.csv`` saved per restart during
    the BO loop (written by ``_log_hyperparameters``).

    Parameters
    ----------
    results_dir : str or Path
        Base simulation output path (``config["simulation_run_path"]``).
    bo_folder : str
        BO experiment folder name (e.g. ``"indgp_bo_squared_grouped"``).
    noise_label : str
        Noise sub-folder (e.g. ``"noise_1e-4_1"``).
    run_subfolder : str, optional
        Extra subfolder between noise_label and "results" (e.g. ``"NEW"``).
    n_restarts_max : int
        Maximum number of restarts to look for (stops at first missing).
    seed_base : int
        Seed for restart 1 (default 42).
    seed_step : int
        Increment between restart seeds (default 1000).

    Returns
    -------
    pd.DataFrame
        Columns: restart, seed, iteration, edge_id, outputscale, noise,
        lengthscale_mean, lengthscale_min, lengthscale_max.
    """
    parts = [Path(results_dir), bo_folder, noise_label]
    if run_subfolder:
        parts.append(run_subfolder)
    parts.append("results")
    base = Path(*parts)
    dfs = []
    for r in range(n_restarts_max):
        seed = seed_base + r * seed_step
        hp_path = base / f"restart_{r+1}_seed-{seed}" / "hyperparameters_per_iteration.csv"
        if not hp_path.exists():
            break
        df = pd.read_csv(hp_path)
        df["restart"] = r + 1
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No hyperparameter CSVs found under {base}")
    return pd.concat(dfs, ignore_index=True)









