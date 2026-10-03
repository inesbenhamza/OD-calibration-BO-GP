from pathlib import Path
from typing import Dict, List, Optional, Union
import copy
import math
import xml.etree.ElementTree as _ET
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.deterministic import GenericDeterministicModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions.errors import ModelFittingError
from gpytorch.utils.errors import NotPSDError



def _randomize_hyperparams(model):
    """Sample hyperparameters uniformly within their constraints."""
    with torch.no_grad():
        ck = model.covar_module 

        if hasattr(ck, "base_kernel") and hasattr(ck.base_kernel, "lengthscale"):
            lb, ub = 0.005, 20.0
            if hasattr(ck.base_kernel, "raw_lengthscale_constraint"):
                c = ck.base_kernel.raw_lengthscale_constraint
                lb, ub = float(c.lower_bound), float(c.upper_bound)
                if not math.isfinite(ub):
                    ub = 20.0
            shape = ck.base_kernel.lengthscale.shape
            ck.base_kernel.lengthscale = torch.empty(shape).uniform_(lb, ub)

        if hasattr(ck, "outputscale") and ck.raw_outputscale.requires_grad:
            lb, ub = 0.05, 2.0
            if hasattr(ck, "raw_outputscale_constraint"):
                c = ck.raw_outputscale_constraint
                c_lb, c_ub = float(c.lower_bound), float(c.upper_bound)
                if math.isfinite(c_lb):
                    lb = c_lb
                if math.isfinite(c_ub) and c_ub > lb:
                    ub = c_ub
            ck.outputscale = torch.empty(1).uniform_(lb, ub)

        if isinstance(model.likelihood, GaussianLikelihood):
            lb, ub = 1e-6, 0.1
            if hasattr(model.likelihood.noise_covar, "raw_noise_constraint"):
                c = model.likelihood.noise_covar.raw_noise_constraint
                lb, ub = float(c.lower_bound), float(c.upper_bound)
                if not math.isfinite(ub):
                    ub = 0.1
            model.likelihood.noise = torch.empty(1).uniform_(lb, ub)


def fit_gpytorch_mll_multistart(mll, n_restarts: int = 5, verbose: bool = False, collect_init_samples: bool = False, collect_converged_samples: bool = False):
    """
    state_dict() is a PyTorch method that returns a flat dictionary of all learnable parameter tensors in the model. 

    can be checked using 

    import pprint
    pprint.pprint(dict(mll.model.state_dict()))

    or for exact shape 

    for name, val in mll.model.state_dict().items():
    print(f"{name}: {val.shape} = {val}")

    mll.model.state_dict() keys :
    {
    'likelihood.noise_covar.raw_noise':                                tensor([...]),   # raw noise (pre-transform)
    'likelihood.noise_covar.raw_noise_constraint.lower_bound':         tensor(...),     # e.g. 1e-3
    'likelihood.noise_covar.raw_noise_constraint.upper_bound':         tensor(...),     # e.g. 1.0
    'mean_module.raw_constant':                                        tensor(...),     # GP constant mean
    'covar_module.raw_outputscale':                                    tensor(...),     # raw outputscale
    'covar_module.raw_outputscale_constraint.lower_bound':             tensor(...),
    'covar_module.raw_outputscale_constraint.upper_bound':             tensor(...),
    'covar_module.base_kernel.raw_lengthscale':                        tensor([[...]]), # shape (1,1) — single lengthscale (not ARD)
    'covar_module.base_kernel.raw_lengthscale_constraint.lower_bound': tensor(...),     # e.g. 0.001
    'covar_module.base_kernel.raw_lengthscale_constraint.upper_bound': tensor(...),     # e.g. 4.0
    'outcome_transform.means':                                         tensor([[...]]), # data mean for Standardize
    'outcome_transform.stdvs':                                         tensor([[...]]), # data std for Standardize
    'outcome_transform._stdvs_sq':                                     tensor([[...]]), # precomputed std²
    'outcome_transform._is_trained':                                   tensor(True),    # whether Standardize has seen data
    }
    """
    original_state = copy.deepcopy(mll.model.state_dict()) # initial hyperparameters before any fitting, default constant set at model cosntruction 

    best_mll_val = -math.inf
    best_state = None
    init_samples = [] if collect_init_samples else None # used to inspect the distribution of random initializations and where they converge 
    converged_samples = [] if collect_converged_samples else None # value of the convegrd hyperpamramerter after fitting and the corresponding MLL, used to inspect the distribution of converged hyperparameters and their quality (MLL)
    # to see see for ex.. that 3 out of 5 restarts converge to the same solution (good), or that they all converge to different places (bad — landscape has many local optima).
    
    
    for i in range(n_restarts):
        mll.model.load_state_dict(original_state)
        _randomize_hyperparams(mll.model)
        candidate_label = f"random restart {i}/{n_restarts - 1}"

        if collect_init_samples:
            with torch.no_grad():
                ck = mll.model.covar_module
                ls = ck.base_kernel.lengthscale.detach().squeeze().cpu().tolist()
                if not isinstance(ls, list):
                    ls = [ls]
                init_samples.append({
                    "restart":     i,
                    "noise":       float(mll.model.likelihood.noise.mean().detach().item()),
                    "outputscale": float(ck.outputscale.detach().item()) if hasattr(ck, "outputscale") else float("nan"),
                    "lengthscale": ls,
                })

        mll.model.train()
        try:
            fit_gpytorch_mll(mll)
            mll.model.train()
            with torch.no_grad():
                output = mll.model(*mll.model.train_inputs)
                mll_val = mll(output, mll.model.train_targets).item()
            if verbose:
                print(f"    {candidate_label}: MLL = {mll_val:.4f}")
            if mll_val > best_mll_val:
                best_mll_val = mll_val
                best_state = copy.deepcopy(mll.model.state_dict())

            if collect_converged_samples:
                with torch.no_grad():
                    ck = mll.model.covar_module
                    ls = ck.base_kernel.lengthscale.detach().squeeze().cpu().tolist()
                    if not isinstance(ls, list):
                        ls = [ls]
                    converged_samples.append({
                        "restart":     i,
                        "label":       candidate_label,
                        "mll_val":     mll_val,
                        "noise":       float(mll.model.likelihood.noise.mean().detach().item()),
                        "outputscale": float(ck.outputscale.detach().item()) if hasattr(ck, "outputscale") else float("nan"),
                        "lengthscale": ls,
                        "is_best":     False,
                    })
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"    {candidate_label}: failed ({str(e)[:60]})")

    if best_state is not None:
        mll.model.load_state_dict(best_state)
    else:
        mll.model.load_state_dict(original_state)

    if collect_converged_samples and converged_samples:
        best_idx = max(range(len(converged_samples)), key=lambda k: converged_samples[k]["mll_val"])
        converged_samples[best_idx]["is_best"] = True

    mll.model.eval()
    if collect_converged_samples:
        return converged_samples
    if collect_init_samples:
        return init_samples

###
######
####




def set_covar_module(kernel: str, constraint_mode):

    if constraint_mode in ("constrained") :
        ls_constraint = Interval(0.005, 20.0)
    else:
        ls_constraint = None


    if kernel == "matern-1p5":
        base_kernel = MaternKernel(nu=1.5, has_lengthscale=True, lengthscale_constraint=ls_constraint)
    elif kernel == "matern-2p5":
        base_kernel = MaternKernel(nu=2.5, has_lengthscale=True, lengthscale_constraint=ls_constraint)
    elif kernel == "rbf":
        base_kernel = RBFKernel(has_lengthscale=True, lengthscale_constraint=ls_constraint)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    scale_kernel = ScaleKernel(base_kernel)

    return scale_kernel





####################
###############################
###################
#scalar gp
def initialize_vanillabo_model(train_X, train_Y, kernel: str, constraint_mode: str = "constrained"):

    dim = train_X.size(-1)

    covar_module = set_covar_module(kernel, constraint_mode=constraint_mode)

    if constraint_mode in ("constrained") :
        noise_constraint = Interval(1e-4, 1)
    else:
        noise_constraint = None

    likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform= Standardize(m=1), 
    ) # standardize as constant mean so need to be centered around 0 

    return model


########
############
#####################

# INDEPENDENT GP MODELS
def initialize_independent_gp_models_with_modellist(train_X, train_Y_multi, kernel: str = "matern-2p5", constraint_mode: str = "constrained"):
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

    individual_models = [] # list to hold each edge-specific GP model, which will be wrapped in ModelListGP; allows for edge-specific hyperparameters and diagnostics (e.g. which edges are harder to fit, how well the model fits each edge, etc.)
    mlls = [] # keep track of MLLs for each edge-specific GP, useful for diagnostics and hyperparameter tuning insights (e.g. which edges are harder to fit, how well the model fits each edge, etc.)

    for edge_idx in range(n_edges):
        train_X_edge          = train_X
        train_Y_edge_filtered = train_Y_multi[:, edge_idx].unsqueeze(-1)  # takes all rows for a single column so whole column edge_idx (one edge)

        covar_module      = set_covar_module(kernel, constraint_mode=constraint_mode)

        outcome_transform = Standardize(m=1)

        if constraint_mode in ("constrained"):
            noise_constraint = Interval(1e-4, 1)
        else:
            noise_constraint = None
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        gp_model = SingleTaskGP(
            train_X_edge,
            train_Y_edge_filtered,
            covar_module=covar_module,
            likelihood=likelihood,
            # fixed ZeroMean (not a learned ConstantMean): Standardize already
            # exactly zero-centers this edge's targets before fitting, so a
            # learned constant is redundant. 
            mean_module=ZeroMean(),
            outcome_transform=outcome_transform,
        )

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        individual_models.append(gp_model)
        mlls.append(mll)

    # Wrap all individual GPs in a ModelListGP
    model_list_gp = ModelListGP(*individual_models)

    return model_list_gp, mlls




def predict_independent_gps(model_list_gp, test_X):
    """for interpretation of the predictions and uncertainties of each edge-specific GP in the ModelListGP,
    called to inspect things like: "how well does the GP fit edge 3"""

    predictions = []
    uncertainties = []


    for gp_model in model_list_gp.models: # Access individual GPs from ModelListGP
        gp_model.eval()

        with torch.no_grad():  # disable gradient tracking, we're predicting, not training
            posterior = gp_model.posterior(test_X)       # ask the GP: "what do you predict at these new OD matrices?"
            mean = posterior.mean.squeeze(-1)             # the GP's best guess for the error at each test point
            variance = posterior.variance.squeeze(-1)     # how uncertain the GP is at each test point
            predictions.append(mean)                     # collect this edge's mean predictions
            uncertainties.append(variance.sqrt())        # std dev = sqrt(variance), more interpretable than variance

    predictions = torch.stack(predictions, dim=-1)
    uncertainties = torch.stack(uncertainties, dim=-1)

    return predictions, uncertainties


class IndGPAggregationModel(Model):
    """
    Wraps a fitted ModelListGP (L per-edge f_l outputs) and exposes scalar
    S(x) = sum_l f_l(x) = nRMSE²(x) for use with ExpectedImprovement.
    """

    def __init__(self, model_list_gp: ModelListGP):
        super().__init__()
        self.model_list_gp = model_list_gp

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        post = self.model_list_gp.posterior(X, observation_noise=observation_noise)
        mean = post.mean      
        var  = post.variance  

        # S = sum_l f_l = nRMSE²  — linear
        mu_S  = mean.sum(dim=-1, keepdim=True)                    # (N, 1) we get N aggragted mean
        var_S = var.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # (N, 1) we get N aggregated variance, one per OD

        mvn = MultivariateNormal(mu_S.squeeze(-1), var_S.squeeze(-1).unsqueeze(-1)) #  Gaussian distribution defined by a mean vector and covariance matrix
        # used to wrap the mean and variance of the aggregated output S into a MultivariateNormal distribution, which is then wrapped in a GPyTorchPosterior for compatibility with BoTorch acquisition functions.  
        return GPyTorchPosterior(mvn) # wrappping the mvn into an object 




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
        noise = float(gp.likelihood.noise.detach().mean().item())

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
    noise = float(model.likelihood.noise.detach().mean().item())

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





def load_indgp_iters(
    sim_base,
    n_bo_iters: int,
    train_X_init,
    train_Y_init,
    best_S_init: float,
    edge_ids: List[str],
    C: float,
    device,
    dtype,
):
    """
    Resume an IndGP BO run by reloading already-simulated iterations from disk.

    Reads od.xml (OD vector) and link_measure_compare.csv (simulated vs GT flows)
    from each iter_* subfolder under sim_base and reconstructs the training tensors
    and convergence history up to the last completed iteration.

    best_S_init and returned best_S are in nRMSE² units (= Σ_l f_l).

    Returns
    -------
    train_X, train_Y, best_S, S_hist, start_iter
    """
    train_X    = train_X_init.clone()
    train_Y    = train_Y_init.clone()
    best_S     = best_S_init
    S_hist     = [best_S_init]
    start_iter = 0

    for _it in range(n_bo_iters):
        od_xml      = Path(sim_base) / f"iter_{_it}" / "od.xml"
        compare_csv = Path(sim_base) / f"iter_{_it}" / "link_measure_compare.csv"
        if not od_xml.exists() or not compare_csv.exists():
            break

        tree    = _ET.parse(str(od_xml))
        rows    = [el.attrib for el in tree.getroot().iter("tazRelation")]
        _counts = pd.DataFrame(rows)["count"].astype(float).values
        _x      = torch.tensor(_counts, device=device, dtype=dtype).view(1, -1)

        _df = pd.read_csv(str(compare_csv)).set_index("edge_id")
        per_edge_errors = []
        for eid in edge_ids:
            if eid in _df.index:
                gt_val  = float(_df.loc[eid, "interval_nVehContrib_gt"])
                sim_val = float(_df.loc[eid, "interval_nVehContrib_sim"])
                per_edge_errors.append(
                    C * (sim_val - gt_val) ** 2 if gt_val > 0 else np.nan
                )
            else:
                per_edge_errors.append(np.nan)

        _y    = torch.tensor(per_edge_errors, device=device, dtype=dtype).view(1, -1)
        _loss = float(np.nansum(per_edge_errors))  # nRMSE² = Σ_l f_l

        train_X = torch.cat([train_X, _x], dim=0)
        train_Y = torch.cat([train_Y, _y], dim=0)
        if _loss < best_S:
            best_S = _loss
        S_hist.append(best_S)
        start_iter = _it + 1

    return train_X, train_Y, best_S, S_hist, start_iter


def log_indgp_hyperparameters(agg_model, edge_ids: List[str], iteration: int, hp_log_path):
    """
    Append one row per edge of GP hyperparameters to a CSV file.

    Parameters
    ----------
    agg_model   : LinearAggregationModel (ModelListGP-like with per-edge GPs)
    edge_ids    : list of edge ID strings
    iteration   : current BO iteration index
    hp_log_path : Path — CSV is created on first call, appended thereafter
    """
    hp   = extract_independent_gp_hyperparameters(agg_model, edge_ids=edge_ids)
    rows = []
    for row in hp["per_edge"]:
        ls      = row["lengthscale"]
        ls_list = ls if isinstance(ls, list) else [float(ls)]
        rows.append({
            "iteration":        iteration,
            "edge_id":          row["edge_id"],
            "outputscale":      row["outputscale"],
            "noise":            row["noise"],
            "lengthscale_mean": float(np.mean(ls_list)),
            "lengthscale_min":  float(np.min(ls_list)),
            "lengthscale_max":  float(np.max(ls_list)),
            **{f"lengthscale_dim_{i}": float(v) for i, v in enumerate(ls_list)},
        })
    hp_log_path = Path(hp_log_path)
    pd.DataFrame(rows).to_csv(hp_log_path, mode="a", header=not hp_log_path.exists(), index=False)
