"""
Leave-One-Out (LOO) surrogate validation for all GP models.

For each held-out point (x_i, S_i):
  - Train the GP on the remaining N-1 points
  - Predict scalar aggregated S at x_i  →  mu_pred, sigma_pred
  - Compute RMSE, 95% coverage, NLPD

Saves:
  - loo_metrics.csv        : one row per model with RMSE / Coverage / NLPD
  - loo_scatter.png        : mu_pred vs S_true scatter per model
  - loo_calibration.png    : sorted z-scores (should be N(0,1) if well-calibrated)

Usage
-----
python scripts/loo_surrogate_validation.py \
    --results_dir  output/quickstart_vanillabo_matern-2p5/results/restart_1_seed-42 \
    --model_name   vanillabo \
    --kernel       matern-2p5 \
    --net_xml      network/quickstart/net.xml          # only needed for graph_gp
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# ---- project root on sys.path ------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.gp_models import (
    initialize_vanillabo_model,
    initialize_independent_gp_models_with_modellist,
    make_linear_aggregation_model_from_error_gps,
)
from src.optimization.io import load_bo_results
from MOGP.helpers_MOGP import (
    initialize_icm_gp,
    LinearAggregationICM,
    fit_gpytorch_mll_with_adam_fallback,
)


# ==============================================================================
# Helpers
# ==============================================================================

def _aggregate_S(Y_errors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum of per-edge errors → scalar S.  NaN edges are skipped."""
    valid = ~torch.isnan(Y_errors)                          # [N, L]
    w = weights.unsqueeze(0).expand_as(Y_errors)            # [N, L]
    num   = (Y_errors.nan_to_num(0.0) * w).sum(-1)         # [N]
    denom = (valid.float() * w).sum(-1).clamp_min(1e-12)   # [N]
    total_w = w.sum(-1)
    return num / denom * total_w                            # re-scale to full weight sum


def _standardize_per_edge(Y: torch.Tensor):
    """Returns (Y_std, y_mean [1,L], y_std [1,L]).  Operates on non-NaN entries."""
    y_mean = torch.nanmean(Y, dim=0, keepdim=True)                     # [1, L]
    y_std  = Y.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6) # [1, L]
    return (Y - y_mean) / y_std, y_mean, y_std


def _filter_nan_samples(X: torch.Tensor, Y: torch.Tensor):
    """Drop any row that has a NaN in Y (used for ICM which needs full rows)."""
    mask = ~torch.isnan(Y).any(dim=1)
    return X[mask], Y[mask]


# ==============================================================================
# Per-model LOO predictors  →  return (mu_pred, sigma_pred) in RAW S space
# ==============================================================================

def _predict_vanillabo(X_train, Y_train, X_test, weights, kernel):
    S_train = _aggregate_S(Y_train, weights).unsqueeze(-1)          # [N-1, 1]
    # filter any inf/nan
    ok = torch.isfinite(S_train).squeeze(-1)
    model = initialize_vanillabo_model(X_train[ok], S_train[ok], kernel)
    mll   = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception:
        pass
    model.eval()
    with torch.no_grad():
        post = model.posterior(X_test)                              # Standardize un-transforms
    return post.mean.squeeze().item(), post.variance.sqrt().squeeze().item()


def _predict_indgp(X_train, Y_train, X_test, weights, kernel):
    model_list, mlls = initialize_independent_gp_models_with_modellist(
        X_train, Y_train, kernel
    )
    for mll in mlls:
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
    weights_np = weights.cpu().numpy()
    # make_linear_aggregation_model_from_error_gps expects uniform or flow weights
    agg = make_linear_aggregation_model_from_error_gps(
        model_list_gp_errors=model_list,
        weights=weights_np,
        ground_truth_flows=None,
        expect_flow_weights=False,
    )
    agg.eval()
    with torch.no_grad():
        post = agg.posterior(X_test)                                # already raw space
    return post.mean.squeeze().item(), post.variance.sqrt().squeeze().item()


def _predict_icm(X_train, Y_train, X_test, weights, rank=1):
    X_tr, Y_tr = _filter_nan_samples(X_train, Y_train)
    if len(X_tr) < 2:
        return float("nan"), float("nan")
    Y_std, y_mean, y_std = _standardize_per_edge(Y_tr)
    model, mll = initialize_icm_gp(X_tr, Y_std, rank=rank)
    fit_gpytorch_mll_with_adam_fallback(mll, verbose=False)
    model.eval()
    w_eff = weights * y_std.squeeze(0)                              # [L]
    agg = LinearAggregationICM(model, w_eff)
    with torch.no_grad():
        post = agg.posterior(X_test)                                # Z-space
    mu_Z    = post.mean.squeeze().item()
    sigma   = post.variance.sqrt().squeeze().item()
    # Un-shift: S_raw = S_Z + sum(w * mu_per_edge)
    shift   = (weights * y_mean.squeeze(0)).sum().item()
    return mu_Z + shift, sigma


def _predict_graph_gp(
    X_train, Y_train, X_test, weights,
    net_xml_path, edge_ids, input_dim,
    adjacency_type, graph_kernel_type,
):
    from MOGP.graph import run_graph_multitask_gp, GraphKernelAggregationModel

    L = len(edge_ids)
    dtype = X_train.dtype
    device = X_train.device

    # stack (x_n, task_l) for each training point and each task
    N_tr = X_train.shape[0]
    task_col = torch.arange(L, dtype=dtype, device=device)
    x_rep    = X_train.unsqueeze(1).expand(N_tr, L, input_dim)     # [N, L, d]
    t_rep    = task_col.view(1, L, 1).expand(N_tr, L, 1)           # [N, L, 1]
    X_stack  = torch.cat([x_rep, t_rep], dim=-1).reshape(N_tr * L, input_dim + 1)
    Y_stack  = Y_train.reshape(N_tr * L, 1)                         # [N*L, 1]

    # drop NaN rows
    ok = ~torch.isnan(Y_stack.squeeze(-1))
    X_stack, Y_stack = X_stack[ok], Y_stack[ok]

    model, _mll, _info, _method = run_graph_multitask_gp(
        net_xml_path  = net_xml_path,
        edge_ids      = edge_ids,
        train_X       = X_stack,
        train_Y       = Y_stack,
        input_dim     = input_dim,
        adjacency_type= adjacency_type,
        graph_kernel_type = graph_kernel_type,
        standardize   = True,
        device        = device,
    )
    model.eval()
    agg = GraphKernelAggregationModel(model, weights)
    with torch.no_grad():
        post = agg.posterior(X_test)                                # raw space (un-standardized inside)
    return post.mean.squeeze().item(), post.variance.sqrt().squeeze().item()


# ==============================================================================
# Core LOO loop
# ==============================================================================

def run_loo(
    X_all: torch.Tensor,          # [N, d]
    Y_errors_all: torch.Tensor,   # [N, L] or [N] for vanilla
    weights: torch.Tensor,        # [L]
    model_name: str,
    kernel: str = "matern-2p5",
    net_xml_path: str = None,
    edge_ids=None,
    adjacency_type: str = "shared_node_weighted",
    graph_kernel_type: str = "diffusion",
    rank: int = 1,
    verbose: bool = True,
):
    N = X_all.shape[0]
    d = X_all.shape[1]

    # Compute true scalar S for every point (used as ground truth)
    if Y_errors_all.dim() == 1:
        S_true = Y_errors_all.tolist()
    else:
        S_true = _aggregate_S(Y_errors_all, weights).tolist()

    mu_preds    = []
    sigma_preds = []

    for i in range(N):
        if verbose:
            print(f"  fold {i+1}/{N}", end="\r", flush=True)

        idx_train = [j for j in range(N) if j != i]
        X_tr = X_all[idx_train]
        Y_tr = Y_errors_all[idx_train] if Y_errors_all.dim() > 1 else Y_errors_all[idx_train].unsqueeze(-1).expand(-1, 1)
        X_te = X_all[i:i+1]            # [1, d]

        if model_name == "vanillabo":
            mu, sig = _predict_vanillabo(X_tr, Y_tr, X_te, weights, kernel)

        elif model_name == "independent_gp":
            mu, sig = _predict_indgp(X_tr, Y_tr, X_te, weights, kernel)

        elif model_name in ("mogp", "icm"):
            mu, sig = _predict_icm(X_tr, Y_tr, X_te, weights, rank=rank)

        elif model_name == "graph_gp":
            assert net_xml_path is not None, "net_xml_path required for graph_gp"
            assert edge_ids     is not None, "edge_ids required for graph_gp"
            mu, sig = _predict_graph_gp(
                X_tr, Y_tr, X_te, weights,
                net_xml_path, edge_ids, d,
                adjacency_type, graph_kernel_type,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        mu_preds.append(mu)
        sigma_preds.append(sig)

    if verbose:
        print()

    return np.array(S_true), np.array(mu_preds), np.array(sigma_preds)


# ==============================================================================
# Metrics
# ==============================================================================

def compute_metrics(S_true, mu_pred, sigma_pred, ci=1.96):
    """
    RMSE            : sqrt(mean((mu - S_true)^2))
    Coverage (95%)  : fraction of true values inside [mu ± ci*sigma]
    NLPD            : mean negative log predictive density (lower = better)
    """
    ok = np.isfinite(S_true) & np.isfinite(mu_pred) & np.isfinite(sigma_pred) & (sigma_pred > 0)
    S, mu, sig = S_true[ok], mu_pred[ok], sigma_pred[ok]

    rmse     = float(np.sqrt(np.mean((mu - S) ** 2)))
    coverage = float(np.mean(np.abs(mu - S) <= ci * sig))
    nlpd     = float(np.mean(0.5 * np.log(2 * math.pi * sig**2) + (S - mu)**2 / (2 * sig**2)))

    return {"RMSE": rmse, "Coverage_95": coverage, "NLPD": nlpd, "n_valid": int(ok.sum())}


# ==============================================================================
# Plots
# ==============================================================================

def plot_scatter(results: dict, save_path: Path):
    """mu_pred vs S_true per model — should be on the diagonal."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, (name, (S_true, mu_pred, sigma_pred)) in zip(axes[0], results.items()):
        ok = np.isfinite(S_true) & np.isfinite(mu_pred)
        ax.errorbar(
            S_true[ok], mu_pred[ok],
            yerr=1.96 * sigma_pred[ok],
            fmt="o", alpha=0.6, markersize=4, linewidth=0.6,
            label="±1.96σ",
        )
        lo = min(S_true[ok].min(), mu_pred[ok].min()) * 0.95
        hi = max(S_true[ok].max(), mu_pred[ok].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        ax.set_xlabel("True S")
        ax.set_ylabel("Predicted S")
        ax.set_title(name)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved → {save_path}")


def plot_calibration(results: dict, save_path: Path):
    """
    Standardized residuals z = (S_true - mu) / sigma.
    For a well-calibrated GP the empirical CDF should match N(0,1).
    """
    from scipy.stats import norm

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, (name, (S_true, mu_pred, sigma_pred)) in zip(axes[0], results.items()):
        ok = np.isfinite(S_true) & np.isfinite(mu_pred) & np.isfinite(sigma_pred) & (sigma_pred > 0)
        z = (S_true[ok] - mu_pred[ok]) / sigma_pred[ok]
        z_sorted = np.sort(z)
        p_emp    = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
        p_ideal  = norm.cdf(z_sorted)

        ax.plot(p_ideal, p_emp, "o-", markersize=4, label="empirical")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.set_xlabel("Theoretical CDF N(0,1)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"{name}\n(PIT plot)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Calibration plot saved → {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",      required=True,
                   help="Path to a restart results dir (contains bo_results_restart_*.pkl or bo_results.pkl)")
    p.add_argument("--model_name",       required=True,
                   choices=["vanillabo", "independent_gp", "mogp", "icm", "graph_gp"])
    p.add_argument("--kernel",           default="matern-2p5")
    p.add_argument("--net_xml",          default=None,
                   help="Path to SUMO net.xml  (required for graph_gp)")
    p.add_argument("--adjacency_type",   default="shared_node_weighted",
                   choices=["shared_node_weighted", "connections"])
    p.add_argument("--graph_kernel_type",default="diffusion",
                   choices=["diffusion", "regularized_laplacian", "random_walk"])
    p.add_argument("--rank",             type=int, default=1,
                   help="ICM rank (mogp/icm only)")
    p.add_argument("--use_flow_weights", action="store_true",
                   help="Use flow-proportional aggregation weights")
    p.add_argument("--n_init",           type=int, default=None,
                   help="Use only the first n_init Sobol points for LOO (default: all)")
    p.add_argument("--out_dir",          default=None,
                   help="Where to save outputs (default: same as results_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load saved BO results ----------------------------------------------
    print(f"Loading results from {results_dir} ...")
    bo_results = load_bo_results(results_dir)

    X_all      = bo_results.all_X          # list[Tensor[d]] or Tensor[N, d]
    Y_all      = bo_results.all_Y_errors   # list[Tensor[L]] or Tensor[N, L]

    # normalise to tensors
    dtype  = torch.float64
    device = torch.device("cpu")

    if isinstance(X_all, list):
        X_all = torch.stack(X_all).to(dtype=dtype, device=device)
    else:
        X_all = X_all.to(dtype=dtype, device=device)

    if isinstance(Y_all, list):
        Y_all = torch.stack(Y_all).to(dtype=dtype, device=device)
    else:
        Y_all = Y_all.to(dtype=dtype, device=device)

    # optionally keep only the first n_init points (initial Sobol design)
    if args.n_init is not None:
        X_all = X_all[:args.n_init]
        Y_all = Y_all[:args.n_init]

    N, d = X_all.shape
    print(f"Dataset: N={N} points,  d={d},  Y shape={tuple(Y_all.shape)}")

    if N < 3:
        print("ERROR: Need at least 3 points for LOO. Aborting.")
        sys.exit(1)

    # ---- weights -----------------------------------------------------------
    L = Y_all.shape[1] if Y_all.dim() == 2 else 1
    if args.use_flow_weights:
        # flow-proportional weights require gt_flows — not available here without
        # the original config, so fall back to uniform with a warning.
        print("WARNING: --use_flow_weights requires gt_flows; falling back to uniform.")
    weights = torch.full((L,), 1.0 / L, dtype=dtype, device=device)

    # ---- edge ids for graph_gp ---------------------------------------------
    edge_ids = None
    if args.model_name == "graph_gp":
        if args.net_xml is None:
            print("ERROR: --net_xml is required for graph_gp")
            sys.exit(1)
        # Try to infer edge ids from metadata or ask the user to pass them.
        # For now we attempt to read from a saved metadata.json.
        meta_path = results_dir / "metadata.json"
        if meta_path.exists():
            import json
            meta = json.load(open(meta_path))
            edge_ids = meta.get("edge_ids", None)
        if edge_ids is None:
            print("ERROR: Could not infer edge_ids from metadata.json. "
                  "Add 'edge_ids' to your metadata or pass them programmatically.")
            sys.exit(1)

    # ---- run LOO -----------------------------------------------------------
    print(f"\nRunning LOO for model '{args.model_name}'  (N={N} folds) ...")
    S_true, mu_pred, sigma_pred = run_loo(
        X_all       = X_all,
        Y_errors_all= Y_all,
        weights     = weights,
        model_name  = args.model_name,
        kernel      = args.kernel,
        net_xml_path= args.net_xml,
        edge_ids    = edge_ids,
        adjacency_type   = args.adjacency_type,
        graph_kernel_type= args.graph_kernel_type,
        rank        = args.rank,
        verbose     = True,
    )

    # ---- metrics -----------------------------------------------------------
    metrics = compute_metrics(S_true, mu_pred, sigma_pred)
    print(f"\n{'='*40}")
    print(f"  Model     : {args.model_name}")
    print(f"  RMSE      : {metrics['RMSE']:.6f}")
    print(f"  Coverage  : {metrics['Coverage_95']*100:.1f}%  (target 95%)")
    print(f"  NLPD      : {metrics['NLPD']:.4f}")
    print(f"  n_valid   : {metrics['n_valid']}/{N}")
    print(f"{'='*40}\n")

    # save CSV
    df = pd.DataFrame({
        "model":       args.model_name,
        "S_true":      S_true,
        "mu_pred":     mu_pred,
        "sigma_pred":  sigma_pred,
    })
    df.to_csv(out_dir / "loo_predictions.csv", index=False)

    metrics_df = pd.DataFrame([{"model": args.model_name, **metrics}])
    metrics_csv = out_dir / "loo_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved → {metrics_csv}")

    # ---- plots -------------------------------------------------------------
    raw_results = {args.model_name: (S_true, mu_pred, sigma_pred)}
    plot_scatter(raw_results,     out_dir / "loo_scatter.png")
    try:
        plot_calibration(raw_results, out_dir / "loo_calibration.png")
    except ImportError:
        print("scipy not found — skipping calibration (PIT) plot.")


if __name__ == "__main__":
    main()
