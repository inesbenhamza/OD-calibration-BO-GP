"""
Convergence comparison on quickstart_grid_3 (n_init=5, 50 BO iters, 5 seeds):
IndGP vs route-overlap-weighted GraphGP (product kernel, "GraphGP-Matern")
vs the new shared+private GraphGP.

Same methodology as plot_grid2d_ninit5_comparison.py / the notebook's
runs_grid2d convention: sqrt(best_S)=nRMSE, pad-to-longest with
nanmean/nanstd, mean +/- 0.5*std band, defensive np.minimum.accumulate.
Safe to run while the shared+private runs are still finishing -- seeds that
haven't reached iteration 50 yet just contribute NaN past their current
iteration and get excluded from the mean/std there via nanmean/nanstd.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn")
output_root_grid3 = BASE / "output" / "quickstart_grid_3_graph"

runs_grid3 = {
    "ScalarGP": (
        output_root_grid3 / "ScalarGP_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "IndGP": (
        output_root_grid3 / "indgp_signed_residual_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "Route-overlap (GraphGP-Matern)": (
        output_root_grid3 / "graph_bo_matern_signed_residual_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "Shared+private (new)": (
        output_root_grid3 / "graph_bo_sharedprivate_signed_residual_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
}

colors_grid3 = ["forestgreen", "steelblue", "crimson", "darkorange"]

eps = 1e-12


def main():
    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, (results_dir, fname, col, sqrt)), color in zip(runs_grid3.items(), colors_grid3):
        curves = []
        for csv in sorted(results_dir.glob(f"restart_*/{fname}")):
            df = pd.read_csv(csv)
            vals = df[col].values.astype(float)
            if sqrt:
                vals = np.sqrt(np.clip(vals, 0, None))
            curve = np.minimum.accumulate(np.where(np.isnan(vals), np.inf, vals))
            curve = np.where(np.isinf(curve), np.nan, curve)
            curves.append(curve)

        if not curves:
            print(f"[SKIP] {label}: no files in {results_dir}")
            continue

        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0, ddof=1)
        lower = np.clip(mean_curve - 0.5 * std_curve, eps, None)
        upper = np.clip(mean_curve + 0.5 * std_curve, eps, None)
        iters = np.arange(max_len)

        ax.plot(iters, np.clip(mean_curve, eps, None),
                label=f"{label} (n={len(curves)})", color=color, lw=2)
        ax.fill_between(iters, lower, upper, color=color, alpha=0.25,
                         where=~np.isnan(lower) & ~np.isnan(upper))

    ax.set_yscale("log")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best nRMSE (log scale)")
    ax.set_title("Convergence comparison — quickstart_grid_3 ($d=16$, $L=34$, n_init=5, 5 seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    out_path = BASE / "scripts" / "grid3_ninit5_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
