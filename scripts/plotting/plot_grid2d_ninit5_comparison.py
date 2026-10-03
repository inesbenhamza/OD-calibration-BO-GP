"""
Convergence comparison on grid_2d_3x3 (n_init=5, 50 BO iters, 5 seeds):
ScalarGP vs IndGP vs route-overlap-weighted GraphGP (product kernel) vs the
new shared+private GraphGP.

Matches the plotting methodology already used in
notebook_for_generated_network/ICMkernel_grid3.ipynb (same run_grid2d-style
dict, sqrt(best_S)=nRMSE, pad-to-longest with nanmean/nanstd, mean +/- 0.5*std
band, defensive np.minimum.accumulate), just pointed at the ninit5 (5 seed,
50 iter, matched-budget) result paths instead of the vanilla 100-iter set.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/inesbenhamza/Desktop/Sumo_od_calibration_bn")
output_root_grid2d = BASE / "output" / "grid_2d_3x3_bo_vanilla_matern-2p5"

# label -> (results_dir, filename, column, sqrt-transform). Fixed order = fixed
# color assignment below -- same convention as the notebook's runs_grid2d dict.
runs_grid2d = {
    "ScalarGP": (
        output_root_grid2d / "ScalarGP_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "IndGP": (
        output_root_grid2d / "IndGP_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "Route-overlap (GraphGP-Matern)": (
        output_root_grid2d / "GraphGP_matern_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "Shared+private (Gram B)": (
        output_root_grid2d / "SharedPrivateGraphGP_mcEI_ninit5" / "results",
        "convergence.csv", "best_S", True,
    ),
    "Shared+private (matern_spectral B, ablation)": (
        output_root_grid2d / "SharedPrivateGraphGP_mcEI_ninit5_matern_spectral" / "results",
        "convergence.csv", "best_S", True,
    ),
}

# same fixed color sequence as the notebook (forestgreen, steelblue, crimson,
# darkorange, purple, ...) so this figure stays consistent if the other rows
# (JacMean/RidgeMean/JacPrior +IndGP) get added back in later.
colors_grid2d = ["forestgreen", "steelblue", "crimson", "darkorange", "purple"]

eps = 1e-12


def main():
    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, (results_dir, fname, col, sqrt)), color in zip(runs_grid2d.items(), colors_grid2d):
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
    ax.set_title("Convergence comparison — grid_2d_3x3 ($d=12$, $L=24$, n_init=5, 5 seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    out_path = BASE / "scripts" / "grid2d_ninit5_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
