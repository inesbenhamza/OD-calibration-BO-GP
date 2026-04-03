#!/usr/bin/env python3
"""
Compute a short statistical summary of final S (best aggregated error) across restarts
for each method (vanilla BO, IND-GP squared). Reports both arithmetic
and geometric mean (geometric = value of the log-space convergence curve at final iter).
Optionally plot a boxplot.

Usage:
  python scripts/summarize_final_s_across_restarts.py
  python scripts/summarize_final_s_across_restarts.py --base output/quickstart_bo_vanilla_matern-2p5
  python scripts/summarize_final_s_across_restarts.py --base output/quickstart_bo_vanilla_matern-2p5 --plot --out-dir output/indgp_hyperparameters_comparison
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "output" / "quickstart_bo_vanilla_matern-2p5"

# Folder names under base that contain results/restart_*_seed-*/convergence.csv
DEFAULT_METHODS = ["vanilla_bo", "indgp_bo_squared"]


def get_final_s_from_convergence_csv(csv_path):
    """Read convergence.csv; return final S (last row's best S). Supports 'best_S' or 'best_S_model'."""
    df = pd.read_csv(csv_path)
    if "best_S" in df.columns:
        return float(df["best_S"].iloc[-1])
    if "best_S_model" in df.columns:
        return float(df["best_S_model"].iloc[-1])
    raise ValueError(f"No best_S or best_S_model column in {csv_path}")


def collect_final_s_per_restart(results_dir):
    """Collect final S from each restart_*_seed-*/convergence.csv under results_dir."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    final_s_list = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("restart_"):
            continue
        conv = sub / "convergence.csv"
        if not conv.exists():
            continue
        try:
            s = get_final_s_from_convergence_csv(conv)
            final_s_list.append(s)
        except Exception as e:
            print(f"  Warning: skip {conv}: {e}")
    return final_s_list


def summarize(values):
    """Return dict with mean_arithmetic, mean_geometric, std, median, min, max, count."""
    if not values:
        return None
    a = np.array(values, dtype=float)
    a_pos = a[a > 0]
    mean_geom = float(10 ** np.nanmean(np.log10(a_pos))) if len(a_pos) > 0 else np.nan
    return {
        "mean_arithmetic": float(np.mean(a)),
        "mean_geometric": mean_geom,
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "count": len(a),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize final S across restarts per method (vanilla, IND-GP squared)."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Base directory containing method folders (e.g. vanilla_bo, indgp_bo_squared)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Subfolder names under --base (default: vanilla_bo indgp_bo_squared)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a boxplot of final S by method",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for summary CSV and boxplot (default: base dir)",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"Base directory not found: {base}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect final S per method
    records = []
    data_for_plot = {}

    for method in args.methods:
        results_dir = base / method / "results"
        final_s_list = collect_final_s_per_restart(results_dir)
        if not final_s_list:
            print(f"No restarts found for {method} (looked in {results_dir})")
            continue
        st = summarize(final_s_list)
        st["method"] = method
        records.append(st)
        data_for_plot[method] = final_s_list
        print(f"  {method}: n={st['count']}  final S: mean_arithmetic={st['mean_arithmetic']:.6f}±{st['std']:.6f}  mean_geometric={st['mean_geometric']:.6f}  median={st['median']:.6f}  min={st['min']:.6f}  max={st['max']:.6f}")

    if not records:
        print("No data collected. Check --base and --methods.")
        return 1

    # Summary table
    df = pd.DataFrame(records)
    csv_path = out_dir / "final_S_summary_across_restarts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print table for paper
    print("\n" + "=" * 80)
    print("Final S (best aggregated error) across restarts — arithmetic and geometric mean")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    print("Note: mean_geometric = value of the 'Comparison: log-space 95% CI' curve at the final iteration.")
    print("      mean_arithmetic = standard average over restarts.")
    print()
    print("Suggested sentence for paper:")
    for _, r in df.iterrows():
        name = r["method"].replace("_", " ").title()
        print(f"  {name}: mean (arithmetic) final S = {r['mean_arithmetic']:.6f} ± {r['std']:.6f}, mean (geometric) = {r['mean_geometric']:.6f} (n = {int(r['count'])} restarts).")

    # Boxplot
    if args.plot and data_for_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            labels = list(data_for_plot.keys())
            # Clean labels for display
            labels_display = [s.replace("_", " ").replace("indgp", "IND-GP").replace("bo", "BO") for s in labels]
            data = [data_for_plot[k] for k in labels]
            bp = ax.boxplot(data, labels=labels_display, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
            ax.set_ylabel("Final S (best aggregated error)")
            ax.set_xlabel("Method")
            ax.set_title("Distribution of final S across restarts")
            ax.set_yscale("log")
            plt.tight_layout()
            plot_path = out_dir / "boxplot_final_S_by_method.pdf"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved: {plot_path}")
        except ImportError:
            print("matplotlib not available; skip --plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
