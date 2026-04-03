"""
Bayesian Optimization pipeline for SUMO OD calibration.

Uses src.simulation.sumo_runner (run_simulation_for_od, parse_simulation_outputs) and
src.utils.edge_flow_grouping (compute_grouped_loss, group_edge_dataframe) for a clean separation
of simulation, parsing, and metric logic. The BO loop delegates to
BayesianOptimizationLoop.
"""

import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
import pandas as pd
import numpy as np
import pprint

from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

# ── traffic-condition demand scales ─────────────────────────────────────────
# For synthetic networks (e.g. quickstart), time-of-day has no effect.
# Congestion vs free-flow is controlled purely by the GT OD demand level
# relative to road capacity:
#   peak    → full GT demand (1.0x)  → road capacity likely approached → congested
#   offpeak → 30% of GT demand       → flows well below capacity       → free-flow
# The sim time window (sim_start_time / sim_end_time) is left unchanged from config.
_DEMAND_SCALES = {"peak": 1.0, "offpeak": 0.30}


def draw_initial_points(method: str, n: int, dim: int, seed: int,
                        dtype, device) -> torch.Tensor:
    """
    Draw *n* initial points in [0,1]^dim using the chosen strategy.

    Parameters
    ----------
    method : {'sobol', 'lhs', 'random'}
    n      : number of points
    dim    : dimension of OD space
    seed   : RNG seed
    dtype, device : torch tensor settings

    Returns
    -------
    Tensor of shape (n, dim), values in [0, 1]
    """
    if method == "sobol":
        engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
        return engine.draw(n).to(dtype=dtype, device=device)

    if method == "lhs":
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=dim, seed=seed)
        pts = sampler.random(n=n)                     # shape (n, dim), in [0,1]
        return torch.tensor(pts, dtype=dtype, device=device)

    if method == "random":
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.rand(n, dim, dtype=dtype, device=device, generator=rng)

    raise ValueError(f"Unknown init_method '{method}'. Choose sobol, lhs, or random.")

from src.optimization import BayesianOptimizationLoop, run_multiple_bo_restarts
from src.simulation.data_loader import load_kwargs_config, od_xml_to_df
from src.simulation.sumo_runner import run_simulation_for_od, parse_simulation_outputs
from src.utils.seed import set_seed
from src.utils.edge_flow_grouping import (
    EDGE_GROUPS,
    get_representative_edges,
    group_edge_dataframe,
    compute_grouped_loss,
)
from src.utils.link_flow_analysis import (
    compute_squared_metric_all_edge,
    compute_squared_metric_per_edge,
)

# SUMO environment
sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    macos_path = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo"
    if os.path.exists(macos_path):
        sumo_home = macos_path
    else:
        default_sumo_paths = [
            "/opt/sumo-1.12/share/sumo",
            "C:/Program Files (x86)/Eclipse/Sumo",
        ]
        sumo_home = next((p for p in default_sumo_paths if os.path.exists(p)), None)
if not sumo_home:
    sys.exit("SUMO_HOME is not set and no default path exists.")
os.environ["SUMO_HOME"] = sumo_home
os.environ["LIBSUMO_AS_TRACI"] = "1"
SUMO_HOME = Path(os.environ["SUMO_HOME"])
tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
if os.path.exists(tools_path):
    sys.path.append(tools_path)
else:
    sys.exit(f"Cannot find SUMO tools at {tools_path}")
SUMO_ROOT = Path(os.environ["SUMO_HOME"]).parents[1]
SUMO_BIN = str(SUMO_ROOT / "bin")
os.environ["PATH"] = SUMO_BIN + os.pathsep + os.environ.get("PATH", "")
base_path = str(project_root)
if " " in base_path:
    raise ValueError("base_path should not contain any spaces.")
os.chdir(base_path)


def _grouped_error_metric(gt_edge_data, sim_edge_data, edge_ids):
    """Wrapper: grouped loss for use as error_metric_func (edge_ids ignored)."""
    return compute_grouped_loss(gt_edge_data, sim_edge_data)


def main():
    parser = argparse.ArgumentParser(description="OD calibration (modular pipeline)")
    parser.add_argument("--network_name", type=str, default="1ramp", choices=["2corridor", "3junction", "4smallRegion", "5fullRegion", "quickstart"])
    parser.add_argument("--model_name", type=str, default="vanillabo", choices=["vanillabo", "independent_gp", "mogp"])
    parser.add_argument("--kernel", type=str, default="matern-2p5", choices=["matern-1p5", "matern-2p5", "rbf"])
    parser.add_argument("--date", type=int, default=221014)
    parser.add_argument("--hour", type=str, default="08-09", choices=["06-07", "08-09", "17-18"])
    parser.add_argument("--eval_measure", type=str, default="count", choices=["count", "speed"])
    parser.add_argument("--routes_per_od", type=str, default="single", choices=["single", "multiple"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_max", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--n_iterations", type=int, default=30)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--error_metric", type=str, default="squared", choices=["squared"])
    parser.add_argument("--weights", type=str, default="uniform", choices=["uniform", "flow_proportional"])
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    parser.add_argument("--use_grouped_edges", action="store_true", help="Use grouped edges (L12, L14, ...) for loss and BO")
    parser.add_argument(
        "--init_method", type=str, default="sobol",
        choices=["sobol", "lhs", "random"],
        help="Strategy for drawing initial evaluation points (sobol | lhs | random)",
    )
    parser.add_argument(
        "--traffic_condition", type=str, default=None,
        choices=["peak", "offpeak"],
        help=(
            "Traffic regime shortcut: 'peak' keeps GT demand as-is (congested), "
            "'offpeak' scales GT demand by 0.3 (free-flow). "
            "Overridden by --demand_scale if both are provided."
        ),
    )
    parser.add_argument(
        "--demand_scale", type=float, default=None,
        help=(
            "Multiply every GT OD count by this factor before simulation. "
            "E.g. 0.3 = free-flow, 1.0 = default congestion, 2.0 = heavy congestion. "
            "Takes precedence over --traffic_condition."
        ),
    )
    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)
    date = args.date
    hour = args.hour
    model_name = args.model_name
    kernel = args.kernel
    network_name = args.network_name
    eval_measure = args.eval_measure
    routes_per_od = args.routes_per_od
    cpu_max = args.cpu_max
    n_iterations = args.n_iterations
    n_restarts = args.n_restarts

    if network_name == "quickstart":
        sim_setup_filename = "sim_setup.json"
    else:
        sim_setup_filename = f"sim_setup_network_{network_name}.json"

    config = load_kwargs_config(
        base_path,
        model_name=model_name,
        sim_setup_filename=sim_setup_filename,
        kernel=kernel,
    )
    config["seed"] = seed

    # ── apply traffic-condition overrides ───────────────────────────────────
    # --demand_scale takes precedence; --traffic_condition is a shortcut
    if args.demand_scale is not None:
        config["_demand_scale"] = args.demand_scale
        print(f"[demand_scale={args.demand_scale}] GT OD counts multiplied by {args.demand_scale}")
    elif args.traffic_condition is not None:
        tc = args.traffic_condition
        config["_demand_scale"] = _DEMAND_SCALES[tc]
        print(f"[traffic_condition={tc}] GT OD demand scaled by {_DEMAND_SCALES[tc]}")
    else:
        config["_demand_scale"] = 1.0

    # ── tag output directory with init method and traffic/demand condition ───
    tag_parts = [args.init_method]
    if args.demand_scale is not None:
        tag_parts.append(f"scale{args.demand_scale}")
    elif args.traffic_condition:
        tag_parts.append(args.traffic_condition)
    run_tag = "_".join(tag_parts)
    config["simulation_run_path"] = f"{config['simulation_run_path']}_{run_tag}"

    if model_name == "independent_gp":
        config["simulation_run_path"] = f"{config['simulation_run_path']}_{args.error_metric}"
    Path(config["simulation_run_path"]).mkdir(parents=True, exist_ok=True)
    pprint.pprint(dict(config))

    print(f"Reading: {config['file_gt_od']}")
    gt_od_df = od_xml_to_df(config["file_gt_od"])
    dim_od = gt_od_df.shape[0]
    print(f"Number of OD pairs: {dim_od}")
    routes_df = pd.read_csv(config["fixed_routes"], index_col=0)

    simulation_gt_run_path = f"{config['simulation_run_path']}/ground_truth"
    prefix_output_gt = f"{simulation_gt_run_path}/sim"
    sim_edge_out_gt = f"{prefix_output_gt}_{config['EDGE_OUT_STR']}"
    new_od_xml = f"{simulation_gt_run_path}/od.xml"
    Path(simulation_gt_run_path).mkdir(parents=True, exist_ok=True)

    base_od = gt_od_df.copy()
    curr_od = base_od["count"].astype(float).to_numpy()
    # scale GT demand to simulate congested (1.0) or free-flow (0.3) regime
    demand_scale = config.get("_demand_scale", 1.0)
    curr_od = curr_od * demand_scale
    base_od["count"] = curr_od
    base_od = base_od.rename(columns={"fromTaz": "from", "toTaz": "to"})

    from src.simulation.sumo_runner import create_taz_xml, simulate_od, parse_loop_data_xml_to_pandas

    create_taz_xml(new_od_xml, base_od, config["od_duration_sec"], base_path)

    true_sensor_file_name = f"gt_link_data_{network_name}_{date}_{hour}.csv"
    sensor_data_path = base_path + f"/sensor_data/{date}/" + true_sensor_file_name

    if network_name == "quickstart" or not Path(sensor_data_path).exists():
        print("Using ground truth simulation output (no sensor data file found)")
        simulate_od(
            str(new_od_xml),
            prefix_output_gt,
            base_path,
            str(config["net_xml"]),
            str(config["taz2edge_xml"]),
            str(config["additional_xml"]),
            routes_df,
            str(config["sim_end_time"]),
            config["TRIPS2ODS_OUT_STR"],
        )
        gt_loop_stats, _, _ = parse_loop_data_xml_to_pandas(
            base_path, sim_edge_out_gt, prefix_output_gt, str(config["SUMO_PATH"])
        )
        gt_edge_data = gt_loop_stats
        if "edge_id" in gt_loop_stats.columns:
            all_edge_ids = gt_loop_stats["edge_id"].tolist()
        else:
            try:
                import sumolib
                net = sumolib.net.readNet(str(config["net_xml"]))
                all_edge_ids = [edge.getID() for edge in net.getEdges()]
            except ImportError:
                all_edge_ids = list(gt_loop_stats.columns) if hasattr(gt_loop_stats, "columns") else []
    else:
        sensor_measure_gt = pd.read_csv(sensor_data_path)
        all_edge_ids = sensor_measure_gt["link_id"].tolist()
        print(f"Number of sensors: {len(all_edge_ids)}")
        gt_edge_data = sensor_measure_gt

    if args.use_grouped_edges:
        try:
            edge_ids = get_representative_edges()
            gt_edge_data = group_edge_dataframe(gt_edge_data)
        except (ValueError, KeyError):
            print("Warning: could not apply edge grouping; using all edges.")
            edge_ids = all_edge_ids
    else:
        edge_ids = all_edge_ids

    gt_od_vals = gt_od_df["count"].values
    device = torch.device(args.device)
    dtype = torch.double
    od_bound_start = config.get("od_bound_start", 0.0)
    od_bound_end = config.get("od_bound_end", 2000.0)
    bounds = torch.tensor(
        [[od_bound_start] * dim_od, [od_bound_end] * dim_od],
        device=device,
        dtype=dtype,
    )
    n_init_search = config["n_init_search"]

    if model_name == "vanillabo":
        error_metric_func = _grouped_error_metric if args.use_grouped_edges else compute_squared_metric_all_edge
    elif model_name in ["independent_gp", "mogp"]:
        error_metric_func = compute_squared_metric_per_edge
    else:
        error_metric_func = compute_squared_metric_all_edge

    print(f"\n[DEBUG] BO configuration: model={model_name}, use_grouped_edges={args.use_grouped_edges}")

    sim_run_path = Path(config["simulation_run_path"])
    path_init = sim_run_path / "initial_search"
    path_init.mkdir(exist_ok=True)
    path_opt = sim_run_path / "bo_iterations"
    path_opt.mkdir(exist_ok=True)
    path_result = sim_run_path / "results"
    path_result.mkdir(exist_ok=True)
    path_init_simul = path_init

    od_pairs = (
        routes_df[["fromTaz", "toTaz"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )





    print(f"RUNNING INITIAL SEARCH ({args.init_method.upper()}, n={n_init_search})")
    train_X_init_norm = draw_initial_points(
        method=args.init_method,
        n=n_init_search,
        dim=dim_od,
        seed=seed,
        dtype=dtype,
        device=device,
    )

    train_Y_errors_init = []
    for i in range(n_init_search):
        print(f"\nEvaluating initial sample {i+1}/{n_init_search}")
        X_real = unnormalize(train_X_init_norm[i : i + 1], bounds).squeeze()
        curr_od = X_real.cpu().numpy()
        sim_dir = path_init_simul / f"{args.init_method}_{i}"

        prefix_output = run_simulation_for_od(
            curr_od,
            od_pairs,
            sim_dir,
            config,
            base_path,
            routes_df,
            seed=seed,
        )
        curr_loop_stats = parse_simulation_outputs(prefix_output, config, base_path)

        if args.use_grouped_edges and model_name == "vanillabo":
            curr_loss = compute_grouped_loss(gt_edge_data, curr_loop_stats)
            train_Y_errors_init.append(float(curr_loss))
        else:
            if args.use_grouped_edges:
                curr_loop_stats = group_edge_dataframe(curr_loop_stats)
            errors = error_metric_func(gt_edge_data, curr_loop_stats, edge_ids)
            if model_name == "vanillabo":
                train_Y_errors_init.append(float(errors))
            else:
                train_Y_errors_init.append(errors)

    train_Y_errors_init = torch.tensor(
        np.array(train_Y_errors_init),
        dtype=dtype,
        device=device,
    )
    print(f"\nInitial search complete. Shape: {train_Y_errors_init.shape}")

    if model_name is not None:
        print("RUNNING OPTIMIZATION LOOP")
        aggregated_results = run_multiple_bo_restarts(
            config=config,
            gt_edge_data=gt_edge_data,
            edge_ids=edge_ids,
            gt_od_vals=gt_od_vals,
            routes_df=routes_df,
            base_path=base_path,
            bounds=bounds,
            device=device,
            dtype=dtype,
            train_X_init=train_X_init_norm,
            train_Y_errors_init=train_Y_errors_init,
            n_restarts=n_restarts,
            n_bo_iterations=n_iterations,
            error_metric_func=error_metric_func,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )

        print("SAVING RESULTS")
        best_result = aggregated_results["overall_best_result"]
        print(f"\nBest result (restart {aggregated_results['overall_best_restart'] + 1}):")
        print(f"  Best S: {best_result.best_S:.6f}")

        all_results = aggregated_results.get("all_results", [])
        if all_results:
            for i, res in enumerate(all_results, start=1):
                restart_seed = 42 + (i - 1) * 1000
                subdir = path_result / f"restart_{i}_seed-{restart_seed}"
                subdir.mkdir(parents=True, exist_ok=True)
                conv_df = pd.DataFrame({
                    "iteration": range(len(res.convergence_curve)),
                    "best_S_model": res.convergence_curve,
                })
                conv_df.to_csv(subdir / "convergence.csv", index=False)

        model_curve = best_result.convergence_curve
        convergence_df = pd.DataFrame({
            "iteration": range(len(model_curve)),
            "best_S_model": model_curve,
        })
        convergence_df.to_csv(path_result / "convergence.csv", index=False)

        if best_result.gp_optimizer_log:
            opt_log = best_result.gp_optimizer_log
            n_adam = opt_log.count("adam")
            pct = 100 * n_adam / len(opt_log)
            print(f"\n[MOGP optimizer log] Adam fallback used in {n_adam}/{len(opt_log)} iterations ({pct:.1f}%)")
            pd.DataFrame({
                "iteration": range(1, len(opt_log) + 1),
                "gp_optimizer": opt_log,
            }).to_csv(path_result / "gp_optimizer_log.csv", index=False)

        from src.optimization.io import save_bo_results
        save_bo_results(
            results=best_result,
            save_dir=path_result,
            restart_idx=None,
            seed=None,
            metadata={
                "model_name": model_name,
                "kernel": kernel,
                "network_name": network_name,
                "best_restart": aggregated_results["overall_best_restart"] + 1,
            },
        )
        print(f"\nResults saved to: {path_result}")


if __name__ == "__main__":
    main()
