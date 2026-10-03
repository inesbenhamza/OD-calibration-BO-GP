# Standard library imports
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# Third-party imports
import pandas as pd


def load_experiment_metadata(config_path: str, sim_setup_filename: str = "sim_setup.json"):
    """
    Load config.json and a chosen sim_setup file from /config/.
    """
    config_file = Path(config_path, "config.json")
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {config_path}")

    config = json.load(open(config_file))
    config["SUMO"] = Path(config["SUMO"])

    sim_setup_file = Path(config_path, sim_setup_filename)
    if not sim_setup_file.exists():
        raise FileNotFoundError(f"{sim_setup_filename} not found in {config_path}")

    sim_setup = json.load(open(sim_setup_file))

    return config, sim_setup


def load_kwargs_config(base_path: str, model_name: str, sim_setup_filename: str = "sim_setup.json", kernel: str = "matern-2p5"):
    """
    Load config + dynamic sim_setup file, and build kwargs_config.
    Unified version that works for both toy and big networks.

    Auto-detects network type based on network_name:
    - Big networks (network_2corridor, etc.): Uses "edge_data.xml" format
    - Toy networks: Uses "edge_data_{network_name}.xml" format

    Parameters
    ----------
    base_path : str
        Base path of the project.
    model_name : str
        Name of the model (e.g., "vanillabo", "independent_gp").
    sim_setup_filename : str, optional
        Name of the sim_setup JSON file. Default: "sim_setup.json"
    kernel : str, optional
        Kernel type for GP models (e.g., "matern-2p5", "matern-1p5", "rbf"). Default: "matern-2p5"

    Returns
    -------
    dictionary 
        Configuration dictionary with all simulation and optimization parameters.
    """
    config_path = Path(base_path, 'config')

    config, sim_setup = load_experiment_metadata(
        config_path,
        sim_setup_filename=sim_setup_filename
    )

    kwargs_config = {}


    # network and kernel config 
    network_name = sim_setup['network_name']
    kwargs_config["network_name"] = network_name
    kwargs_config["model_name"] = model_name or sim_setup.get("model_name", "vanillabo")
    kwargs_config["kernel"] = kernel or sim_setup.get("kernel", "matern-2p5")

    #parameters setup for the ablation study on the effect of the OD bounds 
    kwargs_config["constraint_mode"] = sim_setup.get("constraint_mode", "tight")

    kwargs_config["network_path"] = Path("network", network_name) # network/quickstart if network is quickstart
    kwargs_config["taz2edge_xml"] = Path(base_path, kwargs_config["network_path"], 'taz.xml')
    kwargs_config["net_xml"] = Path(base_path, kwargs_config["network_path"], 'net.xml')

    # Routes file: use config if provided, otherwise default based on network type
    routes_file = sim_setup.get("routes_file")
    if routes_file is None:
        # Auto-detect: big networks typically use "routes.csv", toy networks use "routes_single.csv"
        if "2corridor" in network_name.lower() or "big" in network_name.lower():
            routes_file = "routes.csv"
        else:
            routes_file = "routes_single.csv"
    kwargs_config["fixed_routes"] = Path(base_path, kwargs_config["network_path"], routes_file)

    kwargs_config["file_gt_od"] = Path(base_path, kwargs_config["network_path"], 'od.xml')
    kwargs_config["additional_xml"] = Path(base_path, kwargs_config["network_path"], 'additional.xml')

    # Include kernel in output path (like BO4Mob does)
    kwargs_config["simulation_run_path"] = f"output/{network_name}_{kwargs_config['model_name']}_{kwargs_config['kernel']}"
    
    # Add path keys for prepare_run_paths() function
    kwargs_config["path_init"] = kwargs_config["simulation_run_path"] + "/initial_search"
    kwargs_config["path_opt"] = kwargs_config["simulation_run_path"] + "/bo_iterations"
    kwargs_config["path_result"] = kwargs_config["simulation_run_path"] + "/results"

    # Auto-detect EDGE_OUT_STR format based on network type
    # Big networks: "edge_data.xml", Toy networks: "edge_data_{network_name}.xml"
    if "2corridor" in network_name.lower() or "big" in network_name.lower():
        kwargs_config["EDGE_OUT_STR"] = "edge_data.xml"
    else:
        kwargs_config["EDGE_OUT_STR"] = f'edge_data_{network_name}.xml'

    kwargs_config["TRIPS2ODS_OUT_STR"] = 'trips.xml'
    kwargs_config["SUMO_PATH"] = config["SUMO"]

    kwargs_config["sim_start_time"] = sim_setup['sim_start_time']
    kwargs_config["sim_end_time"] = sim_setup['sim_end_time']
    kwargs_config["sim_stat_freq_sec"] = sim_setup['sim_stat_freq_sec']

    # Warmup exclusion: skip the first stat interval when the simulation starts
    # at t=0, because the network is empty at the start and the first interval
    # shows artificially low flows. For networks with a non-zero sim_start_time
    # (e.g. quickstart at 54000s), traffic is already established so no warmup needed.
    if sim_setup['sim_start_time'] == 0:
        kwargs_config["sensor_start_time"] = sim_setup['sim_stat_freq_sec']
    else:
        kwargs_config["sensor_start_time"] = sim_setup['sim_start_time']
    kwargs_config["sensor_end_time"] = sim_setup['sim_end_time']
    kwargs_config["od_duration_sec"] = sim_setup['od_duration_sec']
    kwargs_config["n_init_search"] = sim_setup['n_init_search']
    kwargs_config["dim_od"] = sim_setup['dim_od']

    # BO PARAMETERS - Add both naming conventions for compatibility
    # Lowercase with BO_ prefix (for direct access)
    kwargs_config["BO_batch_size"] = sim_setup["BO_batch_size"]
    kwargs_config["BO_num_restarts"] = sim_setup["BO_num_restarts"]
    kwargs_config["BO_raw_samples"] = sim_setup.get("BO_raw_samples") or sim_setup.get("BO_sample_shape", 128)
    #kwargs_config["BO_sample_shape"] = sim_setup.get("BO_sample_shape") or sim_setup.get("bo_sample_shape", 128)
    kwargs_config["BO_niter"] = sim_setup["BO_niter"]

    # UPPERCASE keys (used in helpers_od_calibration_BN.py)
    kwargs_config["NITER"] = sim_setup["BO_niter"]
    kwargs_config["BATCH_SIZE"] = sim_setup["BO_batch_size"]
    kwargs_config["NUM_RESTARTS"] = sim_setup["BO_num_restarts"]
    kwargs_config["RAW_SAMPLES"] = sim_setup.get("BO_raw_samples", 128)
    #kwargs_config["SAMPLE_SHAPE"] = sim_setup.get("BO_sample_shape") or sim_setup.get("bo_sample_shape", 128)

    # Optional cleanup setting
    kwargs_config["CLEANUP_INTERMEDIATE_FILES"] = sim_setup.get("CLEANUP_INTERMEDIATE_FILES", False)

    # Early stopping criteria
    kwargs_config["EARLY_STOP_PATIENCE"] = sim_setup.get("EARLY_STOP_PATIENCE", 0)
    kwargs_config["EARLY_STOP_DELTA"] = sim_setup.get("EARLY_STOP_DELTA", 1e-6)
    kwargs_config["EARLY_STOP_MIN_ACQ"] = sim_setup.get("EARLY_STOP_MIN_ACQ", 0.0)
    

    # OD bounds (BO4Mob-style: lb=1 to avoid "No vehicles loaded")
    kwargs_config["od_bound_start"] = sim_setup.get("od_bound_start", 1)
    kwargs_config["od_bound_end"] = sim_setup.get("od_bound_end", 2000)


    return kwargs_config





# XML/IO HELPERS


def od_xml_to_df(file_path):
    """
    Parse OD XML file to DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the OD XML file.

    Returns
    -------
    pd.DataFrame
        DataFrame with OD pairs and counts.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    df = xml2df_str(root, "tazRelation")
    df["count"] = df["count"].astype(float)
    print("Total GT demand:", df["count"].sum())
    return df


def iter_str(root, row_str):
    """
    Iterator for XML elements.
    """
    for element in root.iter(row_str):
        yield element.attrib


def xml2df_str(root, row_str):
    """
    Convert XML elements to DataFrame.
    """
    return pd.DataFrame(iter_str(root, row_str))


def is_in_OD_set(routes_df, orig, dest):
    """
    Check if OD pair is in route set.
    """
    return ((routes_df.fromTaz == orig) and (routes_df.toTaz == dest))









