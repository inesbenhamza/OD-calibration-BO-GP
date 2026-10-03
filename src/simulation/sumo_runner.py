# Standard library imports
import os
import subprocess
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# Local application imports
from src.simulation.data_loader import xml2df_str


# Parsing helper: SUMO edge output XML -> DataFrame
def parse_loop_data_xml_to_pandas(base_dir, sim_edge_file, prefix_output, SUMO_PATH):
    """
    Convert SUMO edge output XML → CSV → Pandas dataframe.
    """
    output_csv = f"{base_dir}/{prefix_output}_loopOutputs.csv"

    convert_cmd = [
        "python", f"{SUMO_PATH}/tools/xml/xml2csv.py",
        sim_edge_file, "-o", output_csv
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"xml2csv.py failed (exit {result.returncode}) for {sim_edge_file}\n"
            f"stderr: {result.stderr.strip()}\n"
            f"stdout: {result.stdout.strip()}"
        )

    df = pd.read_csv(output_csv, sep=";")

    df["interval_nVehContrib"] = df["edge_arrived"] + df["edge_left"] # this is edge_arrived + edge_left
    df["interval_harmonicMeanSpeed"] = df["edge_speed"]

    df_agg = df.groupby("edge_id", as_index=False).agg(
        interval_nVehContrib=("interval_nVehContrib", "sum"),
        interval_harmonicMeanSpeed=("interval_harmonicMeanSpeed", "mean")
    )

    return df_agg, df, output_csv






# XML CREATION HELPERS

def create_taz_xml(file_xml, od_df, od_end_time_seconds, base_dir):
    """
    Writes SUMO TAZ file with proper <data><interval> wrapper.

    Parameters
    ----------
    file_xml : str
        Output XML filename.
    od_df : pd.DataFrame
        DataFrame with OD pairs (columns: from, to, count).
    od_end_time_seconds : int
        End time of OD interval in seconds.
    base_dir : str
        Base directory where XML is written.
    """
    xml_out = f"{base_dir}/{file_xml}"

    od_df.to_xml(
        xml_out,
        attr_cols=["from", "to", "count"],
        root_name="interval",
        row_name="tazRelation",
        index=False
    )

    tree = ET.parse(xml_out)
    root = tree.getroot()
    root.set("id", "DEFAULT_VEHTYPE")
    root.set("begin", "0")
    root.set("end", str(od_end_time_seconds))

    new_root = ET.Element("data")
    new_root.insert(0, root)

    tree = ET.ElementTree(new_root)
    tree.write(xml_out)
    print("Created", xml_out)


def create_od_xml(current_od, base_od_df, file_od, od_end_time_seconds, base_dir):
    """
    Create SUMO OD XML file (tazRelation inside <data><interval> structure).

    Parameters
    ----------
    current_od : array-like
        OD demand vector (one element per OD pair).
    base_od_df : pd.DataFrame
        Two-column TAZ table with keys: fromTaz, toTaz
    file_od : str
        Output filename (e.g. "path/to/od.xml")
    od_end_time_seconds : int
        End of OD interval for SUMO.
    base_dir : str
        Base directory where XML is written.
    """
    import numpy as np
    print(f'total expected GT demand: {np.sum(current_od)}')

    # Insert OD values
    base_od_df['count'] = current_od

    # Round values to 1 decimal
    base_od_df['count'] = [round(elem, 1) for elem in base_od_df['count']]

    # Rename for SUMO format
    base_od_df = base_od_df.rename(columns={'fromTaz': 'from', 'toTaz': 'to'})

    # Delegate XML creation to create_taz_xml
    create_taz_xml(file_od, base_od_df, od_end_time_seconds, base_dir)


def create_od_tazrelation_xml(od_df: pd.DataFrame, output_file: Path, od_end_time_seconds: int) -> None:
    """
    Create a TAZ OD matrix XML file from a DataFrame and wrap it in <data>.
    """
    od_df.to_xml(
        output_file,
        attr_cols=["from", "to", "count"],
        root_name="interval",
        row_name="tazRelation",
        index=False,
    )

    tree = ET.parse(output_file)
    root = tree.getroot()
    root.set("id", "DEFAULT_VEHTYPE")
    root.set("begin", "0")
    root.set("end", str(od_end_time_seconds))

    new_root = ET.Element("data")
    new_root.insert(0, root)
    tree = ET.ElementTree(new_root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Created OD TAZ relation XML at: {output_file}")


def update_trip_routes(in_trips_xml, out_trips_xml, routes_df, seed=0):
    """
    Assign each generated trip to a fixed route from the route set.

    Writes <vehicle> elements with an explicit <route edges="..."/> child,
    so SUMO follows the exact route_edges path instead of recomputing its
    own shortest path between from/to edges (which would ignore the "ratio"
    split for OD pairs with multiple route options sharing the same
    start/last edge).

    Parameters
    ----------
    in_trips_xml : str
        Input trips XML file path.
    out_trips_xml : str
        Output trips XML file path.
    routes_df : pd.DataFrame
        Routes DataFrame with fromTaz, toTaz, route_edges columns.
        May contain multiple rows per (fromTaz, toTaz) pair (split routes),
        optionally weighted by a "ratio" column.
    seed : int, optional
        Seed for the RNG used to assign trips to split routes. Default: 0.
    """
    # Check if input file exists
    if not Path(in_trips_xml).exists():
        raise FileNotFoundError(
            f"Input trips file not found: {in_trips_xml}\n"
            f"This file should have been created by od2trips. Check the od2trips command output."
        )

    tree = ET.parse(in_trips_xml)
    root = tree.getroot()

    trips_df = xml2df_str(root, "trip")
    routes_df = routes_df.copy()
    routes_df["fromTaz"] = routes_df["fromTaz"].astype(str)
    routes_df["toTaz"] = routes_df["toTaz"].astype(str)

    trips_df = trips_df.drop(columns=["to", "from"], errors="ignore")

    has_ratio = "ratio" in routes_df.columns
    rng = np.random.default_rng(seed)

    # For OD pairs with multiple routes (split ratios), assign each trip to
    # exactly one route (sampled according to the ratios), instead of merging
    # to a cross product which would create duplicate vehicle ids.
    assigned_groups = []
    for (from_taz, to_taz), group in trips_df.groupby(["fromTaz", "toTaz"], sort=False):
        od_routes = routes_df[
            (routes_df["fromTaz"] == from_taz) & (routes_df["toTaz"] == to_taz)
        ]
        if od_routes.empty:
            continue

        if len(od_routes) == 1:
            chosen = od_routes.iloc[np.zeros(len(group), dtype=int)]
        else:
            weights = (
                od_routes["ratio"].astype(float).to_numpy()
                if has_ratio
                else np.ones(len(od_routes))
            )
            weights = weights / weights.sum()
            choice_idx = rng.choice(len(od_routes), size=len(group), p=weights)
            chosen = od_routes.iloc[choice_idx]

        group = group.reset_index(drop=True)
        chosen = chosen[["route_edges"]].reset_index(drop=True)
        assigned_groups.append(pd.concat([group, chosen], axis=1))

    merged = pd.concat(assigned_groups, ignore_index=True)
    merged["depart_float"] = merged["depart"].astype(float)
    merged = merged.sort_values(by="depart_float")

    new_root = ET.Element("routes")
    for _, trip_row in merged.iterrows():
        veh = ET.SubElement(new_root, "vehicle", {
            "id": str(trip_row["id"]),
            "depart": str(trip_row["depart"]),
            "type": str(trip_row["type"]),
            "departLane": str(trip_row["departLane"]),
            "departSpeed": str(trip_row["departSpeed"]),
        })
        ET.SubElement(veh, "route", {"edges": str(trip_row["route_edges"])})

    ET.ElementTree(new_root).write(out_trips_xml, encoding="utf-8", xml_declaration=True)





def simulate_od(
    od_xml: str,
    prefix_output: str,
    base_dir: str,
    net_xml: str,
    taz2edge_xml: str,
    additional_xml: str,
    routes_df: pd.DataFrame,
    sim_end_time: str,
    TRIPS2ODS_OUT_STR: str,
    sim_start_time: int = 0,
    seed: int = 0
):
    """
    Run SUMO for an OD matrix and write output XML files.

    Parameters
    ----------
    od_xml : str
        Path to OD XML file.
    prefix_output : str
        Prefix for output files.
    base_dir : str
        Base directory for output files.
    net_xml : str
        Path to network XML file.
    taz2edge_xml : str
        Path to TAZ to edge mapping XML file.
    additional_xml : str
        Path to additional XML file.
    routes_df : pd.DataFrame
        Routes DataFrame.
    sim_end_time : str
        Simulation end time.
    TRIPS2ODS_OUT_STR : str
        Output trips XML filename string.
    sim_start_time : int, optional
        Simulation start time. Default: 0
    seed : int, optional
        Random seed. Default: 0
    """
    trips_before = f"{base_dir}/{prefix_output}_{TRIPS2ODS_OUT_STR[:-4]}_beforeRteUpdates.xml"
    trips_after  = f"{base_dir}/{prefix_output}_{TRIPS2ODS_OUT_STR}"

    # Ensure output directory exists
    trips_before_path = Path(trips_before)
    trips_before_path.parent.mkdir(parents=True, exist_ok=True)
    trips_after_path = Path(trips_after)
    trips_after_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if input files exist
    if not Path(od_xml).exists():
        raise FileNotFoundError(f"OD XML file not found: {od_xml}")
    if not Path(taz2edge_xml).exists():
        raise FileNotFoundError(f"TAZ XML file not found: {taz2edge_xml}")

    # 1. Generate trips
    od2trips_cmd = [
        "od2trips",
        "--spread.uniform",
        "--taz-files", str(taz2edge_xml),
        "--tazrelation-files", str(Path(od_xml).resolve()),
        "--seed", str(seed),
        "-o", str(trips_before)
    ]
    print(" ".join(od2trips_cmd))
    
    # Run command and capture output
    try:
        result = subprocess.run(
            od2trips_cmd,
            capture_output=True,
            text=True,
            check=False  
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"od2trips command not found. Make sure SUMO tools are in your PATH.\n"
            f"Try: export PATH=$PATH:/path/to/sumo/bin"
        )
    
    # Check if od2trips succeeded
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(
            f"od2trips command failed with exit code {result.returncode}\n"
            f"Command: {' '.join(od2trips_cmd)}\n"
            f"Error output:\n{error_msg}\n"
            f"Standard output:\n{result.stdout}"
        )
    
    # Verify the file was created
    if not trips_before_path.exists():
        raise FileNotFoundError(
            f"od2trips did not create expected file: {trips_before}\n"
            f"Command was: {' '.join(od2trips_cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"Check that od2trips is in PATH and the input files exist."
        )

    # 2. Update trips based on fixed routes
    update_trip_routes(trips_before, trips_after, routes_df, seed=seed)
    
    # Verify the output file was created
    if not trips_after_path.exists():
        raise FileNotFoundError(
            f"update_trip_routes did not create expected file: {trips_after}\n"
            f"Input file was: {trips_before}"
        )

    # 3. Run SUMO
    sumo_cmd = (
        f"sumo --output-prefix {prefix_output}_ "
        f"--ignore-route-errors=true "
        f"--net-file={net_xml} "
        f"--routes={trips_after} "
        f"-b {sim_start_time} -e {sim_end_time} "
        f"--additional-files {additional_xml} "
        f"--duration-log.statistics "
        f"--xml-validation never "
        f"--vehroutes routes.vehroutes.xml "
        f"--verbose "
        f"--no-warnings "
        f"--mesosim true "
        f"--seed {seed}"
    )
    print("Running SUMO:", sumo_cmd)
    os.system(sumo_cmd)


def run_simulation_for_od(
    od_vector,
    base_od_df: pd.DataFrame,
    sim_dir,
    config: dict,
    base_path,
    routes_df: pd.DataFrame,
    seed: int = 0,
) -> str:
    """
    Run SUMO for a given OD demand vector.

    Parameters
    ----------
    od_vector : array-like
        OD demand values (one per OD pair), in same order as base_od_df rows.
    base_od_df : pd.DataFrame
        Base OD table with columns fromTaz, toTaz (same order as od_vector).
    sim_dir : str or Path
        Directory for this simulation (OD XML and outputs written here).
    config : dict
        Must contain: od_duration_sec, net_xml, taz2edge_xml, additional_xml,
        sim_end_time, TRIPS2ODS_OUT_STR.
    base_path : str or Path
        Project base directory.
    routes_df : pd.DataFrame
        Routes table for trip updates.
    seed : int, optional
        Random seed for SUMO/od2trips. Default 0.

    Returns
    -------
    str
        Output prefix path relative to base_path (e.g. "path/to/sim_dir/sim").
    """
    sim_dir = Path(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_path)

    create_od_xml(
        od_vector,
        base_od_df.copy(),
        "od.xml",
        config["od_duration_sec"],
        str(sim_dir),
    )

    prefix_output = str(sim_dir / "sim")
    simulate_od(
        str(sim_dir / "od.xml"),
        prefix_output,
        str(base_path),
        str(config["net_xml"]),
        str(config["taz2edge_xml"]),
        str(config["additional_xml"]),
        routes_df,
        str(config["sim_end_time"]),
        config["TRIPS2ODS_OUT_STR"],
        sim_start_time=0,
        seed=seed,
    )
    return prefix_output


def parse_simulation_outputs(
    prefix_output,
    config: dict,
    base_path,
) -> pd.DataFrame:
    """
    Read SUMO edge output and return the edge flow dataframe.

    Parameters
    ----------
    prefix_output : str or Path
        Simulation output prefix (e.g. "path/to/sim_dir/sim", relative to base_path).
    config : dict
        Must contain EDGE_OUT_STR and SUMO_PATH.
    base_path : str or Path
        Project base directory.

    Returns
    -------
    pd.DataFrame
        Edge-level flow data with columns including edge_id and interval_nVehContrib.
    """
    base_path = Path(base_path)
    sim_edge_out = f"{prefix_output}_{config['EDGE_OUT_STR']}"
    df_agg, _, _ = parse_loop_data_xml_to_pandas(
        str(base_path),
        sim_edge_out,
        str(prefix_output),
        str(config["SUMO_PATH"]),
    )
    return df_agg
