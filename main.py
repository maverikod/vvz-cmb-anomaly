import argparse
import json
import os
import sys
from cmb_anomaly.convert import run_convert_command
from cmb_anomaly.multiscale import run_multiscale_anomaly_search
from cmb_anomaly.scale_clustering import run_scale_clustering_analysis
from cmb_anomaly.galactic_correlation import run_galactic_correlation_analysis
from cmb_anomaly.dust_correlation import run_dust_correlation_analysis
from cmb_anomaly.region_match import find_similar_regions
from cmb_anomaly.array_backend import cp, np, array_load, array_save, print_backend_info

CONFIG_ENV_VAR = "CMB_ANOMALY_CONF"

HELP_SEQUENCE = """
Последовательность команд для полного анализа:

1. Конвертация FITS в .npy:
   cmb-anomaly convert -i data/COM_CMB_IQU-smica_2048_R3.00_full.fits -o data/cmb_temperature.npy
   cmb-anomaly convert -i data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits -o data/mask_common.npy

2. Автоматический поиск аномалий с вариацией радиуса:
   cmb-anomaly multiscale-anomaly --temperature data/cmb_temperature.npy --mask data/mask_common.npy
   cmb-anomaly multiscale-anomaly --temperature data/cmb_temperature.npy --no-mask --results results_anomalies_nomask.csv

3. Кластеризация аномалий по дискретным масштабам (без маски):
   cmb-anomaly scale-clustering --csv results_anomalies_nomask.csv [--alpha 0.7] [--r0 <min_radius>] [--top 10]

4. Корреляция аномалий с Галактикой:
   cmb-anomaly galactic-correlation --csv results_anomalies_nomask.csv

5. Dust correlation (CMB anomalies vs dust map):
   cmb-anomaly dust-correlation --anomalies galactic_corr_anomalies_1deg.csv --dust path/to/dust_map.fits

(Далее будут добавляться новые команды для расширенного анализа)
"""

def load_config(args) -> dict:
    """
    Load configuration from file specified by --config/-c or environment variable CMB_ANOMALY_CONF.
    Args:
        args: argparse.Namespace with parsed arguments
    Returns:
        dict: loaded config
    Raises:
        FileNotFoundError: if config file is not found
        json.JSONDecodeError: if config is not valid JSON
    """
    config_path = None
    if getattr(args, 'config', None):
        config_path = args.config
    elif os.environ.get(CONFIG_ENV_VAR):
        config_path = os.environ[CONFIG_ENV_VAR]
    else:
        print("Error: Config file must be specified via --config/-c or CMB_ANOMALY_CONF environment variable.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(
        description="CMB Anomaly Analysis: Statistical analysis of large anomalies on the CMB map (Planck SMICA 2018)"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Main analysis CLI
    parser.add_argument(
        "-c", "--config", type=str, help="Path to JSON config file (overrides CMB_ANOMALY_CONF)"
    )
    parser.add_argument(
        "--help-config", action="store_true", help="Show config file format and exit"
    )
    parser.add_argument(
        "--help-sequence", action="store_true", help="Show recommended command sequence for full analysis"
    )

    # Subcommand: convert
    convert_parser = subparsers.add_parser(
        "convert", help="Convert Planck SMICA FITS file to NumPy .npy array (temperature or mask)"
    )
    convert_parser.add_argument("--input", "-i", required=True, help="Path to input FITS file")
    convert_parser.add_argument("--output", "-o", required=True, help="Path to output .npy file")

    # Subcommand: multiscale-anomaly
    ms_parser = subparsers.add_parser(
        "multiscale-anomaly", help="Automatic search for anomalies with radius variation (multi-scale)"
    )
    ms_parser.add_argument("--temperature", required=True, help="Path to .npy file with temperature map")
    ms_parser.add_argument("--mask", required=False, help="Path to .npy file with mask (omit or use --no-mask for no mask)")
    ms_parser.add_argument("--no-mask", action="store_true", help="Do not use any mask (analyze all pixels)")
    ms_parser.add_argument("--results", default="results_anomalies_multi.csv", help="Path to output CSV file")
    ms_parser.add_argument("--step", type=int, default=5, help="Step in degrees for grid (default: 5)")
    ms_parser.add_argument("--top", type=int, default=20, help="Number of top anomalies to visualize (default: 20)")

    # Subcommand: scale-clustering
    sc_parser = subparsers.add_parser(
        "scale-clustering", help="Analyze clustering of anomaly radii by discrete theoretical levels (no mask)"
    )
    sc_parser.add_argument("--csv", required=True, help="Path to CSV with anomalies (must have 'radius_deg')")
    sc_parser.add_argument("--alpha", type=float, default=0.7, help="Logarithmic step alpha (default: 0.7)")
    sc_parser.add_argument("--r0", type=float, default=None, help="Minimal radius r0 (default: min(radius_deg) in file)")
    sc_parser.add_argument("--top", type=int, default=10, help="Number of top levels to print (default: 10)")

    # Subcommand: galactic-correlation
    gc_parser = subparsers.add_parser(
        "galactic-correlation", help="Analyze correlation of anomaly clusters with Galactic structures"
    )
    gc_parser.add_argument("--csv", required=True, help="Path to CSV with anomalies (must have 'radius_deg', 'l', 'b')")
    gc_parser.add_argument("--radii", nargs='+', type=int, default=[1,5,25], help="Radii to analyze (default: 1 5 25)")
    gc_parser.add_argument("--out-prefix", default="galactic_corr", help="Prefix for output files")
    gc_parser.add_argument("--top", type=int, default=20, help="Number of top anomalies to plot on map (default: 20)")

    # Subcommand: dust-correlation
    dust_parser = subparsers.add_parser(
        "dust-correlation", help="Analyze correlation between CMB anomalies and dust map"
    )
    dust_parser.add_argument("--anomalies", required=True, help="CSV with anomalies (l, b, radius_deg, zone)")
    dust_parser.add_argument("--dust", required=True, help="FITS file with dust map (HEALPix)")
    dust_parser.add_argument("--out-prefix", default="dust_corr", help="Prefix for output files")
    dust_parser.add_argument("--n-control", type=int, default=1000, help="Number of control samples per zone/scale (default: 1000)")
    dust_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Subcommand: region-match
    match_parser = subparsers.add_parser(
        "region-match", help="Find and analyze regions similar to known anomalies from YAML"
    )
    match_parser.add_argument("--anomalies", required=True, help="CSV with found anomalies (from multiscale-anomaly or galactic-correlation)")
    match_parser.add_argument("--known-yaml", required=True, help="YAML file with known regions")
    match_parser.add_argument("--output", required=False, help="Output CSV for matches")
    match_parser.add_argument("--radius-tol", type=float, default=0.2, help="Relative tolerance for radius matching (default: 0.2)")
    match_parser.add_argument("--top", type=int, default=3, help="Number of top matches per region (default: 3)")

    args = parser.parse_args()

    if args.command == "convert":
        run_convert_command(args.input, args.output)
        return

    if args.command == "multiscale-anomaly":
        mask_path = None if getattr(args, 'no_mask', False) else args.mask
        run_multiscale_anomaly_search(
            temperature_path=args.temperature,
            mask_path=mask_path,
            results_csv=args.results,
            step_deg=args.step,
            top_n=args.top
        )
        return

    if args.command == "scale-clustering":
        run_scale_clustering_analysis(
            csv_path=args.csv,
            alpha=args.alpha,
            r0=args.r0,
            top_n=args.top
        )
        return

    if args.command == "galactic-correlation":
        run_galactic_correlation_analysis(
            csv_path=args.csv,
            radii=args.radii,
            out_prefix=args.out_prefix,
            top_n=args.top
        )
        return

    if args.command == "dust-correlation":
        run_dust_correlation_analysis(
            anomaly_csv=args.anomalies,
            dust_fits=args.dust,
            out_prefix=args.out_prefix,
            n_control=args.n_control,
            seed=args.seed
        )
        return

    if args.command == "region-match":
        find_similar_regions(
            anomaly_csv=args.anomalies,
            known_yaml=args.known_yaml,
            output_csv=args.output,
            radius_tol=args.radius_tol,
            top_n=args.top
        )
        print("Region matching complete.")
        return

    if args.help_config:
        print(json.dumps({
            "data_file": "Path to Planck SMICA 2018 FITS file (e.g. data/COM_CMB_IQU-smica_2048_R3.00_full.fits)",
            "mask_file": "Path to mask FITS file (optional)",
            "anomaly_yaml": "Path to YAML file with anomaly regions (optional)"
        }, indent=2))
        sys.exit(0)

    if args.help_sequence:
        print(HELP_SEQUENCE)
        sys.exit(0)

    config = load_config(args)
    print("Config loaded successfully:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    # TODO: implement main logic

if __name__ == "__main__":
    main() 