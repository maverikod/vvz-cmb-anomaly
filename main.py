import argparse
import json
import logging
import os
import sys
from cmb_anomaly.convert import run_convert_command
from cmb_anomaly.multiscale import run_multiscale_anomaly_search
from cmb_anomaly.scale_clustering import run_scale_clustering_analysis
from cmb_anomaly.galactic_correlation import run_galactic_correlation_analysis
from cmb_anomaly.dust_correlation import run_dust_correlation_analysis
from cmb_anomaly.region_match import find_similar_regions

# Array backend imports removed - not used in main.py
from cmb_anomaly.structure_detector import run_structure_detection_from_paths
from cmb_anomaly.correlation_function import run_correlation_analysis
from cmb_anomaly.utils import ensure_dir_for_file

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
    if getattr(args, "config", None):
        config_path = args.config
    elif os.environ.get(CONFIG_ENV_VAR):
        config_path = os.environ[CONFIG_ENV_VAR]
    else:
        print(
            "Error: Config file must be specified via --config/-c or CMB_ANOMALY_CONF environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_path(arg_value, config_value, default_value):
    if arg_value is not None:
        return arg_value
    elif config_value is not None:
        return config_value
    else:
        return default_value


def main():
    parser = argparse.ArgumentParser(
        description="CMB Anomaly Analysis: Statistical analysis of large anomalies on the CMB map (Planck SMICA 2018)"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Main analysis CLI
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to JSON config file (overrides CMB_ANOMALY_CONF)",
    )
    parser.add_argument(
        "--help-config", action="store_true", help="Show config file format and exit"
    )
    parser.add_argument(
        "--help-sequence",
        action="store_true",
        help="Show recommended command sequence for full analysis",
    )

    # Subcommand: convert
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert Planck SMICA FITS file to NumPy .npy array (temperature or mask)",
    )
    convert_parser.add_argument(
        "--input", "-i", required=True, help="Path to input FITS file"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Path to output .npy file"
    )

    # Subcommand: multiscale-anomaly
    ms_parser = subparsers.add_parser(
        "multiscale-anomaly",
        help="Automatic search for anomalies with radius variation (multi-scale)",
    )
    ms_parser.add_argument(
        "--temperature", required=True, help="Path to .npy file with temperature map"
    )
    ms_parser.add_argument(
        "--mask",
        required=False,
        help="Path to .npy file with mask (omit or use --no-mask for no mask)",
    )
    ms_parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not use any mask (analyze all pixels)",
    )
    ms_parser.add_argument(
        "--results",
        default="results_anomalies_multi.csv",
        help="Path to output CSV file",
    )
    ms_parser.add_argument(
        "--step", type=int, default=5, help="Step in degrees for grid (default: 5)"
    )
    ms_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top anomalies to visualize (default: 20)",
    )

    # Subcommand: scale-clustering
    sc_parser = subparsers.add_parser(
        "scale-clustering",
        help="Analyze clustering of anomaly radii by discrete theoretical levels (no mask)",
    )
    sc_parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with anomalies (must have 'radius_deg')",
    )
    sc_parser.add_argument(
        "--alpha", type=float, default=0.7, help="Logarithmic step alpha (default: 0.7)"
    )
    sc_parser.add_argument(
        "--r0",
        type=float,
        default=None,
        help="Minimal radius r0 (default: min(radius_deg) in file)",
    )
    sc_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top levels to print (default: 10)",
    )

    # Subcommand: galactic-correlation
    gc_parser = subparsers.add_parser(
        "galactic-correlation",
        help="Analyze correlation of anomaly clusters with Galactic structures",
    )
    gc_parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with anomalies (must have 'radius_deg', 'l', 'b')",
    )
    gc_parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[1, 5, 25],
        help="Radii to analyze (default: 1 5 25)",
    )
    gc_parser.add_argument(
        "--out-prefix", default="galactic_corr", help="Prefix for output files"
    )
    gc_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top anomalies to plot on map (default: 20)",
    )

    # Subcommand: dust-correlation
    dust_parser = subparsers.add_parser(
        "dust-correlation",
        help="Analyze correlation between CMB anomalies and dust map",
    )
    dust_parser.add_argument(
        "--anomalies", required=True, help="CSV with anomalies (l, b, radius_deg, zone)"
    )
    dust_parser.add_argument(
        "--dust", required=True, help="FITS file with dust map (HEALPix)"
    )
    dust_parser.add_argument(
        "--out-prefix", default="dust_corr", help="Prefix for output files"
    )
    dust_parser.add_argument(
        "--n-control",
        type=int,
        default=1000,
        help="Number of control samples per zone/scale (default: 1000)",
    )
    dust_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Subcommand: region-match
    match_parser = subparsers.add_parser(
        "region-match",
        help="Find and analyze regions similar to known anomalies from YAML",
    )
    match_parser.add_argument(
        "--anomalies",
        required=True,
        help="CSV with found anomalies (from multiscale-anomaly or galactic-correlation)",
    )
    match_parser.add_argument(
        "--known-yaml", required=True, help="YAML file with known regions"
    )
    match_parser.add_argument("--output", required=False, help="Output CSV for matches")
    match_parser.add_argument(
        "--radius-tol",
        type=float,
        default=0.2,
        help="Relative tolerance for radius matching (default: 0.2)",
    )
    match_parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top matches per region (default: 3)",
    )

    # Subcommand: structure-detect (ring-like oscillatory patterns)
    sd_parser = subparsers.add_parser(
        "structure-detect",
        help="Detect ring-like oscillatory structures via radial matched filtering to cos(kappa r + phi)",
    )
    sd_parser.add_argument(
        "--temperature", required=True, help="Path to .npy file with temperature map"
    )
    sd_parser.add_argument(
        "--mask",
        required=False,
        help="Path to .npy file with mask (omit to analyze all pixels)",
    )
    sd_parser.add_argument(
        "--results",
        default="results/02_multiscale/structure_detections.csv",
        help="Path to output CSV file with detections",
    )
    sd_parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step in degrees for center grid (default: 5)",
    )
    sd_parser.add_argument(
        "--rmax",
        type=float,
        default=15.0,
        help="Max radius for radial profile in degrees (default: 15.0)",
    )
    sd_parser.add_argument(
        "--binsize",
        type=float,
        default=0.5,
        help="Radial bin size in degrees (default: 0.5)",
    )
    sd_parser.add_argument(
        "--pth",
        type=float,
        default=1e-3,
        help="p-value threshold for reporting detections (default: 1e-3)",
    )
    sd_parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU_count - 1)",
    )
    sd_parser.add_argument(
        "--block-size",
        type=int,
        default=500,
        help="Number of centers per block for block-based processing (default: 500)",
    )
    sd_parser.add_argument(
        "--block-number",
        type=int,
        default=None,
        help="Process only this block number (0-indexed). Use --count-blocks to see total blocks.",
    )
    sd_parser.add_argument(
        "--count-blocks",
        action="store_true",
        help="Count and print total number of blocks, then exit",
    )
    sd_parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration if available (default: True)",
    )
    sd_parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU acceleration",
    )
    sd_parser.add_argument(
        "--long-wavelength",
        action="store_true",
        help="Enable long-wavelength search mode (kappa_min=0.01, max_radius=60deg, bin_size=1.0deg)",
    )
    sd_parser.add_argument(
        "--kappa-min",
        type=float,
        default=None,
        help="Minimum κ in 1/deg (overrides default or long-wavelength mode)",
    )
    sd_parser.add_argument(
        "--kappa-max",
        type=float,
        default=None,
        help="Maximum κ in 1/deg (overrides default)",
    )

    # Subcommand: correlation-function (two-point correlation analysis)
    cf_parser = subparsers.add_parser(
        "correlation-function",
        help="Compute two-point correlation function ξ(θ) for CMB map (probes base substrate structure)",
    )
    cf_parser.add_argument(
        "--temperature", required=True, help="Path to .npy file with temperature map"
    )
    cf_parser.add_argument(
        "--mask",
        required=False,
        help="Path to .npy file with mask (omit to analyze all pixels)",
    )
    cf_parser.add_argument(
        "--output",
        default="results/correlation_function/xi_theta.csv",
        help="Path to output CSV file",
    )
    cf_parser.add_argument(
        "--theta-min",
        type=float,
        default=0.1,
        help="Minimum angular separation in degrees (default: 0.1)",
    )
    cf_parser.add_argument(
        "--theta-max",
        type=float,
        default=180.0,
        help="Maximum angular separation in degrees (default: 180.0)",
    )
    cf_parser.add_argument(
        "--n-bins",
        type=int,
        default=1000,
        help="Number of angular separation bins (default: 1000)",
    )
    cf_parser.add_argument(
        "--kappa-min",
        type=float,
        default=0.1,
        help="Minimum κ for fitting in 1/deg (default: 0.1)",
    )
    cf_parser.add_argument(
        "--kappa-max",
        type=float,
        default=3.0,
        help="Maximum κ for fitting in 1/deg (default: 3.0)",
    )
    cf_parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration if available",
    )

    args = parser.parse_args()

    if args.command == "convert":
        output_path = get_path(args.output, None, None)
        ensure_dir_for_file(output_path)
        run_convert_command(args.input, output_path)
        return

    if args.command == "multiscale-anomaly":
        mask_path = None if getattr(args, "no_mask", False) else args.mask
        results_csv = get_path(args.results, None, "results_anomalies_multi.csv")
        ensure_dir_for_file(results_csv)
        run_multiscale_anomaly_search(
            temperature_path=args.temperature,
            mask_path=mask_path,
            results_csv=results_csv,
            step_deg=args.step,
            top_n=args.top,
        )
        return

    if args.command == "scale-clustering":
        csv_path = get_path(args.csv, None, None)
        run_scale_clustering_analysis(
            csv_path=csv_path, alpha=args.alpha, r0=args.r0, top_n=args.top
        )
        return

    if args.command == "galactic-correlation":
        csv_path = get_path(args.csv, None, None)
        out_prefix = get_path(args.out_prefix, None, "galactic_corr")
        ensure_dir_for_file(f"{out_prefix}_anomalies_1deg.csv")
        run_galactic_correlation_analysis(
            csv_path=csv_path, radii=args.radii, out_prefix=out_prefix, top_n=args.top
        )
        return

    if args.command == "dust-correlation":
        anomaly_csv = get_path(args.anomalies, None, None)
        dust_fits = get_path(args.dust, None, None)
        out_prefix = get_path(args.out_prefix, None, "dust_corr")
        ensure_dir_for_file(f"{out_prefix}_summary.csv")
        run_dust_correlation_analysis(
            anomaly_csv=anomaly_csv,
            dust_fits=dust_fits,
            out_prefix=out_prefix,
            n_control=args.n_control,
            seed=args.seed,
        )
        return

    if args.command == "region-match":
        anomaly_csv = get_path(args.anomalies, None, None)
        known_yaml = get_path(args.known_yaml, None, None)
        output_csv = get_path(args.output, None, None)
        if output_csv:
            ensure_dir_for_file(output_csv)
        find_similar_regions(
            anomaly_csv=anomaly_csv,
            known_yaml=known_yaml,
            output_csv=output_csv,
            radius_tol=args.radius_tol,
            top_n=args.top,
        )
        print("Region matching complete.")
        return

    if args.command == "structure-detect":
        # Configure logging for structure detection
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        # This must be done before any multiprocessing operations
        if args.use_gpu:
            import multiprocessing as mp

            try:
                current_method = mp.get_start_method(allow_none=True)
                if current_method != "spawn":
                    logging.info(
                        "Setting multiprocessing start method to 'spawn' for CUDA compatibility"
                    )
                    mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, ignore
                pass
        results_csv = get_path(
            args.results, None, "results/02_multiscale/structure_detections.csv"
        )
        ensure_dir_for_file(results_csv)

        # Apply long-wavelength mode defaults if enabled
        if args.long_wavelength:
            kappa_min = args.kappa_min if args.kappa_min is not None else 0.01
            kappa_max = args.kappa_max if args.kappa_max is not None else 0.1
            max_radius = 60.0
            bin_size = 1.0
            logging.info("Long-wavelength search mode enabled:")
            logging.info(f"  κ range: [{kappa_min:.3f}, {kappa_max:.3f}] per deg")
            logging.info(f"  Max radius: {max_radius} deg, bin size: {bin_size} deg")
        else:
            kappa_min = args.kappa_min if args.kappa_min is not None else 0.1
            kappa_max = args.kappa_max if args.kappa_max is not None else 3.0
            max_radius = args.rmax
            bin_size = args.binsize

            run_structure_detection_from_paths(
                temperature_path=args.temperature,
                mask_path=args.mask,
                results_csv_path=results_csv,
                step_deg=args.step,
                max_radius_deg=max_radius,
                bin_size_deg=bin_size,
                p_value_threshold=args.pth,
                n_jobs=args.n_jobs,
                block_size=args.block_size,
                use_gpu=args.use_gpu,
                kappa_min_per_deg=kappa_min,
                kappa_max_per_deg=kappa_max,
                block_number=args.block_number,
                count_blocks_only=args.count_blocks,
            )
        return

    if args.command == "correlation-function":
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        run_correlation_analysis(
            temperature_path=args.temperature,
            mask_path=args.mask,
            output_csv=args.output,
            theta_min_deg=args.theta_min,
            theta_max_deg=args.theta_max,
            n_bins=args.n_bins,
            kappa_min_per_deg=args.kappa_min,
            kappa_max_per_deg=args.kappa_max,
            use_gpu=args.use_gpu,
        )
        return

    if args.help_config:
        print(
            json.dumps(
                {
                    "data_file": "Path to Planck SMICA 2018 FITS file (e.g. data/COM_CMB_IQU-smica_2048_R3.00_full.fits)",
                    "mask_file": "Path to mask FITS file (optional)",
                    "anomaly_yaml": "Path to YAML file with anomaly regions (optional)",
                },
                indent=2,
            )
        )
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
