"""
Postprocessing module for CMB anomaly clusters.

This module provides functions to filter anomalies from the raw CSV output
(results_anomalies_multi.csv) to obtain a physically meaningful set of clusters:
for each unique center (l, b), only the radius with the maximum S is kept.
Optionally, close anomalies can be merged into a single cluster.

Usage:
    python -m cmb_anomaly.cluster_postprocess --input results_anomalies_multi.csv --output unique_clusters.csv
"""

import pandas as pd
import argparse
from typing import Optional, List
import logging
from .utils import ensure_dir_for_file
import numpy as np
from tqdm import tqdm

def log_cluster_stats(df, stage=""):
    """
    Log statistics (min/mean/max) for S, radius, npix in the DataFrame.
    """
    if df.empty:
        logging.info(f"[{stage}] No clusters to log.")
        return
    s_vals = df["S"]
    r_vals = df["radius_deg"] if "radius_deg" in df.columns else None
    npix_vals = df["npix"] if "npix" in df.columns else None
    logging.info(f"[{stage}] S: min={s_vals.min():.2f}, mean={s_vals.mean():.2f}, max={s_vals.max():.2f}")
    if r_vals is not None:
        logging.info(f"[{stage}] radius_deg: min={r_vals.min():.2f}, mean={r_vals.mean():.2f}, max={r_vals.max():.2f}")
    if npix_vals is not None:
        logging.info(f"[{stage}] npix: min={npix_vals.min()}, mean={npix_vals.mean():.1f}, max={npix_vals.max()}")


def aggregate_by_max_s(df, center_columns: List[str], s_column: str) -> pd.DataFrame:
    """
    Group by center columns, keep row with maximal S in each group.
    Logs the number of unique clusters.
    """
    unique_anomalies = []
    for _, group in df.groupby(center_columns):
        best_row = group.loc[group[s_column].idxmax()]
        unique_anomalies.append(best_row)
    result_df = pd.DataFrame(unique_anomalies)
    logging.info(f"Aggregated to {len(result_df)} unique clusters by {center_columns}")
    return result_df


def dbscan_clusters(df, center_columns: List[str], radius_column: str, s_column: str, eps: float, min_samples: int) -> pd.DataFrame:
    """
    Cluster anomalies using DBSCAN in (l, b, r) space, keep max S in each cluster.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("scikit-learn is required for DBSCAN clustering.")
    X = df[center_columns + [radius_column]].values
    X_scaled = X.copy()
    X_scaled[:,2] /= eps  # scale radius to eps
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    df['cluster'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    logging.info(f"DBSCAN found {n_clusters} clusters (noise: {(db.labels_==-1).sum()})")
    unique_anomalies = []
    for cl in sorted(df['cluster'].unique()):
        group = df[df['cluster'] == cl]
        best_row = group.loc[group[s_column].idxmax()]
        unique_anomalies.append(best_row)
    result_df = pd.DataFrame(unique_anomalies)
    return result_df


def filter_clusters(df, s_column: str = "S", npix_column: str = "npix", s_threshold: float = 0.0, min_npix: int = 0) -> pd.DataFrame:
    """
    Filter clusters by S and npix thresholds.
    """
    before = len(df)
    filtered = df[(df[s_column] >= s_threshold) & (df[npix_column] >= min_npix)]
    after = len(filtered)
    logging.info(f"Filtered clusters by S>={s_threshold}, npix>={min_npix}: {before} -> {after}")
    return filtered


def filter_unique_clusters(
    input_csv: str,
    output_csv: str,
    center_columns: Optional[list] = None,
    s_column: str = "S",
    radius_column: str = "radius_deg",
    round_decimals: int = 4,
    use_dbscan: bool = False,
    dbscan_eps: float = 1.0,
    dbscan_min_samples: int = 1,
    s_threshold: float = 0.0,
    min_npix: int = 0,
    summary_report: bool = True
) -> None:
    """
    Filter anomalies to keep only one (with maximal S) per unique center or cluster.
    Optionally, cluster close anomalies using DBSCAN.
    Optionally, filter by S and npix.
    Logs statistics at each stage.
    Parameters:
        input_csv (str): Path to input CSV file with raw anomalies.
        output_csv (str): Path to save filtered unique clusters.
        center_columns (list, optional): Columns to use as center (default: ["l_deg", "b_deg"]).
        s_column (str): Name of the S column (default: "S").
        radius_column (str): Name of the radius column (default: "radius_deg").
        round_decimals (int): Decimals to round coordinates for grouping (default: 4).
        use_dbscan (bool): If True, use DBSCAN clustering in (l, b, r) space.
        dbscan_eps (float): DBSCAN epsilon (in degrees).
        dbscan_min_samples (int): DBSCAN min_samples.
        s_threshold (float): Minimum S for cluster to be kept.
        min_npix (int): Minimum npix for cluster to be kept.
        summary_report (bool): If True, log summary statistics for clusters.
    """
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} anomalies from {input_csv}")
    if center_columns is None:
        if "l" in df.columns and "b" in df.columns:
            center_columns = ["l", "b"]
        elif "l_deg" in df.columns and "b_deg" in df.columns:
            center_columns = ["l_deg", "b_deg"]
        elif "center_pix" in df.columns:
            center_columns = ["center_pix"]
        else:
            raise ValueError("Cannot determine center columns. Please specify explicitly.")
    # Round coordinates to avoid floating point duplicates
    for col in center_columns:
        if df[col].dtype.kind in "fc":
            df[col] = df[col].round(round_decimals)
    if summary_report:
        log_cluster_stats(df, stage="raw")
    if use_dbscan:
        logging.info(f"Clustering anomalies with DBSCAN: eps={dbscan_eps}, min_samples={dbscan_min_samples}")
        result_df = dbscan_clusters(df, center_columns, radius_column, s_column, dbscan_eps, dbscan_min_samples)
        if summary_report:
            log_cluster_stats(result_df, stage="dbscan")
    else:
        # Индикатор прогресса для агрегации по max S
        print(f"[CLUSTERING] Aggregating by max S for {len(df.groupby(center_columns))} unique centers...")
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            tqdm = None
            use_tqdm = False
        unique_anomalies = []
        groups = df.groupby(center_columns)
        if use_tqdm:
            group_iter = tqdm(groups, total=len(groups), desc="Aggregating")
        else:
            group_iter = groups
        for i, (_, group) in enumerate(group_iter):
            best_row = group.loc[group[s_column].idxmax()]
            unique_anomalies.append(best_row)
            if not use_tqdm and i % 10000 == 0 and i > 0:
                print(f"[CLUSTERING] Processed {i} / {len(groups)} centers...")
        result_df = pd.DataFrame(unique_anomalies)
        logging.info(f"Aggregated to {len(result_df)} unique clusters by {center_columns}")
        if summary_report:
            log_cluster_stats(result_df, stage="aggregated")
    # Фильтрация по S и npix
    result_df = filter_clusters(result_df, s_column=s_column, npix_column="npix" if "npix" in result_df.columns else None, s_threshold=s_threshold, min_npix=min_npix)
    if summary_report:
        log_cluster_stats(result_df, stage="filtered")
    ensure_dir_for_file(output_csv)
    result_df.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(result_df)} unique clusters to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter unique CMB anomaly clusters by center or DBSCAN.")
    parser.add_argument("--input", required=True, help="Input CSV file (raw anomalies)")
    parser.add_argument("--output", required=True, help="Output CSV file (unique clusters)")
    parser.add_argument("--center", nargs="*", default=None, help="Columns for center (default: l b)")
    parser.add_argument("--s_column", default="S", help="Column for S value (default: S)")
    parser.add_argument("--radius_column", default="radius_deg", help="Column for radius (default: radius_deg)")
    parser.add_argument("--round_decimals", type=int, default=4, help="Decimals for rounding coordinates (default: 4)")
    parser.add_argument("--use_dbscan", action="store_true", help="Use DBSCAN clustering in (l, b, r) space")
    parser.add_argument("--dbscan_eps", type=float, default=1.0, help="DBSCAN epsilon (in degrees)")
    parser.add_argument("--dbscan_min_samples", type=int, default=1, help="DBSCAN min_samples")
    parser.add_argument("--s_threshold", type=float, default=0.0, help="Minimum S for cluster to be kept")
    parser.add_argument("--min_npix", type=int, default=0, help="Minimum npix for cluster to be kept")
    parser.add_argument("--no_summary", action="store_true", help="Disable summary report logging")
    args = parser.parse_args()
    filter_unique_clusters(
        input_csv=args.input,
        output_csv=args.output,
        center_columns=args.center,
        s_column=args.s_column,
        radius_column=args.radius_column,
        round_decimals=args.round_decimals,
        use_dbscan=args.use_dbscan,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        s_threshold=args.s_threshold,
        min_npix=args.min_npix,
        summary_report=not args.no_summary
    ) 