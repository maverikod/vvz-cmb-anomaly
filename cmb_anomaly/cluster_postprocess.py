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
from typing import Optional

def filter_unique_clusters(
    input_csv: str,
    output_csv: str,
    center_columns: Optional[list] = None,
    s_column: str = "S",
    radius_column: str = "radius_deg",
    round_decimals: int = 4
) -> None:
    """
    Filter anomalies to keep only one (with maximal S) per unique center.

    Parameters:
        input_csv (str): Path to input CSV file with raw anomalies.
        output_csv (str): Path to save filtered unique clusters.
        center_columns (list, optional): Columns to use as center (default: ["l_deg", "b_deg"]).
        s_column (str): Name of the S column (default: "S").
        radius_column (str): Name of the radius column (default: "radius_deg").
        round_decimals (int): Decimals to round coordinates for grouping (default: 4).
    """
    df = pd.read_csv(input_csv)
    if center_columns is None:
        # Try to use galactic coordinates by default
        if "l_deg" in df.columns and "b_deg" in df.columns:
            center_columns = ["l_deg", "b_deg"]
        elif "center_pix" in df.columns:
            center_columns = ["center_pix"]
        else:
            raise ValueError("Cannot determine center columns. Please specify explicitly.")

    # Round coordinates to avoid floating point duplicates
    for col in center_columns:
        if df[col].dtype.kind in "fc":
            df[col] = df[col].round(round_decimals)

    # Group by center, keep row with maximal S
    unique_anomalies = []
    for _, group in df.groupby(center_columns):
        best_row = group.loc[group[s_column].idxmax()]
        unique_anomalies.append(best_row)
    result_df = pd.DataFrame(unique_anomalies)
    result_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter unique CMB anomaly clusters by center.")
    parser.add_argument("--input", required=True, help="Input CSV file (raw anomalies)")
    parser.add_argument("--output", required=True, help="Output CSV file (unique clusters)")
    parser.add_argument("--center", nargs="*", default=None, help="Columns for center (default: l_deg b_deg)")
    parser.add_argument("--s_column", default="S", help="Column for S value (default: S)")
    parser.add_argument("--radius_column", default="radius_deg", help="Column for radius (default: radius_deg)")
    parser.add_argument("--round_decimals", type=int, default=4, help="Decimals for rounding coordinates (default: 4)")
    args = parser.parse_args()
    filter_unique_clusters(
        input_csv=args.input,
        output_csv=args.output,
        center_columns=args.center,
        s_column=args.s_column,
        radius_column=args.radius_column,
        round_decimals=args.round_decimals
    ) 