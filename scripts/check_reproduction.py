"""Compare an observed DGMF summary with the manuscript reference metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "results"
        / "reference"
        / "paper_dgmf_primary_metrics_full_precision.csv",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    observed = pd.read_csv(args.observed)
    reference = pd.read_csv(args.reference)
    merge_keys = ["task", "primary_metric"]
    if "variant" in reference.columns:
        merge_keys.append("variant")
    merged = reference.merge(
        observed[[*merge_keys, "mean"]].rename(columns={"mean": "observed_mean"}),
        on=merge_keys,
        how="left",
    )
    merged["absolute_difference"] = (merged["observed_mean"] - merged["mean"]).abs()
    merged["within_tolerance"] = merged["absolute_difference"] <= args.absolute_tolerance
    display_columns = ["endpoint"]
    if "variant" in merged.columns:
        display_columns.append("variant")
    display_columns.extend(
        [
            "primary_metric",
            "mean",
            "observed_mean",
            "absolute_difference",
            "within_tolerance",
        ]
    )
    print(merged[display_columns].to_string(index=False))
    if merged["observed_mean"].isna().any() or not merged["within_tolerance"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
