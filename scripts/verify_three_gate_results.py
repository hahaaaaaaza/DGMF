from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TASKS = {
    "bioavailability_ma",
    "hia_hou",
    "pgp_broccatelli",
    "bbb_martins",
    "vdss_lombardo",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_veith",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "herg",
    "ld50_zhu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def validate_variant(root: Path, expected_variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(root / "seed_metrics.csv")
    summary = pd.read_csv(root / "summary.csv")
    counts = seeds.groupby("task")["seed"].nunique()
    predictions = len(list(root.glob("*/seed_*/model_*/test_predictions.csv")))
    errors = []
    if set(seeds["task"]) != EXPECTED_TASKS:
        errors.append("task set mismatch")
    if len(seeds) != 60 or not (counts == 5).all():
        errors.append("expected 12 tasks x 5 seeds")
    if len(summary) != 12 or not (summary["seeds"] == 5).all():
        errors.append("summary is incomplete")
    if predictions != 60:
        errors.append(f"expected 60 predictions, found {predictions}")
    if not np.isfinite(seeds["value"]).all():
        errors.append("non-finite metric")
    if set(seeds["variant"]) != {expected_variant}:
        errors.append(f"unexpected variant labels: {set(seeds['variant'])}")
    if errors:
        raise RuntimeError(f"{expected_variant}: " + "; ".join(errors))
    return seeds, summary


def adjusted_value(metric: pd.Series, value: pd.Series) -> pd.Series:
    return np.where(metric.isin(["roc", "spearman"]), value, -value)


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.parameter_audit)
    if len(audit) != 6 or not audit["exact_match"].all():
        raise RuntimeError("Three-way parameter audit is incomplete or not exact")

    full_seeds, full_summary = validate_variant(args.full_root, "full")
    shared_seeds, shared_summary = validate_variant(
        args.shared_root, "matched_shared_gate"
    )
    direction_seeds, direction_summary = validate_variant(
        args.direction_root, "direction_id_gate"
    )

    keys = ["endpoint", "task", "task_type", "primary_metric"]
    summary = full_summary[keys + ["mean", "std"]].rename(
        columns={"mean": "independent_mean", "std": "independent_sd"}
    )
    summary = summary.merge(
        shared_summary[keys + ["mean", "std"]].rename(
            columns={"mean": "shared_mean", "std": "shared_sd"}
        ),
        on=keys,
        validate="one_to_one",
    ).merge(
        direction_summary[keys + ["mean", "std"]].rename(
            columns={"mean": "direction_id_mean", "std": "direction_id_sd"}
        ),
        on=keys,
        validate="one_to_one",
    )
    for name in ("shared", "direction_id", "independent"):
        summary[f"{name}_adjusted"] = adjusted_value(
            summary["primary_metric"], summary[f"{name}_mean"]
        )
    adjusted_columns = [
        "shared_adjusted",
        "direction_id_adjusted",
        "independent_adjusted",
    ]
    summary["best_variant"] = (
        summary[adjusted_columns]
        .idxmax(axis=1)
        .str.replace("_adjusted", "", regex=False)
    )
    summary.drop(columns=adjusted_columns).to_csv(
        args.output_root / "three_gate_summary_comparison.csv", index=False
    )

    seed_keys = ["endpoint", "task", "seed", "task_type", "primary_metric"]
    paired = full_seeds[seed_keys + ["value"]].rename(
        columns={"value": "independent_value"}
    )
    paired = paired.merge(
        shared_seeds[seed_keys + ["value"]].rename(
            columns={"value": "shared_value"}
        ),
        on=seed_keys,
        validate="one_to_one",
    ).merge(
        direction_seeds[seed_keys + ["value"]].rename(
            columns={"value": "direction_id_value"}
        ),
        on=seed_keys,
        validate="one_to_one",
    )
    for name in ("shared", "direction_id", "independent"):
        paired[f"{name}_adjusted"] = adjusted_value(
            paired["primary_metric"], paired[f"{name}_value"]
        )
    paired["best_variant"] = (
        paired[[f"{name}_adjusted" for name in ("shared", "direction_id", "independent")]]
        .idxmax(axis=1)
        .str.replace("_adjusted", "", regex=False)
    )
    paired.to_csv(args.output_root / "three_gate_paired_seed_comparison.csv", index=False)
    wins = paired["best_variant"].value_counts().rename_axis("variant").reset_index(name="seed_wins")
    wins.to_csv(args.output_root / "three_gate_seed_wins.csv", index=False)

    completion = pd.DataFrame(
        [
            {
                "variants": 3,
                "endpoints_per_variant": 12,
                "seeds_per_endpoint": 5,
                "seed_runs_per_variant": 60,
                "prediction_files_per_variant": 60,
                "parameter_match_exact": True,
            }
        ]
    )
    completion.to_csv(args.output_root / "completion_audit.csv", index=False)
    print(summary.drop(columns=adjusted_columns, errors="ignore").to_string(index=False))
    print("\nSeed-level wins:\n" + wins.to_string(index=False))


if __name__ == "__main__":
    main()
