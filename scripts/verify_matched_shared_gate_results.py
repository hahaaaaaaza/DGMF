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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_path = args.variant_root / "seed_metrics.csv"
    summary_path = args.variant_root / "summary.csv"
    if not seed_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Missing seed_metrics.csv or summary.csv")

    seeds = pd.read_csv(seed_path)
    summary = pd.read_csv(summary_path)
    observed_tasks = set(seeds["task"])
    counts = seeds.groupby("task")["seed"].nunique()
    prediction_count = len(list(args.variant_root.glob("*/seed_*/model_*/test_predictions.csv")))
    errors = []
    if observed_tasks != EXPECTED_TASKS:
        errors.append(f"task mismatch: {sorted(observed_tasks ^ EXPECTED_TASKS)}")
    if len(seeds) != 60:
        errors.append(f"expected 60 seed rows, found {len(seeds)}")
    if not (counts == 5).all():
        errors.append(f"incomplete seeds:\n{counts[counts != 5].to_string()}")
    if len(summary) != 12 or not (summary["seeds"] == 5).all():
        errors.append("summary does not contain 12 endpoints with five seeds each")
    if prediction_count != 60:
        errors.append(f"expected 60 prediction files, found {prediction_count}")
    if not np.isfinite(seeds["value"]).all():
        errors.append("non-finite primary metric found")
    if set(seeds["variant"]) != {"matched_shared_gate"}:
        errors.append(f"unexpected variants: {sorted(set(seeds['variant']))}")
    if errors:
        raise RuntimeError("\n".join(errors))

    report = {
        "completed_seed_runs": len(seeds),
        "completed_endpoints": len(summary),
        "prediction_files": prediction_count,
        "all_finite": True,
    }
    pd.DataFrame([report]).to_csv(args.variant_root / "completion_audit.csv", index=False)

    if args.full_summary:
        full = pd.read_csv(args.full_summary).rename(
            columns={"mean": "full_mean", "std": "full_std", "seeds": "full_seeds"}
        )
        matched = summary.rename(
            columns={
                "mean": "matched_shared_mean",
                "std": "matched_shared_std",
                "seeds": "matched_shared_seeds",
            }
        )
        keys = ["endpoint", "task", "task_type", "primary_metric"]
        comparison = full.merge(matched, on=keys, validate="one_to_one")
        higher_is_better = comparison["primary_metric"].isin(["roc", "spearman"])
        comparison["full_minus_matched_adjusted"] = np.where(
            higher_is_better,
            comparison["full_mean"] - comparison["matched_shared_mean"],
            comparison["matched_shared_mean"] - comparison["full_mean"],
        )
        comparison["better_model"] = np.where(
            comparison["full_minus_matched_adjusted"] > 0,
            "full_direction_specific",
            np.where(
                comparison["full_minus_matched_adjusted"] < 0,
                "matched_shared_gate",
                "tie",
            ),
        )
        comparison.to_csv(args.variant_root / "comparison_vs_full.csv", index=False)
        print(comparison.to_string(index=False))

    print(f"COMPLETE: {report}")


if __name__ == "__main__":
    main()
