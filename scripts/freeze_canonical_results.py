"""Validate and freeze the complete deterministic paper result suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN_VARIANTS = (
    "full",
    "concat",
    "without_semantic",
    "without_topological",
    "without_geometric",
)
MECHANISM_VARIANTS = ("target_agnostic", "shared_gate")
SEEDS = (1, 2, 3, 4, 5)


def selected_variants(suite: str) -> tuple[str, ...]:
    if suite == "main":
        return MAIN_VARIANTS
    if suite == "mechanism":
        return MECHANISM_VARIANTS
    return MAIN_VARIANTS + MECHANISM_VARIANTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tasks() -> tuple[str, ...]:
    endpoints = json.loads((ROOT / "configs" / "endpoints.json").read_text(encoding="utf-8"))
    return tuple(endpoints)


def validate_summary(
    summary: pd.DataFrame, tasks: tuple[str, ...], variants: tuple[str, ...]
) -> None:
    required = {
        "endpoint",
        "task",
        "task_type",
        "variant",
        "primary_metric",
        "mean",
        "std",
        "seeds",
    }
    missing_columns = required - set(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary is missing columns: {sorted(missing_columns)}")

    expected = {(variant, task) for variant in variants for task in tasks}
    observed = set(zip(summary["variant"], summary["task"]))
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(f"Incomplete canonical summary. Missing={missing}; unexpected={unexpected}")
    if len(summary) != len(expected):
        raise ValueError("Canonical summary contains duplicate variant/task rows")
    if not (summary["seeds"] == len(SEEDS)).all():
        incomplete = summary.loc[summary["seeds"] != len(SEEDS), ["variant", "task", "seeds"]]
        raise ValueError(f"Non-five-seed rows:\n{incomplete.to_string(index=False)}")
    if summary[["mean", "std"]].isna().any().any():
        raise ValueError("Canonical summary contains missing mean or standard deviation values")


def validate_variant_manifest(path: Path, variant: str, tasks: tuple[str, ...]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    if experiment.get("variant") != variant:
        raise ValueError(f"Variant mismatch in {path}")
    if tuple(experiment.get("tasks", [])) != tasks:
        raise ValueError(f"Task list mismatch in {path}")
    if tuple(experiment.get("seeds", [])) != SEEDS:
        raise ValueError(f"Seed list mismatch in {path}")
    if not experiment.get("data_seed_equals_scaffold_seed"):
        raise ValueError(f"Data seed policy is not fixed in {path}")
    if not experiment.get("pytorch_seed_equals_scaffold_seed"):
        raise ValueError(f"PyTorch seed policy is not fixed in {path}")


def artifact_rows(
    output_root: Path, tasks: tuple[str, ...], variants: tuple[str, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in variants:
        manifest_path = output_root / variant / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        validate_variant_manifest(manifest_path, variant, tasks)
        paths = [manifest_path, output_root / variant / "seed_metrics.csv", output_root / variant / "summary.csv"]
        for task in tasks:
            for seed in SEEDS:
                seed_root = output_root / variant / task / f"seed_{seed}"
                paths.extend(
                    [
                        seed_root / "command.txt",
                        seed_root / "config.toml",
                        seed_root / "model_0" / "test_predictions.csv",
                        seed_root / "model_0" / "best.pt",
                    ]
                )
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "variant": variant,
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "canonical")
    parser.add_argument("--suite", choices=["main", "mechanism", "all"], default="main")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    summary_path = output_root / "paper_suite_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    tasks = expected_tasks()
    variants = selected_variants(args.suite)
    summary = pd.read_csv(summary_path)
    validate_summary(summary, tasks, variants)

    paper_metrics = summary.copy()
    paper_metrics["paper_value"] = [
        f"{mean:.4f} ({std:.4f})" for mean, std in zip(paper_metrics["mean"], paper_metrics["std"])
    ]
    paper_metrics.to_csv(output_root / "paper_metrics_4dp.csv", index=False)

    artifacts = pd.DataFrame(artifact_rows(output_root, tasks, variants))
    artifacts.to_csv(output_root / "canonical_artifact_manifest.csv", index=False)
    freeze = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "variants": list(variants),
        "tasks": list(tasks),
        "seeds": list(SEEDS),
        "summary_sha256": sha256(summary_path),
        "paper_metrics_sha256": sha256(output_root / "paper_metrics_4dp.csv"),
        "artifact_manifest_sha256": sha256(output_root / "canonical_artifact_manifest.csv"),
        "artifact_count": len(artifacts),
    }
    (output_root / "canonical_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Frozen {len(summary)} paper rows and {len(artifacts)} artifacts under {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
