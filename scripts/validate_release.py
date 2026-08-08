"""Dependency-free structural validation for a DGMF source release."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    endpoints = json.loads((ROOT / "configs" / "endpoints.json").read_text(encoding="utf-8"))
    if len(endpoints) != 12:
        raise RuntimeError(f"Expected 12 endpoints, found {len(endpoints)}")

    with (ROOT / "configs" / "best_hyperparameters.csv").open(encoding="utf-8", newline="") as handle:
        params = list(csv.DictReader(handle))
    if len(params) != 12:
        raise RuntimeError(f"Expected 12 hyperparameter rows, found {len(params)}")

    reference_dir = ROOT / "results" / "reference"
    with (reference_dir / "paper_dgmf_primary_metrics_full_precision.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        primary_rows = list(csv.DictReader(handle))
    if len(primary_rows) != 12 or {row["task"] for row in primary_rows} != set(endpoints):
        raise RuntimeError("The full-precision primary reference must contain all 12 endpoints")

    with (reference_dir / "paper_ablation_metrics_full_precision.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        ablation_rows = list(csv.DictReader(handle))
    expected_variants = {
        "concat",
        "without_semantic",
        "without_topological",
        "without_geometric",
    }
    if len(ablation_rows) != 48:
        raise RuntimeError(f"Expected 48 ablation reference rows, found {len(ablation_rows)}")
    for task in endpoints:
        variants = {row["variant"] for row in ablation_rows if row["task"] == task}
        if variants != expected_variants:
            raise RuntimeError(f"Incomplete ablation variants for {task}: {sorted(variants)}")

    source_path = ROOT / "chemprop" / "nn" / "fingerprint_encoder.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if "DGMFFusionEncoder" not in classes:
        raise RuntimeError("DGMFFusionEncoder was not found")

    public_api = (ROOT / "dgmf" / "__init__.py").read_text(encoding="utf-8")
    if "DGMFFusionEncoder" not in public_api:
        raise RuntimeError("The public dgmf package does not export DGMFFusionEncoder")

    missing = []
    for task, config in endpoints.items():
        for seed in range(1, 6):
            path = ROOT / "data" / "split_manifests" / task / f"seed_{seed}.csv"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected = int(config["retained_size"])
            if len(rows) != expected:
                raise RuntimeError(
                    f"{path.relative_to(ROOT)} has {len(rows)} rows; expected {expected}"
                )
            keys = [row["sample_key"] for row in rows]
            if len(set(keys)) != len(keys) or any(len(key) != 64 for key in keys):
                raise RuntimeError(f"Invalid or duplicate hashes in {path.relative_to(ROOT)}")
            if {row["split"] for row in rows} - {"train", "val", "test"}:
                raise RuntimeError(f"Invalid split label in {path.relative_to(ROOT)}")
    if missing:
        raise RuntimeError("Missing split manifests:\n" + "\n".join(missing))
    print(
        "DGMF release structure is valid: 12 endpoints, 60 complete split "
        "manifests, full-precision primary and ablation references, and the "
        "DGMF model class."
    )


if __name__ == "__main__":
    main()
