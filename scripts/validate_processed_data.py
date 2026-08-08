"""Verify processed training CSVs against the committed hash-only split manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prepare_tdc12 import sample_key


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_mapping(path: Path, processed: bool) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None:
            raise RuntimeError(f"Missing header: {path}")
        required = {"Drug_ID", "Drug", "Y", "split"} if processed else {"sample_key", "split"}
        missing = required - set(rows.fieldnames)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        for row in rows:
            key = sample_key(row["Drug_ID"], row["Drug"], row["Y"]) if processed else row["sample_key"]
            split = row["split"].strip().lower()
            if key in mapping:
                raise RuntimeError(f"Duplicate sample key in {path}: {key}")
            mapping[key] = split
    return mapping


def main() -> None:
    args = parse_args()
    endpoints = json.loads((ROOT / "configs" / "endpoints.json").read_text(encoding="utf-8"))
    tasks = list(endpoints) if args.tasks == ["all"] else args.tasks
    records = []
    for task in tasks:
        for seed in range(1, 6):
            processed_path = args.data_root / task / f"seed_{seed}.csv"
            manifest_path = ROOT / "data" / "split_manifests" / task / f"seed_{seed}.csv"
            if not processed_path.exists():
                raise FileNotFoundError(processed_path)
            processed = load_mapping(processed_path, processed=True)
            manifest = load_mapping(manifest_path, processed=False)
            if processed != manifest:
                missing = set(manifest) - set(processed)
                unexpected = set(processed) - set(manifest)
                changed = {key for key in set(processed) & set(manifest) if processed[key] != manifest[key]}
                raise RuntimeError(
                    f"Data mismatch for {task}/seed_{seed}: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}, changed_split={len(changed)}"
                )
            records.append({"task": task, "seed": seed, "samples": len(processed), "status": "exact"})
            print(f"[exact] {task}/seed_{seed}: {len(processed)} samples")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(records)} task-seed files against the committed manifests.")


if __name__ == "__main__":
    main()
