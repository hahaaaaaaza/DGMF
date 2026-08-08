"""Build hash-only split manifests from the processed paper CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "split_manifests",
    )
    return parser.parse_args()


def canonical_label(value: str) -> str:
    try:
        return format(float(value), ".15g")
    except ValueError:
        return value.strip()


def sample_key(drug_id: str, smiles: str, label: str) -> str:
    payload = f"{drug_id.strip()}\x1f{smiles.strip()}\x1f{canonical_label(label)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(source: Path, output: Path) -> int:
    lookup: dict[str, str] = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"Drug_ID", "Drug", "Y", "split"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise RuntimeError(f"{source} must contain {sorted(required)}")
        for row in rows:
            key = sample_key(row["Drug_ID"], row["Drug"], row["Y"])
            split = row["split"].strip().lower()
            if split not in {"train", "val", "test"}:
                raise RuntimeError(f"Invalid split {split!r} in {source}")
            previous = lookup.setdefault(key, split)
            if previous != split:
                raise RuntimeError(
                    f"The same molecule/label key occurs in both {previous} and {split}: {source}"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_key", "split"])
        writer.writerows(sorted(lookup.items()))
    return len(lookup)


def main() -> None:
    args = parse_args()
    endpoints = json.loads((ROOT / "configs" / "endpoints.json").read_text(encoding="utf-8"))
    total = 0
    for task in endpoints:
        for seed in range(1, 6):
            source = args.source_root / task / f"seed_{seed}.csv"
            if not source.exists():
                raise FileNotFoundError(source)
            output = args.output_root / task / f"seed_{seed}.csv"
            count = build_manifest(source, output)
            total += count
            print(f"{task} seed {seed}: {count} unique sample keys")
    print(f"Wrote 60 split manifests containing {total} task-seed assignments.")


if __name__ == "__main__":
    main()
