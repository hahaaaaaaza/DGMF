"""Rebuild the 12 manuscript datasets from TDC and fixed split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--tdc-cache", type=Path, default=ROOT / "data" / "tdc_cache")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def canonical_label(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return format(float(value), ".15g")
    except (TypeError, ValueError):
        return str(value).strip()


def sample_key(drug_id: object, smiles: object, label: object) -> str:
    payload = (
        f"{str(drug_id).strip()}\x1f{str(smiles).strip()}\x1f{canonical_label(label)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_endpoint_config() -> dict[str, dict[str, str]]:
    path = ROOT / "configs" / "endpoints.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_tdc_frame(config: dict[str, str], cache_dir: Path) -> pd.DataFrame:
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError as exc:
        raise RuntimeError('PyTDC is required. Install this project with `pip install -e ".[paper]"`.') from exc

    loader_cls = ADME if config["loader"] == "ADME" else Tox
    dataset = loader_cls(name=config["tdc_name"], path=str(cache_dir))
    frame = dataset.get_data().copy()
    required = {"Drug", "Y"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"TDC dataset {config['tdc_name']} is missing columns: {sorted(missing)}")
    if "Drug_ID" not in frame.columns:
        frame.insert(0, "Drug_ID", [f"row_{index}" for index in range(len(frame))])
    return frame[["Drug_ID", "Drug", "Y"]]


def load_split_lookup(task: str, seed: int) -> dict[str, str]:
    path = ROOT / "data" / "split_manifests" / task / f"seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row["sample_key"]: row["split"] for row in rows}


def validate_and_clean(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    valid_rows = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.Y) or not str(row.Drug).strip():
            continue
        if Chem.MolFromSmiles(str(row.Drug)) is None:
            continue
        valid_rows.append({"Drug_ID": row.Drug_ID, "Drug": row.Drug, "Y": row.Y})
    cleaned = pd.DataFrame(valid_rows)
    if cleaned.empty:
        raise RuntimeError(f"No valid molecules remained for {task}")
    cleaned["sample_key"] = [
        sample_key(drug_id, smiles, label)
        for drug_id, smiles, label in zip(
            cleaned["Drug_ID"], cleaned["Drug"], cleaned["Y"]
        )
    ]
    if cleaned["sample_key"].duplicated().any():
        raise RuntimeError(f"Duplicate TDC identifier/SMILES/label tuples found for {task}")
    return cleaned


def prepare_task(
    task: str,
    config: dict[str, str],
    output_root: Path,
    cache_dir: Path,
    force: bool,
) -> None:
    frame = validate_and_clean(load_tdc_frame(config, cache_dir), task)
    observed_keys = set(frame["sample_key"])
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(1, 6):
        output_path = task_dir / f"seed_{seed}.csv"
        if output_path.exists() and not force:
            print(f"[skip] {output_path}")
            continue
        lookup = load_split_lookup(task, seed)
        if len(lookup) != int(config["retained_size"]):
            raise RuntimeError(
                f"Manifest size mismatch for {task}/seed_{seed}: "
                f"expected {config['retained_size']}, found {len(lookup)}"
            )
        missing = observed_keys - set(lookup)
        unexpected = set(lookup) - observed_keys
        if missing or unexpected:
            raise RuntimeError(
                f"Dataset version mismatch for {task}/seed_{seed}: "
                f"{len(missing)} unrecognized downloaded samples and "
                f"{len(unexpected)} manifest samples not found."
            )
        output = frame[["Drug_ID", "Drug", "Y"]].copy()
        output["split"] = frame["sample_key"].map(lookup)
        if output["split"].isna().any():
            raise RuntimeError(f"Incomplete split assignment for {task}/seed_{seed}")
        output.to_csv(output_path, index=False)
        counts = output["split"].value_counts().to_dict()
        print(f"[write] {output_path} | {counts}")


def main() -> None:
    args = parse_args()
    configs = load_endpoint_config()
    tasks = list(configs) if args.tasks == ["all"] else args.tasks
    unknown = sorted(set(tasks) - set(configs))
    if unknown:
        raise ValueError(f"Unknown endpoints: {unknown}")
    args.tdc_cache.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        prepare_task(task, configs[task], args.output_root, args.tdc_cache, args.force)


if __name__ == "__main__":
    main()
