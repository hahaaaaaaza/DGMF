"""Run the deterministic DGMF experiments reported in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--suite", choices=["main", "mechanism", "all"], default="main")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "canonical")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--molformer-cache-dir", type=Path, default=ROOT / "molformer_1d_cache")
    parser.add_argument("--geometry-cache-dir", type=Path, default=ROOT / "gotennet_3d_cache")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_variants(suite: str) -> tuple[str, ...]:
    if suite == "main":
        return MAIN_VARIANTS
    if suite == "mechanism":
        return MECHANISM_VARIANTS
    return MAIN_VARIANTS + MECHANISM_VARIANTS


def run_variant(args: argparse.Namespace, variant: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_dgmf.py"),
        "--variant",
        variant,
        "--tasks",
        *args.tasks,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--output-root",
        str(args.output_root),
        "--data-root",
        str(args.data_root),
        "--molformer-model",
        args.molformer_model,
        "--molformer-cache-dir",
        str(args.molformer_cache_dir),
        "--geometry-cache-dir",
        str(args.geometry_cache_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num-workers",
        str(args.num_workers),
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def combine_summaries(output_root: Path, variants: tuple[str, ...]) -> Path:
    frames = []
    for variant in variants:
        path = output_root / variant / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed summary: {path}")
        frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "paper_suite_summary.csv"
    combined.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    variants = selected_variants(args.suite)
    for variant in variants:
        run_variant(args, variant)
    if not args.dry_run:
        output_path = combine_summaries(args.output_root, variants)
        print(f"Saved canonical paper summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
