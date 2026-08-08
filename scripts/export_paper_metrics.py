"""Format a canonical suite summary exactly as values should appear in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decimals", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
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
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not (frame["seeds"] == 5).all():
        incomplete = frame.loc[frame["seeds"] != 5, ["endpoint", "variant", "seeds"]]
        raise ValueError(f"Incomplete five-seed results:\n{incomplete.to_string(index=False)}")
    format_string = f"{{:.{args.decimals}f}} ({{:.{args.decimals}f}})"
    frame["paper_value"] = [
        format_string.format(mean, std) for mean, std in zip(frame["mean"], frame["std"])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} paper rows to {args.output}")


if __name__ == "__main__":
    main()
