from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chemprop.nn.fingerprint_encoder import DGMFFusionEncoder


VARIANTS = ("matched-shared-gate", "direction-id-gate", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build(d_h: int, variant: str) -> DGMFFusionEncoder:
    return DGMFFusionEncoder(
        d_h=d_h,
        d_xd_in=776,
        d_xd_out=128,
        d_3d=8,
        use_3d_graph=False,
        graph_num_layers=2,
        fp_groups=128,
        dropout=0.0,
        nhead=4,
        fusion_variant=variant,
    )


def gate_parameter_count(module: DGMFFusionEncoder) -> int:
    prefixes = (
        "cross_gate.",
        "shared_gate_input_scale",
        "shared_gate_output_scale",
        "direction_gate_trunk.",
        "direction_gate_trunk_scale",
        "direction_gate_heads.",
    )
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes)
    )


def main() -> None:
    args = parse_args()
    rows = []
    for d_h in (300, 600):
        modules = {variant: build(d_h, variant) for variant in VARIANTS}
        reference_gate = gate_parameter_count(modules["full"])
        reference_total = sum(p.numel() for p in modules["full"].parameters())
        for variant, module in modules.items():
            gate_count = gate_parameter_count(module)
            total_count = sum(p.numel() for p in module.parameters())
            rows.append(
                {
                    "message_hidden_dim": d_h,
                    "variant": variant,
                    "gate_parameters": gate_count,
                    "gate_difference_vs_full": gate_count - reference_gate,
                    "fusion_encoder_parameters": total_count,
                    "encoder_difference_vs_full": total_count - reference_total,
                    "exact_match": gate_count == reference_gate
                    and total_count == reference_total,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame["exact_match"].all():
        raise RuntimeError(f"Parameter matching failed:\n{frame.to_string(index=False)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
