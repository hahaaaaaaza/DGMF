from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chemprop.nn.fingerprint_encoder import DGMFFusionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    prefixes = ("cross_gate.", "shared_gate_input_scale", "shared_gate_output_scale")
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes)
    )


def main() -> None:
    args = parse_args()
    rows = []
    for d_h in (300, 600):
        full = build(d_h, "full")
        matched = build(d_h, "matched-shared-gate")
        full_gate = gate_parameter_count(full)
        matched_gate = gate_parameter_count(matched)
        full_total = sum(parameter.numel() for parameter in full.parameters())
        matched_total = sum(parameter.numel() for parameter in matched.parameters())
        rows.append(
            {
                "message_hidden_dim": d_h,
                "embedding_dim": 128,
                "full_gate_parameters": full_gate,
                "matched_shared_gate_parameters": matched_gate,
                "gate_parameter_difference": matched_gate - full_gate,
                "full_fusion_encoder_parameters": full_total,
                "matched_shared_fusion_encoder_parameters": matched_total,
                "fusion_encoder_parameter_difference": matched_total - full_total,
                "exact_match": full_gate == matched_gate and full_total == matched_total,
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
