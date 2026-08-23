from chemprop.nn.fingerprint_encoder import DGMFFusionEncoder


def gate_parameter_count(module: DGMFFusionEncoder) -> int:
    prefixes = ("cross_gate.", "shared_gate_input_scale", "shared_gate_output_scale")
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes)
    )


def build(variant: str) -> DGMFFusionEncoder:
    return DGMFFusionEncoder(
        d_h=32,
        d_xd_in=24,
        d_xd_out=16,
        d_3d=8,
        use_3d_graph=False,
        graph_num_layers=1,
        fp_groups=4,
        dropout=0.0,
        nhead=4,
        fusion_variant=variant,
    )


def test_matched_shared_gate_has_exact_full_gate_parameter_count() -> None:
    full = build("full")
    matched_shared = build("matched-shared-gate")

    assert gate_parameter_count(matched_shared) == gate_parameter_count(full)
    assert sum(p.numel() for p in matched_shared.parameters()) == sum(
        p.numel() for p in full.parameters()
    )


def test_target_agnostic_matched_shared_gate_has_exact_full_parameter_count() -> None:
    full = build("full")
    matched_shared_target_agnostic = build("matched-shared-target-agnostic")

    assert gate_parameter_count(matched_shared_target_agnostic) == gate_parameter_count(full)
    assert sum(p.numel() for p in matched_shared_target_agnostic.parameters()) == sum(
        p.numel() for p in full.parameters()
    )
