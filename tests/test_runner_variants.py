from argparse import Namespace
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_dgmf", ROOT / "scripts" / "run_dgmf.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def build(variant: str) -> list[str]:
    args = Namespace(
        variant=variant,
        data_root=ROOT / "data" / "split_manifests",
        output_root=ROOT / "results" / "test",
        epochs=80,
        patience=15,
        num_workers=0,
        accelerator="gpu",
        devices="1",
        molformer_model="model",
        molformer_cache_dir=ROOT / "cache" / "molformer",
        geometry_cache_dir=ROOT / "cache" / "geometry",
        save_directional_messages=False,
    )
    config = {
        "display_name": "HIA",
        "task_type": "classification",
        "primary_metric": "roc",
    }
    params = pd.read_csv(ROOT / "configs" / "best_hyperparameters.csv").set_index("endpoint").loc["HIA"]
    return RUNNER.build_command(
        args,
        "hia_hou",
        config,
        params,
        1,
        args.output_root / variant / "hia_hou" / "seed_1",
    )


@pytest.mark.parametrize(
    ("variant", "encoder", "molformer", "gotennet", "no_1d"),
    [
        ("full", "dgmf", True, True, False),
        ("concat", "threeway", True, True, False),
        ("without_semantic", "2d3d", False, True, True),
        ("without_topological", "1d3d", True, True, False),
        ("without_geometric", "attention", True, False, False),
    ],
)
def test_paper_variant_commands(
    variant: str,
    encoder: str,
    molformer: bool,
    gotennet: bool,
    no_1d: bool,
) -> None:
    command = build(variant)
    assert command[command.index("--x-d-encoder") + 1] == encoder
    assert ("--use-molformer-1d" in command) is molformer
    assert ("--use-gotennet-3d-graph" in command) is gotennet
    assert ("--no-1d-fingerprints" in command) is no_1d
    assert command[command.index("--data-seed") + 1] == "1"
    assert command[command.index("--pytorch-seed") + 1] == "1"


@pytest.mark.parametrize(
    ("variant", "fusion_variant"),
    [
        ("shared_gate", "shared-gate"),
        ("matched_shared_gate", "matched-shared-gate"),
        ("target_agnostic", "matched-target-agnostic"),
    ],
)
def test_mechanism_variant_commands(variant: str, fusion_variant: str) -> None:
    command = build(variant)
    assert command[command.index("--x-d-encoder") + 1] == "dgmf"
    assert command[command.index("--embedding-fusion-variant") + 1] == fusion_variant
