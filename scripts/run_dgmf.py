"""Train DGMF on the 12 fixed TDC scaffold-split endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full": {
        "args": ["--x-d-encoder", "dgmf", "--embedding-fusion-variant", "full"],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "concat": {
        "args": ["--x-d-encoder", "threeway"],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "without_semantic": {
        "args": ["--x-d-encoder", "2d3d", "--no-1d-fingerprints"],
        "use_molformer": False,
        "use_gotennet": True,
    },
    "without_topological": {
        "args": ["--x-d-encoder", "1d3d"],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "without_geometric": {
        "args": ["--x-d-encoder", "attention"],
        "use_molformer": True,
        "use_gotennet": False,
    },
    "shared_gate": {
        "args": ["--x-d-encoder", "dgmf", "--embedding-fusion-variant", "shared-gate"],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "target_agnostic": {
        "args": [
            "--x-d-encoder",
            "dgmf",
            "--embedding-fusion-variant",
            "matched-target-agnostic",
        ],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "no_residual": {
        "args": ["--x-d-encoder", "dgmf", "--embedding-fusion-variant", "no-residual"],
        "use_molformer": True,
        "use_gotennet": True,
    },
    "self_attention": {
        "args": [
            "--x-d-encoder",
            "dgmf",
            "--embedding-fusion-variant",
            "self-attention",
        ],
        "use_molformer": True,
        "use_gotennet": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "main")
    parser.add_argument("--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--molformer-cache-dir", type=Path, default=ROOT / "molformer_1d_cache")
    parser.add_argument("--geometry-cache-dir", type=Path, default=ROOT / "gotennet_3d_cache")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--save-directional-messages", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_configs() -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    endpoints = json.loads((ROOT / "configs" / "endpoints.json").read_text(encoding="utf-8"))
    params = pd.read_csv(ROOT / "configs" / "best_hyperparameters.csv").set_index("endpoint")
    return endpoints, params


def model_identifier(value: str) -> str:
    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    source_roots = [ROOT / "dgmf", ROOT / "chemprop", ROOT / "scripts"]
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def data_hashes(tasks: list[str], seeds: list[int], data_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for task in tasks:
        for seed in seeds:
            path = data_root / task / f"seed_{seed}.csv"
            hashes[f"{task}/seed_{seed}.csv"] = file_sha256(path)
    return hashes


def nvidia_smi_output() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_run_manifest(args: argparse.Namespace, tasks: list[str], variant_root: Path) -> None:
    endpoints_path = ROOT / "configs" / "endpoints.json"
    hyperparameters_path = ROOT / "configs" / "best_hyperparameters.csv"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "chemprop",
                "transformers",
                "torch-geometric",
                "rdkit",
                "optuna",
            ]
        },
        "experiment": {
            "variant": args.variant,
            "tasks": tasks,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "molformer_model": model_identifier(args.molformer_model),
            "data_root": str(args.data_root.resolve()),
            "data_seed_equals_scaffold_seed": True,
            "pytorch_seed_equals_scaffold_seed": True,
        },
        "configuration_hashes": {
            "endpoints.json": file_sha256(endpoints_path),
            "best_hyperparameters.csv": file_sha256(hyperparameters_path),
            "source_tree": source_tree_sha256(),
        },
        "data_hashes": data_hashes(tasks, args.seeds, args.data_root),
        "nvidia_smi": nvidia_smi_output(),
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    (variant_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_command_record(output_dir: Path, command: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def completed_prediction(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("model_*/test_predictions.csv"))
    return candidates[0] if candidates else None


def build_command(
    args: argparse.Namespace,
    task: str,
    config: dict[str, str],
    params: pd.Series,
    seed: int,
    output_dir: Path,
) -> list[str]:
    data_path = args.data_root / task / f"seed_{seed}.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run `python scripts/prepare_tdc12.py` first."
        )
    metrics = ["roc"] if config["task_type"] == "classification" else ["mae", "spearman"]
    warmup = min(int(params["warmup_epochs"]), max(0, args.epochs - 1))
    variant = VARIANTS[args.variant]
    command = [
        sys.executable,
        "-u",
        "-m",
        "dgmf",
        "train",
        "--data-path",
        str(data_path.resolve()),
        "--smiles-columns",
        "Drug",
        "--target-columns",
        "Y",
        "--splits-column",
        "split",
        "--data-seed",
        str(seed),
        "--pytorch-seed",
        str(seed),
        "--task-type",
        config["task_type"],
        "--metrics",
        *metrics,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--warmup-epochs",
        str(warmup),
        "--init-lr",
        str(params["init_lr"]),
        "--max-lr",
        str(params["max_lr"]),
        "--final-lr",
        str(params["final_lr"]),
        "--batch-size",
        str(int(params["batch_size"])),
        "--message-hidden-dim",
        str(int(params["message_hidden_dim"])),
        "--depth",
        str(int(params["depth"])),
        "--ffn-hidden-dim",
        str(int(params["ffn_hidden_dim"])),
        "--ffn-num-layers",
        str(int(params["ffn_num_layers"])),
        "--dropout",
        str(params["dropout"]),
        "--x-d-fp-encoder",
        "itransformer",
        "--x-d-fp-groups",
        str(int(params["fp_groups"])),
        "--x-d-embed-dim",
        str(int(params["embed_dim"])),
        "--x-d-encoder-heads",
        str(int(params["encoder_heads"])),
        "--show-individual-scores",
        "--output-dir",
        str(output_dir.resolve()),
        "--num-workers",
        str(args.num_workers),
        "--accelerator",
        args.accelerator,
        "--devices",
        args.devices,
        *variant["args"],
    ]
    if variant["use_molformer"]:
        command.extend(
            [
                "--use-molformer-1d",
                "--molformer-cache-dir",
                str(args.molformer_cache_dir.resolve()),
                "--molformer-model",
                model_identifier(args.molformer_model),
                "--molformer-device",
                "cpu",
                "--molformer-max-length",
                "256",
                "--molformer-pooling",
                "pooler",
                "--molformer-batch-size",
                "32",
            ]
        )
    if variant["use_gotennet"]:
        command.extend(
            [
                "--use-gotennet-3d-graph",
                "--gotennet-cutoff",
                str(params["gotennet_cutoff"]),
                "--gotennet-pooling",
                str(params["gotennet_pooling"]),
                "--gotennet-lr-scale",
                str(params["gotennet_lr_scale"]),
                "--geometry-cache-dir",
                str(args.geometry_cache_dir.resolve()),
                "--geometry-num-conformers",
                "8",
            ]
        )
    if config["primary_metric"] == "spearman":
        command.extend(["--tracking-metric", "spearman"])
    if args.save_directional_messages and args.variant != "concat":
        command.append("--save-dgmf-messages")
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    pytorch_seed = command[command.index("--pytorch-seed") + 1]
    env["PYTHONHASHSEED"] = pytorch_seed
    env["PYTHONPATH"] = str(ROOT) if not env.get("PYTHONPATH") else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def metric_row(
    task: str,
    config: dict[str, str],
    seed: int,
    data_path: Path,
    prediction_path: Path,
    variant: str,
) -> dict[str, object]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    data = pd.read_csv(data_path)
    truth = data[data["split"] == "test"][["Drug", "Y"]].reset_index(drop=True)
    predictions = pd.read_csv(prediction_path)
    prediction_column = predictions.columns[1]
    if len(truth) == len(predictions):
        y_true = pd.to_numeric(truth["Y"], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(predictions[prediction_column], errors="coerce").to_numpy(float)
    else:
        merged = truth.merge(
            predictions[["Drug", prediction_column]].rename(columns={prediction_column: "prediction"}),
            on="Drug",
            how="inner",
        )
        y_true = pd.to_numeric(merged["Y"], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(merged["prediction"], errors="coerce").to_numpy(float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    metric = config["primary_metric"]
    if metric == "roc":
        value = float(roc_auc_score(y_true.astype(int), y_pred))
    elif metric == "mae":
        value = float(mean_absolute_error(y_true, y_pred))
    else:
        value = float(spearmanr(y_true, y_pred).statistic)
    return {
        "endpoint": config["display_name"],
        "task": task,
        "task_type": config["task_type"],
        "variant": variant,
        "seed": seed,
        "primary_metric": metric,
        "value": value,
        "n_test": len(y_true),
    }


def save_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "seed_metrics.csv", index=False)
    grouped = (
        frame.groupby(["endpoint", "task", "task_type", "variant", "primary_metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), seeds=("value", "count"))
    )
    grouped.to_csv(output_root / "summary.csv", index=False)
    print("\n" + grouped.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    endpoints, params = load_configs()
    tasks = list(endpoints) if args.tasks == ["all"] else args.tasks
    unknown = sorted(set(tasks) - set(endpoints))
    if unknown:
        raise ValueError(f"Unknown endpoints: {unknown}")

    variant_root = args.output_root / args.variant
    if not args.dry_run:
        write_run_manifest(args, tasks, variant_root)

    rows: list[dict[str, object]] = []
    for task in tasks:
        config = endpoints[task]
        display_name = config["display_name"]
        if display_name not in params.index:
            raise KeyError(f"No selected hyperparameters for {display_name}")
        for seed in args.seeds:
            output_dir = variant_root / task / f"seed_{seed}"
            prediction_path = completed_prediction(output_dir)
            if args.force or prediction_path is None:
                print(f"\n[run] {task} | {args.variant} | seed {seed}", flush=True)
                command = build_command(args, task, config, params.loc[display_name], seed, output_dir)
                if not args.dry_run:
                    write_command_record(output_dir, command)
                run_command(command, args.dry_run)
                prediction_path = completed_prediction(output_dir)
            else:
                print(f"[skip] {task} | {args.variant} | seed {seed}", flush=True)
            if not args.dry_run:
                if prediction_path is None:
                    raise RuntimeError(f"No test predictions found under {output_dir}")
                rows.append(
                    metric_row(
                        task,
                        config,
                        seed,
                        args.data_root / task / f"seed_{seed}.csv",
                        prediction_path,
                        args.variant,
                    )
                )
                save_summary(rows, variant_root)


if __name__ == "__main__":
    main()
