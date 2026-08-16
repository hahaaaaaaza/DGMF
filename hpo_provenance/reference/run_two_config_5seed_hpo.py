"""Five-seed averaged Optuna HPO for two 1D/2D/3D fusion configs.

This is intentionally separate from endpoint config selection: both requested
configs get their own study, and each Optuna trial is scored by the mean
validation objective across seeds 1..5.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from run_framework_comparison import SPEARMAN_REGRESSION_TASKS
from run_framework_comparison import configs as regression_configs
from run_framework_comparison import prepare_clean_csv
from run_framework_comparison import REGRESSION_TASKS
from run_framework_comparison import regression_metrics_for_task, spearman_corr
from run_framework_comparison_classification import CLASSIFICATION_TASKS
from run_framework_comparison_classification import configs as classification_configs

try:
    import optuna
except ImportError as exc:  # pragma: no cover - user environment guard
    optuna = None
    OPTUNA_IMPORT_ERROR = exc
else:
    OPTUNA_IMPORT_ERROR = None


NEWDATA_DIR = Path(__file__).parent
REPO_ROOT = NEWDATA_DIR.parents[2]
os.chdir(NEWDATA_DIR)
os.environ["PYTHONIOENCODING"] = "utf-8"

DEFAULT_CONFIGS = [
    "fusion_1d2d3d_molformer_gotennet",
    "fusion_1d2d3d_tri_pair_gated_xattn_molformer_gotennet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--task", default="caco2_wang")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--hpo-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-root", default="results_two_config_5seed_hpo")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-accelerator", default=None)
    parser.add_argument("--train-devices", default=None)
    parser.add_argument("--x-d-fp-encoder", choices=["itransformer", "duet"], default="itransformer")
    parser.add_argument("--molformer-cache-dir", default="molformer_1d_cache")
    parser.add_argument("--molformer-model", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--molformer-device", default="cpu")
    parser.add_argument("--molformer-max-length", type=int, default=256)
    parser.add_argument("--molformer-pooling", choices=["pooler", "cls", "mean"], default="pooler")
    parser.add_argument("--molformer-batch-size", type=int, default=32)
    parser.add_argument("--unimol-cache-dir", default="unimol_3d_cache")
    parser.add_argument("--unimol-model-name", default="unimolv1")
    parser.add_argument("--unimol-model-path", default=None)
    parser.add_argument("--unimol-device", default="cpu")
    parser.add_argument("--unimol-batch-size", type=int, default=32)
    parser.add_argument("--unimol-dim", type=int, default=512)
    parser.add_argument("--unimol-remove-hs", action="store_true")
    parser.add_argument("--trainable-molformer-1d", action="store_true")
    parser.add_argument("--molformer-unfreeze-layers", type=int, default=2)
    parser.add_argument("--molformer-lr-scale", type=float, default=0.05)
    parser.add_argument("--geometry-cache-dir", default="gotennet_3d_cache")
    parser.add_argument("--geometry-num-conformers", type=int, default=8)
    parser.add_argument("--max-lr-low", type=float, default=5e-5)
    parser.add_argument("--max-lr-high", type=float, default=4e-4)
    parser.add_argument("--fixed-batch-size", type=int, default=None)
    parser.add_argument("--fixed-embed-dim", type=int, default=None)
    parser.add_argument("--fixed-x-d-encoder-heads", type=int, default=None)
    parser.add_argument("--fixed-gotennet-cutoff", type=float, default=None)
    parser.add_argument("--fixed-gotennet-pooling", choices=["mean", "mean_max"], default=None)
    parser.add_argument("--fixed-gotennet-lr-scale", type=float, default=None)
    parser.add_argument("--save-xattn-alpha", action="store_true")
    parser.add_argument("--print-xattn-alpha", action="store_true")
    parser.add_argument("--skip-final", action="store_true", help="Only run HPO trials; do not retrain best params.")
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--final-patience", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def objective_metric(task_type: str, task: str) -> str:
    if task_type == "regression" and task in SPEARMAN_REGRESSION_TASKS:
        return "val/spearman"
    return "val_loss" if task_type == "regression" else "val/roc"


def objective_direction(task_type: str, task: str) -> str:
    metric = objective_metric(task_type, task)
    return "minimize" if metric in {"val_loss", "val/mae", "val/rmse"} else "maximize"


def read_objective(output_dir: Path, metric: str) -> float:
    metric_files = sorted(output_dir.glob("**/metrics.csv"))
    for metric_file in reversed(metric_files):
        df = pd.read_csv(metric_file)
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not values.empty:
            return float(values.min() if metric in {"val_loss", "val/rmse", "val/mae"} else values.max())

    if metric == "val_loss":
        checkpoint_values = []
        for checkpoint_file in sorted(output_dir.glob("**/best-epoch=*-val_loss=*.ckpt")):
            try:
                checkpoint_values.append(float(checkpoint_file.stem.rsplit("val_loss=", 1)[1]))
            except (IndexError, ValueError):
                continue
        if checkpoint_values:
            return float(min(checkpoint_values))

    raise RuntimeError(f"Metric {metric!r} was not found under {output_dir}")


def source_configs(
    task_type: str, embed_dim: int, n_heads: int, unimol_dim: int = 512
) -> dict[str, list[str]]:
    source = classification_configs if task_type == "classification" else regression_configs
    return source(embed_dim, n_heads, unimol_dim)


def complete_params_from_trial(
    trial: optuna.trial.BaseTrial | optuna.trial.FrozenTrial,
    args: argparse.Namespace,
) -> dict[str, int | float | str]:
    max_lr = float(trial.params["max_lr"])
    init_lr_ratio = float(trial.params["init_lr_ratio"])
    final_lr_ratio = float(trial.params["final_lr_ratio"])
    embed_dim = int(args.fixed_embed_dim or trial.params["x_d_embed_dim"])
    n_heads = int(args.fixed_x_d_encoder_heads or trial.params["x_d_encoder_heads"])
    return {
        "batch_size": int(args.fixed_batch_size or trial.params["batch_size"]),
        "dropout": float(trial.params["dropout"]),
        "message_hidden_dim": int(trial.params["message_hidden_dim"]),
        "depth": int(trial.params["depth"]),
        "ffn_hidden_dim": int(trial.params["ffn_hidden_dim"]),
        "ffn_num_layers": int(trial.params["ffn_num_layers"]),
        "x_d_embed_dim": embed_dim,
        "x_d_encoder_heads": n_heads,
        "x_d_fp_groups": int(trial.params["x_d_fp_groups"]),
        "gotennet_cutoff": float(args.fixed_gotennet_cutoff or trial.params["gotennet_cutoff"]),
        "gotennet_pooling": str(args.fixed_gotennet_pooling or trial.params["gotennet_pooling"]),
        "gotennet_lr_scale": float(args.fixed_gotennet_lr_scale or trial.params["gotennet_lr_scale"]),
        "max_lr": max_lr,
        "init_lr": max_lr * init_lr_ratio,
        "final_lr": max_lr * final_lr_ratio,
        "warmup_epochs": int(trial.params["warmup_epochs"]),
    }


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, int | float | str]:
    max_lr = trial.suggest_float("max_lr", args.max_lr_low, args.max_lr_high, log=True)
    trial.suggest_float("init_lr_ratio", 0.04, 0.12, log=True)
    trial.suggest_float("final_lr_ratio", 0.04, 0.12, log=True)
    embed_dim = args.fixed_embed_dim or trial.suggest_categorical("x_d_embed_dim", [128])
    valid_heads = [h for h in [4, 8] if embed_dim % h == 0]
    if args.fixed_x_d_encoder_heads is None:
        trial.suggest_categorical("x_d_encoder_heads", valid_heads)

    if args.fixed_batch_size is None:
        trial.suggest_categorical("batch_size", [16, 32])
    trial.suggest_float("dropout", 0.2, 0.35, step=0.05)
    trial.suggest_categorical("message_hidden_dim", [300, 600])
    trial.suggest_int("depth", 3, 4)
    trial.suggest_categorical("ffn_hidden_dim", [300, 512])
    trial.suggest_int("ffn_num_layers", 1, 2)
    trial.suggest_categorical("x_d_fp_groups", [64, 128])
    if args.fixed_gotennet_cutoff is None:
        trial.suggest_categorical("gotennet_cutoff", [4.0, 5.0])
    if args.fixed_gotennet_pooling is None:
        trial.suggest_categorical("gotennet_pooling", ["mean", "mean_max"])
    if args.fixed_gotennet_lr_scale is None:
        trial.suggest_categorical("gotennet_lr_scale", [0.1, 0.3])
    trial.suggest_int("warmup_epochs", 2, max(2, min(4, args.epochs - 1)))
    return complete_params_from_trial(trial, args)


def data_file_for(task: str, seed: int) -> Path:
    data_file = prepare_clean_csv(NEWDATA_DIR / "tdc_admet_group_merged_3d" / task / f"seed_{seed}.csv")
    if data_file is None:
        raise RuntimeError(f"Missing data for {task}/seed_{seed}")
    return data_file


def prediction_file(output_dir: Path) -> Path:
    return output_dir / "model_0" / "test_predictions.csv"


def predictions_have_finite_values(output_dir: Path) -> bool:
    pred_file = prediction_file(output_dir)
    if not pred_file.exists():
        return False
    try:
        pred_df = pd.read_csv(pred_file)
        if pred_df.shape[1] < 2:
            return False
        values = pd.to_numeric(pred_df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    except Exception:
        return False
    return bool(np.isfinite(values).any())


def build_train_cmd(
    args: argparse.Namespace,
    config_name: str,
    params: dict[str, int | float | str],
    seed: int,
    output_dir: Path,
    epochs: int,
    patience: int,
    save_xattn_alpha: bool = False,
) -> list[str]:
    metrics = regression_metrics_for_task(args.task) if args.task_type == "regression" else ["roc", "prc", "accuracy"]
    fp_encoder = "molformer" if args.trainable_molformer_1d else args.x_d_fp_encoder
    all_configs = source_configs(
        args.task_type,
        int(params["x_d_embed_dim"]),
        int(params["x_d_encoder_heads"]),
        args.unimol_dim,
    )
    if config_name not in all_configs:
        raise ValueError(f"Unknown config {config_name}. Available configs: {list(all_configs)}")

    cmd = [
        sys.executable,
        "-m",
        "chemprop.cli.main",
        "train",
        "--data-path",
        str(data_file_for(args.task, seed)),
        "--smiles-columns",
        "Drug",
        "--target-columns",
        "Y",
        "--splits-column",
        "split",
        "--task-type",
        args.task_type,
        "--metrics",
        *metrics,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--warmup-epochs",
        str(params["warmup_epochs"]),
        "--init-lr",
        str(params["init_lr"]),
        "--max-lr",
        str(params["max_lr"]),
        "--final-lr",
        str(params["final_lr"]),
        "--batch-size",
        str(params["batch_size"]),
        "--message-hidden-dim",
        str(params["message_hidden_dim"]),
        "--depth",
        str(params["depth"]),
        "--ffn-hidden-dim",
        str(params["ffn_hidden_dim"]),
        "--ffn-num-layers",
        str(params["ffn_num_layers"]),
        "--dropout",
        str(params["dropout"]),
        "--x-d-fp-encoder",
        fp_encoder,
        "--x-d-fp-groups",
        str(params["x_d_fp_groups"]),
        "--show-individual-scores",
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(args.num_workers),
        "--geometry-cache-dir",
        str((NEWDATA_DIR / args.geometry_cache_dir).resolve()),
        "--geometry-num-conformers",
        str(args.geometry_num_conformers),
        "--molformer-cache-dir",
        str((NEWDATA_DIR / args.molformer_cache_dir).resolve()),
        "--molformer-model",
        args.molformer_model,
        "--molformer-device",
        args.molformer_device,
        "--molformer-max-length",
        str(args.molformer_max_length),
        "--molformer-pooling",
        args.molformer_pooling,
        "--molformer-batch-size",
        str(args.molformer_batch_size),
        "--unimol-cache-dir",
        str((NEWDATA_DIR / args.unimol_cache_dir).resolve()),
        "--unimol-model-name",
        args.unimol_model_name,
        "--unimol-device",
        args.unimol_device,
        "--unimol-batch-size",
        str(args.unimol_batch_size),
        "--gotennet-cutoff",
        str(params["gotennet_cutoff"]),
        "--gotennet-pooling",
        str(params["gotennet_pooling"]),
        "--gotennet-lr-scale",
        str(params["gotennet_lr_scale"]),
        *all_configs[config_name],
    ]
    if args.trainable_molformer_1d:
        cmd.extend(
            [
                "--trainable-molformer-1d",
                "--molformer-unfreeze-layers",
                str(args.molformer_unfreeze_layers),
                "--molformer-lr-scale",
                str(args.molformer_lr_scale),
            ]
        )
    if args.unimol_model_path is not None:
        cmd.extend(["--unimol-model-path", args.unimol_model_path])
    if args.unimol_remove_hs:
        cmd.append("--unimol-remove-hs")
    if args.task_type == "regression" and args.task in SPEARMAN_REGRESSION_TASKS:
        cmd.extend(["--tracking-metric", "spearman"])
    if save_xattn_alpha:
        cmd.append("--save-xattn-alpha")
        if args.print_xattn_alpha:
            cmd.append("--print-xattn-alpha")
    if args.train_accelerator is not None:
        cmd.extend(["--accelerator", args.train_accelerator])
    if args.train_devices is not None:
        cmd.extend(["--devices", args.train_devices])
    return cmd


def run_cmd(cmd: list[str], args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = str(REPO_ROOT) if not env.get("PYTHONPATH") else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    if args.train_accelerator is not None and args.train_accelerator.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
    return result.returncode


def run_one_seed(
    trial: optuna.Trial,
    args: argparse.Namespace,
    config_name: str,
    params: dict[str, int | float | str],
    seed: int,
    hpo_dir: Path,
) -> float:
    metric = objective_metric(args.task_type, args.task)
    output_dir = hpo_dir / f"trial_{trial.number:04d}" / f"seed_{seed}"
    if output_dir.exists() and not args.force:
        try:
            return read_objective(output_dir, metric)
        except RuntimeError:
            pass

    cmd = build_train_cmd(args, config_name, params, seed, output_dir, args.epochs, args.patience)
    returncode = run_cmd(cmd, args)
    if returncode != 0:
        raise optuna.TrialPruned(
            f"Trial {trial.number} failed for {config_name}/{args.task}/seed_{seed}: exit {returncode}"
        )
    return read_objective(output_dir, metric)


def test_metrics(args: argparse.Namespace, output_dir: Path, seed: int) -> dict[str, float | int | str] | None:
    pred_file = prediction_file(output_dir)
    if not pred_file.exists():
        return None
    pred_df = pd.read_csv(pred_file)
    if pred_df.shape[1] < 2:
        return None
    pred_col = pred_df.columns[1]
    data_df = pd.read_csv(data_file_for(args.task, seed))
    test_df = data_df[data_df["split"] == "test"][["Drug", "Y"]]
    if len(test_df) == len(pred_df):
        y_true = pd.to_numeric(test_df["Y"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=float)
    else:
        pred = pred_df[["Drug", pred_col]].rename(columns={pred_col: "Y_pred"})
        merged = test_df.merge(pred, on="Drug", how="inner")
        y_true = pd.to_numeric(merged["Y"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(merged["Y_pred"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    if len(y_true) == 0:
        return None

    row: dict[str, float | int | str] = {"seed": seed, "n": int(len(y_true))}
    if args.task_type == "regression":
        row.update(
            {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
                "spearman": spearman_corr(y_true, y_pred),
            }
        )
    else:
        y_true_int = y_true.astype(int)
        y_label = (y_pred >= 0.5).astype(int)
        row["roc"] = float(roc_auc_score(y_true_int, y_pred)) if len(np.unique(y_true_int)) > 1 else float("nan")
        row["prc"] = float(average_precision_score(y_true_int, y_pred)) if len(np.unique(y_true_int)) > 1 else float("nan")
        row["accuracy"] = float(accuracy_score(y_true_int, y_label))
    return row


def summarize_metric_rows(rows: list[dict[str, float | int | str]], args: argparse.Namespace) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "task_type": args.task_type,
        "task": args.task,
        "seeds": len(rows),
    }
    metrics = ["mae", "rmse", "r2", "spearman"] if args.task_type == "regression" else ["roc", "prc", "accuracy"]
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows if metric in row], dtype=float)
        values = values[np.isfinite(values)]
        out[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
        out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def save_test_summary(
    args: argparse.Namespace,
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    prefix: str,
    config_name: str,
) -> dict[str, float | int | str] | None:
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_seed_metrics.csv", index=False)
    summary = summarize_metric_rows(rows, args)
    summary["config"] = config_name
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    return summary


def run_final_best_params(
    args: argparse.Namespace,
    config_name: str,
    params: dict[str, int | float | str],
    final_dir: Path,
) -> dict[str, float | int | str] | None:
    rows: list[dict[str, float | int | str]] = []
    epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    patience = args.final_patience if args.final_patience is not None else args.patience
    for seed in args.hpo_seeds:
        output_dir = final_dir / config_name / f"seed_{seed}"
        alpha_file = output_dir / "model_0" / "xattn_alpha" / "xattn_alpha.csv"
        needs_alpha = args.save_xattn_alpha and not alpha_file.exists()
        if args.force or needs_alpha or not predictions_have_finite_values(output_dir):
            cmd = build_train_cmd(
                args,
                config_name,
                params,
                seed,
                output_dir,
                epochs,
                patience,
                save_xattn_alpha=args.save_xattn_alpha,
            )
            returncode = run_cmd(cmd, args)
            if returncode != 0:
                raise RuntimeError(f"Final run failed for {config_name}/{args.task}/seed_{seed}: exit {returncode}")
        row = test_metrics(args, output_dir, seed)
        if row is not None:
            rows.append(row)
    return save_test_summary(args, rows, final_dir / config_name, "final", config_name)


def run_config_study(args: argparse.Namespace, config_name: str, root: Path) -> dict[str, object]:
    hpo_dir = root / "hpo" / config_name
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{args.task_type}_{args.task}_{config_name}_5seed",
        storage=f"sqlite:///{hpo_dir / 'study.db'}",
        direction=objective_direction(args.task_type, args.task),
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args)
        values = [run_one_seed(trial, args, config_name, params, seed, hpo_dir) for seed in args.hpo_seeds]
        value = float(np.mean(values))
        if not np.isfinite(value):
            raise optuna.TrialPruned(f"Trial {trial.number} produced non-finite objective")
        trial.set_user_attr("per_seed_values", json.dumps(values))
        trial.set_user_attr("params_json", json.dumps(params, sort_keys=True))
        return value

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    remaining_trials = max(0, args.n_trials - len(completed))
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, timeout=args.timeout)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No completed Optuna trials for {config_name}")

    trials_csv = hpo_dir / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)
    params = json.loads(study.best_trial.user_attrs.get("params_json", "{}"))
    if not params:
        params = complete_params_from_trial(study.best_trial, args)
    result: dict[str, object] = {
        "config": config_name,
        "objective_metric": objective_metric(args.task_type, args.task),
        "direction": objective_direction(args.task_type, args.task),
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": params,
    }
    (hpo_dir / "best_params.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    best_rows = []
    best_trial_dir = hpo_dir / f"trial_{study.best_trial.number:04d}"
    for seed in args.hpo_seeds:
        row = test_metrics(args, best_trial_dir / f"seed_{seed}", seed)
        if row is not None:
            best_rows.append(row)
    best_summary = save_test_summary(args, best_rows, hpo_dir, "best_trial_test", config_name)
    if best_summary is not None:
        for key, value in best_summary.items():
            if key not in {"config", "task", "task_type"}:
                result[f"best_trial_test_{key}"] = value

    if not args.skip_final:
        final_summary = run_final_best_params(args, config_name, params, root / "final")
        if final_summary is not None:
            for key, value in final_summary.items():
                if key not in {"config", "task", "task_type"}:
                    result[f"final_{key}"] = value

    return result


def validate_args(args: argparse.Namespace) -> None:
    tasks = REGRESSION_TASKS if args.task_type == "regression" else CLASSIFICATION_TASKS
    if args.task not in tasks:
        raise ValueError(f"{args.task!r} is not a known {args.task_type} task.")
    for config_name in args.configs:
        configs = source_configs(
            args.task_type,
            args.fixed_embed_dim or 128,
            args.fixed_x_d_encoder_heads or 4,
            args.unimol_dim,
        )
        if config_name not in configs:
            raise ValueError(f"Unknown config {config_name}. Available configs: {list(configs)}")


def main() -> None:
    args = parse_args()
    if optuna is None:
        raise SystemExit("Optuna is not installed. Activate chemprop-optuna or install optuna first.") from OPTUNA_IMPORT_ERROR
    validate_args(args)
    root = NEWDATA_DIR / args.output_root / args.task_type / args.task
    root.mkdir(parents=True, exist_ok=True)

    rows = [run_config_study(args, config_name, root) for config_name in args.configs]
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "two_config_5seed_hpo_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved outputs under {root}")


if __name__ == "__main__":
    main()
