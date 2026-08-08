from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.conf import LIGHTNING_26_COMPAT_ARGS
from chemprop.data import (
    BatchGeometryGraph,
    BatchMolGraph,
    MulticomponentTrainingBatch,
    TrainingBatch,
)
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch


class MPNN(pl.LightningModule):
    r"""An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine.
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
        x_d_encoder: nn.Module | None = None,
        motif_2d_encoder: nn.Module | None = None,
        gotennet_lr_scale: float = 1.0,
        molformer_lr_scale: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "X_d_transform",
                "x_d_encoder",
                "motif_2d_encoder",
                "message_passing",
                "agg",
                "predictor",
            ]
        )
        self.hparams["X_d_transform"] = X_d_transform
        if x_d_encoder is not None:
            self.hparams["x_d_encoder"] = x_d_encoder
        if motif_2d_encoder is not None:
            self.hparams["motif_2d_encoder"] = motif_2d_encoder
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()
        self.x_d_encoder = x_d_encoder
        self.motif_2d_encoder = motif_2d_encoder
        self._xattn_alpha_collect_enabled = False
        self._xattn_alpha_records: list[dict[str, Tensor | int]] = []

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.gotennet_lr_scale = gotennet_lr_scale
        self.molformer_lr_scale = molformer_lr_scale

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
        X_3d: BatchGeometryGraph | None = None,
    ) -> Tensor:
        if getattr(self.message_passing, "is_pharmhgt_backbone", False):
            H = self.message_passing(bmg, V_d)
            H = self.bn(H)
            if X_d is None:
                return H
            X_d = self.X_d_transform(X_d)
            if self.x_d_encoder is not None:
                if getattr(self.x_d_encoder, "requires_graph_context", False):
                    if getattr(self.x_d_encoder, "use_3d_graph", False):
                        return self.x_d_encoder(H, X_d, X_3d)
                    return self.x_d_encoder(H, X_d)
                Z_1d = self.x_d_encoder(X_d)
                return torch.cat((H, Z_1d), dim=1)
            return torch.cat((H, X_d), dim=1)

        H_v = self.message_passing(bmg, V_d)
        if getattr(self.message_passing, "returns_graph_embedding", False):
            H = H_v
        else:
            H = self.agg(H_v, bmg.batch)
            if self.motif_2d_encoder is not None:
                H = self.motif_2d_encoder(H, H_v, bmg)
        H = self.bn(H)

        if X_d is None:
            if (
                self.x_d_encoder is not None
                and getattr(self.x_d_encoder, "requires_graph_context", False)
                and getattr(self.x_d_encoder, "use_3d_graph", False)
            ):
                return self.x_d_encoder(H, X_d, X_3d)
            return H

        X_d = self.X_d_transform(X_d)

        if self.x_d_encoder is not None:
            if getattr(self.x_d_encoder, "requires_graph_context", False):
                if getattr(self.x_d_encoder, "use_3d_graph", False):
                    return self.x_d_encoder(H, X_d, X_3d)
                return self.x_d_encoder(H, X_d)
            Z_1d = self.x_d_encoder(X_d)
            return torch.cat((H, Z_1d), dim=1)
        else:
            return torch.cat((H, X_d), dim=1)

    def encoding(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
        X_3d: BatchGeometryGraph | None = None,
        i: int = -1,
    ) -> Tensor:
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d, X_3d), i)

    def forward(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
        X_3d: BatchGeometryGraph | None = None,
    ) -> Tensor:
        Z = self.fingerprint(bmg, V_d, X_d, X_3d)
        return self.predictor(Z)

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, X_3d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d, X_3d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, X_3d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        Z = self.fingerprint(bmg, V_d, X_d, X_3d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, X_3d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d, X_3d)
        weights = torch.ones_like(weights)
        if self.predictor.n_targets > 1:
            preds = preds[..., 0]
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, V_d, X_d, X_3d, *_ = batch
        preds = self(bmg, V_d, X_d, X_3d)
        self._capture_xattn_alpha(batch_idx)
        return preds

    def enable_xattn_alpha_collection(self, enabled: bool = True) -> None:
        self._xattn_alpha_collect_enabled = enabled
        if enabled:
            self._xattn_alpha_records = []

    def get_xattn_alpha_records(self, clear: bool = False) -> list[dict[str, Tensor | int]]:
        records = self._xattn_alpha_records
        if clear:
            self._xattn_alpha_records = []
        return records

    def _capture_xattn_alpha(self, batch_idx: int) -> None:
        if not self._xattn_alpha_collect_enabled or self.x_d_encoder is None:
            return

        weights = getattr(self.x_d_encoder, "last_attention_weights", None)
        if weights is None:
            return

        record: dict[str, Tensor | int] = {
            "batch_idx": batch_idx,
            "attention_weights": weights.detach().cpu(),
        }

        gates = getattr(self.x_d_encoder, "last_modality_gates", None)
        if gates is not None:
            record["modality_gates"] = gates.detach().cpu()

        pair_gates = getattr(self.x_d_encoder, "last_pair_gates", None)
        if pair_gates is not None:
            record["pair_gates"] = pair_gates.detach().cpu()

        directional_weights = getattr(self.x_d_encoder, "last_directional_weights", None)
        if directional_weights is not None:
            record["directional_weights"] = directional_weights.detach().cpu()

        entropy = getattr(self.x_d_encoder, "last_attention_entropy", None)
        if entropy is not None:
            record["attention_entropy"] = entropy.detach().cpu()

        alpha = getattr(self.x_d_encoder, "alpha", None)
        if isinstance(alpha, nn.Parameter):
            record["residual_alpha_tanh"] = alpha.detach().tanh().cpu()

        self._xattn_alpha_records.append(record)

    def transfer_batch_to_device(self, batch: BatchType, device, dataloader_idx: int):
        if isinstance(batch, TrainingBatch):
            bmg, V_d, X_d, X_3d, targets, weights, lt_mask, gt_mask = batch
            self._move_graph_to_device(bmg, device)
            self._move_graph_to_device(X_3d, device)
            return TrainingBatch(
                bmg,
                self._move_to_device(V_d, device),
                self._move_to_device(X_d, device),
                X_3d,
                self._move_to_device(targets, device),
                self._move_to_device(weights, device),
                self._move_to_device(lt_mask, device),
                self._move_to_device(gt_mask, device),
            )

        if isinstance(batch, MulticomponentTrainingBatch):
            bmgs, V_ds, X_d, X_3d, targets, weights, lt_mask, gt_mask = batch
            for bmg in bmgs:
                self._move_graph_to_device(bmg, device)
            self._move_graph_to_device(X_3d, device)
            return MulticomponentTrainingBatch(
                bmgs,
                [self._move_to_device(V_d, device) for V_d in V_ds],
                self._move_to_device(X_d, device),
                X_3d,
                self._move_to_device(targets, device),
                self._move_to_device(weights, device),
                self._move_to_device(lt_mask, device),
                self._move_to_device(gt_mask, device),
            )

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @staticmethod
    def _move_to_device(x, device):
        return None if x is None else x.to(device)

    @staticmethod
    def _move_graph_to_device(graph, device) -> None:
        if graph is not None and hasattr(graph, "to"):
            graph.to(device)

    def configure_optimizers(self):
        special_groups: list[dict] = []
        special_param_ids: set[int] = set()
        for class_name, lr_scale in [
            ("GotenNetGeometryEncoder", self.gotennet_lr_scale),
            ("TrainableMolFormerFingerprintEncoder", self.molformer_lr_scale),
        ]:
            params: list[nn.Parameter] = []
            for module in self.modules():
                if module.__class__.__name__ == class_name:
                    source = (
                        module.molformer
                        if class_name == "TrainableMolFormerFingerprintEncoder"
                        and hasattr(module, "molformer")
                        else module
                    )
                    for param in source.parameters():
                        if param.requires_grad:
                            params.append(param)
            if params and lr_scale != 1.0:
                special_param_ids.update(id(param) for param in params)
                special_groups.append({"params": params, "lr": self.init_lr * lr_scale})

        other_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in special_param_ids
        ]
        if special_groups:
            param_groups = []
            if other_params:
                param_groups.append({"params": other_params, "lr": self.init_lr})
            param_groups.extend(special_groups)
            opt = optim.Adam(param_groups)
        else:
            trainable_params = [param for param in self.parameters() if param.requires_grad]
            opt = optim.Adam(trainable_params, self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch
        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}
        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    def get_batch_size(self, batch: TrainingBatch) -> int:
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(
                f"{traceback.format_exc()}\nModel loading failed (full stacktrace above)! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'chemprop convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint."
            )

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        # Restore x_d_encoder: stored as the nn.Module itself in hparams
        if "x_d_encoder" in hparams and hparams["x_d_encoder"] is not None:
            submodules["x_d_encoder"] = hparams.pop("x_d_encoder")
        else:
            hparams.pop("x_d_encoder", None)

        if "motif_2d_encoder" in hparams and hparams["motif_2d_encoder"] is not None:
            submodules["motif_2d_encoder"] = hparams.pop("motif_2d_encoder")
        else:
            hparams.pop("motif_2d_encoder", None)

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v
            for k, v in kwargs.items()
            if k in ["message_passing", "agg", "predictor", "motif_2d_encoder"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)
        # Pass stored optional encoders explicitly so Lightning forwards them to __init__
        if "x_d_encoder" in submodules:
            kwargs["x_d_encoder"] = submodules["x_d_encoder"]
        if "motif_2d_encoder" in submodules:
            kwargs["motif_2d_encoder"] = submodules["motif_2d_encoder"]

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(
            buffer, map_location, hparams_file, strict, **LIGHTNING_26_COMPAT_ARGS, **kwargs
        )

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)  # includes x_d_encoder if present

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
