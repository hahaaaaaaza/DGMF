import logging
from typing import Iterable

import torch
from torch import Tensor
from torch import nn

from chemprop.data import BatchGeometryGraph, BatchMolGraph, MulticomponentTrainingBatch
from chemprop.models.model import MPNN
from chemprop.nn import Aggregation, MulticomponentMessagePassing, Predictor
from chemprop.nn.metrics import ChempropMetric
from chemprop.nn.transforms import ScaleTransform

logger = logging.getLogger(__name__)


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
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
        dmr_aux_weight: float = 0.1,
        fusion_weight_decay: float = 0.0,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
            x_d_encoder,
            dmr_aux_weight,
            fusion_weight_decay,
        )
        self.message_passing: MulticomponentMessagePassing

    def fingerprint(
        self,
        bmgs: Iterable[BatchMolGraph],
        V_ds: Iterable[Tensor | None],
        X_d: Tensor | None = None,
        X_3d: BatchGeometryGraph | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        if X_d is None:
            return H
        X_d = self.X_d_transform(X_d)
        if self.x_d_encoder is not None:
            if getattr(self.x_d_encoder, "requires_graph_context", False):
                if getattr(self.x_d_encoder, "use_3d_graph", False):
                    return self.x_d_encoder(H, X_d, X_3d)
                return self.x_d_encoder(H, X_d)
            Z = self.x_d_encoder(X_d)
            return torch.cat((H, Z), dim=1)
        return torch.cat((H, X_d), dim=1)

    def on_validation_model_eval(self) -> None:
        self.eval()
        for block in self.message_passing.blocks:
            block.V_d_transform.train()
            block.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def get_batch_size(self, batch: MulticomponentTrainingBatch) -> int:
        return len(batch[0][0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(
                f"Model loading failed! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'chemprop convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint."
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

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]
        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams
