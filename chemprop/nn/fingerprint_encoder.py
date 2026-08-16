"""Encoders for 1D fingerprints and MMFF 3D geometry graphs fused with DMPNN output."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.data.geometry import BatchGeometryGraph
from chemprop.featurizers.molformer import resolve_molformer_model_path
from chemprop.nn.message_passing.hignn import HiGNNMessagePassing

MACCS_FINGERPRINT_DIM = 166
RDKIT_FINGERPRINT_DIM = 2048
ECFP_FINGERPRINT_DIM = 2048
NATIVE_1D_FINGERPRINT_DIM = MACCS_FINGERPRINT_DIM + RDKIT_FINGERPRINT_DIM + ECFP_FINGERPRINT_DIM


def _clone_activation(activation: nn.Module | None = None) -> nn.Module:
    return copy.deepcopy(activation) if activation is not None else nn.ReLU()


def _eca_kernel_size(channels: int, gamma: int = 2, bias: int = 1) -> int:
    channels = max(2, int(channels))
    kernel = int(abs(math.log2(channels) / gamma + bias / gamma))
    if kernel % 2 == 0:
        kernel += 1
    return max(3, kernel)


class ECAChannelAttention(nn.Module):
    """Efficient Channel Attention over descriptor channels."""

    def __init__(self, channels: int, kernel_size: int | None = None):
        super().__init__()
        kernel_size = kernel_size or _eca_kernel_size(channels)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def weights(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"ECAChannelAttention expects [batch, channels], got {tuple(x.shape)}")
        return torch.sigmoid(self.conv(x.unsqueeze(1))).squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weights(x)


class MLPDescriptorEncoder(nn.Module):
    """Two-layer MLP: flattened descriptors -> fixed-size embedding."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        d_h = max(d_out * hidden_mult, 512)
        act = activation if activation is not None else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_h, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ECADescriptorEncoder(nn.Module):
    """Projection + ECA channel attention: flattened descriptors -> fixed-size embedding."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            act,
            nn.Dropout(dropout),
            ECAChannelAttention(d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _pool_molformer_output(outputs, attention_mask: Tensor, pooling: str) -> Tensor:
    if pooling == "pooler":
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        pooling = "cls"

    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unknown MoLFormer pooling mode: {pooling}")


def _hidden_size_from_config(config) -> int | None:
    for key in ("hidden_size", "d_model", "n_embd", "embedding_size"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    return None


def _largest_module_list(module: nn.Module) -> nn.ModuleList | None:
    candidates: list[nn.ModuleList] = []
    for _, submodule in module.named_modules():
        if not isinstance(submodule, nn.ModuleList) or len(submodule) == 0:
            continue
        if any(any(True for _ in child.parameters()) for child in submodule):
            candidates.append(submodule)
    if not candidates:
        return None
    return max(candidates, key=len)


class TrainableMolFormerFingerprintEncoder(nn.Module):
    """Fine-tunable MoLFormer 1D encoder carried through ``X_d`` token features."""

    def __init__(
        self,
        model_name: str,
        d_out: int,
        max_length: int = 256,
        pooling: str = "pooler",
        unfreeze_layers: int = 0,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")
        if pooling not in {"pooler", "cls", "mean"}:
            raise ValueError("MoLFormer pooling must be one of: pooler, cls, mean.")

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "Trainable MoLFormer requires the optional `transformers` dependency. "
                "Install it with `pip install -e .[molformer]` or `pip install transformers`."
            ) from exc

        self.model_name = resolve_molformer_model_path(model_name)
        self.max_length = int(max_length)
        self.pooling = pooling
        self.unfreeze_layers = int(unfreeze_layers)
        self.molformer = AutoModel.from_pretrained(
            self.model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.model_input_length = min(
            self.max_length,
            int(getattr(self.molformer.config, "max_position_embeddings", self.max_length)),
        )
        self._set_molformer_trainable(self.unfreeze_layers)

        hidden_size = _hidden_size_from_config(self.molformer.config)
        if hidden_size is None:
            with torch.no_grad():
                input_ids = torch.zeros((1, self.model_input_length), dtype=torch.long)
                attention_mask = torch.ones((1, self.model_input_length), dtype=torch.long)
                outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden_size = int(
                    _pool_molformer_output(outputs, attention_mask, self.pooling).shape[-1]
                )

        act = activation if activation is not None else nn.ReLU()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, d_out),
            nn.LayerNorm(d_out),
            act,
            nn.Dropout(dropout),
        )

    def _set_molformer_trainable(self, unfreeze_layers: int) -> None:
        for param in self.molformer.parameters():
            param.requires_grad = False

        if unfreeze_layers < 0:
            for param in self.molformer.parameters():
                param.requires_grad = True
            return

        if unfreeze_layers == 0:
            return

        layers = _largest_module_list(self.molformer)
        if layers is None:
            return
        for layer in layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        token_width = 2 * self.max_length
        if x.shape[1] % token_width != 0:
            raise ValueError(
                f"Trainable MoLFormer expected X_d dim to be a multiple of {token_width}, "
                f"got {x.shape[1]}."
            )

        n_components = x.shape[1] // token_width
        tokens = x.reshape(x.shape[0], n_components, token_width)
        input_ids = (
            tokens[:, :, : self.model_input_length]
            .reshape(-1, self.model_input_length)
            .round()
            .long()
        )
        attention_mask = (
            tokens[:, :, self.max_length : self.max_length + self.model_input_length]
            .reshape(-1, self.model_input_length)
            .round()
            .long()
        )

        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _pool_molformer_output(outputs, attention_mask, self.pooling)
        pooled = pooled.reshape(x.shape[0], n_components, -1).mean(dim=1)
        return self.proj(pooled)


def _mean_by_index(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if dim_size == 0:
        return values.new_zeros((0, values.shape[-1]))

    out = values.new_zeros((dim_size, values.shape[-1]))
    if values.numel() == 0 or index.numel() == 0:
        return out

    out.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=dim_size).to(values.dtype).unsqueeze(1)
    return out / counts.clamp_min(1)


class MotifEnhancedGraphEncoder(nn.Module):
    """Conservative BRICS motif enhancement on top of the Chemprop DMPNN graph embedding."""

    def __init__(
        self,
        d_graph: int,
        d_hidden: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        use_eca: bool = True,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        d_hidden = d_hidden or d_graph
        act = activation if activation is not None else nn.ReLU()
        self.depth = depth
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.motif_proj = nn.Sequential(
            nn.Linear(d_graph, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
            nn.Dropout(dropout),
        )
        self.motif_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.LayerNorm(d_hidden),
                    _clone_activation(act),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )
        self.motif_out = nn.Linear(d_hidden, d_graph)
        fusion_layers: list[nn.Module] = [
            nn.Linear(2 * d_graph, d_graph),
            nn.LayerNorm(d_graph),
            _clone_activation(act),
            nn.Dropout(dropout),
        ]
        if use_eca:
            fusion_layers.append(ECAChannelAttention(d_graph))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, graph_h: Tensor, atom_h: Tensor, bmg: BatchMolGraph) -> Tensor:
        motif_atom_index = getattr(bmg, "motif_atom_index", None)
        motif_edge_index = getattr(bmg, "motif_edge_index", None)
        motif_batch = getattr(bmg, "motif_batch", None)
        if (
            motif_atom_index is None
            or motif_batch is None
            or motif_atom_index.numel() == 0
            or motif_batch.numel() == 0
        ):
            return graph_h

        n_motifs = motif_batch.numel()
        motif_h = _mean_by_index(atom_h[motif_atom_index[1]], motif_atom_index[0], n_motifs)
        motif_h = self.motif_proj(motif_h)

        for update in self.motif_updates:
            if motif_edge_index is not None and motif_edge_index.numel():
                msg = _mean_by_index(motif_h[motif_edge_index[0]], motif_edge_index[1], n_motifs)
            else:
                msg = torch.zeros_like(motif_h)
            motif_h = motif_h + update(motif_h + msg)

        motif_graph_h = _mean_by_index(self.motif_out(motif_h), motif_batch, graph_h.shape[0])
        delta = self.fusion(torch.cat((graph_h, motif_graph_h), dim=1))
        return graph_h + torch.tanh(self.residual_scale) * delta


class PharmHGTGraphEncoder(nn.Module):
    """PharmHGT-style atom/BRICS-fragment heterogeneous 2D encoder.

    This adapts the public PharmHGT design to Chemprop's native ``BatchMolGraph``
    instead of requiring a separate DGL dataloader: atoms and BRICS fragments are
    connected by junction memberships, fragments carry MACCS + RDKit
    pharmacophore features, and BRICS reaction-rule edges pass messages between
    fragments.
    """

    def __init__(
        self,
        d_graph: int,
        d_hidden: int | None = None,
        pharm_dim: int = 194,
        reac_dim: int = 34,
        depth: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        d_hidden = d_hidden or d_graph
        act = activation if activation is not None else nn.ReLU()
        self.depth = depth
        self.pharm_dim = pharm_dim
        self.reac_dim = reac_dim
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        self.atom_proj = nn.Sequential(
            nn.Linear(d_graph, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
            nn.Dropout(dropout),
        )
        self.pharm_proj = nn.Sequential(
            nn.Linear(pharm_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
            nn.Dropout(dropout),
        )
        self.reac_proj = nn.Sequential(
            nn.Linear(reac_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
        )
        self.atom_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_hidden, d_hidden),
                    nn.LayerNorm(d_hidden),
                    _clone_activation(act),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                )
                for _ in range(depth)
            ]
        )
        self.fragment_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_hidden, d_hidden),
                    nn.LayerNorm(d_hidden),
                    _clone_activation(act),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                )
                for _ in range(depth)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(d_graph + 2 * d_hidden, d_graph),
            nn.LayerNorm(d_graph),
            _clone_activation(act),
            nn.Dropout(dropout),
            ECAChannelAttention(d_graph),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_h: Tensor, atom_h: Tensor, bmg: BatchMolGraph) -> Tensor:
        motif_atom_index = getattr(bmg, "motif_atom_index", None)
        motif_edge_index = getattr(bmg, "motif_edge_index", None)
        motif_batch = getattr(bmg, "motif_batch", None)
        motif_features = getattr(bmg, "motif_features", None)
        motif_edge_features = getattr(bmg, "motif_edge_features", None)

        if (
            motif_atom_index is None
            or motif_batch is None
            or motif_atom_index.numel() == 0
            or motif_batch.numel() == 0
        ):
            return graph_h

        n_motifs = motif_batch.numel()
        atom_state = self.atom_proj(atom_h)
        motif_from_atoms = _mean_by_index(atom_state[motif_atom_index[1]], motif_atom_index[0], n_motifs)

        if motif_features is not None and motif_features.numel():
            if motif_features.shape[1] != self.pharm_dim:
                raise ValueError(
                    f"PharmHGT motif features expected dim {self.pharm_dim}, got {motif_features.shape[1]}."
                )
            motif_state = self.pharm_proj(motif_features) + motif_from_atoms
        else:
            motif_state = motif_from_atoms

        for atom_update, fragment_update in zip(self.atom_updates, self.fragment_updates):
            atom_from_motifs = _mean_by_index(
                motif_state[motif_atom_index[0]], motif_atom_index[1], atom_state.shape[0]
            )

            if motif_edge_index is not None and motif_edge_index.numel():
                src, dst = motif_edge_index
                edge_msg = motif_state[src]
                if motif_edge_features is not None and motif_edge_features.numel():
                    if motif_edge_features.shape[1] != self.reac_dim:
                        raise ValueError(
                            "PharmHGT motif edge features expected dim "
                            f"{self.reac_dim}, got {motif_edge_features.shape[1]}."
                        )
                    edge_msg = edge_msg + self.reac_proj(motif_edge_features)
                motif_from_edges = _mean_by_index(edge_msg, dst, n_motifs)
            else:
                motif_from_edges = torch.zeros_like(motif_state)

            motif_from_atoms = _mean_by_index(
                atom_state[motif_atom_index[1]], motif_atom_index[0], n_motifs
            )
            atom_state = atom_state + self.dropout(
                atom_update(torch.cat([atom_state, atom_from_motifs], dim=1))
            )
            motif_state = motif_state + self.dropout(
                fragment_update(torch.cat([motif_state, motif_from_atoms, motif_from_edges], dim=1))
            )

        atom_graph_h = _mean_by_index(atom_state, bmg.batch, graph_h.shape[0])
        motif_graph_h = _mean_by_index(motif_state, motif_batch, graph_h.shape[0])
        delta = self.readout(torch.cat([graph_h, atom_graph_h, motif_graph_h], dim=1))
        return graph_h + torch.tanh(self.residual_scale) * self.dropout(delta)


class HiGNNGatedGraphEncoder(nn.Module):
    """Adaptive gate between the Chemprop DMPNN embedding and a HiGNN-style branch."""

    def __init__(
        self,
        d_graph: int,
        atom_dim: int,
        bond_dim: int,
        d_hidden: int | None = None,
        depth: int = 3,
        slices: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        feature_attention: bool = True,
    ):
        super().__init__()
        d_hidden = d_hidden or d_graph
        act = activation if activation is not None else nn.ReLU()
        self.hignn = HiGNNMessagePassing(
            d_v=atom_dim,
            d_e=bond_dim,
            d_h=d_hidden,
            depth=depth,
            slices=slices,
            dropout=dropout,
            feature_attention=feature_attention,
            activation=activation if activation is not None else "RELU",
        )
        self.hignn_proj = (
            nn.Identity()
            if d_hidden == d_graph
            else nn.Sequential(
                nn.Linear(d_hidden, d_graph),
                act,
                nn.Dropout(dropout),
                nn.LayerNorm(d_graph),
            )
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * d_graph, d_graph),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_graph, d_graph),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Linear(2 * d_graph, d_graph),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_graph, d_graph),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_gate: Tensor | None = None

    def forward(self, graph_h: Tensor, atom_h: Tensor, bmg: BatchMolGraph) -> Tensor:
        del atom_h
        hignn_h = self.hignn_proj(self.hignn(bmg))
        gate = self.gate(torch.cat([graph_h, hignn_h, torch.abs(graph_h - hignn_h)], dim=1))
        delta = self.delta(torch.cat([graph_h, hignn_h], dim=1))
        self.last_gate = gate.detach()
        return graph_h + torch.tanh(self.alpha) * gate * delta


class PharmHGTBackboneEncoder(nn.Module):
    """Standalone PharmHGT-style BRICS fragment/pharmacophore 2D backbone."""

    returns_graph_embedding = True

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        d_hidden: int = 300,
        pharm_dim: int = 194,
        reac_dim: int = 34,
        depth: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.output_dim = 2 * d_hidden
        self.hparams = {
            "cls": self.__class__,
            "atom_dim": atom_dim,
            "bond_dim": bond_dim,
            "d_hidden": d_hidden,
            "pharm_dim": pharm_dim,
            "reac_dim": reac_dim,
            "depth": depth,
            "dropout": dropout,
        }
        self.is_pharmhgt_backbone = True
        self.depth = depth
        self.pharm_dim = pharm_dim
        self.reac_dim = reac_dim
        self.V_d_transform = nn.Identity()
        self.graph_transform = nn.Identity()

        self.atom_proj = nn.Sequential(
            nn.Linear(atom_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
            nn.Dropout(dropout),
        )
        self.bond_proj = nn.Sequential(
            nn.Linear(bond_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
        )
        self.pharm_proj = nn.Sequential(
            nn.Linear(pharm_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
            nn.Dropout(dropout),
        )
        self.reac_proj = nn.Sequential(
            nn.Linear(reac_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            _clone_activation(act),
        )

        self.atom_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_hidden, d_hidden),
                    nn.LayerNorm(d_hidden),
                    _clone_activation(act),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                )
                for _ in range(depth)
            ]
        )
        self.fragment_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * d_hidden, d_hidden),
                    nn.LayerNorm(d_hidden),
                    _clone_activation(act),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def _empty_fragment_state(self, atom_state: Tensor, bmg: BatchMolGraph) -> tuple[Tensor, Tensor]:
        graph_h = _mean_by_index(atom_state, bmg.batch, len(bmg))
        return graph_h, torch.zeros_like(graph_h)

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        del V_d
        atom_state = self.atom_proj(bmg.V)
        motif_atom_index = getattr(bmg, "motif_atom_index", None)
        motif_edge_index = getattr(bmg, "motif_edge_index", None)
        motif_batch = getattr(bmg, "motif_batch", None)
        motif_features = getattr(bmg, "motif_features", None)
        motif_edge_features = getattr(bmg, "motif_edge_features", None)

        if (
            motif_atom_index is None
            or motif_batch is None
            or motif_atom_index.numel() == 0
            or motif_batch.numel() == 0
        ):
            atom_graph_h, motif_graph_h = self._empty_fragment_state(atom_state, bmg)
            return self.output_norm(torch.cat([atom_graph_h, motif_graph_h], dim=1))

        n_motifs = motif_batch.numel()
        motif_from_atoms = _mean_by_index(atom_state[motif_atom_index[1]], motif_atom_index[0], n_motifs)
        if motif_features is not None and motif_features.numel():
            if motif_features.shape[1] != self.pharm_dim:
                raise ValueError(
                    f"PharmHGT motif features expected dim {self.pharm_dim}, got {motif_features.shape[1]}."
                )
            motif_state = self.pharm_proj(motif_features) + motif_from_atoms
        else:
            motif_state = motif_from_atoms

        for atom_update, fragment_update in zip(self.atom_updates, self.fragment_updates):
            bond_msg = atom_state[bmg.edge_index[0]] + self.bond_proj(bmg.E)
            atom_from_bonds = _mean_by_index(bond_msg, bmg.edge_index[1], atom_state.shape[0])
            atom_from_motifs = _mean_by_index(
                motif_state[motif_atom_index[0]], motif_atom_index[1], atom_state.shape[0]
            )

            if motif_edge_index is not None and motif_edge_index.numel():
                src, dst = motif_edge_index
                edge_msg = motif_state[src]
                if motif_edge_features is not None and motif_edge_features.numel():
                    if motif_edge_features.shape[1] != self.reac_dim:
                        raise ValueError(
                            "PharmHGT motif edge features expected dim "
                            f"{self.reac_dim}, got {motif_edge_features.shape[1]}."
                        )
                    edge_msg = edge_msg + self.reac_proj(motif_edge_features)
                motif_from_edges = _mean_by_index(edge_msg, dst, n_motifs)
            else:
                motif_from_edges = torch.zeros_like(motif_state)

            motif_from_atoms = _mean_by_index(
                atom_state[motif_atom_index[1]], motif_atom_index[0], n_motifs
            )
            atom_state = atom_state + self.dropout(
                atom_update(torch.cat([atom_state, atom_from_bonds, atom_from_motifs], dim=1))
            )
            motif_state = motif_state + self.dropout(
                fragment_update(torch.cat([motif_state, motif_from_atoms, motif_from_edges], dim=1))
            )

        atom_graph_h = _mean_by_index(atom_state, bmg.batch, len(bmg))
        motif_graph_h = _mean_by_index(motif_state, motif_batch, len(bmg))
        return self.output_norm(torch.cat([atom_graph_h, motif_graph_h], dim=1))


class FingerprintPatchEmbedding(nn.Module):
    """Split one fingerprint view into fixed-size descriptor patches."""

    def __init__(
        self,
        d_in: int,
        patch_size: int,
        d_model: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.patch_size = patch_size
        self.n_patches = math.ceil(d_in / patch_size)
        self.padded_dim = self.n_patches * patch_size
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.d_in:
            raise ValueError(f"Expected fingerprint view dim {self.d_in}, got {x.shape[1]}.")
        pad = self.padded_dim - self.d_in
        if pad:
            x = F.pad(x, (0, pad))
        x = x.reshape(x.shape[0], self.n_patches, self.patch_size)
        return self.patch_proj(x) + self.pos_embedding


class PatchMLPBlock(nn.Module):
    """MLP-Mixer style patch mixing followed by feature mixing."""

    def __init__(
        self,
        n_patches: int,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        patch_hidden = max(n_patches * 2, 16)
        feature_hidden = d_model * 2
        self.patch_norm = nn.LayerNorm(d_model)
        self.patch_mlp = nn.Sequential(
            nn.Linear(n_patches, patch_hidden),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(patch_hidden, n_patches),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, feature_hidden),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.patch_norm(x).transpose(1, 2)
        y = self.patch_mlp(y).transpose(1, 2)
        x = x + self.dropout(y)
        return x + self.dropout(self.feature_mlp(self.feature_norm(x)))


class AttentivePatchPooling(nn.Module):
    """Learn which descriptor patches matter for a fingerprint view."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class FingerprintViewEncoder(nn.Module):
    """Patch embedding -> PatchMLP blocks -> attentive pooling for one fingerprint."""

    def __init__(
        self,
        d_in: int,
        patch_size: int,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.embedding = FingerprintPatchEmbedding(d_in, patch_size, d_model)
        self.blocks = nn.ModuleList(
            [
                PatchMLPBlock(
                    n_patches=self.embedding.n_patches,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentivePatchPooling(d_model, dropout=dropout, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(self.norm(x))


class GatedViewFusion(nn.Module):
    """Softmax-gated fusion over fingerprint view embeddings."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, views: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(views), dim=1)
        return self.norm((weights * views).sum(dim=1))


class PatchFingerprintEncoder(nn.Module):
    """Three-view MACCS/RDKit/ECFP patch encoder for native 1D fingerprints."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.has_native_fingerprints = d_in >= NATIVE_1D_FINGERPRINT_DIM

        if not self.has_native_fingerprints:
            self.fallback = MLPDescriptorEncoder(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                activation=activation,
            )
            return

        self.extra_dim = d_in - NATIVE_1D_FINGERPRINT_DIM
        self.maccs_encoder = FingerprintViewEncoder(
            d_in=MACCS_FINGERPRINT_DIM,
            patch_size=16,
            d_model=d_out,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        self.rdkit_encoder = FingerprintViewEncoder(
            d_in=RDKIT_FINGERPRINT_DIM,
            patch_size=32,
            d_model=d_out,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        self.ecfp_encoder = FingerprintViewEncoder(
            d_in=ECFP_FINGERPRINT_DIM,
            patch_size=32,
            d_model=d_out,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        self.extra_encoder = (
            MLPDescriptorEncoder(
                d_in=self.extra_dim,
                d_out=d_out,
                dropout=dropout,
                activation=activation,
            )
            if self.extra_dim > 0
            else None
        )
        self.view_fusion = GatedViewFusion(d_out, dropout=dropout, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_native_fingerprints:
            return self.fallback(x)

        fp = x[:, -NATIVE_1D_FINGERPRINT_DIM:]
        offset = 0
        x_maccs = fp[:, offset : offset + MACCS_FINGERPRINT_DIM]
        offset += MACCS_FINGERPRINT_DIM
        x_rdkit = fp[:, offset : offset + RDKIT_FINGERPRINT_DIM]
        offset += RDKIT_FINGERPRINT_DIM
        x_ecfp = fp[:, offset : offset + ECFP_FINGERPRINT_DIM]

        views = [
            self.maccs_encoder(x_maccs),
            self.rdkit_encoder(x_rdkit),
            self.ecfp_encoder(x_ecfp),
        ]
        if self.extra_encoder is not None:
            views.append(self.extra_encoder(x[:, : self.extra_dim]))
        return self.view_fusion(torch.stack(views, dim=1))


class DUETClusterBlock(nn.Module):
    """DUET-inspired dual clustering over descriptor groups and fingerprint views."""

    def __init__(
        self,
        d_model: int,
        num_group_clusters: int = 8,
        num_view_clusters: int = 3,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.d_model = d_model
        self.group_prototypes = nn.Parameter(torch.empty(num_group_clusters, d_model))
        self.view_prototypes = nn.Parameter(torch.empty(num_view_clusters, d_model))
        nn.init.trunc_normal_(self.group_prototypes, std=0.02)
        nn.init.trunc_normal_(self.view_prototypes, std=0.02)

        self.group_norm1 = nn.LayerNorm(d_model)
        self.group_norm2 = nn.LayerNorm(d_model)
        self.view_norm1 = nn.LayerNorm(d_model)
        self.view_norm2 = nn.LayerNorm(d_model)
        self.group_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            act,
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.view_ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.view_gate = nn.Linear(d_model, 1)
        self.group_to_view = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _cluster_context(self, tokens: Tensor, prototypes: Tensor) -> Tensor:
        logits = torch.matmul(tokens, prototypes.t()) / math.sqrt(self.d_model)
        assignment = torch.softmax(logits, dim=-1)
        denom = assignment.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        centroids = torch.einsum("bnc,bnd->bcd", assignment, tokens) / denom
        return torch.einsum("bnc,bcd->bnd", assignment, centroids)

    def forward(self, group_tokens: Tensor, view_tokens: Tensor) -> tuple[Tensor, Tensor]:
        group_context = self._cluster_context(self.group_norm1(group_tokens), self.group_prototypes)
        view_context = self._cluster_context(self.view_norm1(view_tokens), self.view_prototypes)

        view_weights = torch.softmax(self.view_gate(view_context), dim=1)
        view_summary = (view_weights * view_context).sum(dim=1, keepdim=True)
        group_tokens = group_tokens + self.dropout(group_context + view_summary)
        group_tokens = group_tokens + self.dropout(self.group_ffn(self.group_norm2(group_tokens)))

        group_summary = self.group_to_view(group_tokens.mean(dim=1, keepdim=True))
        view_tokens = view_tokens + self.dropout(view_context + group_summary)
        view_tokens = view_tokens + self.dropout(self.view_ffn(self.view_norm2(view_tokens)))
        return group_tokens, view_tokens


class DUETFingerprintEncoder(nn.Module):
    """DUET-inspired fingerprint encoder with group/view clustering and raw residual."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_layers: int = 2,
        num_groups: int = 64,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_groups = num_groups
        self.has_native_fingerprints = d_in >= NATIVE_1D_FINGERPRINT_DIM

        if not self.has_native_fingerprints:
            self.fallback = MLPDescriptorEncoder(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                activation=activation,
            )
            return

        act = activation if activation is not None else nn.ReLU()
        self.maccs_to_groups = nn.Linear(MACCS_FINGERPRINT_DIM, num_groups)
        self.sequence_norm = nn.LayerNorm(3)
        self.group_proj = nn.Linear(3, d_out)
        self.coarse_proj = nn.Linear(3, d_out)
        self.group_embedding = nn.Parameter(torch.zeros(1, num_groups, d_out))
        self.view_proj = nn.Linear(num_groups, d_out)
        self.view_embedding = nn.Parameter(torch.zeros(1, 3, d_out))
        nn.init.trunc_normal_(self.group_embedding, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        num_group_clusters = max(4, min(16, num_groups // 8))
        self.blocks = nn.ModuleList(
            [
                DUETClusterBlock(
                    d_model=d_out,
                    num_group_clusters=num_group_clusters,
                    num_view_clusters=3,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.group_pool = AttentivePatchPooling(d_out, dropout=dropout, activation=activation)
        self.view_pool = AttentivePatchPooling(d_out, dropout=dropout, activation=activation)
        self.ts_fusion = nn.Sequential(
            nn.Linear(2 * d_out, d_out),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )
        self.raw_encoder = MLPDescriptorEncoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            activation=activation,
        )
        self.out = nn.Sequential(
            nn.Linear(2 * d_out, d_out),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    @staticmethod
    def _group_mean(x: Tensor, num_groups: int) -> Tensor:
        pad = (-x.shape[1]) % num_groups
        if pad:
            x = F.pad(x, (0, pad))
        return x.reshape(x.shape[0], num_groups, -1).mean(dim=-1)

    def _split_native(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        fp = x[:, -NATIVE_1D_FINGERPRINT_DIM:]
        offset = 0
        x_maccs = fp[:, offset : offset + MACCS_FINGERPRINT_DIM]
        offset += MACCS_FINGERPRINT_DIM
        x_rdkit = fp[:, offset : offset + RDKIT_FINGERPRINT_DIM]
        offset += RDKIT_FINGERPRINT_DIM
        x_ecfp = fp[:, offset : offset + ECFP_FINGERPRINT_DIM]
        return x_maccs, x_rdkit, x_ecfp

    def _group_sequence(self, x: Tensor) -> Tensor:
        x_maccs, x_rdkit, x_ecfp = self._split_native(x)
        maccs_group = self.maccs_to_groups(x_maccs)
        rdkit_group = self._group_mean(x_rdkit, self.num_groups)
        ecfp_group = self._group_mean(x_ecfp, self.num_groups)
        return self.sequence_norm(torch.stack([maccs_group, rdkit_group, ecfp_group], dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_native_fingerprints:
            return self.fallback(x)

        sequence = self._group_sequence(x)
        group_tokens = self.group_proj(sequence) + self.group_embedding

        coarse = F.avg_pool1d(sequence.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        coarse_tokens = self.coarse_proj(coarse).transpose(1, 2)
        coarse_tokens = F.interpolate(
            coarse_tokens,
            size=self.num_groups,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        group_tokens = group_tokens + coarse_tokens

        view_tokens = self.view_proj(sequence.transpose(1, 2)) + self.view_embedding
        for block in self.blocks:
            group_tokens, view_tokens = block(group_tokens, view_tokens)

        z_group = self.group_pool(group_tokens)
        z_view = self.view_pool(view_tokens)
        z_ts = self.ts_fusion(torch.cat([z_group, z_view], dim=1))
        z_raw = self.raw_encoder(x)
        return self.out(torch.cat([z_ts, z_raw], dim=1))


class ITransformerFingerprintEncoder(nn.Module):
    """iTransformer-inspired FP encoder using fingerprint views as variate tokens."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_layers: int = 2,
        num_groups: int = 128,
        nhead: int = 4,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if d_out % nhead != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by nhead ({nhead})")

        self.d_in = d_in
        self.d_out = d_out
        self.num_groups = num_groups
        self.has_native_fingerprints = d_in >= NATIVE_1D_FINGERPRINT_DIM

        if not self.has_native_fingerprints:
            self.fallback = MLPDescriptorEncoder(
                d_in=d_in,
                d_out=d_out,
                dropout=dropout,
                activation=activation,
            )
            return

        act = activation if activation is not None else nn.ReLU()
        activation_name = "gelu" if isinstance(act, nn.GELU) else "relu"
        self.maccs_to_groups = nn.Linear(MACCS_FINGERPRINT_DIM, num_groups)
        self.view_norm = nn.LayerNorm(num_groups)
        self.value_embedding = nn.Linear(num_groups, d_out)
        self.view_embedding = nn.Parameter(torch.zeros(1, 3, d_out))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out,
            nhead=nhead,
            dim_feedforward=4 * d_out,
            dropout=dropout,
            activation=activation_name,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_out),
        )
        self.ts_pool = AttentivePatchPooling(d_out, dropout=dropout, activation=activation)
        self.raw_encoder = MLPDescriptorEncoder(
            d_in=d_in,
            d_out=d_out,
            dropout=dropout,
            activation=activation,
        )
        self.out = nn.Sequential(
            nn.Linear(2 * d_out, d_out),
            _clone_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    @staticmethod
    def _group_mean(x: Tensor, num_groups: int) -> Tensor:
        pad = (-x.shape[1]) % num_groups
        if pad:
            x = F.pad(x, (0, pad))
        return x.reshape(x.shape[0], num_groups, -1).mean(dim=-1)

    def _split_native(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        fp = x[:, -NATIVE_1D_FINGERPRINT_DIM:]
        offset = 0
        x_maccs = fp[:, offset : offset + MACCS_FINGERPRINT_DIM]
        offset += MACCS_FINGERPRINT_DIM
        x_rdkit = fp[:, offset : offset + RDKIT_FINGERPRINT_DIM]
        offset += RDKIT_FINGERPRINT_DIM
        x_ecfp = fp[:, offset : offset + ECFP_FINGERPRINT_DIM]
        return x_maccs, x_rdkit, x_ecfp

    def _view_tokens(self, x: Tensor) -> Tensor:
        x_maccs, x_rdkit, x_ecfp = self._split_native(x)
        maccs_group = self.maccs_to_groups(x_maccs)
        rdkit_group = self._group_mean(x_rdkit, self.num_groups)
        ecfp_group = self._group_mean(x_ecfp, self.num_groups)
        x_fp = torch.stack([maccs_group, rdkit_group, ecfp_group], dim=1)
        return self.view_norm(x_fp)

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_native_fingerprints:
            return self.fallback(x)

        # iTransformer inversion: each fingerprint view is a variate token,
        # and the aligned descriptor-group vector is embedded as its token value.
        view_tokens = self.value_embedding(self._view_tokens(x)) + self.view_embedding
        view_tokens = self.encoder(view_tokens)
        z_ts = self.ts_pool(view_tokens)
        z_raw = self.raw_encoder(x)
        return self.out(torch.cat([z_ts, z_raw], dim=1))


def build_fingerprint_encoder(
    encoder: str,
    d_in: int,
    d_out: int,
    num_layers: int = 2,
    num_groups: int = 128,
    nhead: int = 4,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
    molformer_model: str = "ibm-research/MoLFormer-XL-both-10pct",
    molformer_max_length: int = 256,
    molformer_pooling: str = "pooler",
    molformer_unfreeze_layers: int = 0,
) -> nn.Module:
    match encoder:
        case "itransformer":
            return ITransformerFingerprintEncoder(
                d_in=d_in,
                d_out=d_out,
                num_layers=num_layers,
                num_groups=num_groups,
                nhead=nhead,
                dropout=dropout,
                activation=activation,
            )
        case "duet":
            return DUETFingerprintEncoder(
                d_in=d_in,
                d_out=d_out,
                num_layers=num_layers,
                num_groups=num_groups,
                dropout=dropout,
                activation=activation,
            )
        case "molformer":
            return TrainableMolFormerFingerprintEncoder(
                model_name=molformer_model,
                d_out=d_out,
                max_length=molformer_max_length,
                pooling=molformer_pooling,
                unfreeze_layers=molformer_unfreeze_layers,
                dropout=dropout,
                activation=activation,
            )
        case _:
            raise ValueError(f"Unknown fingerprint encoder: {encoder}")


class OneDOnlyEncoder(nn.Module):
    """Use only datapoint descriptors/fingerprints and ignore the DMPNN graph vector."""

    requires_graph_context = True

    def __init__(
        self,
        d_xd_in: int,
        d_xd_out: int,
        num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        nhead: int = 4,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        **fp_encoder_kwargs,
    ):
        super().__init__()
        self.net = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_xd_in,
            d_out=d_xd_out,
            num_layers=num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d_out = d_xd_out

    def forward(self, H: Tensor, X_d: Tensor) -> Tensor:
        del H
        if X_d is None:
            raise ValueError("X_d is required for the 1D-only encoder.")
        return self.net(X_d)


class MultiHeadTokenFusion(nn.Module):
    """Fuse modality tokens with multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_tokens: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        act = activation if activation is not None else nn.ReLU()
        self.token_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.token_embedding[:, : tokens.shape[1], :]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))
        return tokens.mean(dim=1)


class GeometryGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        act = activation if activation is not None else nn.ReLU()
        self.nhead = nhead
        self.distance_bias = nn.Sequential(
            nn.Linear(2, nhead),
            nn.Tanh(),
        )
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, distances: Tensor, adjacency: Tensor, mask: Tensor) -> Tensor:
        edge_features = torch.stack(
            [distances, torch.exp(-distances.clamp_min(0.0))],
            dim=-1,
        )
        attn_mask = self.distance_bias(edge_features).permute(0, 3, 1, 2)
        attn_mask = attn_mask.masked_fill(~adjacency[:, None, :, :], -1e4)
        attn_mask = attn_mask.reshape(x.shape[0] * self.nhead, x.shape[1], x.shape[1])

        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class GeometryGraphTransformerEncoder(nn.Module):
    """MMFF geometry graph -> GraphTransformer -> global average pooling."""

    def __init__(
        self,
        node_fdim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.node_proj = nn.Linear(node_fdim, d_model)
        self.layers = nn.ModuleList(
            [
                GeometryGraphTransformerLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_out = d_model

    def forward(self, graph: BatchGeometryGraph) -> Tensor:
        V = torch.nan_to_num(graph.V, nan=0.0, posinf=0.0, neginf=0.0)
        distances = torch.nan_to_num(graph.distances, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.node_proj(V)
        for layer in self.layers:
            x = layer(x, distances, graph.adjacency, graph.mask)
        x = self.norm(x)
        mask = graph.mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class GeometryMeanPoolEncoder(nn.Module):
    """3D graph node descriptors -> masked mean pooling -> fixed-size embedding."""

    def __init__(
        self,
        node_fdim: int,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.proj = nn.Sequential(
            nn.Linear(node_fdim, d_model),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.d_out = d_model

    def forward(self, graph: BatchGeometryGraph) -> Tensor:
        x = self.proj(torch.nan_to_num(graph.V, nan=0.0, posinf=0.0, neginf=0.0))
        mask = graph.mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class GotenNetGeometryEncoder(nn.Module):
    """Trainable GotenNet 3D backbone over conformer atomic numbers and coordinates."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.0,
        cutoff: float = 5.0,
        pooling: str = "mean",
        n_rbf: int = 32,
        lmax: int = 2,
        coordinate_control: str = "none",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unknown GotenNet pooling mode: {pooling}")
        if coordinate_control not in {"none", "permute"}:
            raise ValueError(f"Unknown GotenNet coordinate control: {coordinate_control}")
        try:
            from gotennet import GotenNet
            from gotennet.models.components.layers import CosineCutoff
        except ImportError as exc:
            raise ImportError(
                "GotenNet 3D graphs require the optional `gotennet` package. "
                "Install this project with `pip install -e .[gotennet]` or install `gotennet`."
            ) from exc

        self.cutoff = cutoff
        self.pooling = pooling
        self.coordinate_control = coordinate_control
        self.representation = GotenNet(
            n_atom_basis=d_model,
            n_interactions=num_layers,
            n_rbf=n_rbf,
            cutoff_fn=CosineCutoff(cutoff),
            activation="swish",
            max_z=100,
            num_heads=nhead,
            attn_dropout=dropout,
            edge_updates=True,
            lmax=lmax,
            aggr="add",
            scale_edge=False,
            sep_htr=True,
            sep_dir=True,
            sep_tensor=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = (
            nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
            if pooling == "mean_max"
            else nn.Identity()
        )
        self.d_out = d_model

    def forward(self, graph: BatchGeometryGraph) -> Tensor:
        mask = graph.mask
        batch_size, max_nodes = mask.shape
        if not mask.any():
            return graph.coordinates.new_zeros((batch_size, self.d_out))

        atomic_number_chunks = []
        position_chunks = []
        batch_index_chunks = []
        edge_index_chunks = []
        edge_diff_chunks = []
        edge_vec_chunks = []

        node_offset = 0
        for graph_idx in range(batch_size):
            graph_mask = mask[graph_idx]
            num_nodes = int(graph_mask.sum().item())
            if num_nodes == 0:
                continue

            positions_i = graph.coordinates[graph_idx, graph_mask]
            atomic_numbers_i = graph.atomic_numbers[graph_idx, graph_mask].clamp(min=0, max=99)
            if self.coordinate_control == "permute" and num_nodes > 1:
                positions_i = torch.roll(positions_i, shifts=1, dims=0)
            atomic_number_chunks.append(atomic_numbers_i)
            position_chunks.append(positions_i)
            batch_index_chunks.append(
                torch.full((num_nodes,), graph_idx, dtype=torch.long, device=positions_i.device)
            )

            if num_nodes > 1:
                dist_i = torch.cdist(positions_i, positions_i)
                edge_mask_i = (dist_i <= self.cutoff) & (dist_i > 1e-8)
                local_edge_index = edge_mask_i.nonzero(as_tuple=False).t().contiguous()
                if local_edge_index.numel() > 0:
                    edge_index_chunks.append(local_edge_index + node_offset)
                    edge_diff_chunks.append(dist_i[local_edge_index[0], local_edge_index[1]])
                    edge_vec_chunks.append(
                        positions_i[local_edge_index[0]] - positions_i[local_edge_index[1]]
                    )

            node_offset += num_nodes

        if not atomic_number_chunks:
            return graph.coordinates.new_zeros((batch_size, self.d_out))

        atomic_numbers = torch.cat(atomic_number_chunks, dim=0)
        positions = torch.cat(position_chunks, dim=0)
        batch_index = torch.cat(batch_index_chunks, dim=0)
        if edge_index_chunks:
            edge_index = torch.cat(edge_index_chunks, dim=1).contiguous()
            edge_diff = torch.cat(edge_diff_chunks, dim=0)
            edge_vec = torch.cat(edge_vec_chunks, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=positions.device)
            edge_diff = positions.new_empty((0,))
            edge_vec = positions.new_empty((0, 3))

        h, _ = self.representation(atomic_numbers, edge_index, edge_diff, edge_vec)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.norm(h)
        pooled = h.new_zeros((batch_size, h.shape[-1]))
        pooled.index_add_(0, batch_index, h)
        counts = torch.bincount(batch_index, minlength=batch_size).to(h.dtype).unsqueeze(-1)
        mean_pooled = pooled / counts.clamp_min(1.0)
        if self.pooling == "mean":
            return mean_pooled

        max_pooled = h.new_zeros((batch_size, h.shape[-1]))
        for graph_idx in range(batch_size):
            node_mask = batch_index == graph_idx
            if node_mask.any():
                max_pooled[graph_idx] = h[node_mask].max(dim=0).values

        return torch.nan_to_num(
            self.pool_proj(torch.cat([mean_pooled, max_pooled], dim=-1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


def build_geometry_graph_encoder(
    pooler: str,
    node_fdim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
    gotennet_cutoff: float = 5.0,
    gotennet_pooling: str = "mean",
    gotennet_coordinate_control: str = "none",
) -> nn.Module:
    match pooler:
        case "graph_transformer":
            return GeometryGraphTransformerEncoder(
                node_fdim=node_fdim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
            )
        case "mean":
            return GeometryMeanPoolEncoder(
                node_fdim=node_fdim,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
            )
        case "gotennet":
            return GotenNetGeometryEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                cutoff=gotennet_cutoff,
                pooling=gotennet_pooling,
                coordinate_control=gotennet_coordinate_control,
            )
        case _:
            raise ValueError(f"Unknown 3D graph pooler: {pooler}")


class ThreeWayGatedFusionEncoder(nn.Module):
    """Concatenate 2D graph, 1D fingerprint, and 3D geometry embeddings.

    The 1D branch compresses concatenated native-dimensional fingerprints to
    ``d_xd_out``. The 3D branch either compresses descriptor columns or encodes
    an MMFF geometry graph with GraphTransformer + GAP to ``d_xd_out``. The
    returned vector is ``[2D ; 1D ; 3D]`` and is passed to the predictor head.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )
        self.d_out = d_h + 2 * d_xd_out

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        return torch.cat([H, Z_1d, Z_3d], dim=1)


class ConcatPreservingResidualCrossAttentionFusionEncoder(nn.Module):
    """Three-way concat with a zero-initialized residual cross-attention branch.

    The output is exactly ``[2D ; 1D ; 3D]`` at initialization because ``alpha``
    starts at zero. Cross-attention can then learn a residual correction without
    destroying the concat baseline.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        n_slots: int = 4,
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if d_xd_out % nhead != 0:
            raise ValueError(f"d_xd_out ({d_xd_out}) must be divisible by nhead ({nhead})")
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.n_slots = n_slots
        self.d_out = d_h + 2 * d_xd_out

        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )

        self.graph_proj = nn.Sequential(nn.Linear(d_h, d_xd_out), act, nn.Dropout(dropout))
        self.query_proj = nn.Sequential(
            nn.Linear(self.d_out, d_xd_out),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_xd_out),
        )
        self.slot_gens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_xd_out, n_slots * d_xd_out),
                    act,
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, 3, n_slots, d_xd_out))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_xd_out, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(self.d_out, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(3 * d_xd_out, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_out),
        )
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.last_attention_weights: Tensor | None = None
        self.last_modality_gates: Tensor | None = None
        self.last_pair_gates: Tensor | None = None

    def _encode_modalities(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if X_d is None:
            raise ValueError("X_d is required for CPR-XAttn because it carries the 1D branch.")

        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        Z_2d = self.graph_proj(H)
        return Z_1d, Z_2d, Z_3d

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        Z_1d, Z_2d, Z_3d = self._encode_modalities(H, X_d, X_3d_graph)
        h_concat = torch.cat([H, Z_1d, Z_3d], dim=1)
        query = self.query_proj(h_concat).unsqueeze(1)

        attn_outputs = []
        attn_weights = []
        for i, Z in enumerate([Z_1d, Z_2d, Z_3d]):
            slots = self.slot_gens[i](Z).reshape(Z.shape[0], self.n_slots, -1)
            slots = slots + self.slot_embedding[:, i, :, :]
            out, weights = self.cross_attn(
                query,
                slots,
                slots,
                need_weights=True,
                average_attn_weights=True,
            )
            attn_outputs.append(out.squeeze(1))
            attn_weights.append(weights.squeeze(1))

        gates = self.gate(h_concat)
        residual_tokens = [gates[:, i : i + 1] * u for i, u in enumerate(attn_outputs)]
        residual = self.residual_proj(torch.cat(residual_tokens, dim=1))

        self.last_attention_weights = torch.stack(attn_weights, dim=1).detach()
        self.last_modality_gates = gates.detach()
        return h_concat + self.alpha.tanh() * self.dropout(residual)


class TwoDCentricCrossAttentionFusionEncoder(nn.Module):
    """2D-centric embedding cross-attention over 1D, 2D, and 3D modalities.

    This follows the summary-centric CircuitFusion pattern at molecule level:
    the stable 2D DMPNN embedding supplies the query, while projected 1D, 2D,
    and 3D embeddings supply keys and values. The returned vector is
    ``[1D ; enhanced_2D ; 3D]`` for the standard predictor FFN.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if d_xd_out % nhead != 0:
            raise ValueError(f"d_xd_out ({d_xd_out}) must be divisible by nhead ({nhead})")
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.d_out = 3 * d_xd_out

        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(d_h, d_xd_out),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_xd_out),
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
                nn.LayerNorm(d_xd_out),
            )
        )
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_xd_out))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        self.cross_attn = nn.MultiheadAttention(d_xd_out, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_xd_out)
        self.last_attention_weights: Tensor | None = None

    def _encode_modalities(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if X_d is None:
            raise ValueError("X_d is required for 2D-centric XAttn because it carries the 1D branch.")

        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        Z_2d = self.graph_proj(H)
        return Z_1d, Z_2d, Z_3d

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        Z_1d, Z_2d, Z_3d = self._encode_modalities(H, X_d, X_3d_graph)
        tokens = torch.stack([Z_1d, Z_2d, Z_3d], dim=1) + self.modality_embedding
        query = tokens[:, 1:2, :]
        Z_2d_cross, attn_weights = self.cross_attn(
            query,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        Z_2d_enhanced = self.norm(Z_2d + self.dropout(Z_2d_cross.squeeze(1)))

        self.last_attention_weights = attn_weights.squeeze(1).detach()
        return torch.cat([Z_1d, Z_2d_enhanced, Z_3d], dim=1)


class TriPairGatedCrossAttentionFusionEncoder(nn.Module):
    """Pairwise gated cross-attention over 1D, 2D, and 3D modality embeddings.

    One projected token is built for each modality. For each modality pair, a
    pair query attends only over the two tokens in that pair, producing an
    interaction token. A soft gate estimates modality reliability from the
    three modality tokens and the three pair tokens. The output keeps the same
    shape as the stable ``[raw 2D ; 1D ; 3D]`` concat baseline and only applies
    a small feature-scale calibration to each modality branch.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if d_xd_out % nhead != 0:
            raise ValueError(f"d_xd_out ({d_xd_out}) must be divisible by nhead ({nhead})")
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.pairs = ((0, 1), (0, 2), (1, 2))
        self.d_out = d_h + 2 * d_xd_out

        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(d_h, d_xd_out),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_xd_out),
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
                nn.LayerNorm(d_xd_out),
            )
        )

        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_xd_out))
        self.pair_embedding = nn.Parameter(torch.zeros(1, 3, d_xd_out))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.pair_embedding, std=0.02)

        self.pair_query = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4 * d_xd_out, hidden),
                    act,
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_xd_out),
                    nn.LayerNorm(d_xd_out),
                )
                for _ in self.pairs
            ]
        )
        self.cross_attn = nn.MultiheadAttention(d_xd_out, nhead, dropout=dropout, batch_first=True)
        self.pair_norm = nn.LayerNorm(d_xd_out)
        self.gate = nn.Sequential(
            nn.Linear(6 * d_xd_out, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Softmax(dim=-1),
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)
        self.modality_scale = 0.1
        self.last_attention_weights: Tensor | None = None
        self.last_modality_gates: Tensor | None = None
        self.last_pair_gates: Tensor | None = None

    def _encode_modalities(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if X_d is None:
            raise ValueError("X_d is required for tri-pair gated XAttn because it carries the 1D branch.")

        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        Z_2d = self.graph_proj(H)
        return Z_1d, Z_2d, Z_3d

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        Z_1d, Z_2d, Z_3d = self._encode_modalities(H, X_d, X_3d_graph)
        tokens = torch.stack([Z_1d, Z_2d, Z_3d], dim=1) + self.modality_embedding

        pair_tokens = []
        pair_weights = []
        for pair_idx, (left_idx, right_idx) in enumerate(self.pairs):
            left = tokens[:, left_idx, :]
            right = tokens[:, right_idx, :]
            query_in = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
            query = self.pair_query[pair_idx](query_in).unsqueeze(1) + self.pair_embedding[:, pair_idx : pair_idx + 1, :]
            key_value = tokens[:, [left_idx, right_idx], :]
            out, weights = self.cross_attn(
                query,
                key_value,
                key_value,
                need_weights=True,
                average_attn_weights=True,
            )
            pair_tokens.append(self.pair_norm(out.squeeze(1) + 0.5 * (left + right)))
            pair_weights.append(weights.squeeze(1))

        pair_stack = torch.stack(pair_tokens, dim=1)
        branch_tokens = torch.cat([tokens, pair_stack], dim=1)
        gates = self.gate(branch_tokens.flatten(start_dim=1))

        self.last_attention_weights = torch.stack(pair_weights, dim=1).detach()
        modality_gates = gates[:, :3]
        modality_gates = modality_gates / modality_gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.last_modality_gates = modality_gates.detach()
        self.last_pair_gates = gates[:, 3:].detach()

        scales = 1.0 + self.modality_scale * (modality_gates - (1.0 / 3.0))
        Z_1d = Z_1d * scales[:, 0:1]
        H = H * scales[:, 1:2]
        Z_3d = Z_3d * scales[:, 2:3]
        return torch.cat([H, Z_1d, Z_3d], dim=1)


class DGMFFusionEncoder(nn.Module):
    """Directed gated fusion over semantic, topological, and geometric embeddings.

    This module does not turn modalities into attention tokens. Instead, it
    computes feature-wise gates directly between whole-molecule modality
    embeddings. Controlled variants isolate target conditioning, direction-
    specific parameters, the residual path, and tri-modal self-attention.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        fusion_variant: str = "full",
        gotennet_coordinate_control: str = "none",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")
        valid_variants = {
            "full",
            "matched-target-agnostic",
            "shared-gate",
            "matched-shared-gate",
            "direction-id-gate",
            "no-residual",
            "self-attention",
        }
        if fusion_variant not in valid_variants:
            raise ValueError(
                f"Unknown embedding fusion variant {fusion_variant!r}; "
                f"expected one of {sorted(valid_variants)}"
            )

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.fusion_variant = fusion_variant
        self.directed_pairs = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
        self.d_out = d_h + 2 * d_xd_out

        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(d_h, d_xd_out),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_xd_out),
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
                gotennet_coordinate_control=gotennet_coordinate_control,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
                nn.LayerNorm(d_xd_out),
            )
        )

        if fusion_variant == "self-attention":
            self.value_proj = nn.ModuleList()
            self.cross_gate = nn.ModuleList()
            self.modality_attn = nn.MultiheadAttention(
                d_xd_out,
                nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.modality_attn_norm = nn.LayerNorm(d_xd_out)
            self.modality_ffn = nn.Sequential(
                nn.Linear(d_xd_out, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
            self.modality_ffn_norm = nn.LayerNorm(d_xd_out)
        else:
            self.value_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_xd_out, d_xd_out),
                        act,
                        nn.Dropout(dropout),
                        nn.LayerNorm(d_xd_out),
                    )
                    for _ in range(3)
                ]
            )

            def build_gate(
                gate_hidden: int = hidden, *, include_sigmoid: bool = True
            ) -> nn.Sequential:
                layers: list[nn.Module] = [
                    nn.Linear(4 * d_xd_out, hidden),
                    _clone_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_xd_out),
                ]
                if gate_hidden != hidden:
                    layers[0] = nn.Linear(4 * d_xd_out, gate_hidden)
                    layers[3] = nn.Linear(gate_hidden, d_xd_out)
                if include_sigmoid:
                    layers.append(nn.Sigmoid())
                return nn.Sequential(*layers)

            self.shared_gate_input_scale: nn.Parameter | None = None
            self.shared_gate_output_scale: nn.Parameter | None = None
            self.direction_gate_trunk: nn.Sequential | None = None
            self.direction_gate_trunk_scale: nn.Parameter | None = None
            self.direction_gate_heads = nn.ModuleList()
            if fusion_variant == "matched-shared-gate":
                self.cross_gate = nn.ModuleList(
                    [build_gate(6 * hidden, include_sigmoid=False)]
                )
                self.shared_gate_input_scale = nn.Parameter(torch.ones(4 * d_xd_out))
                self.shared_gate_output_scale = nn.Parameter(torch.ones(d_xd_out))
            elif fusion_variant == "direction-id-gate":
                # The shared 4d -> 3h trunk extracts pair features for every
                # direction. Selecting one of six equal-width output heads is
                # the direction-ID conditioning operation. The active trunk
                # scale makes the gate budget exactly equal to six independent
                # 4d -> h -> d gates without introducing unused parameters.
                direction_hidden = 3 * hidden
                self.cross_gate = nn.ModuleList()
                self.direction_gate_trunk = nn.Sequential(
                    nn.Linear(4 * d_xd_out, direction_hidden),
                    _clone_activation(activation),
                    nn.Dropout(dropout),
                )
                self.direction_gate_trunk_scale = nn.Parameter(
                    torch.ones(direction_hidden)
                )
                self.direction_gate_heads = nn.ModuleList(
                    [
                        nn.Linear(direction_hidden, d_xd_out)
                        for _ in self.directed_pairs
                    ]
                )
            else:
                num_gates = 1 if fusion_variant == "shared-gate" else len(self.directed_pairs)
                self.cross_gate = nn.ModuleList([build_gate() for _ in range(num_gates)])
        self.message_norm = nn.ModuleList([nn.LayerNorm(d_xd_out) for _ in range(3)])
        self.graph_delta = nn.Sequential(
            nn.Linear(d_xd_out, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, d_h),
        )
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(3))
        self.last_attention_weights: Tensor | None = None
        self.last_modality_gates: Tensor | None = None
        self.last_pair_gates: Tensor | None = None
        self.last_directional_weights: Tensor | None = None

    def _encode_modalities(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if X_d is None:
            raise ValueError("X_d is required for DGMF because it carries the semantic branch.")

        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        Z_2d = self.graph_proj(H)
        return Z_1d, Z_2d, Z_3d

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        Z_1d, Z_2d, Z_3d = self._encode_modalities(H, X_d, X_3d_graph)
        modalities = [Z_1d, Z_2d, Z_3d]
        messages = [torch.zeros_like(Z_1d), torch.zeros_like(Z_2d), torch.zeros_like(Z_3d)]
        gate_means: list[Tensor] = []
        directional_scores: list[Tensor] = []

        if self.fusion_variant == "self-attention":
            tokens = torch.stack(modalities, dim=1)
            attn_out, attn_weights = self.modality_attn(
                tokens,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            attn_hidden = self.modality_attn_norm(tokens + attn_out)
            enhanced = self.modality_ffn_norm(attn_hidden + self.modality_ffn(attn_hidden))
            message_stack = enhanced - tokens
            messages = [message_stack[:, idx] for idx in range(3)]
            for target_idx, source_idx in self.directed_pairs:
                pair_weight = attn_weights[:, target_idx, source_idx]
                gate_means.append(pair_weight)
                directional_scores.append(pair_weight.clamp_min(1e-8))
        else:
            if self.fusion_variant == "direction-id-gate":
                if (
                    self.direction_gate_trunk is None
                    or self.direction_gate_trunk_scale is None
                    or len(self.direction_gate_heads) != len(self.directed_pairs)
                ):
                    raise RuntimeError("Direction-ID gate modules are incomplete.")
                for direction_idx, (target_idx, source_idx) in enumerate(
                    self.directed_pairs
                ):
                    target = modalities[target_idx]
                    source = modalities[source_idx]
                    gate_input = torch.cat(
                        [target, source, torch.abs(target - source), target * source],
                        dim=-1,
                    )
                    shared_features = self.direction_gate_trunk(gate_input)
                    shared_features = (
                        shared_features * self.direction_gate_trunk_scale
                    )
                    gate = torch.sigmoid(
                        self.direction_gate_heads[direction_idx](shared_features)
                    )
                    pair_message = gate * self.value_proj[source_idx](source)
                    messages[target_idx] = messages[target_idx] + pair_message
                    gate_means.append(gate.mean(dim=1))
                    directional_scores.append(
                        torch.linalg.vector_norm(pair_message, dim=1)
                        / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-8)
                    )
            else:
                gate_modules = (
                    [self.cross_gate[0]] * len(self.directed_pairs)
                    if self.fusion_variant in {"shared-gate", "matched-shared-gate"}
                    else list(self.cross_gate)
                )
                for gate_module, (target_idx, source_idx) in zip(
                    gate_modules, self.directed_pairs
                ):
                    target = modalities[target_idx]
                    source = modalities[source_idx]
                    if self.fusion_variant == "matched-target-agnostic":
                        zeros = torch.zeros_like(source)
                        gate_input = torch.cat([source, source, zeros, source * source], dim=-1)
                    else:
                        gate_input = torch.cat(
                            [target, source, torch.abs(target - source), target * source],
                            dim=-1,
                        )
                    if self.fusion_variant == "matched-shared-gate":
                        if self.shared_gate_input_scale is None or self.shared_gate_output_scale is None:
                            raise RuntimeError("Matched shared-gate calibration parameters are missing.")
                        gate_logits = gate_module(gate_input * self.shared_gate_input_scale)
                        gate = torch.sigmoid(gate_logits * self.shared_gate_output_scale)
                    else:
                        gate = gate_module(gate_input)
                    pair_message = gate * self.value_proj[source_idx](source)
                    messages[target_idx] = messages[target_idx] + pair_message
                    gate_means.append(gate.mean(dim=1))
                    directional_scores.append(
                        torch.linalg.vector_norm(pair_message, dim=1)
                        / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-8)
                    )

        message_scale = 1.0 if self.fusion_variant == "self-attention" else 2.0
        messages = [
            self.message_norm[i](message / message_scale)
            for i, message in enumerate(messages)
        ]
        alpha = self.alpha.tanh()
        residual_1d = self.dropout(messages[0])
        residual_2d = self.dropout(self.graph_delta(messages[1]))
        residual_3d = self.dropout(messages[2])
        update_scale = torch.ones_like(alpha) if self.fusion_variant == "no-residual" else alpha

        contribution_scores = torch.stack(
            [
                torch.linalg.vector_norm(update_scale[0] * residual_1d, dim=1)
                / torch.linalg.vector_norm(Z_1d, dim=1).clamp_min(1e-8),
                torch.linalg.vector_norm(update_scale[1] * residual_2d, dim=1)
                / torch.linalg.vector_norm(H, dim=1).clamp_min(1e-8),
                torch.linalg.vector_norm(update_scale[2] * residual_3d, dim=1)
                / torch.linalg.vector_norm(Z_3d, dim=1).clamp_min(1e-8),
            ],
            dim=1,
        ).clamp_min(1e-8)
        contribution_weights = contribution_scores / contribution_scores.sum(dim=1, keepdim=True)

        if self.fusion_variant == "no-residual":
            Z_1d = residual_1d
            H = residual_2d
            Z_3d = residual_3d
        else:
            Z_1d = Z_1d + alpha[0] * residual_1d
            H = H + alpha[1] * residual_2d
            Z_3d = Z_3d + alpha[2] * residual_3d

        gate_stack = torch.stack(gate_means, dim=1)
        incoming = torch.stack(
            [
                gate_stack[:, [0, 1]].mean(dim=1),
                gate_stack[:, [2, 3]].mean(dim=1),
                gate_stack[:, [4, 5]].mean(dim=1),
            ],
            dim=1,
        )
        modality_gates = torch.softmax(incoming, dim=1)
        pair_gates = torch.stack(
            [
                gate_stack[:, [0, 2]].mean(dim=1),
                gate_stack[:, [1, 4]].mean(dim=1),
                gate_stack[:, [3, 5]].mean(dim=1),
            ],
            dim=1,
        )
        pair_gates = pair_gates / pair_gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        directional_stack = torch.stack(directional_scores, dim=1).clamp_min(1e-8)
        directional_weights = directional_stack / directional_stack.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        self.last_attention_weights = contribution_weights.detach()
        self.last_modality_gates = modality_gates.detach()
        self.last_pair_gates = pair_gates.detach()
        self.last_directional_weights = directional_weights.detach()
        return torch.cat([H, Z_1d, Z_3d], dim=1)


# Backward-compatible name used by checkpoints and pre-release experiment scripts.
EmbeddingCrossGatedAttentionFusionEncoder = DGMFFusionEncoder


class TwoDThreeDAnchoredOneDGateFusionEncoder(nn.Module):
    """Use a stable 2D+3D anchor and add 1D only as gated residual adaptation.

    This is a MAG/FiLM-inspired reliability gate: the 2D graph embedding and
    3D geometry embedding form the protected main path, while frozen or
    trainable MoLFormer 1D features can only contribute through a
    zero-initialized residual branch. At initialization the module is exactly
    equivalent to the 2D+3D concat backbone.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)
        anchor_dim = d_h + d_xd_out

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.d_out = anchor_dim

        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )

        self.anchor_norm = nn.LayerNorm(anchor_dim)
        self.one_d_align = nn.Sequential(
            nn.Linear(d_xd_out, anchor_dim),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(anchor_dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(d_xd_out, anchor_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(anchor_dim, anchor_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * anchor_dim, max(2 * d_xd_out, anchor_dim)),
            act,
            nn.Dropout(dropout),
            nn.Linear(max(2 * d_xd_out, anchor_dim), anchor_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.last_modality_gates: Tensor | None = None
        self.last_attention_weights: Tensor | None = None

    def _encode_modalities(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> tuple[Tensor, Tensor]:
        if X_d is None:
            raise ValueError("X_d is required for anchored-gated fusion because it carries the 1D branch.")

        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        return Z_1d, Z_3d

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        Z_1d, Z_3d = self._encode_modalities(H, X_d, X_3d_graph)
        anchor = self.anchor_norm(torch.cat([H, Z_3d], dim=1))
        z1d_anchor = self.one_d_align(Z_1d)
        gate_in = torch.cat([anchor, z1d_anchor, torch.abs(anchor - z1d_anchor)], dim=1)
        gate = self.gate(gate_in)
        residual = gate * self.delta(Z_1d)

        gate_scalar = gate.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        anchor_share = (1.0 - gate_scalar) / 2.0
        self.last_modality_gates = torch.cat(
            [gate_scalar, anchor_share, anchor_share],
            dim=1,
        ).detach()
        self.last_attention_weights = self.last_modality_gates.detach()

        return self.anchor_norm(anchor + self.alpha.tanh() * self.dropout(residual))


class TaskAwareEntropyGatedModalityCrossAttentionEncoder(nn.Module):
    """Task-query cross-attention over 1D, 2D, and 3D modality embeddings.

    This adapts the CircuitFusion/DREAM fusion pattern to molecule-level ADMET
    prediction: each task owns a learnable query and attends over one token per
    molecular modality. The normalized attention entropy gates how much the
    cross-attended context is trusted against a lightweight residual mixture.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_model: int,
        n_tasks: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_model * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.n_tasks = n_tasks
        self.d_out = d_model

        self.graph_proj = nn.Sequential(
            nn.Linear(d_h, d_model),
            act,
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_model,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_model,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.LayerNorm(d_model),
            )
        )

        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, d_model))
        self.task_queries = nn.Parameter(torch.empty(1, n_tasks, d_model))
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.task_queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model + 1, d_model),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            act,
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.last_attention_weights: Tensor | None = None
        self.last_attention_entropy: Tensor | None = None

    def _modal_tokens(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None
    ) -> Tensor:
        if X_d is None:
            raise ValueError("X_d is required for TEG-MCA because it carries the 1D branch.")

        Z_2d = self.graph_proj(H)
        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        return torch.stack([Z_1d, Z_2d, Z_3d], dim=1) + self.modality_embedding

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        tokens = self._modal_tokens(H, X_d, X_3d_graph)
        queries = self.task_queries.expand(tokens.shape[0], -1, -1)
        attn_out, weights = self.cross_attn(
            queries,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1)
        entropy = entropy / math.log(tokens.shape[1])
        confidence = (1.0 - entropy).unsqueeze(-1)
        residual = torch.bmm(weights, tokens)

        gate_in = torch.cat([queries, attn_out, residual, entropy.unsqueeze(-1)], dim=-1)
        gate = self.gate(gate_in) * confidence
        fused = gate * attn_out + (1.0 - gate) * residual
        fused = self.norm(queries + self.dropout(fused))
        fused = self.norm(fused + self.dropout(self.ffn(fused)))

        self.last_attention_weights = weights.detach()
        self.last_attention_entropy = entropy.detach()
        return fused


class OneDThreeDFusionEncoder(nn.Module):
    """Concatenate 1D fingerprint and 3D geometry/descriptor embeddings.

    This branch intentionally does not concatenate the DMPNN/2D graph vector,
    so it can be used as the "1D+3D only" comparison against 2D and 3-way
    fusion models.
    """

    requires_graph_context = True

    def __init__(
        self,
        d_xd_in: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        if not use_3d_graph and (d_3d <= 0 or d_3d >= d_xd_in):
            raise ValueError(f"d_3d must be in [1, d_xd_in-1], got d_3d={d_3d}, d_xd_in={d_xd_in}")

        d_1d = d_xd_in if use_3d_graph else d_xd_in - d_3d
        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.fp_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_1d,
            d_out=d_xd_out,
            num_layers=graph_num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )
        self.d_out = 2 * d_xd_out

    def forward(
        self, H: Tensor, X_d: Tensor, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        del H
        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            X_1d = X_d
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            X_3d = X_d[:, : self.d_3d]
            X_1d = X_d[:, self.d_3d :]
            Z_3d = self.d3_proj(X_3d)

        Z_1d = self.fp_proj(X_1d)
        return torch.cat([Z_1d, Z_3d], dim=1)


class ThreeDOnlyEncoder(nn.Module):
    """Use only 3D descriptors or 3D graph node descriptors and ignore 1D/2D inputs."""

    requires_graph_context = True

    def __init__(
        self,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
    ):
        super().__init__()
        if not use_3d_graph and d_3d <= 0:
            raise ValueError(f"d_3d must be > 0 for descriptor 3D, got d_3d={d_3d}.")

        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)
        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )
        self.d_out = d_xd_out

    def forward(
        self, H: Tensor, X_d: Tensor | None, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        del H
        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            return self.d3_proj(X_3d_graph)

        if X_d is None:
            raise ValueError("X_d is required when using descriptor 3D.")
        return self.d3_proj(X_d[:, : self.d_3d])


class TwoDThreeDFusionEncoder(nn.Module):
    """Concatenate DMPNN/2D graph and 3D geometry embeddings."""

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_out: int,
        d_3d: int = 0,
        use_3d_graph: bool = False,
        node_fdim_3d: int = 8,
        graph_pooler: str = "graph_transformer",
        graph_num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        gotennet_cutoff: float = 5.0,
        gotennet_pooling: str = "mean",
    ):
        super().__init__()
        if not use_3d_graph and d_3d <= 0:
            raise ValueError(f"d_3d must be > 0 for descriptor 3D, got d_3d={d_3d}.")

        act = activation if activation is not None else nn.ReLU()
        hidden = max(d_xd_out * 2, 256)

        self.d_3d = d_3d
        self.use_3d_graph = use_3d_graph
        self.d3_proj = (
            build_geometry_graph_encoder(
                pooler=graph_pooler,
                node_fdim=node_fdim_3d,
                d_model=d_xd_out,
                nhead=nhead,
                num_layers=graph_num_layers,
                dropout=dropout,
                activation=activation,
                gotennet_cutoff=gotennet_cutoff,
                gotennet_pooling=gotennet_pooling,
            )
            if use_3d_graph
            else nn.Sequential(
                nn.Linear(d_3d, hidden),
                act,
                nn.Dropout(dropout),
                nn.Linear(hidden, d_xd_out),
            )
        )
        self.d_out = d_h + d_xd_out

    def forward(
        self, H: Tensor, X_d: Tensor | None, X_3d_graph: BatchGeometryGraph | None = None
    ) -> Tensor:
        if self.use_3d_graph:
            if X_3d_graph is None:
                raise ValueError("X_3d_graph is required when use_3d_graph=True.")
            Z_3d = self.d3_proj(X_3d_graph)
        else:
            if X_d is None:
                raise ValueError("X_d is required when using descriptor 3D.")
            Z_3d = self.d3_proj(X_d[:, : self.d_3d])

        return torch.cat([H, Z_3d], dim=1)


class TwoWayAttentionFusionEncoder(nn.Module):
    """Fuse DMPNN/2D graph embedding and DUET-FP encoded 1D descriptors with MHA."""

    requires_graph_context = True

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        nhead: int = 4,
        num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        **fp_encoder_kwargs,
    ):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.graph_proj = nn.Sequential(nn.Linear(d_h, d_xd_out), act, nn.Dropout(dropout))
        self.xd_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_xd_in,
            d_out=d_xd_out,
            num_layers=num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        self.attn_fusion = MultiHeadTokenFusion(d_xd_out, nhead, 2, dropout, activation)
        self.norm = nn.LayerNorm(d_xd_out)
        self.d_out = d_h + d_xd_out

    def forward(self, H: Tensor, X_d: Tensor) -> Tensor:
        G = self.graph_proj(H)
        Z = self.xd_proj(X_d)
        fused_desc = self.norm(self.attn_fusion(torch.stack([G, Z], dim=1)) + G)
        return torch.cat([H, fused_desc], dim=1)


class GatedFusionEncoder(nn.Module):
    """Gate-controlled fusion of DMPNN graph embedding H and 1D descriptor embedding Z.

    Instead of a raw concat, a sigmoid gate learns how much of Z to let through:
        g = sigmoid(W_g [ H ; Z_1d ])
        output = [ H ; g * Z_1d ]

    This lets the model suppress the 1D branch per-sample when it is not informative.
    """

    def __init__(
        self,
        d_h: int,
        d_xd_in: int,
        d_xd_out: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        num_layers: int = 2,
        fp_groups: int = 128,
        fp_encoder: str = "itransformer",
        nhead: int = 4,
        **fp_encoder_kwargs,
    ):
        super().__init__()
        self.xd_proj = build_fingerprint_encoder(
            encoder=fp_encoder,
            d_in=d_xd_in,
            d_out=d_xd_out,
            num_layers=num_layers,
            num_groups=fp_groups,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            **fp_encoder_kwargs,
        )
        # Gate: maps [H ; Z] -> d_xd_out scalars
        self.gate = nn.Sequential(
            nn.Linear(d_h + d_xd_out, d_xd_out),
            nn.Sigmoid(),
        )
        self.d_out = d_h + d_xd_out

    def forward(self, H: Tensor, X_d: Tensor) -> Tensor:
        """Returns fused [H ; g*Z] of shape [B, d_h + d_xd_out]."""
        Z = self.xd_proj(X_d)          # [B, d_xd_out]
        g = self.gate(torch.cat([H, Z], dim=1))   # [B, d_xd_out]
        return torch.cat([H, g * Z], dim=1)        # [B, d_h + d_xd_out]


class FingerprintTransformerEncoder(nn.Module):
    """Patch the descriptor vector, run a Transformer encoder, mean-pool to one vector."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        patch_size: int = 128,
        activation: str = "relu",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_in = d_in
        self.patch_size = patch_size
        n_patches = math.ceil(d_in / patch_size)

        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, d_in]
        d_in = x.shape[1]
        pad = (-d_in) % self.patch_size
        if pad:
            x = F.pad(x, (0, pad))
        n_patches = x.shape[1] // self.patch_size
        x = x.view(x.shape[0], n_patches, self.patch_size)
        x = self.patch_proj(x)
        if n_patches != self.pos_embedding.shape[1]:
            # Rare: if d_in changes vs construction; re-interpolate position table
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=n_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_embedding[:, :n_patches, :]
        x = x + pos
        x = self.encoder(x)
        return x.mean(dim=1)
