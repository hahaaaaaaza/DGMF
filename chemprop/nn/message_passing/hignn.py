from __future__ import annotations

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


def _sum_by_index(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = values.new_zeros((dim_size, values.shape[-1]))
    if values.numel() == 0 or index.numel() == 0:
        return out
    out.index_add_(0, index, values)
    return out


def _mean_by_index(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = _sum_by_index(values, index, dim_size)
    if values.numel() == 0 or index.numel() == 0:
        return out
    counts = torch.bincount(index, minlength=dim_size).to(values.dtype).unsqueeze(1)
    return out / counts.clamp_min(1)


def _max_by_index(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = values.new_full((dim_size, values.shape[-1]), -torch.inf)
    if values.numel() == 0 or index.numel() == 0:
        return values.new_zeros((dim_size, values.shape[-1]))
    if hasattr(out, "scatter_reduce_"):
        expanded = index.unsqueeze(1).expand_as(values)
        out.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    else:
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                out[i] = values[mask].max(dim=0).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _softmax_by_index(scores: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if scores.numel() == 0:
        return scores
    max_per = _max_by_index(scores.unsqueeze(1), index, dim_size).squeeze(1)
    exp = torch.exp(scores - max_per[index])
    denom = torch.zeros(dim_size, dtype=exp.dtype, device=exp.device)
    denom.index_add_(0, index, exp)
    return exp / denom[index].clamp_min(1e-12)


class _FeatureAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor, batch: Tensor, size: int) -> Tensor:
        max_result = _max_by_index(x, batch, size)
        sum_result = _sum_by_index(x, batch, size)
        weights = torch.sigmoid(self.mlp(max_result) + self.mlp(sum_result))
        return x * weights[batch]


class _NTNConv(nn.Module):
    def __init__(self, d_hidden: int, slices: int, dropout: float):
        super().__init__()
        if d_hidden % slices != 0:
            raise ValueError(f"HiGNN hidden dim ({d_hidden}) must be divisible by slices ({slices}).")
        self.d_hidden = d_hidden
        self.slices = slices
        self.dropout = dropout
        self.bilinear = nn.Bilinear(d_hidden, d_hidden, slices, bias=False)
        self.linear = nn.Linear(3 * d_hidden, slices)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        src, dst = edge_index
        x_j = x[src]
        x_i = x[dst]
        score = self.bilinear(x_i, x_j)
        block_score = self.linear(torch.cat((x_i, edge_attr, x_j), dim=1))
        alpha = torch.tanh(score + block_score)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        dim_split = self.d_hidden // self.slices
        msg_base = torch.maximum(x_j, edge_attr).reshape(-1, self.slices, dim_split)
        msg = (msg_base * alpha.unsqueeze(-1)).reshape(-1, self.d_hidden)
        return _sum_by_index(msg, dst, x.shape[0])


class HiGNNMessagePassing(MessagePassing, HyperparametersMixin):
    """HiGNN-style hierarchical BRICS fragment encoder for the 2D branch.

    This adapts the public HiGNN backbone to Chemprop's native ``BatchMolGraph``:
    atom and BRICS-fragment graphs are updated with NTNConv-style gated message
    passing, feature-wise attention is applied at both levels, and molecule-
    fragment attention injects fragment context into the molecule representation.
    """

    returns_graph_embedding = True

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
        depth: int = 3,
        slices: int = 2,
        dropout: float = 0.2,
        feature_attention: bool = True,
        attention_reduction: int = 4,
        activation: str | nn.Module | Activation = Activation.RELU,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        ignore_list = ["V_d_transform", "graph_transform"]
        if isinstance(activation, nn.Module):
            ignore_list.append("activation")
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["graph_transform"] = graph_transform
        if isinstance(activation, nn.Module):
            self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.depth = depth
        self.d_h = d_h
        self.feature_attention = feature_attention
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

        atom_input_dim = d_v + (d_vd or 0)
        self.atom_proj = nn.Linear(atom_input_dim, d_h, bias=bias)
        self.bond_proj = nn.Linear(d_e, d_h, bias=bias)
        self.fragment_edge_proj = nn.LazyLinear(d_h, bias=bias)
        self.fragment_init = nn.Linear(d_h, d_h, bias=bias)

        self.atom_convs = nn.ModuleList([_NTNConv(d_h, slices=slices, dropout=dropout) for _ in range(depth)])
        self.fragment_convs = nn.ModuleList([_NTNConv(d_h, slices=slices, dropout=dropout) for _ in range(depth)])
        self.atom_gate = nn.Linear(3 * d_h, d_h)
        self.fragment_gate = nn.Linear(3 * d_h, d_h)

        self.atom_feature_attention = _FeatureAttention(d_h, attention_reduction)
        self.fragment_feature_attention = _FeatureAttention(d_h, attention_reduction)

        self.fragment_query = nn.Linear(d_h, d_h)
        self.fragment_key = nn.Linear(d_h, d_h)
        self.fragment_value = nn.Linear(d_h, d_h)
        self.readout = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            get_activation_function(activation),
            nn.Dropout(dropout),
            nn.LayerNorm(d_h),
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.d_h

    def _atom_input(self, bmg: BatchMolGraph, V_d: Tensor | None) -> Tensor:
        if V_d is None:
            return bmg.V

        V_d = self.V_d_transform(V_d)
        if V_d.shape[0] != bmg.V.shape[0]:
            raise InvalidShapeError("V_d", V_d.shape, [len(bmg.V), V_d.shape[1]])
        return torch.cat((bmg.V, V_d), dim=1)

    def _fragment_edges(self, bmg: BatchMolGraph, motif_h: Tensor) -> tuple[Tensor, Tensor]:
        motif_edge_index = getattr(bmg, "motif_edge_index", None)
        motif_edge_features = getattr(bmg, "motif_edge_features", None)
        if motif_edge_index is None or motif_edge_index.numel() == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=motif_h.device),
                motif_h.new_zeros((0, self.d_h)),
            )
        if motif_edge_features is not None and motif_edge_features.numel():
            edge_attr = self.fragment_edge_proj(motif_edge_features)
        else:
            edge_attr = motif_h.new_zeros((motif_edge_index.shape[1], self.d_h))
        return motif_edge_index, edge_attr

    def _molecule_fragment_attention(
        self, mol_vec: Tensor, motif_h: Tensor, motif_batch: Tensor
    ) -> Tensor:
        if motif_h.numel() == 0:
            return torch.zeros_like(mol_vec)
        q = self.fragment_query(mol_vec)[motif_batch]
        k = self.fragment_key(motif_h)
        scores = (q * k).sum(dim=1) / (self.d_h**0.5)
        attn = _softmax_by_index(scores, motif_batch, mol_vec.shape[0])
        values = self.fragment_value(motif_h) * attn.unsqueeze(1)
        return _sum_by_index(values, motif_batch, mol_vec.shape[0])

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)

        atom_h = F.relu(self.atom_proj(self._atom_input(bmg, V_d)))
        bond_h = F.relu(self.bond_proj(bmg.E))
        graph_batch = bmg.batch
        n_graphs = len(bmg)

        motif_atom_index = getattr(bmg, "motif_atom_index", None)
        motif_batch = getattr(bmg, "motif_batch", None)
        if motif_atom_index is None or motif_batch is None:
            motif_atom_index = torch.empty((2, 0), dtype=torch.long, device=atom_h.device)
            motif_batch = torch.empty((0,), dtype=torch.long, device=atom_h.device)

        n_motifs = motif_batch.numel()
        if n_motifs:
            motif_h = self.fragment_init(
                _mean_by_index(atom_h[motif_atom_index[1]], motif_atom_index[0], n_motifs)
            )
        else:
            motif_h = atom_h.new_zeros((0, self.d_h))

        for atom_conv, fragment_conv in zip(self.atom_convs, self.fragment_convs):
            atom_msg = F.relu(atom_conv(atom_h, bmg.edge_index, bond_h))
            atom_beta = torch.sigmoid(self.atom_gate(torch.cat((atom_h, atom_msg, atom_h - atom_msg), dim=1)))
            atom_h = atom_beta * atom_h + (1.0 - atom_beta) * atom_msg

            if n_motifs:
                motif_edge_index, motif_edge_h = self._fragment_edges(bmg, motif_h)
                if motif_edge_index.numel():
                    motif_msg = F.relu(fragment_conv(motif_h, motif_edge_index, motif_edge_h))
                else:
                    motif_msg = torch.zeros_like(motif_h)
                motif_atom_msg = _mean_by_index(atom_h[motif_atom_index[1]], motif_atom_index[0], n_motifs)
                motif_msg = motif_msg + motif_atom_msg
                motif_beta = torch.sigmoid(
                    self.fragment_gate(torch.cat((motif_h, motif_msg, motif_h - motif_msg), dim=1))
                )
                motif_h = motif_beta * motif_h + (1.0 - motif_beta) * motif_msg

        if self.feature_attention:
            atom_h = self.atom_feature_attention(atom_h, graph_batch, n_graphs)
            if n_motifs:
                motif_h = self.fragment_feature_attention(motif_h, motif_batch, n_graphs)

        mol_vec = F.relu(_sum_by_index(atom_h, graph_batch, n_graphs))
        fra_vec = self._molecule_fragment_attention(mol_vec, motif_h, motif_batch)
        out = self.readout(torch.cat((mol_vec, fra_vec), dim=1))
        return torch.nan_to_num(out)
