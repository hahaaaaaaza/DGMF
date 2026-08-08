from __future__ import annotations

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


def _mean_by_index(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if dim_size == 0:
        return values.new_zeros((0, values.shape[-1]))

    out = values.new_zeros((dim_size, values.shape[-1]))
    if values.numel() == 0 or index.numel() == 0:
        return out

    out.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=dim_size).to(values.dtype).unsqueeze(1)
    return out / counts.clamp_min(1)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_h: int,
        dropout: float,
        activation: str | nn.Module | Activation,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            get_activation_function(activation),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_h),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        return self.norm(h + self.net(x))


class HimNetMessagePassing(MessagePassing, HyperparametersMixin):
    """BRICS motif-aware hierarchical graph encoder for the 2D branch.

    This module keeps only the hierarchical atom/motif/molecule graph encoder idea from HimNet and
    deliberately omits HimNet's molecular fingerprint fusion, because this project already has a
    separate 1D MoLFormer/fingerprint branch.
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
        dropout: float = 0.0,
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
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

        atom_input_dim = d_v + (d_vd or 0)
        self.atom_proj = nn.Linear(atom_input_dim, d_h, bias=bias)
        self.bond_proj = nn.Linear(d_e, d_h, bias=bias)
        self.motif_proj = nn.Linear(d_h, d_h, bias=bias)
        self.global_proj = nn.Linear(d_h, d_h, bias=bias)

        self.atom_update = _ResidualBlock(4 * d_h, d_h, dropout, activation)
        self.motif_update = _ResidualBlock(4 * d_h, d_h, dropout, activation)
        self.global_update = _ResidualBlock(3 * d_h, d_h, dropout, activation)
        self.readout = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            get_activation_function(activation),
            nn.Dropout(dropout),
            nn.LayerNorm(d_h),
        )

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

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)

        atom_h = self.atom_proj(self._atom_input(bmg, V_d))
        graph_batch = bmg.batch
        n_graphs = len(bmg)

        motif_atom_index = getattr(bmg, "motif_atom_index", None)
        motif_batch = getattr(bmg, "motif_batch", None)
        motif_edge_index = getattr(bmg, "motif_edge_index", None)
        if motif_atom_index is None or motif_batch is None:
            motif_atom_index = torch.empty((2, 0), dtype=torch.long, device=atom_h.device)
            motif_batch = torch.empty((0,), dtype=torch.long, device=atom_h.device)
        if motif_edge_index is None:
            motif_edge_index = torch.empty((2, 0), dtype=torch.long, device=atom_h.device)

        n_motifs = motif_batch.numel()
        if n_motifs:
            motif_h = self.motif_proj(
                _mean_by_index(atom_h[motif_atom_index[1]], motif_atom_index[0], n_motifs)
            )
        else:
            motif_h = atom_h.new_zeros((0, self.d_h))

        global_h = self.global_proj(_mean_by_index(atom_h, graph_batch, n_graphs))

        edge_src, edge_dst = bmg.edge_index
        for _ in range(self.depth):
            edge_msg = atom_h[edge_src] + self.bond_proj(bmg.E)
            atom_from_atoms = _mean_by_index(edge_msg, edge_dst, atom_h.shape[0])

            if n_motifs:
                motif_from_atoms = _mean_by_index(
                    atom_h[motif_atom_index[1]], motif_atom_index[0], n_motifs
                )
                atom_from_motifs = _mean_by_index(
                    motif_h[motif_atom_index[0]], motif_atom_index[1], atom_h.shape[0]
                )
                if motif_edge_index.numel():
                    motif_from_motifs = _mean_by_index(
                        motif_h[motif_edge_index[0]], motif_edge_index[1], n_motifs
                    )
                else:
                    motif_from_motifs = torch.zeros_like(motif_h)
                motif_from_global = global_h[motif_batch]
                motif_update = torch.cat(
                    (motif_h, motif_from_atoms, motif_from_motifs, motif_from_global), dim=1
                )
                motif_h = self.motif_update(motif_h, motif_update)
            else:
                atom_from_motifs = torch.zeros_like(atom_h)

            atom_from_global = global_h[graph_batch]
            atom_update = torch.cat(
                (atom_h, atom_from_atoms, atom_from_motifs, atom_from_global), dim=1
            )
            atom_h = self.atom_update(atom_h, atom_update)

            graph_from_atoms = _mean_by_index(atom_h, graph_batch, n_graphs)
            if n_motifs:
                graph_from_motifs = _mean_by_index(motif_h, motif_batch, n_graphs)
            else:
                graph_from_motifs = torch.zeros_like(graph_from_atoms)
            global_update = torch.cat((global_h, graph_from_atoms, graph_from_motifs), dim=1)
            global_h = self.global_update(global_h, global_update)

        graph_atom_h = _mean_by_index(atom_h, graph_batch, n_graphs)
        if n_motifs:
            graph_motif_h = _mean_by_index(motif_h, motif_batch, n_graphs)
        else:
            graph_motif_h = torch.zeros_like(graph_atom_h)
        graph_h = self.readout(torch.cat((graph_atom_h, graph_motif_h, global_h), dim=1))
        return torch.nan_to_num(graph_h)
