from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import CuikBatchedDatum, Datum, MolAtomBondDatum
from chemprop.data.geometry import BatchGeometryGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import BatchCuikMolGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""
    motif_atom_index: Tensor = field(init=False)
    """A tensor of shape ``2 x M`` mapping batched motif indices to batched atom indices."""
    motif_edge_index: Tensor = field(init=False)
    """A tensor of shape ``2 x E_m`` containing directed motif-motif edges."""
    motif_batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` for each motif node"""
    motif_features: Tensor = field(init=False)
    """Optional PharmHGT-style BRICS fragment/pharmacophore feature matrix."""
    motif_edge_features: Tensor = field(init=False)
    """Optional BRICS reaction-rule edge features for motif-motif edges."""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        motif_atom_indexes = []
        motif_edge_indexes = []
        motif_batch_indexes = []
        motif_features = []
        motif_edge_features = []

        num_nodes = 0
        num_edges = 0
        num_motifs = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))

            mg_num_motifs = int(getattr(mg, "num_motifs", 0) or 0)
            motif_atom_index = getattr(mg, "motif_atom_index", None)
            if mg_num_motifs > 0 and motif_atom_index is not None and motif_atom_index.size:
                motif_atom_index = np.asarray(motif_atom_index, dtype=int).copy()
                motif_atom_index[0] += num_motifs
                motif_atom_index[1] += num_nodes
                motif_atom_indexes.append(motif_atom_index)

                motif_edge_index = getattr(mg, "motif_edge_index", None)
                if motif_edge_index is not None and motif_edge_index.size:
                    motif_edge_index = np.asarray(motif_edge_index, dtype=int).copy()
                    motif_edge_index += num_motifs
                    motif_edge_indexes.append(motif_edge_index)
                    mg_motif_edge_features = getattr(mg, "motif_edge_features", None)
                    if mg_motif_edge_features is not None and len(mg_motif_edge_features):
                        motif_edge_features.append(np.asarray(mg_motif_edge_features, dtype=np.single))

                motif_batch_indexes.append([i] * mg_num_motifs)
                mg_motif_features = getattr(mg, "motif_features", None)
                if mg_motif_features is not None and len(mg_motif_features):
                    motif_features.append(np.asarray(mg_motif_features, dtype=np.single))
                num_motifs += mg_num_motifs

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        self.motif_atom_index = (
            torch.from_numpy(np.hstack(motif_atom_indexes)).long()
            if motif_atom_indexes
            else torch.empty((2, 0), dtype=torch.long)
        )
        self.motif_edge_index = (
            torch.from_numpy(np.hstack(motif_edge_indexes)).long()
            if motif_edge_indexes
            else torch.empty((2, 0), dtype=torch.long)
        )
        self.motif_batch = (
            torch.tensor(np.concatenate(motif_batch_indexes)).long()
            if motif_batch_indexes
            else torch.empty((0,), dtype=torch.long)
        )
        self.motif_features = (
            torch.from_numpy(np.vstack(motif_features)).float()
            if motif_features
            else torch.empty((0, 0), dtype=torch.float)
        )
        self.motif_edge_features = (
            torch.from_numpy(np.vstack(motif_edge_features)).float()
            if motif_edge_features
            else torch.empty((0, 0), dtype=torch.float)
        )

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        self.motif_atom_index = self.motif_atom_index.to(device)
        self.motif_edge_index = self.motif_edge_index.to(device)
        self.motif_batch = self.motif_batch.to(device)
        self.motif_features = self.motif_features.to(device)
        self.motif_edge_features = self.motif_edge_features.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph | BatchCuikMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    X_3d: BatchGeometryGraph | None
    Y: Tensor | None
    w: Tensor | None
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks, x_3ds = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if x_3ds[0] is None else BatchGeometryGraph(x_3ds),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def collate_cuik_batch(batch: CuikBatchedDatum) -> TrainingBatch:
    bmg, V_d, X_d, X_3d, Y, weights, lt_mask, gt_mask = batch
    return TrainingBatch(
        bmg,
        None if V_d is None else torch.from_numpy(V_d).float(),
        None if X_d is None else torch.from_numpy(X_d).float(),
        None if X_3d is None else BatchGeometryGraph(X_3d),
        None if Y is None else torch.from_numpy(Y).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_mask is None else torch.from_numpy(lt_mask),
        None if gt_mask is None else torch.from_numpy(gt_mask),
    )


@dataclass(repr=False, eq=False, slots=True)
class BatchMolAtomBondGraph(BatchMolGraph):
    bond_batch: Tensor = field(init=False)
    """A tensor of indices that show which :class:`MolGraph` each bond belongs to in the batch"""

    def __post_init__(self, mgs: Sequence[MolGraph]):
        # inheriting a dataclass with slots=True requires explicit arguments to super
        super(BatchMolAtomBondGraph, self).__post_init__(mgs)

        bond_batch_indexes = []
        for i, mg in enumerate(mgs):
            bond_batch_indexes.append([i] * len(mg.E))

        self.bond_batch = torch.tensor(np.concatenate(bond_batch_indexes)).long()

    def to(self, device):
        super(BatchMolAtomBondGraph, self).to(device)
        self.bond_batch = self.bond_batch.to(device)


class MolAtomBondTrainingBatch(NamedTuple):
    bmg: BatchMolAtomBondGraph
    V_d: Tensor | None
    E_d: Tensor | None
    X_d: Tensor | None
    Ys: tuple[Tensor | None, Tensor | None, Tensor | None]
    w: tuple[Tensor | None, Tensor | None, Tensor | None]
    lt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    gt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    constraints: tuple[Tensor | None, Tensor | None]


def collate_mol_atom_bond_batch(batch: Iterable[MolAtomBondDatum]) -> MolAtomBondTrainingBatch:
    mgs, V_ds, E_ds, x_ds, yss, weights, lt_maskss, gt_maskss, constraintss = zip(*batch)

    weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    weights_tensors = []
    for ys in zip(*yss):
        if ys[0] is None:
            weights_tensors.append(None)
        elif ys[0].ndim == 1:
            weights_tensors.append(weights)
        else:
            repeats = torch.tensor([y.shape[0] for y in ys])
            weights_tensors.append(torch.repeat_interleave(weights, repeats, dim=0))

    if constraintss[0][0] is None and constraintss[0][1] is None:
        constraintss = None
    else:
        constraintss = [
            None if constraints[0] is None else torch.from_numpy(np.array(constraints)).float()
            for constraints in zip(*constraintss)
        ]

    return MolAtomBondTrainingBatch(
        BatchMolAtomBondGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        [None if ys[0] is None else torch.from_numpy(np.vstack(ys)).float() for ys in zip(*yss)],
        weights_tensors,
        [
            None if lt_masks[0] is None else torch.from_numpy(np.vstack(lt_masks))
            for lt_masks in zip(*lt_maskss)
        ],
        [
            None if gt_masks[0] is None else torch.from_numpy(np.vstack(gt_masks))
            for gt_masks in zip(*gt_maskss)
        ],
        constraintss,
    )


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    X_3d: BatchGeometryGraph | None
    Y: Tensor | None
    w: Tensor | None
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].X_3d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
