from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import Mol
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
import torch
from torch import Tensor

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer, GraphFeaturizer
from chemprop.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin
from chemprop.utils.utils import is_cuikmolmaker_available

if is_cuikmolmaker_available():
    import cuik_molmaker


PHARMHGT_PHARM_DIM = 194
PHARMHGT_REACTION_DIM = 34
_PHARMHGT_FDEF = str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
_PHARMHGT_FACTORY = ChemicalFeatures.BuildFeatureFactory(_PHARMHGT_FDEF)
_PHARMHGT_FEATURE_TYPES = [
    feature_def.split(".")[1] for feature_def in _PHARMHGT_FACTORY.GetFeatureDefs().keys()
]


def _unique_cliques(cliques: list[set[int]]) -> list[list[int]]:
    """Return deterministic, non-empty motif atom sets."""

    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for clique in cliques:
        key = tuple(sorted(clique))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(list(key))

    return unique


def _merge_overlapping_cliques(cliques: list[set[int]]) -> list[set[int]]:
    merged = [set(c) for c in cliques if c]
    changed = True
    while changed:
        changed = False
        next_cliques: list[set[int]] = []
        while merged:
            base = merged.pop()
            i = 0
            while i < len(merged):
                if base & merged[i]:
                    base |= merged.pop(i)
                    changed = True
                else:
                    i += 1
            next_cliques.append(base)
        merged = next_cliques

    return merged


def _pharmhgt_reaction_feature(rule_pair: tuple[str, str]) -> list[int]:
    left, right = rule_pair
    left_idx = int(left) if left not in {"7a", "7b"} else 7
    right_idx = int(right) if right not in {"7a", "7b"} else 7
    feat = [0] * PHARMHGT_REACTION_DIM
    feat[left_idx] = 1
    feat[17 + right_idx] = 1
    return feat


def _pharmhgt_fragment_feature(mol: Chem.Mol, atom_indices: list[int]) -> list[int]:
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_indices)
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            raise ValueError("invalid fragment")
        maccs = [int(x) for x in MACCSkeys.GenMACCSKeys(frag)]
        feature_types = [feature.GetType() for feature in _PHARMHGT_FACTORY.GetFeaturesForMol(frag)]
        pharm = [int(ftype in feature_types) for ftype in _PHARMHGT_FEATURE_TYPES]
        return maccs + pharm
    except Exception:
        return [0] * PHARMHGT_PHARM_DIM


def _brics_motif_graph(
    mol: Chem.Mol, n_graph_atoms: int
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Build BRICS motif memberships and directed motif edges for a molecule."""

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        motif_atom_index = np.array([[0], [0]], dtype=int)
        motif_edge_index = np.empty((2, 0), dtype=int)
        motif_features = np.zeros((1, PHARMHGT_PHARM_DIM), dtype=np.single)
        motif_edge_features = np.empty((0, PHARMHGT_REACTION_DIM), dtype=np.single)
        return motif_atom_index, motif_edge_index, 1, motif_features, motif_edge_features

    cliques: list[set[int]] = [
        {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in mol.GetBonds()
    ]
    brics_bonds_with_rules = list(BRICS.FindBRICSBonds(mol))
    brics_bonds = {tuple(sorted(map(int, bond))) for bond, _ in brics_bonds_with_rules}

    if brics_bonds:
        kept_cliques: list[set[int]] = []
        for clique in cliques:
            if len(clique) == 2 and tuple(sorted(clique)) in brics_bonds:
                continue
            kept_cliques.append(clique)
        cliques = kept_cliques
        for a1, a2 in brics_bonds:
            cliques.append({a1})
            cliques.append({a2})

    if not cliques:
        cliques = [{atom_idx} for atom_idx in range(n_atoms)]

    cliques = _merge_overlapping_cliques(cliques)
    covered_atoms = set().union(*cliques) if cliques else set()
    for atom_idx in range(n_atoms):
        if atom_idx not in covered_atoms:
            cliques.append({atom_idx})

    motif_cliques = _unique_cliques(cliques)
    atom_to_motif: dict[int, int] = {}
    membership_rows: list[tuple[int, int]] = []
    for motif_idx, clique in enumerate(motif_cliques):
        for atom_idx in clique:
            if atom_idx >= n_graph_atoms:
                continue
            atom_to_motif[atom_idx] = motif_idx
            membership_rows.append((motif_idx, atom_idx))

    if not membership_rows:
        membership_rows = [(0, 0)]
        motif_cliques = [[0]]
        atom_to_motif = {0: 0}

    motif_edges: set[tuple[int, int]] = set()
    motif_edge_features_by_edge: dict[tuple[int, int], list[int]] = {}
    for bond, rule_pair in brics_bonds_with_rules:
        a1, a2 = map(int, bond)
        m1 = atom_to_motif.get(a1)
        m2 = atom_to_motif.get(a2)
        if m1 is not None and m2 is not None and m1 != m2:
            motif_edges.add((m1, m2))
            motif_edges.add((m2, m1))
            motif_edge_features_by_edge[(m1, m2)] = _pharmhgt_reaction_feature(rule_pair)
            motif_edge_features_by_edge[(m2, m1)] = _pharmhgt_reaction_feature((rule_pair[1], rule_pair[0]))

    motif_atom_index = np.array(membership_rows, dtype=int).T
    if motif_edges:
        sorted_edges = sorted(motif_edges)
        motif_edge_index = np.array(sorted_edges, dtype=int).T
        motif_edge_features = np.array(
            [motif_edge_features_by_edge.get(edge, [0] * PHARMHGT_REACTION_DIM) for edge in sorted_edges],
            dtype=np.single,
        )
    else:
        motif_edge_index = np.empty((2, 0), dtype=int)
        motif_edge_features = np.empty((0, PHARMHGT_REACTION_DIM), dtype=np.single)

    motif_features = np.array(
        [_pharmhgt_fragment_feature(mol, clique) for clique in motif_cliques],
        dtype=np.single,
    )

    return motif_atom_index, motif_edge_index, len(motif_cliques), motif_features, motif_edge_features


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizer`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=MultiHotAtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizer, default=MultiHotBondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim), dtype=np.single)
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.single)
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for bond in mol.GetBonds():
            x_e = self.bond_featurizer(bond)
            if bond_features_extra is not None:
                x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]), dtype=np.single)

            E[i : i + 2] = x_e

            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

            i += 2

        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)
        (
            motif_atom_index,
            motif_edge_index,
            num_motifs,
            motif_features,
            motif_edge_features,
        ) = _brics_motif_graph(mol, len(V))

        return MolGraph(
            V,
            E,
            edge_index,
            rev_edge_index,
            motif_atom_index,
            motif_edge_index,
            num_motifs,
            motif_features,
            motif_edge_features,
        )


@dataclass(repr=False, eq=False, slots=True)
class BatchCuikMolGraph:
    V: Tensor
    """the atom feature matrix"""
    E: Tensor
    """the bond feature matrix"""
    edge_index: Tensor
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self):
        self.__size = self.batch[-1].item() + 1

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


@dataclass
class CuikmolmakerMolGraphFeaturizer(Featurizer[list[str], BatchCuikMolGraph]):
    """A :class:`CuikmolmakerMolGraphFeaturizer` featurizes a list of molecules at once instead of
    one molecule at a time for efficiency.

    Parameters
    ----------
    atom_featurizer_mode: str, default="V2"
        The mode of the atom featurizer (V1, V2, ORGANIC, RIGR) to use.
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    add_h: bool, default=False
        whether to add hydrogens to the `Chem.Mol` objects created from the input SMILES strings
    """

    atom_featurizer_mode: Literal["V1", "V2", "ORGANIC", "RIGR"] = "V2"
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0
    add_h: bool = False

    atom_fdim: int = field(init=False)
    bond_fdim: int = field(init=False)

    def __post_init__(self):
        if not is_cuikmolmaker_available():
            raise ImportError(
                "CuikmolmakerMolGraphFeaturizer requires cuik-molmaker package to be installed. "
                f"Please install it using `python {Path(__file__).parents[2] / Path('scripts/check_and_install_cuik_molmaker.py')}`."
            )
        atom_props_float = ["aromatic", "mass"]
        bond_props = ["is-null", "bond-type-onehot", "conjugated", "in-ring", "stereo"]
        self.bond_fdim = 14

        self.atom_featurizer_mode = self.atom_featurizer_mode.upper()
        if self.atom_featurizer_mode == "V1":
            atom_props_onehot = [
                "atomic-number",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization",
            ]
            self.atom_fdim = 133
        elif self.atom_featurizer_mode == "V2":
            atom_props_onehot = [
                "atomic-number-common",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-expanded",
            ]
            self.atom_fdim = 72
        elif self.atom_featurizer_mode == "ORGANIC":
            atom_props_onehot = [
                "atomic-number-organic",
                "total-degree",
                "formal-charge",
                "chirality",
                "num-hydrogens",
                "hybridization-organic",
            ]
            self.atom_fdim = 44
        elif self.atom_featurizer_mode == "RIGR":
            atom_props_onehot = ["atomic-number-common", "total-degree", "num-hydrogens"]
            atom_props_float = ["mass"]
            bond_props = ["is-null", "in-ring"]
            self.atom_fdim = 52
            self.bond_fdim = 2
        else:
            raise ValueError(f"Invalid atom featurizer mode: {self.atom_featurizer_mode}")

        self.atom_property_list_onehot = cuik_molmaker.atom_onehot_feature_names_to_tensor(
            atom_props_onehot
        )

        self.atom_property_list_float = cuik_molmaker.atom_float_feature_names_to_tensor(
            atom_props_float
        )

        self.bond_property_list = cuik_molmaker.bond_feature_names_to_tensor(bond_props)

        self.atom_fdim += self.extra_atom_fdim
        self.bond_fdim += self.extra_bond_fdim

    def __call__(
        self,
        smiles_list: list[str],
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> BatchCuikMolGraph:
        offset_carbon, duplicate_edges, add_self_loop = False, True, False

        (
            atom_feats,
            bond_feats,
            edge_index,
            rev_edge_index,
            batch,
        ) = cuik_molmaker.batch_mol_featurizer(
            smiles_list,
            self.atom_property_list_onehot,
            self.atom_property_list_float,
            self.bond_property_list,
            self.add_h,
            offset_carbon,
            duplicate_edges,
            add_self_loop,
        )

        if atom_features_extra is not None:
            atom_features_extra = torch.tensor(atom_features_extra, dtype=torch.float32)
            atom_feats = torch.cat((atom_feats, atom_features_extra), dim=1)
        if bond_features_extra is not None:
            bond_features_extra = np.repeat(bond_features_extra, repeats=2, axis=0)
            bond_features_extra = torch.tensor(bond_features_extra, dtype=torch.float32)
            bond_feats = torch.cat((bond_feats, bond_features_extra), dim=1)

        return BatchCuikMolGraph(
            V=atom_feats,
            E=bond_feats,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            batch=batch,
        )
