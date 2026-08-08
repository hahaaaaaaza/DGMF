from typing import NamedTuple

import numpy as np


class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute."""
    motif_atom_index: np.ndarray | None = None
    """An optional array of shape ``2 x M`` mapping motif indices to atom indices."""
    motif_edge_index: np.ndarray | None = None
    """An optional array of shape ``2 x E_m`` containing directed motif-motif edges."""
    num_motifs: int = 0
    """The number of BRICS motifs in the molecule."""
    motif_features: np.ndarray | None = None
    """Optional PharmHGT-style BRICS fragment/pharmacophore features."""
    motif_edge_features: np.ndarray | None = None
    """Optional BRICS reaction-rule features for directed motif-motif edges."""
