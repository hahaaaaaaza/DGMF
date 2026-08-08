from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol

ECFP_DIM = 2048
RDKIT_FINGERPRINT_DIM = 2048


def _morgan_ecfp_bits(
    mol: Mol,
    radius: int = 2,
    n_bits: int = ECFP_DIM,
    use_chirality: bool = True,
) -> np.ndarray:
    if mol is None:
        raise ValueError("`mol` is None in _morgan_ecfp_bits.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _maccs_bits(mol: Mol) -> np.ndarray:
    if mol is None:
        raise ValueError("`mol` is None in _maccs_bits.")

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr[1:]


def _rdkit_bits(mol: Mol, n_bits: int = RDKIT_FINGERPRINT_DIM) -> np.ndarray:
    if mol is None:
        raise ValueError("`mol` is None in _rdkit_bits.")

    fp = RDKFingerprint(mol, fpSize=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def concatenated_native_1d_fingerprint(
    mol_or_smiles: Mol | str,
    radius: int = 2,
    n_bits: int = ECFP_DIM,
    use_chirality: bool = True,
) -> np.ndarray:
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles

    if mol is None:
        raise ValueError(
            f"Invalid molecule / SMILES in concatenated_native_1d_fingerprint: {mol_or_smiles!r}"
        )

    maccs = _maccs_bits(mol)
    rdkit = _rdkit_bits(mol)
    ecfp = _morgan_ecfp_bits(mol, radius=radius, n_bits=n_bits, use_chirality=use_chirality)
    return np.concatenate([maccs, rdkit, ecfp], axis=0)


def concatenated_1d_fingerprint(
    mol_or_smiles: Mol | str,
    radius: int = 2,
    n_bits: int = ECFP_DIM,
    use_chirality: bool = True,
) -> np.ndarray:
    return concatenated_native_1d_fingerprint(
        mol_or_smiles,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )

def ecfp_maccs_fingerprint(
    mol_or_smiles: Mol | str,
    radius: int = 2,
    n_bits: int = ECFP_DIM,
    use_chirality: bool = True,
) -> np.ndarray:
    return concatenated_1d_fingerprint(
        mol_or_smiles,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )
