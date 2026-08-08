from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from chemprop.data.geometry import GeometryGraph

GEOMETRY_NODE_FDIM = 8


def _zero_geometry_graph(node_fdim: int = GEOMETRY_NODE_FDIM) -> GeometryGraph:
    return GeometryGraph(
        V=np.zeros((1, node_fdim), dtype=np.float32),
        distances=np.zeros((1, 1), dtype=np.float32),
        adjacency=np.ones((1, 1), dtype=bool),
        atomic_numbers=np.zeros(1, dtype=np.int64),
        coordinates=np.zeros((1, 3), dtype=np.float32),
    )


def _atom_features(atom: Chem.Atom) -> list[float]:
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 5.0,
        atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
        float(atom.GetIsAromatic()),
        atom.GetMass() / 200.0,
        float(atom.IsInRing()),
        float(atom.HasProp("_ChiralityPossible")),
    ]


def mmff_geometry_graph(
    mol: Mol,
    distance_cutoff: float = 4.5,
    max_iters: int = 200,
    random_seed: int = 0xC0FFEE,
    num_conformers: int = 1,
    conformer_selection: str = "lowest",
    selection_seed: int | None = None,
) -> GeometryGraph:
    if mol is None:
        return _zero_geometry_graph()
    if conformer_selection not in {"lowest", "random", "highest"}:
        raise ValueError(
            "conformer_selection must be one of 'lowest', 'random', or 'highest', "
            f"got {conformer_selection!r}"
        )

    mol3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True

    try:
        num_conformers = max(1, int(num_conformers))
        if num_conformers == 1:
            conf_ids = [AllChem.EmbedMolecule(mol3d, params)]
            if conf_ids[0] < 0:
                return _zero_geometry_graph()
        else:
            conf_ids = list(AllChem.EmbedMultipleConfs(mol3d, numConfs=num_conformers, params=params))
            if not conf_ids:
                return _zero_geometry_graph()

        props = AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant="MMFF94")
        conformer_energies: list[tuple[int, float]] = []
        for conf_id in conf_ids:
            if props is not None:
                AllChem.MMFFOptimizeMolecule(
                    mol3d, mmffVariant="MMFF94", maxIters=max_iters, confId=int(conf_id)
                )
                force_field = AllChem.MMFFGetMoleculeForceField(
                    mol3d, props, confId=int(conf_id)
                )
            else:
                AllChem.UFFOptimizeMolecule(mol3d, maxIters=max_iters, confId=int(conf_id))
                force_field = AllChem.UFFGetMoleculeForceField(mol3d, confId=int(conf_id))
            energy = force_field.CalcEnergy() if force_field is not None else float("inf")
            if np.isfinite(energy):
                conformer_energies.append((int(conf_id), float(energy)))

        if not conformer_energies:
            return _zero_geometry_graph()
        if conformer_selection == "lowest":
            selected_conf_id = min(conformer_energies, key=lambda item: item[1])[0]
        elif conformer_selection == "highest":
            selected_conf_id = max(conformer_energies, key=lambda item: item[1])[0]
        else:
            rng = np.random.default_rng(
                random_seed if selection_seed is None else selection_seed
            )
            selected_conf_id = conformer_energies[int(rng.integers(len(conformer_energies)))][0]
    except (ImportError, FileNotFoundError):
        raise
    except Exception:
        return _zero_geometry_graph()

    conf = mol3d.GetConformer(selected_conf_id)
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    coords = coords - coords.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float32)
    adjacency = distances <= distance_cutoff

    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency[i, j] = True
        adjacency[j, i] = True
    np.fill_diagonal(adjacency, True)

    atom_features = np.asarray([_atom_features(atom) for atom in mol3d.GetAtoms()], dtype=np.float32)
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol3d.GetAtoms()], dtype=np.int64)
    return GeometryGraph(
        V=atom_features,
        distances=distances,
        adjacency=adjacency,
        atomic_numbers=atomic_numbers,
        coordinates=coords,
    )


def _rdkit_mol_to_ase_atoms(mol: Mol):
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "MACE 3D graphs require ASE. Install MACE with its ASE dependency, for example "
            "`pip install mace-torch ase`, before using `--use-mace-3d-graph`."
        ) from exc

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=np.asarray(conf.GetPositions(), dtype=np.float64))


@lru_cache(maxsize=8)
def _get_mace_calculator(
    model: str,
    model_path: str | None,
    device: str,
    default_dtype: str,
):
    try:
        from mace.calculators import MACECalculator, mace_off
    except ImportError as exc:
        raise ImportError(
            "MACE 3D graphs require the optional `mace-torch` package. Install it, then rerun "
            "with `--use-mace-3d-graph`."
        ) from exc

    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MACE model path does not exist: {path}")
        try:
            return MACECalculator(model_paths=str(path), device=device, default_dtype=default_dtype)
        except TypeError:
            return MACECalculator(model_path=str(path), device=device, default_dtype=default_dtype)

    return mace_off(model=model, device=device, default_dtype=default_dtype)


@lru_cache(maxsize=8)
def _get_mace_descriptor_dim(
    model: str,
    model_path: str | None,
    device: str,
    default_dtype: str,
    invariants_only: bool,
    num_layers: int | None,
) -> int:
    descriptor_num_layers = -1 if num_layers is None else num_layers
    methane = Chem.AddHs(Chem.MolFromSmiles("C"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(methane, params) < 0:
        return GEOMETRY_NODE_FDIM

    calc = _get_mace_calculator(model, model_path, device, default_dtype)
    atoms = _rdkit_mol_to_ase_atoms(methane)
    descriptors = calc.get_descriptors(
        atoms, invariants_only=invariants_only, num_layers=descriptor_num_layers
    )
    descriptors = np.asarray(descriptors)
    return int(np.prod(descriptors.shape[1:]))


def mace_geometry_graph(
    mol: Mol,
    distance_cutoff: float = 4.5,
    max_iters: int = 200,
    random_seed: int = 0xC0FFEE,
    model: str = "medium",
    model_path: str | None = None,
    device: str = "cpu",
    default_dtype: str = "float32",
    optimize: bool = True,
    fmax: float = 0.03,
    invariants_only: bool = True,
    num_layers: int | None = None,
) -> GeometryGraph:
    descriptor_num_layers = -1 if num_layers is None else num_layers
    node_fdim = _get_mace_descriptor_dim(
        model, model_path, device, default_dtype, invariants_only, num_layers
    )
    if mol is None:
        return _zero_geometry_graph(node_fdim)

    mol3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True

    try:
        if AllChem.EmbedMolecule(mol3d, params) < 0:
            return _zero_geometry_graph(node_fdim)
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=max(10, min(max_iters, 50)))

        atoms = _rdkit_mol_to_ase_atoms(mol3d)
        calc = _get_mace_calculator(model, model_path, device, default_dtype)
        atoms.calc = calc

        if optimize:
            from ase.optimize import FIRE

            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=fmax, steps=max_iters)

        coords = np.asarray(atoms.get_positions(), dtype=np.float32)
        if not np.isfinite(coords).all():
            return _zero_geometry_graph(node_fdim)
        coords = coords - coords.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(
            np.float32
        )
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = distances <= distance_cutoff

        for bond in mol3d.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjacency[i, j] = True
            adjacency[j, i] = True
        np.fill_diagonal(adjacency, True)

        descriptors = calc.get_descriptors(
            atoms, invariants_only=invariants_only, num_layers=descriptor_num_layers
        )
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim > 2:
            descriptors = descriptors.reshape(descriptors.shape[0], -1)
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol3d.GetAtoms()], dtype=np.int64)

        return GeometryGraph(
            V=descriptors,
            distances=distances,
            adjacency=adjacency,
            atomic_numbers=atomic_numbers,
            coordinates=coords,
        )
    except (ImportError, FileNotFoundError):
        raise
    except Exception:
        return _zero_geometry_graph(node_fdim)


def geometry_graph_node_fdim(
    graphs: list[GeometryGraph | None], default: int = GEOMETRY_NODE_FDIM
) -> int:
    for graph in graphs:
        if graph is not None and graph.V.ndim == 2 and graph.V.shape[1] > 0:
            return int(graph.V.shape[1])
    return default


def combine_geometry_graphs(graphs: list[GeometryGraph]) -> GeometryGraph:
    if not graphs:
        return _zero_geometry_graph()
    if len(graphs) == 1:
        return graphs[0]

    n_nodes = sum(graph.V.shape[0] for graph in graphs)
    node_dim = graphs[0].V.shape[1]
    V = np.zeros((n_nodes, node_dim), dtype=np.float32)
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    atomic_numbers = np.zeros(n_nodes, dtype=np.int64)
    coordinates = np.zeros((n_nodes, 3), dtype=np.float32)

    offset = 0
    for graph in graphs:
        n = graph.V.shape[0]
        V[offset : offset + n] = graph.V
        distances[offset : offset + n, offset : offset + n] = graph.distances
        adjacency[offset : offset + n, offset : offset + n] = graph.adjacency
        if graph.atomic_numbers is not None:
            atomic_numbers[offset : offset + n] = graph.atomic_numbers
        if graph.coordinates is not None:
            coordinates[offset : offset + n] = graph.coordinates
        offset += n

    return GeometryGraph(
        V=V,
        distances=distances,
        adjacency=adjacency,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
    )
