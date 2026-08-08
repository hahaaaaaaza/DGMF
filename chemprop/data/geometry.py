from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GeometryGraph:
    V: np.ndarray
    distances: np.ndarray
    adjacency: np.ndarray
    atomic_numbers: np.ndarray | None = None
    coordinates: np.ndarray | None = None


@dataclass(repr=False, eq=False, slots=True)
class BatchGeometryGraph:
    graphs: InitVar[Sequence[GeometryGraph]]
    V: Tensor = field(init=False)
    distances: Tensor = field(init=False)
    adjacency: Tensor = field(init=False)
    mask: Tensor = field(init=False)
    atomic_numbers: Tensor = field(init=False)
    coordinates: Tensor = field(init=False)

    __size: int = field(init=False)

    def __post_init__(self, graphs: Sequence[GeometryGraph]):
        self.__size = len(graphs)
        max_nodes = max(graph.V.shape[0] for graph in graphs)
        node_dim = graphs[0].V.shape[1]

        Vs = np.zeros((len(graphs), max_nodes, node_dim), dtype=np.float32)
        distances = np.zeros((len(graphs), max_nodes, max_nodes), dtype=np.float32)
        adjacency = np.zeros((len(graphs), max_nodes, max_nodes), dtype=bool)
        mask = np.zeros((len(graphs), max_nodes), dtype=bool)
        atomic_numbers = np.zeros((len(graphs), max_nodes), dtype=np.int64)
        coordinates = np.zeros((len(graphs), max_nodes, 3), dtype=np.float32)

        for i, graph in enumerate(graphs):
            n_nodes = graph.V.shape[0]
            Vs[i, :n_nodes] = graph.V.astype(np.float32, copy=False)
            distances[i, :n_nodes, :n_nodes] = graph.distances.astype(np.float32, copy=False)
            adjacency[i, :n_nodes, :n_nodes] = graph.adjacency.astype(bool, copy=False)
            mask[i, :n_nodes] = True
            graph_atomic_numbers = getattr(graph, "atomic_numbers", None)
            graph_coordinates = getattr(graph, "coordinates", None)
            if graph_atomic_numbers is not None:
                atomic_numbers[i, :n_nodes] = graph_atomic_numbers.astype(np.int64, copy=False)
            if graph_coordinates is not None:
                coordinates[i, :n_nodes] = graph_coordinates.astype(np.float32, copy=False)

        Vs = np.nan_to_num(Vs, nan=0.0, posinf=0.0, neginf=0.0)
        distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

        self.V = torch.from_numpy(Vs).float()
        self.distances = torch.from_numpy(distances).float()
        self.adjacency = torch.from_numpy(adjacency)
        self.mask = torch.from_numpy(mask)
        self.atomic_numbers = torch.from_numpy(atomic_numbers).long()
        self.coordinates = torch.from_numpy(coordinates).float()

    def __len__(self) -> int:
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.distances = self.distances.to(device)
        self.adjacency = self.adjacency.to(device)
        self.mask = self.mask.to(device)
        self.atomic_numbers = self.atomic_numbers.to(device)
        self.coordinates = self.coordinates.to(device)
