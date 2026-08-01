"""Build shortest paths through selected E-path atom graphs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import heapq
import math
import operator

import numpy as np
from scipy.spatial import cKDTree

from epath.clustering import euclidean_distance
from epath.selection import EpathScoredAtom


DEFAULT_PATH_EDGE_DISTANCE_ANGSTROM = 4.0


class EpathPathError(ValueError):
    """Raised when an E-path atom path cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathPathEdge:
    """One weighted edge in an E-path atom graph."""

    target_index: int
    distance: float


EpathPathGraph = tuple[tuple[EpathPathEdge, ...], ...]


@dataclass(frozen=True, slots=True)
class EpathPath:
    """Shortest path through a selected E-path atom graph."""

    atom_indices: tuple[int, ...]
    atoms: tuple[EpathScoredAtom, ...]
    edge_lengths: tuple[float, ...]
    total_length: float

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the path."""

        return len(self.atoms)

    @property
    def step_count(self) -> int:
        """Return the number of atom-to-atom steps in the path."""

        return len(self.edge_lengths)

    @property
    def largest_step(self) -> float:
        """Return the longest atom-to-atom step in the path."""

        return max(self.edge_lengths, default=0.0)


def find_shortest_path(
    atoms: Iterable[EpathScoredAtom],
    *,
    source_index: int,
    target_index: int,
    edge_distance: float = DEFAULT_PATH_EDGE_DISTANCE_ANGSTROM,
) -> EpathPath | None:
    """Return the shortest path between two selected atom indexes, if connected."""

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return None

    source_index = _atom_index(
        source_index,
        atom_count=len(atom_tuple),
        name="source_index",
    )
    target_index = _atom_index(
        target_index,
        atom_count=len(atom_tuple),
        name="target_index",
    )

    graph = build_path_graph(atom_tuple, edge_distance=edge_distance)
    total_length, path_indices = dijkstra_shortest_path(
        graph,
        source_index=source_index,
        target_index=target_index,
    )
    if not path_indices:
        return None

    return path_from_indices(
        atom_tuple,
        path_indices,
        total_length=total_length,
    )


def build_path_graph(
    atoms: Iterable[EpathScoredAtom],
    *,
    edge_distance: float = DEFAULT_PATH_EDGE_DISTANCE_ANGSTROM,
) -> EpathPathGraph:
    """Build a weighted adjacency graph between atoms within edge_distance."""

    edge_distance = _positive_finite_float(edge_distance, name="edge_distance")
    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return ()

    coordinates = _coordinates(atom_tuple)
    tree = cKDTree(coordinates)
    raw_neighbours = tree.query_ball_point(
        coordinates,
        r=edge_distance,
        p=2.0,
        workers=-1,
    )

    graph: list[tuple[EpathPathEdge, ...]] = []
    for atom_index, neighbours in enumerate(raw_neighbours):
        edges: list[EpathPathEdge] = []
        for neighbour_index in sorted(int(index) for index in neighbours):
            if neighbour_index == atom_index:
                continue

            distance = euclidean_distance(
                coordinates[atom_index],
                coordinates[neighbour_index],
            )
            if distance <= edge_distance:
                edges.append(
                    EpathPathEdge(
                        target_index=neighbour_index,
                        distance=distance,
                    )
                )

        graph.append(tuple(edges))

    return tuple(graph)


def dijkstra_shortest_path(
    graph: Sequence[Sequence[EpathPathEdge]],
    *,
    source_index: int,
    target_index: int,
) -> tuple[float, tuple[int, ...]]:
    """Return the shortest graph distance and node indexes between two atoms."""

    node_count = len(graph)
    if node_count == 0:
        raise EpathPathError("path graph is empty.")

    source_index = _atom_index(
        source_index,
        atom_count=node_count,
        name="source_index",
    )
    target_index = _atom_index(
        target_index,
        atom_count=node_count,
        name="target_index",
    )

    distances = [math.inf] * node_count
    previous_indices = [-1] * node_count
    distances[source_index] = 0.0

    queue: list[tuple[float, int]] = [(0.0, source_index)]
    visited_indices: set[int] = set()

    while queue:
        current_distance, current_index = heapq.heappop(queue)
        if current_index in visited_indices:
            continue

        visited_indices.add(current_index)
        if current_index == target_index:
            break

        for edge in graph[current_index]:
            _validate_graph_edge(edge, node_count=node_count)
            next_distance = current_distance + edge.distance
            if next_distance < distances[edge.target_index]:
                distances[edge.target_index] = next_distance
                previous_indices[edge.target_index] = current_index
                heapq.heappush(queue, (next_distance, edge.target_index))

    total_length = float(distances[target_index])
    if math.isinf(total_length):
        return math.inf, ()

    return total_length, _reconstruct_path_indices(
        previous_indices,
        source_index=source_index,
        target_index=target_index,
    )


def path_from_indices(
    atoms: Sequence[EpathScoredAtom],
    atom_indices: Sequence[int],
    *,
    total_length: float | None = None,
) -> EpathPath:
    """Build an EpathPath result from atom indexes."""

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise EpathPathError("cannot build a path from an empty atom list.")

    path_indices = tuple(
        _atom_index(index, atom_count=len(atom_tuple), name="atom_index")
        for index in atom_indices
    )
    if not path_indices:
        raise EpathPathError("path must contain at least one atom index.")

    path_atoms = tuple(atom_tuple[index] for index in path_indices)
    edge_lengths = tuple(
        euclidean_distance(
            _coordinate_for_atom(first_atom),
            _coordinate_for_atom(second_atom),
        )
        for first_atom, second_atom in zip(path_atoms, path_atoms[1:], strict=False)
    )

    if total_length is None:
        total_length = float(sum(edge_lengths))
    else:
        total_length = _non_negative_finite_float(
            total_length,
            name="total_length",
        )

    return EpathPath(
        atom_indices=path_indices,
        atoms=path_atoms,
        edge_lengths=edge_lengths,
        total_length=total_length,
    )


def _coordinates(atoms: Sequence[EpathScoredAtom]) -> np.ndarray:
    coordinates = np.asarray(
        [_coordinate_for_atom(scored_atom) for scored_atom in atoms],
        dtype=float,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise EpathPathError("atom coordinates must have shape (N, 3).")

    if not np.all(np.isfinite(coordinates)):
        raise EpathPathError("atom coordinates must be finite.")

    return coordinates


def _coordinate_for_atom(scored_atom: EpathScoredAtom) -> tuple[float, float, float]:
    atom = scored_atom.atom
    return (float(atom.x), float(atom.y), float(atom.z))


def _reconstruct_path_indices(
    previous_indices: Sequence[int],
    *,
    source_index: int,
    target_index: int,
) -> tuple[int, ...]:
    path: list[int] = []
    current_index = target_index
    while current_index != -1:
        path.append(current_index)
        if current_index == source_index:
            break

        current_index = previous_indices[current_index]

    path.reverse()
    if not path or path[0] != source_index:
        return ()

    return tuple(path)


def _validate_graph_edge(edge: EpathPathEdge, *, node_count: int) -> None:
    _atom_index(edge.target_index, atom_count=node_count, name="edge.target_index")
    _non_negative_finite_float(edge.distance, name="edge.distance")


def _atom_index(value: int, *, atom_count: int, name: str) -> int:
    if isinstance(value, bool):
        raise EpathPathError(f"{name} must be an integer index, got {value!r}.")

    try:
        index = operator.index(value)
    except TypeError as error:
        raise EpathPathError(
            f"{name} must be an integer index, got {value!r}."
        ) from error

    if index < 0 or index >= atom_count:
        raise EpathPathError(
            f"{name} is out of range for {atom_count} atoms: {index!r}."
        )

    return index


def _positive_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise EpathPathError(
            f"{name} must be a positive number, got {number!r}."
        )

    return number


def _non_negative_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise EpathPathError(
            f"{name} must be zero or positive, got {number!r}."
        )

    return number


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathPathError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathPathError(
            f"{name} must be numeric, got {value!r}."
        ) from error

    if not math.isfinite(number):
        raise EpathPathError(f"{name} must be finite, got {number!r}.")

    return number
