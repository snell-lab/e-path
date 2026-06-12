"""Cluster selected E-path atoms by spatial connectivity."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from epath.selection import EpathScoredAtom


DEFAULT_CLUSTER_DISTANCE_ANGSTROM = 4.0


class EpathClusteringError(ValueError):
    """Raised when E-path atom clustering cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathClusterMetrics:
    """Geometric metrics for one E-path atom cluster."""

    mst_total_length: float
    mst_largest_edge: float
    span: float
    mean_step: float


@dataclass(frozen=True, slots=True)
class EpathCluster:
    """One connected component of selected E-path atoms."""

    cluster_id: int
    atoms: tuple[EpathScoredAtom, ...]
    metrics: EpathClusterMetrics

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in this cluster."""

        return len(self.atoms)


def cluster_selected_atoms(
    atoms: Iterable[EpathScoredAtom],
    *,
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
) -> tuple[EpathCluster, ...]:
    """Find connected components among selected atoms using a distance cutoff."""

    distance_cutoff = _positive_finite_float(
        distance_cutoff,
        name="distance_cutoff",
    )
    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return ()

    coordinates = _coordinates(atom_tuple)
    neighbours_by_index = _neighbour_indices_by_atom(
        coordinates,
        distance_cutoff=distance_cutoff,
    )
    components = _connected_components(neighbours_by_index)
    components.sort(key=len, reverse=True)

    return tuple(
        EpathCluster(
            cluster_id=cluster_index,
            atoms=tuple(atom_tuple[index] for index in component),
            metrics=cluster_metrics_for_atoms(
                tuple(atom_tuple[index] for index in component)
            ),
        )
        for cluster_index, component in enumerate(components, start=1)
    )


def keep_largest_cluster(
    clusters: Iterable[EpathCluster],
) -> tuple[EpathCluster, ...]:
    """Return only the largest cluster, preserving an empty input."""

    cluster_tuple = tuple(clusters)
    if not cluster_tuple:
        return ()

    largest_cluster = max(cluster_tuple, key=lambda cluster: cluster.atom_count)
    return (
        EpathCluster(
            cluster_id=1,
            atoms=largest_cluster.atoms,
            metrics=largest_cluster.metrics,
        ),
    )


def cluster_metrics_for_atoms(
    atoms: Sequence[EpathScoredAtom],
) -> EpathClusterMetrics:
    """Calculate MST and span metrics for a group of atoms."""

    if not atoms:
        return EpathClusterMetrics(
            mst_total_length=0.0,
            mst_largest_edge=0.0,
            span=0.0,
            mean_step=0.0,
        )

    return mst_length_stats(_coordinates(atoms))


def mst_length_stats(coordinates: np.ndarray) -> EpathClusterMetrics:
    """Return Prim-style MST metrics for a coordinate array."""

    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
        raise EpathClusteringError("coordinates must have shape (N, 3).")

    if not np.all(np.isfinite(coordinate_array)):
        raise EpathClusteringError("coordinates must contain only finite numbers.")

    atom_count = int(coordinate_array.shape[0])
    if atom_count <= 1:
        return EpathClusterMetrics(
            mst_total_length=0.0,
            mst_largest_edge=0.0,
            span=0.0,
            mean_step=0.0,
        )

    in_tree = np.zeros(atom_count, dtype=bool)
    minimum_edges = np.full(atom_count, np.inf)
    in_tree[0] = True
    for index in range(1, atom_count):
        minimum_edges[index] = euclidean_distance(
            coordinate_array[0],
            coordinate_array[index],
        )

    total_length = 0.0
    largest_edge = 0.0
    for _ in range(atom_count - 1):
        next_index = int(np.argmin(np.where(in_tree, np.inf, minimum_edges)))
        if in_tree[next_index]:
            break

        in_tree[next_index] = True
        edge_length = float(minimum_edges[next_index])
        total_length += edge_length
        largest_edge = max(largest_edge, edge_length)

        for candidate_index in range(atom_count):
            if in_tree[candidate_index]:
                continue

            distance = euclidean_distance(
                coordinate_array[next_index],
                coordinate_array[candidate_index],
            )
            if distance < minimum_edges[candidate_index]:
                minimum_edges[candidate_index] = distance

    span = maximum_pairwise_distance(coordinate_array)
    mean_step = total_length / float(max(atom_count - 1, 1))

    return EpathClusterMetrics(
        mst_total_length=total_length,
        mst_largest_edge=largest_edge,
        span=span,
        mean_step=mean_step,
    )


def euclidean_distance(
    first_coordinate: Sequence[float] | np.ndarray,
    second_coordinate: Sequence[float] | np.ndarray,
) -> float:
    """Return the Euclidean distance between two 3D coordinates."""

    first = np.asarray(first_coordinate, dtype=float)
    second = np.asarray(second_coordinate, dtype=float)
    if first.shape != (3,) or second.shape != (3,):
        raise EpathClusteringError("coordinates must contain exactly 3 values.")

    difference = first - second
    return float(np.sqrt(np.dot(difference, difference)))


def maximum_pairwise_distance(coordinates: np.ndarray) -> float:
    """Return the largest pairwise distance in a coordinate array."""

    coordinate_array = np.asarray(coordinates, dtype=float)
    atom_count = int(coordinate_array.shape[0])
    maximum_distance = 0.0
    for first_index in range(atom_count):
        for second_index in range(first_index + 1, atom_count):
            maximum_distance = max(
                maximum_distance,
                euclidean_distance(
                    coordinate_array[first_index],
                    coordinate_array[second_index],
                ),
            )

    return maximum_distance


def _coordinates(atoms: Sequence[EpathScoredAtom]) -> np.ndarray:
    coordinates = np.asarray(
        [
            (
                scored_atom.atom.x,
                scored_atom.atom.y,
                scored_atom.atom.z,
            )
            for scored_atom in atoms
        ],
        dtype=float,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise EpathClusteringError("atom coordinates must have shape (N, 3).")

    if not np.all(np.isfinite(coordinates)):
        raise EpathClusteringError("atom coordinates must be finite.")

    return coordinates


def _neighbour_indices_by_atom(
    coordinates: np.ndarray,
    *,
    distance_cutoff: float,
) -> tuple[tuple[int, ...], ...]:
    tree = cKDTree(coordinates)
    raw_neighbours = tree.query_ball_point(
        coordinates,
        r=distance_cutoff,
        p=2.0,
        workers=-1,
    )
    return tuple(
        tuple(int(index) for index in neighbours)
        for neighbours in raw_neighbours
    )


def _connected_components(
    neighbours_by_index: Sequence[Sequence[int]],
) -> list[list[int]]:
    components: list[list[int]] = []
    visited_indices: set[int] = set()

    for start_index in range(len(neighbours_by_index)):
        if start_index in visited_indices:
            continue

        component: list[int] = []
        stack = [start_index]
        visited_indices.add(start_index)

        while stack:
            atom_index = stack.pop()
            component.append(atom_index)
            for neighbour_index in neighbours_by_index[atom_index]:
                if neighbour_index in visited_indices:
                    continue

                visited_indices.add(neighbour_index)
                stack.append(neighbour_index)

        components.append(component)

    return components


def _positive_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathClusteringError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathClusteringError(
            f"{name} must be numeric, got {value!r}."
        ) from error

    if not math.isfinite(number) or number <= 0.0:
        raise EpathClusteringError(
            f"{name} must be a finite positive number, got {number!r}."
        )

    return number
