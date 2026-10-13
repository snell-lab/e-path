"""Distance and spanning-tree geometry shared by the E-path analysis layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MstEdge:
    """One edge of a minimum spanning tree."""

    first_index: int
    second_index: int
    length: float


def row_norms(differences: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Euclidean length of each row of a difference array.

    The squared terms are summed in coordinate order, so the result matches a
    per-pair dot product exactly.
    """
    squared = differences * differences
    lengths: NDArray[np.float64] = np.sqrt(
        squared[:, 0] + squared[:, 1] + squared[:, 2]
    )
    return lengths


def minimum_spanning_tree_edges(
    coordinates: NDArray[np.float64],
) -> tuple[MstEdge, ...]:
    """Return the minimum spanning tree of a coordinate array.

    Edges are returned in the order Prim's algorithm adds them, so the total
    and largest edge lengths derived from them match the reported cluster
    geometry exactly.
    """
    atom_count = int(coordinates.shape[0])
    if atom_count <= 1:
        return ()

    in_tree = np.zeros(atom_count, dtype=bool)
    parents = np.zeros(atom_count, dtype=int)
    minimum_edges = _distances_from(coordinates, 0)
    minimum_edges[0] = np.inf
    in_tree[0] = True

    edges: list[MstEdge] = []
    for _ in range(atom_count - 1):
        next_index = int(np.argmin(np.where(in_tree, np.inf, minimum_edges)))
        in_tree[next_index] = True
        edges.append(
            MstEdge(
                first_index=int(parents[next_index]),
                second_index=next_index,
                length=float(minimum_edges[next_index]),
            )
        )

        distances = _distances_from(coordinates, next_index)
        improved = np.logical_and(~in_tree, distances < minimum_edges)
        parents[improved] = next_index
        minimum_edges[improved] = distances[improved]

    return tuple(edges)


def maximum_pairwise_distance_pair(
    coordinates: NDArray[np.float64],
) -> tuple[int, int, float]:
    """Return the two most separated points and the distance between them.

    The pair defines the span, which the analysis reports as the straight-line
    end-to-end distance of a path, so these two atoms are also its endpoints.
    """
    atom_count = int(coordinates.shape[0])
    if atom_count < 2:
        return 0, 0, 0.0

    first_endpoint = 0
    second_endpoint = 0
    maximum_distance = 0.0
    for first_index in range(atom_count - 1):
        distances = row_norms(coordinates[first_index + 1 :] - coordinates[first_index])
        offset = int(np.argmax(distances))
        distance = float(distances[offset])
        if distance > maximum_distance:
            maximum_distance = distance
            first_endpoint = first_index
            second_endpoint = first_index + 1 + offset

    return first_endpoint, second_endpoint, maximum_distance


def neighbour_pairs(
    coordinates: NDArray[np.float64],
    *,
    distance_cutoff: float,
) -> tuple[tuple[int, int], ...]:
    """Return every pair of points within a distance cutoff of each other."""
    point_count = int(coordinates.shape[0])
    pairs: list[tuple[int, int]] = []
    for first_index in range(point_count - 1):
        distances = row_norms(coordinates[first_index + 1 :] - coordinates[first_index])
        for offset in np.nonzero(distances <= distance_cutoff)[0]:
            pairs.append((first_index, first_index + 1 + int(offset)))

    return tuple(pairs)


def _distances_from(
    coordinates: NDArray[np.float64],
    index: int,
) -> NDArray[np.float64]:
    """Return the distance from one point to every point in an array."""
    return row_norms(coordinates - coordinates[index])
