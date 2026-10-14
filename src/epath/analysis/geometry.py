"""Distances and spanning trees over the atoms of one E-path cluster.

`clustering` takes a cluster's spanning-tree lengths from here, `pathways`
its steps, and `cofactors` the distances from one point to a set of atoms.
Coordinates are arrays of shape `(n, 3)`, and lengths come back in the units
of the coordinates, which are angstroms for every caller.

The spanning tree measures every pair of points, which is fast enough for the
atoms of one cluster. Finding the atoms within a cutoff of each other is left
to the KD-tree in `spatial`.

Nothing is checked here. The coordinates must be finite, which `clustering`
checks for the selected atoms before it forms the clusters.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MstEdge:
    """One edge of a minimum spanning tree.

    The indices are rows of the coordinate array. `first_index` is the point
    that was already in the tree and `second_index` the point the edge added.
    """

    first_index: int
    second_index: int
    length: float


def row_norms(differences: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Euclidean length of each row of an `(n, 3)` difference array.

    The squared terms are summed in coordinate order, so the result matches a
    per-pair dot product exactly.
    """
    squared = differences * differences
    return np.sqrt(squared[:, 0] + squared[:, 1] + squared[:, 2])


def minimum_spanning_tree_edges(
    coordinates: NDArray[np.float64],
) -> tuple[MstEdge, ...]:
    """Return the minimum spanning tree of a coordinate array.

    Prim's algorithm grows the tree from point 0, each time adding the outside
    point nearest to the tree, and the lowest index when several are equally
    near. Edges are returned in the order they are added, so the total and
    largest edge lengths derived from them match the reported cluster geometry
    exactly. Fewer than two points give no edges.
    """
    point_count = len(coordinates)
    if point_count <= 1:
        return ()

    in_tree = np.zeros(point_count, dtype=bool)
    in_tree[0] = True
    # For each point outside the tree, the tree point nearest to it and the
    # distance between the two.
    parents = np.zeros(point_count, dtype=int)
    minimum_edges = row_norms(coordinates - coordinates[0])

    edges: list[MstEdge] = []
    for _ in range(point_count - 1):
        next_index = int(np.argmin(np.where(in_tree, np.inf, minimum_edges)))
        in_tree[next_index] = True
        edges.append(
            MstEdge(
                first_index=int(parents[next_index]),
                second_index=next_index,
                length=float(minimum_edges[next_index]),
            )
        )

        distances = row_norms(coordinates - coordinates[next_index])
        improved = ~in_tree & (distances < minimum_edges)
        parents[improved] = next_index
        minimum_edges[improved] = distances[improved]

    return tuple(edges)
