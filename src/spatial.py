"""Radius neighbour searches backed by SciPy's KD-tree.

SciPy ships no type stubs, so this is the one module that talks to
`scipy.spatial` directly. Callers get typed functions and never see the tree.
"""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


class _KDTree(Protocol):
    def query_ball_point(
        self,
        x: NDArray[np.float64],
        r: float,
        p: float = 2.0,
        *,
        return_length: bool = False,
        workers: int = 1,
    ) -> object: ...


class _KDTreeFactory(Protocol):
    def __call__(self, data: NDArray[np.float64]) -> _KDTree: ...


_build_tree = cast(_KDTreeFactory, cKDTree)


def radius_neighbour_indices(
    points: NDArray[np.float64],
    radius: float,
    *,
    minkowski_p: float = 2.0,
) -> tuple[tuple[int, ...], ...]:
    """Return, for each point, the indices of all points within `radius`.

    Each point's own index is included, since it lies at distance zero.
    """
    tree = _build_tree(points)
    raw_neighbours = cast(
        list[list[int]],
        tree.query_ball_point(points, radius, minkowski_p, workers=-1),
    )
    return tuple(
        tuple(int(index) for index in neighbours) for neighbours in raw_neighbours
    )


def radius_neighbour_counts(
    tree_points: NDArray[np.float64],
    query_points: NDArray[np.float64],
    radius: float,
) -> NDArray[np.int64]:
    """Return how many `tree_points` lie within `radius` of each query point."""
    tree = _build_tree(tree_points)
    raw_counts = tree.query_ball_point(
        query_points,
        radius,
        return_length=True,
        workers=-1,
    )
    return np.reshape(np.asarray(raw_counts, dtype=np.int64), (-1,))
