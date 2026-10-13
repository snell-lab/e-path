"""Radius neighbour searches backed by SciPy's KD-tree.

A KD-tree finds the points near each query without measuring the distance to
every point, which would be too slow for the atoms of a whole crystal.

SciPy ships no type stubs, so this is the one module that talks to
`scipy.spatial` directly. The two protocols below describe the part of its
tree that is used here. Callers get typed functions and never see the tree.
"""

from collections.abc import Iterable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

# Passed to SciPy as `workers`, this runs a search on every processor core.
_ALL_CORES = -1


class _KDTree(Protocol):
    def query_ball_point(
        self,
        x: NDArray[np.float64],
        r: float,
        *,
        return_length: bool = False,
        workers: int = 1,
    ) -> object: ...


class _KDTreeFactory(Protocol):
    def __call__(self, data: NDArray[np.float64]) -> _KDTree: ...


_build_tree = cast(_KDTreeFactory, KDTree)


def radius_neighbour_indices(
    points: NDArray[np.float64],
    radius: float,
) -> tuple[tuple[int, ...], ...]:
    """Return, for each point, the indices of all points within `radius`.

    A point at exactly `radius` is included, and so is each point's own index,
    since it lies at distance zero.
    """
    neighbours = cast(
        Iterable[list[int]],
        _build_tree(points).query_ball_point(points, radius, workers=_ALL_CORES),
    )
    return tuple(tuple(indices) for indices in neighbours)


def radius_neighbour_counts(
    tree_points: NDArray[np.float64],
    query_points: NDArray[np.float64],
    radius: float,
) -> NDArray[np.int64]:
    """Return how many `tree_points` lie within `radius` of each query point.

    A point at exactly `radius` is counted.
    """
    counts = _build_tree(tree_points).query_ball_point(
        query_points,
        radius,
        return_length=True,
        workers=_ALL_CORES,
    )
    return np.asarray(counts, dtype=np.int64)
