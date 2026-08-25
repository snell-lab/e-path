"""KD-tree neighbour searches: inclusive radius and self-inclusion."""

import numpy as np

from spatial import radius_neighbour_counts, radius_neighbour_indices


def test_indices_include_self_and_points_exactly_at_radius() -> None:
    points = np.asarray([[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [10.0, 0, 0]])

    neighbours = radius_neighbour_indices(points, 1.5)

    assert sorted(neighbours[0]) == [0, 1]
    assert sorted(neighbours[1]) == [0, 1, 2]  # 2.5 - 1.0 == 1.5 is inside
    assert sorted(neighbours[2]) == [1, 2]
    assert neighbours[3] == (3,)


def test_counts_use_separate_tree_and_query_points() -> None:
    tree_points = np.asarray([[0.0, 0, 0], [3.0, 0, 0], [0, 4.0, 0]])
    queries = np.asarray([[0.0, 0, 0], [100.0, 0, 0]])

    counts = radius_neighbour_counts(tree_points, queries, 4.0)

    assert counts.tolist() == [3, 0]
    assert counts.dtype == np.int64
