"""Distance geometry, spanning trees, and connectivity clustering."""

import numpy as np
import pytest
from support import approx, scored_atoms_from_coordinates

from epath.analysis.clustering import (
    EpathClusteringError,
    cluster_selected_atoms,
    component_statistics,
    mst_length_stats,
)
from epath.analysis.geometry import (
    maximum_pairwise_distance_pair,
    minimum_spanning_tree_edges,
)

LINE = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0), (4.5, 0.0, 0.0)]


def test_minimum_spanning_tree_of_a_line() -> None:
    edges = minimum_spanning_tree_edges(np.asarray(LINE))

    assert [(e.first_index, e.second_index) for e in edges] == [(0, 1), (1, 2), (2, 3)]
    assert [e.length for e in edges] == approx([1.5, 1.5, 1.5])
    assert minimum_spanning_tree_edges(np.zeros((1, 3))) == ()


def test_minimum_spanning_tree_prefers_short_edges() -> None:
    # A square with a far-off point: the MST uses three sides of the square
    # plus the shortest link to the outlier, never the diagonal.
    points = np.asarray(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (5, 1, 0)], dtype=float
    )
    edges = minimum_spanning_tree_edges(points)

    assert sorted(e.length for e in edges) == approx([1.0, 1.0, 1.0, 4.0])
    assert sum(e.length for e in edges) == approx(7.0)


def test_cluster_metrics() -> None:
    metrics = mst_length_stats(np.asarray(LINE))
    assert (
        metrics.mst_total_length,
        metrics.mst_largest_edge,
        metrics.span,
        metrics.mean_step,
    ) == approx((4.5, 1.5, 4.5, 1.5))

    single = mst_length_stats(np.asarray(LINE[:1]))
    assert (single.mst_total_length, single.span, single.mean_step) == (0.0, 0.0, 0.0)


def test_span_endpoints() -> None:
    points = np.asarray([(0, 0, 0), (1, 0, 0), (0, 3, 0), (0, 0, 2)], dtype=float)

    first, second, span = maximum_pairwise_distance_pair(points)
    assert (first, second) == (2, 3) and span == approx(np.sqrt(13))
    assert maximum_pairwise_distance_pair(points[:1]) == (0, 0, 0.0)


def test_non_finite_coordinates_are_rejected() -> None:
    with pytest.raises(EpathClusteringError, match="finite"):
        cluster_selected_atoms(
            scored_atoms_from_coordinates([*LINE, (0.0, 0.0, float("nan"))])
        )


def test_clusters_are_connected_components_ordered_by_size() -> None:
    coordinates = [
        (0.0, 0.0, 0.0),
        (3.9, 0.0, 0.0),
        (7.8, 0.0, 0.0),  # a chain of three, 3.9 A apart
        (50.0, 0.0, 0.0),  # isolated
        (0.0, 20.0, 0.0),
        (0.0, 24.0, 0.0),  # exactly 4.0 A apart: connected
        (0.0, 30.0, 0.0),
        (0.0, 34.001, 0.0),  # just over: not connected
    ]
    atoms = scored_atoms_from_coordinates(coordinates)

    clusters = cluster_selected_atoms(atoms, distance_cutoff=4.0)

    members = [sorted(a.record.source_atom_index for a in c.atoms) for c in clusters]
    assert members == [[0, 1, 2], [4, 5], [3], [6], [7]]
    assert [c.cluster_id for c in clusters] == [1, 2, 3, 4, 5]
    assert clusters[0].atom_count == 3
    assert clusters[0].metrics.mst_total_length == approx(7.8)
    assert clusters[1].metrics.span == approx(4.0)
    assert cluster_selected_atoms(()) == ()

    statistics = component_statistics(clusters)
    assert (statistics.total_count, statistics.largest_atom_count) == (5, 3)
    assert statistics.largest_metrics == clusters[0].metrics
    assert component_statistics(()).largest_metrics is None
