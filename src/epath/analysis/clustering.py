"""Cluster selected E-path atoms by spatial connectivity."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from epath.analysis.geometry import (
    maximum_pairwise_distance_pair,
    minimum_spanning_tree_edges,
)
from epath.analysis.selection import EpathScoredAtom
from spatial import radius_neighbour_indices

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


@dataclass(frozen=True, slots=True)
class ComponentStatistics:
    """Connectivity statistics for every component found, before size filtering.

    Components smaller than the reporting threshold are not pathways, but the
    largest one still describes what the connectivity analysis found. Recording
    it keeps a model that reported nothing distinguishable from a model whose
    largest component fell just short.
    """

    total_count: int
    largest_atom_count: int
    largest_metrics: EpathClusterMetrics | None


def component_statistics(
    clusters: Sequence[EpathCluster],
) -> ComponentStatistics:
    """Summarize every connected component, whether or not it is reportable.

    Args:
        clusters: All components, before the reporting threshold is applied.
            They are ordered largest first by `cluster_selected_atoms`.

    Returns:
        The component count and the geometry of the largest component.
    """
    if not clusters:
        return ComponentStatistics(
            total_count=0,
            largest_atom_count=0,
            largest_metrics=None,
        )

    largest = max(clusters, key=lambda cluster: cluster.atom_count)

    return ComponentStatistics(
        total_count=len(clusters),
        largest_atom_count=largest.atom_count,
        largest_metrics=largest.metrics,
    )


def cluster_selected_atoms(
    atoms: Iterable[EpathScoredAtom],
    *,
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
) -> tuple[EpathCluster, ...]:
    """Find connected components among selected atoms using a distance cutoff."""
    if not math.isfinite(distance_cutoff) or distance_cutoff <= 0.0:
        raise EpathClusteringError(
            "distance_cutoff must be a finite positive number, "
            f"got {distance_cutoff!r}."
        )

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return ()

    coordinates = coordinates_for_atoms(atom_tuple)
    # Every later geometry step works on subsets of these atoms, so checking
    # them once here covers the whole analysis.
    if not np.all(np.isfinite(coordinates)):
        raise EpathClusteringError("atom coordinates must be finite.")

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


def cluster_metrics_for_atoms(
    atoms: Sequence[EpathScoredAtom],
) -> EpathClusterMetrics:
    """Calculate MST and span metrics for a group of atoms."""
    return mst_length_stats(coordinates_for_atoms(atoms))


def mst_length_stats(coordinates: NDArray[np.float64]) -> EpathClusterMetrics:
    """Return Prim-style MST metrics for a coordinate array."""
    atom_count = int(coordinates.shape[0])
    if atom_count <= 1:
        return EpathClusterMetrics(
            mst_total_length=0.0,
            mst_largest_edge=0.0,
            span=0.0,
            mean_step=0.0,
        )

    edges = minimum_spanning_tree_edges(coordinates)
    total_length = 0.0
    largest_edge = 0.0
    for edge in edges:
        total_length += edge.length
        largest_edge = max(largest_edge, edge.length)

    _, _, span = maximum_pairwise_distance_pair(coordinates)
    mean_step = total_length / (atom_count - 1)

    return EpathClusterMetrics(
        mst_total_length=total_length,
        mst_largest_edge=largest_edge,
        span=span,
        mean_step=mean_step,
    )


def coordinates_for_atoms(
    atoms: Sequence[EpathScoredAtom],
) -> NDArray[np.float64]:
    """Return the coordinate array for a sequence of scored atoms."""
    return np.asarray(
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


def _neighbour_indices_by_atom(
    coordinates: NDArray[np.float64],
    *,
    distance_cutoff: float,
) -> tuple[tuple[int, ...], ...]:
    return radius_neighbour_indices(coordinates, distance_cutoff)


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
