"""Cluster selected E-path atoms by spatial connectivity.

Two selected atoms are connected when they lie within a distance cutoff of
each other, and a cluster is a group of atoms joined by a chain of such
connections. `epath.analysis.selection` supplies the atoms, `spatial` finds
the atoms within the cutoff of each one, and `epath.analysis.geometry`
measures each cluster. `epath.analysis.pathways` then describes each cluster
as a path.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from epath.analysis.geometry import minimum_spanning_tree_edges
from epath.analysis.selection import EpathScoredAtom
from spatial import radius_neighbour_indices

DEFAULT_CLUSTER_DISTANCE_ANGSTROM = 4.0


class EpathClusteringError(ValueError):
    """Raised when E-path atom clustering cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathClusterMetrics:
    """Geometric metrics for one group of atoms, all in angstroms.

    `mst_total_length` is the summed length of the edges of the minimum
    spanning tree over the atoms, `mst_largest_edge` the longest of them, and
    `mean_step` their mean. All three are zero for a single atom.
    """

    mst_total_length: float
    mst_largest_edge: float
    mean_step: float


@dataclass(frozen=True, slots=True)
class EpathCluster:
    """One connected component of selected E-path atoms.

    `cluster_id` counts from 1, with the largest component first.
    """

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

    `clusters` holds all the components, before the reporting threshold is
    applied, in the order `cluster_selected_atoms` returns them, so the first
    is the largest. With no components the counts are zero and there are no
    metrics.
    """
    if not clusters:
        return ComponentStatistics(
            total_count=0,
            largest_atom_count=0,
            largest_metrics=None,
        )

    largest = clusters[0]

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
    """Find the connected components of the selected atoms.

    Two atoms are connected when they are `distance_cutoff` angstroms apart or
    closer, so a pair at exactly the cutoff is connected. Every atom ends up in
    one component, and an atom with no neighbour is a component of its own.

    The components are returned largest first, and those of equal size in the
    order of their first atom in `atoms`. Within a component the atoms are in
    the order the search reached them, not in input order. No atoms give no
    components.

    The cutoff must be a finite positive number, which the command line
    enforces and is not checked here. An `EpathClusteringError` is raised when
    a coordinate is not finite.
    """
    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return ()

    coordinates = coordinates_for_atoms(atom_tuple)
    # Every later geometry step works on subsets of these atoms, so checking
    # them once here covers the whole analysis.
    if not np.all(np.isfinite(coordinates)):
        raise EpathClusteringError("atom coordinates must be finite.")

    components = _connected_components(
        radius_neighbour_indices(coordinates, distance_cutoff)
    )
    # The sort is stable, so components of equal size keep their order.
    components.sort(key=len, reverse=True)

    return tuple(
        EpathCluster(
            cluster_id=cluster_id,
            atoms=tuple(atom_tuple[index] for index in component),
            metrics=mst_length_stats(coordinates[component]),
        )
        for cluster_id, component in enumerate(components, start=1)
    )


def mst_length_stats(coordinates: NDArray[np.float64]) -> EpathClusterMetrics:
    """Measure a group of atoms from its coordinates.

    `coordinates` has shape `(n, 3)` and is in angstroms. A single atom has no
    spanning tree, so all of its metrics are zero. The coordinates must be
    finite, which `cluster_selected_atoms` checks.
    """
    edges = minimum_spanning_tree_edges(coordinates)
    # The lengths are added one at a time because `sum` compensates for
    # rounding, which would change the reported totals in their last digits.
    total_length = 0.0
    largest_edge = 0.0
    for edge in edges:
        total_length += edge.length
        largest_edge = max(largest_edge, edge.length)

    return EpathClusterMetrics(
        mst_total_length=total_length,
        mst_largest_edge=largest_edge,
        mean_step=total_length / len(edges) if edges else 0.0,
    )


def coordinates_for_atoms(
    atoms: Sequence[EpathScoredAtom],
) -> NDArray[np.float64]:
    """Return the coordinates of the atoms, in angstroms, one row per atom."""
    return np.asarray(
        [(scored.record.x, scored.record.y, scored.record.z) for scored in atoms],
        dtype=float,
    )


def _connected_components(
    neighbours_by_index: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Group the indices of a neighbour graph into its connected components.

    The components are in order of their lowest index.
    """
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
                if neighbour_index not in visited_indices:
                    visited_indices.add(neighbour_index)
                    stack.append(neighbour_index)

        components.append(component)

    return components
