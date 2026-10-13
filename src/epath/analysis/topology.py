"""Describe the graph topology of an E-path cluster.

The pathway framework distinguishes linear, branched, and clustered networks
using span-to-length ratios and node connectivity. Both measures come from the
geometry the cluster is already reported with, so a path's topology is
consistent with its reported length and span.

Linearity is the span divided by the total MST length: a path that runs from
one end to the other approaches one, while a compact region spends its length
filling a small volume and stays low. Atom-level paths detour along every side
chain they pass through, so an extended pathway does not approach the ratio a
straight line would give.

Connectivity is measured between residues rather than between atoms. Atoms
within one side chain are all mutually close, and the ring of an aromatic
residue contains junctions of its own, so atom connectivity mostly reports the
shape of the residues a path runs through. A fork in the path itself is a
residue in contact with three or more other residues of the same cluster.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.clustering import EpathClusterMetrics
from epath.analysis.geometry import neighbour_pairs

TopologyClass = Literal["linear", "branched", "clustered"]

# A path this elongated relative to its own length reads as a corridor rather
# than a compact region.
DEFAULT_LINEAR_RATIO = 0.20

# Above this fraction of junction residues the component spreads through a
# volume by branching instead of carrying a single route through it.
DEFAULT_BRANCHED_NODE_FRACTION = 0.25

# Residues in contact with at least this many other residues of the cluster are
# junctions, because the path can leave them by more than one route.
JUNCTION_RESIDUE_DEGREE = 3


@dataclass(frozen=True, slots=True)
class PathTopology:
    """Topological description of one E-path cluster."""

    topology: TopologyClass
    linearity: float
    branch_fraction: float


def analyze_topology(
    coordinates: NDArray[np.float64],
    residue_indices: Sequence[int],
    metrics: EpathClusterMetrics,
    *,
    distance_cutoff: float,
    linear_ratio: float = DEFAULT_LINEAR_RATIO,
    branched_node_fraction: float = DEFAULT_BRANCHED_NODE_FRACTION,
) -> PathTopology:
    """Classify a cluster as a linear, branched, or clustered network.

    Args:
        coordinates: Atom coordinates of the cluster.
        residue_indices: The residue each atom belongs to, as an index into the
            cluster's residue list.
        metrics: Cluster geometry, used for the span-to-length ratio.
        distance_cutoff: Contact distance used to build the residue graph. This
            is the cutoff the cluster itself was built with, so two residues are
            in contact exactly when their atoms were connected in the path.
        linear_ratio: Span-to-length ratio at or above which a component that is
            not branched is reported as linear.
        branched_node_fraction: Junction-residue fraction at or above which a
            component is reported as branched.

    Returns:
        The topology class and the two measures it was derived from.
    """
    degrees = _residue_degrees(
        coordinates,
        residue_indices,
        distance_cutoff=distance_cutoff,
    )
    junction_count = sum(1 for degree in degrees if degree >= JUNCTION_RESIDUE_DEGREE)
    branch_fraction = junction_count / len(degrees)
    linearity = (
        metrics.span / metrics.mst_total_length
        if metrics.mst_total_length > 0.0
        else 0.0
    )

    if branch_fraction >= branched_node_fraction:
        topology: TopologyClass = "branched"
    elif linearity >= linear_ratio:
        topology = "linear"
    else:
        topology = "clustered"

    return PathTopology(
        topology=topology,
        linearity=linearity,
        branch_fraction=branch_fraction,
    )


def _residue_degrees(
    coordinates: NDArray[np.float64],
    residue_indices: Sequence[int],
    *,
    distance_cutoff: float,
) -> list[int]:
    """Return how many other residues of the cluster each residue contacts."""
    contacts: list[set[int]] = [set() for _ in range(max(residue_indices) + 1)]
    for first_index, second_index in neighbour_pairs(
        coordinates,
        distance_cutoff=distance_cutoff,
    ):
        first_residue = residue_indices[first_index]
        second_residue = residue_indices[second_index]
        if first_residue == second_residue:
            continue

        contacts[first_residue].add(second_residue)
        contacts[second_residue].add(first_residue)

    return [len(residue_contacts) for residue_contacts in contacts]
