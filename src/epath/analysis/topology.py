"""Describe the graph topology of an E-path cluster.

The pathway framework distinguishes linear, branched, and clustered networks
using span-to-length ratios and node connectivity. Both measures come from the
geometry the cluster is already reported with, so a path's topology is
consistent with its reported length and span.

Linearity is the span divided by the total MST length. A path that runs from
one end to the other approaches one, while a compact region spends its length
filling a small volume and stays low. Atom-level paths detour along every side
chain they pass through, so an extended pathway does not approach the ratio a
straight line would give.

Connectivity is measured between residues rather than between atoms. Atoms
within one side chain are all mutually close, and the ring of an aromatic
residue contains junctions of its own, so atom connectivity mostly reports the
shape of the residues a path runs through. A fork in the path itself is a
residue in contact with three or more other residues of the same cluster.

`epath.analysis.pathways` calls this once for each reported cluster, with the
metrics `epath.analysis.clustering` calculated for it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.clustering import EpathClusterMetrics
from spatial import radius_neighbour_indices

TopologyClass = Literal["linear", "branched", "clustered"]

# A path this elongated relative to its own length reads as a corridor rather
# than a compact region.
LINEAR_RATIO = 0.20

# At or above this fraction of junction residues the cluster spreads through a
# volume by branching instead of carrying a single route through it.
BRANCHED_NODE_FRACTION = 0.25

# Residues in contact with at least this many other residues of the cluster are
# junctions, because the path can leave them by more than one route.
JUNCTION_RESIDUE_DEGREE = 3


@dataclass(frozen=True, slots=True)
class PathTopology:
    """Topological description of one E-path cluster.

    `linearity` is the span divided by the total MST length, from zero to one.
    `branch_fraction` is the fraction of the cluster's residues that are
    junctions. `topology` is the class the two give.
    """

    topology: TopologyClass
    linearity: float
    branch_fraction: float


def analyze_topology(
    coordinates: NDArray[np.float64],
    residue_indices: Sequence[int],
    metrics: EpathClusterMetrics,
    *,
    distance_cutoff: float,
) -> PathTopology:
    """Classify a cluster as a linear, branched, or clustered network.

    `coordinates` holds the position of each atom of the cluster, in angstroms,
    with shape `(n, 3)`. `residue_indices` gives the residue of each atom as an
    index into the cluster's residue list, and `metrics` is the geometry the
    cluster is reported with, which supplies the span and the MST length.

    Two residues are in contact when an atom of one lies within
    `distance_cutoff` angstroms of an atom of the other, the cutoff itself
    included. This is the cutoff the cluster itself was built with, and the
    atoms in range are found by the same search, so two residues are in
    contact exactly when their atoms were connected in the path.

    A cluster whose junction-residue fraction is at least
    `BRANCHED_NODE_FRACTION` is branched, whatever its linearity. Any other
    cluster is linear when its linearity is at least `LINEAR_RATIO` and
    clustered when it is not.
    """
    degrees = _residue_degrees(
        coordinates,
        residue_indices,
        distance_cutoff=distance_cutoff,
    )
    junction_count = sum(1 for degree in degrees if degree >= JUNCTION_RESIDUE_DEGREE)
    branch_fraction = junction_count / len(degrees)
    # A cluster of one atom has no length to divide by.
    linearity = (
        metrics.span / metrics.mst_total_length
        if metrics.mst_total_length > 0.0
        else 0.0
    )

    if branch_fraction >= BRANCHED_NODE_FRACTION:
        topology: TopologyClass = "branched"
    elif linearity >= LINEAR_RATIO:
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
    # Sets, because two residues usually touch through several pairs of atoms.
    contacts: list[set[int]] = [set() for _ in range(max(residue_indices) + 1)]
    neighbours = radius_neighbour_indices(coordinates, distance_cutoff)
    for residue, atom_neighbours in zip(residue_indices, neighbours, strict=True):
        contacts[residue].update(residue_indices[index] for index in atom_neighbours)

    # Every atom is in range of itself and of the other atoms of its residue,
    # so a residue is taken out of its own contacts.
    return [
        len(residue_contacts - {residue})
        for residue, residue_contacts in enumerate(contacts)
    ]
