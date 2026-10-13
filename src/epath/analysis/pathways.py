"""Describe each E-path cluster as a pathway.

This module turns the geometric clusters produced by connectivity analysis
into the pathway description the E-path analysis reports. A description gives
the residues and protected atoms along the path, the environments its two ends
terminate in, the cofactor it comes nearest to, and its graph topology and
residue chemistry.

These are measurements, reported as the basis for interpreting a path with
the protein's known chemistry in view.

`epath.analysis.clustering` supplies the clusters and their geometry. The
topology comes from `epath.analysis.topology`, the chemistry from
`epath.analysis.residues`, and the endpoints and the anchor from
`epath.analysis.endpoints`. This module gathers them into one report for each
cluster and adds the residue listing, the steps, and the descriptor.
`epath.pipeline` calls `analyze_paths` once for a model, and `epath.outputs`
and `epath.cli.report` write the reports out.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    EpathCluster,
    coordinates_for_atoms,
)
from epath.analysis.endpoints import (
    CofactorSite,
    EndpointContext,
    PathAnchor,
    PathEndpoint,
    annotate_endpoint,
    build_endpoint_context,
    find_path_anchor,
)
from epath.analysis.geometry import (
    maximum_pairwise_distance_pair,
    minimum_spanning_tree_edges,
)
from epath.analysis.labels import (
    atom_token,
    format_residue_label,
    residue_key,
    residue_label,
)
from epath.analysis.residues import (
    ResidueClass,
    ResidueComposition,
    analyze_composition,
    class_fractions,
)
from epath.analysis.selection import EpathScoredAtom
from epath.analysis.topology import PathTopology, TopologyClass, analyze_topology
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord

_TOPOLOGY_NOUNS: dict[TopologyClass, str] = {
    "linear": "corridor",
    "branched": "network",
    "clustered": "cluster",
}


StepKind = Literal["intra_residue", "peptide", "tertiary", "interchain"]


@dataclass(frozen=True, slots=True)
class PathStep:
    """One step of a path, which is an edge of its minimum spanning tree.

    Steps are listed in the order the spanning tree was built, which is
    deterministic but is not a walk from one end of the path to the other.
    """

    step_index: int
    length: float
    kind: StepKind
    first_atom: str
    first_residue: str
    second_atom: str
    second_residue: str


@dataclass(frozen=True, slots=True)
class EdgeProvenance:
    """How the steps of a path connect the residues it runs through.

    A path that only follows the backbone is a rigid stretch of chain, which is
    a much weaker coupling candidate than one of the same size whose steps jump
    between residues that are far apart in sequence. Steps are counted on the
    minimum spanning tree, so they are the same steps the reported length and
    mean step describe.

    Peptide steps are identified by sequence adjacency within a chain rather
    than by a modelled backbone bond, so a chain break numbered consecutively
    counts as peptide-contiguous.
    """

    intra_residue: int
    peptide: int
    tertiary: int
    interchain: int

    @property
    def between_residues(self) -> int:
        """Return the number of steps that leave the residue they start in."""
        return self.peptide + self.tertiary + self.interchain

    @property
    def peptide_fraction(self) -> float:
        """Return the share of between-residue steps that follow the sequence."""
        return self.peptide / self.between_residues if self.between_residues else 0.0


@dataclass(frozen=True, slots=True)
class PathResidue:
    """One residue of a path and the atoms of it that were selected."""

    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_names: tuple[str, ...]

    @property
    def label(self) -> str:
        """Return the residue label in Trp76 form."""
        return format_residue_label(
            self.residue_name,
            self.residue_number,
            self.insertion_code,
        )


@dataclass(frozen=True, slots=True)
class PathReport:
    """One E-path cluster described as a pathway.

    `residues` are in ascending residue order, chain by chain, and `chains`
    lists the chains in the order `residues` reaches them. `residue_path` is
    the same listing as text.

    `start_endpoint` and `end_endpoint` describe the two atoms of the cluster
    that lie farthest apart. A path has no direction, the start is simply the
    one of the two that comes first in the cluster's atom order. `anchor` is
    the cofactor nearest to any atom of the path, or None when none is within
    range. `steps` are the edges of the minimum spanning tree.
    """

    cluster: EpathCluster
    chains: tuple[str, ...]
    residues: tuple[PathResidue, ...]
    residue_path: str
    start_endpoint: PathEndpoint
    end_endpoint: PathEndpoint
    anchor: PathAnchor | None
    topology: PathTopology
    composition: ResidueComposition
    steps: tuple[PathStep, ...]
    structural_descriptor: str

    @property
    def is_cofactor_anchored(self) -> bool:
        """Return whether any atom of this path lies in a cofactor's environment."""
        return self.anchor is not None

    @property
    def edges(self) -> EdgeProvenance:
        """Return how the steps of this path connect its residues."""
        counts = Counter(step.kind for step in self.steps)

        return EdgeProvenance(
            intra_residue=counts["intra_residue"],
            peptide=counts["peptide"],
            tertiary=counts["tertiary"],
            interchain=counts["interchain"],
        )

    @property
    def path_id(self) -> int:
        """Return the path identifier, which is the cluster identifier."""
        return self.cluster.cluster_id

    @property
    def node_count(self) -> int:
        """Return the number of selected atoms, the nodes of the path."""
        return self.cluster.atom_count

    @property
    def residue_count(self) -> int:
        """Return the number of residues the path runs through."""
        return len(self.residues)

    @property
    def total_length(self) -> float:
        """Return the total MST length of the path in Angstroms."""
        return self.cluster.metrics.mst_total_length

    @property
    def anchor_summary(self) -> str:
        """Return the anchoring cofactor and its distance, or a note that there is none."""
        if self.anchor is None:
            return "no cofactor in range"

        return f"{self.anchor.label} at {self.anchor.distance:.1f} A"

    @property
    def endpoint_summary(self) -> str:
        """Return both endpoint labels, the start first."""
        return f"{self.start_endpoint.label} -> {self.end_endpoint.label}"


@dataclass(frozen=True, slots=True)
class PathAnalysisResult:
    """Described pathways for one model and the cofactors they were labelled against."""

    reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]


def analyze_paths(
    clusters: Sequence[EpathCluster],
    *,
    scored_atoms: Sequence[EpathScoredAtom],
    cleaned_atoms: Sequence[PreparedAtom],
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
) -> PathAnalysisResult:
    """Describe every reported cluster as a pathway.

    The reports are in the order of `clusters`. `distance_cutoff` must be the
    cutoff the clusters were built with, in angstroms.
    """
    context = build_endpoint_context(
        scored_atoms=scored_atoms,
        cleaned_atoms=cleaned_atoms,
    )
    background = class_fractions(
        record.residue_name
        for record in _unique_residue_atoms([atom.record for atom in scored_atoms])
    )

    return PathAnalysisResult(
        reports=tuple(
            _build_path_report(
                cluster,
                context=context,
                background=background,
                distance_cutoff=distance_cutoff,
            )
            for cluster in clusters
        ),
        cofactor_sites=context.sites,
    )


def _build_path_report(
    cluster: EpathCluster,
    *,
    context: EndpointContext,
    background: dict[ResidueClass, float],
    distance_cutoff: float,
) -> PathReport:
    """Describe one cluster as a pathway.

    `context` and `background` describe the whole model, not the cluster, so
    `analyze_paths` builds each once and passes it to every cluster.
    """
    atoms = [scored_atom.record for scored_atom in cluster.atoms]
    coordinates = coordinates_for_atoms(cluster.atoms)
    residues, residue_indices = _path_residues(atoms)
    composition = analyze_composition(
        (residue.residue_name for residue in residues),
        background=background,
    )
    topology = analyze_topology(
        coordinates,
        residue_indices,
        cluster.metrics,
        distance_cutoff=distance_cutoff,
    )
    first_index, second_index, _ = maximum_pairwise_distance_pair(coordinates)
    start_endpoint = annotate_endpoint(cluster.atoms[first_index], context)
    end_endpoint = annotate_endpoint(cluster.atoms[second_index], context)

    return PathReport(
        cluster=cluster,
        chains=tuple(dict.fromkeys(residue.chain_id for residue in residues)),
        residues=residues,
        residue_path=_format_residue_path(residues),
        start_endpoint=start_endpoint,
        end_endpoint=end_endpoint,
        anchor=find_path_anchor(atoms, context),
        topology=topology,
        composition=composition,
        steps=_path_steps(atoms, coordinates),
        structural_descriptor=_structural_descriptor(
            composition=composition,
            topology=topology,
            endpoints=(start_endpoint, end_endpoint),
            residue_count=len(residues),
        ),
    )


def _path_residues(
    atoms: Sequence[AtomRecord],
) -> tuple[tuple[PathResidue, ...], list[int]]:
    """Group a path's atoms into residues, in ascending residue order.

    Chains are ordered by the lowest residue number of the path they
    contribute, with ties broken by chain identifier, and residues within a
    chain by residue number and then insertion code. This is the order the
    E-path table lists them in. The atoms of a residue keep the order they
    have in `atoms`.

    The residues are returned with, for each atom, the index of its residue
    among them.
    """
    grouped: dict[tuple[str, int | None, str], list[AtomRecord]] = {}
    for atom in atoms:
        grouped.setdefault(residue_key(atom), []).append(atom)

    first_number_by_chain: dict[str, int] = {}
    for chain_id, residue_number, _ in grouped:
        number = _sort_number(residue_number)
        first_number_by_chain[chain_id] = min(
            number,
            first_number_by_chain.get(chain_id, number),
        )

    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            first_number_by_chain[key[0]],
            key[0],
            _sort_number(key[1]),
            key[2],
        ),
    )

    residues = tuple(
        PathResidue(
            chain_id=key[0],
            residue_name=grouped[key][0].residue_name.strip().upper(),
            residue_number=key[1],
            insertion_code=key[2],
            atom_names=tuple(atom.atom_name.strip() for atom in grouped[key]),
        )
        for key in ordered_keys
    )
    position_by_key = {key: position for position, key in enumerate(ordered_keys)}
    residue_indices = [position_by_key[residue_key(atom)] for atom in atoms]

    return residues, residue_indices


def _path_steps(
    atoms: Sequence[AtomRecord],
    coordinates: NDArray[np.float64],
) -> tuple[PathStep, ...]:
    """Return each step of a path with its length and how it connects residues.

    The steps are numbered from one, in the order the spanning tree was built.
    """
    edges = minimum_spanning_tree_edges(coordinates)

    return tuple(
        PathStep(
            step_index=step_index,
            length=edge.length,
            kind=_edge_kind(atoms[edge.first_index], atoms[edge.second_index]),
            first_atom=atom_token(atoms[edge.first_index]),
            first_residue=residue_label(atoms[edge.first_index]),
            second_atom=atom_token(atoms[edge.second_index]),
            second_residue=residue_label(atoms[edge.second_index]),
        )
        for step_index, edge in enumerate(edges, start=1)
    )


def _edge_kind(first: AtomRecord, second: AtomRecord) -> StepKind:
    """Return how a step between two atoms connects their residues.

    A residue without a number is never a sequence neighbour.
    """
    if residue_key(first) == residue_key(second):
        return "intra_residue"

    if first.chain_id != second.chain_id:
        return "interchain"

    if first.residue_number is None or second.residue_number is None:
        return "tertiary"

    # Residues that share a number and differ in insertion code are sequence
    # neighbours too, which is the case of a difference of zero.
    if abs(first.residue_number - second.residue_number) <= 1:
        return "peptide"

    return "tertiary"


def _format_residue_path(residues: Sequence[PathResidue]) -> str:
    """Return the residue-and-node listing used by the E-path pathway table."""
    parts: list[str] = []
    current_chain: str | None = None
    for residue in residues:
        prefix = ""
        if residue.chain_id != current_chain:
            current_chain = residue.chain_id
            prefix = f"Chain {residue.chain_id} - "

        atom_names = ", ".join(residue.atom_names)
        parts.append(f"{prefix}{residue.label} ({atom_names})")

    return " -> ".join(parts)


def _structural_descriptor(
    *,
    composition: ResidueComposition,
    topology: PathTopology,
    endpoints: Sequence[PathEndpoint],
    residue_count: int,
) -> str:
    """Name a path by its dominant chemistry, its shape, and its enrichment.

    A path with no dominant class, which is one made only of residues outside
    the named classes, is an "Unassigned cluster". A path within one residue
    is a hotspot. Any other is an interface motif when either endpoint lies at
    an interface, and otherwise takes its noun from its topology. The classes
    it is enriched in are added unless they are already named as dominant.
    """
    dominant = composition.dominant_classes
    if not dominant:
        return "Unassigned cluster"

    # A class name reads as an adjective as it is, except "sulfur". The
    # "other" class is never dominant or enriched, so it needs no wording.
    adjective = "-".join(
        "sulfur-rich" if name == "sulfur" else name for name in dominant
    )
    if residue_count <= 1:
        return f"Single {adjective} hotspot"

    if any(endpoint.kind == "interface" for endpoint in endpoints):
        noun = "interface motif"
    else:
        noun = _TOPOLOGY_NOUNS[topology.topology]

    enriched = [name for name in composition.enriched_classes if name not in dominant]
    descriptor = f"{adjective.capitalize()} {noun}"
    if not enriched:
        return descriptor

    return f"{descriptor} with {'/'.join(enriched)} enrichment"


def _unique_residue_atoms(atoms: Sequence[AtomRecord]) -> list[AtomRecord]:
    """Return the first atom of each residue, in the order the residues appear."""
    seen: dict[tuple[str, int | None, str], AtomRecord] = {}
    for atom in atoms:
        seen.setdefault(residue_key(atom), atom)

    return list(seen.values())


def _sort_number(residue_number: int | None) -> int:
    """Return the number a residue sorts by, which is 0 when it has none."""
    return residue_number if residue_number is not None else 0
