"""Describe each E-path cluster as a pathway.

This module turns the geometric clusters produced by connectivity analysis
into the pathway description the E-path analysis reports: the residues and
protected atoms along the path, the environments its two ends terminate in,
the cofactor it comes nearest to, and its graph topology and residue
chemistry.

These are measurements. Whether a path conducts, modulates, or merely packs
is a judgement that also draws on what is known about the protein, so no
functional role is assigned here; that interpretation is left to the analyst.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from epath.analysis.atoms import (
    EpathAtom,
    atom_token,
    format_residue_label,
    residue_key,
    residue_label,
)
from epath.analysis.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    EpathCluster,
    coordinates_for_atoms,
)
from epath.analysis.endpoints import (
    DEFAULT_COFACTOR_CONTACT_ANGSTROM,
    DEFAULT_COFACTOR_PROXIMITY_ANGSTROM,
    DEFAULT_INTERFACE_ANGSTROM,
    DEFAULT_SURFACE_PERCENTILE,
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
from epath.analysis.residues import (
    ResidueClass,
    ResidueComposition,
    analyze_composition,
    class_fractions,
)
from epath.analysis.topology import (
    DEFAULT_BRANCHED_NODE_FRACTION,
    DEFAULT_LINEAR_RATIO,
    PathTopology,
    analyze_topology,
)
from rabdam.structure.models import PreparedAtom

# The adjective describes what a path is mostly made of; the plain class name
# is used when listing the classes it is merely enriched in.
_CLASS_ADJECTIVES: dict[ResidueClass, str] = {
    "aromatic": "aromatic",
    "sulfur": "sulfur-rich",
    "charged": "charged",
    "polar": "polar",
    "hydrophobic": "hydrophobic",
    "other": "mixed",
}

_CLASS_NAMES: dict[ResidueClass, str] = {
    "aromatic": "aromatic",
    "sulfur": "sulfur",
    "charged": "charged",
    "polar": "polar",
    "hydrophobic": "hydrophobic",
    "other": "mixed",
}

_TOPOLOGY_NOUNS: dict[str, str] = {
    "linear": "corridor",
    "branched": "network",
    "clustered": "cluster",
}


@dataclass(frozen=True, slots=True)
class PathAnalysisOptions:
    """Thresholds used to describe E-path clusters."""

    linear_ratio: float = DEFAULT_LINEAR_RATIO
    branched_node_fraction: float = DEFAULT_BRANCHED_NODE_FRACTION
    cofactor_contact_distance: float = DEFAULT_COFACTOR_CONTACT_ANGSTROM
    cofactor_proximity_distance: float = DEFAULT_COFACTOR_PROXIMITY_ANGSTROM
    interface_distance: float = DEFAULT_INTERFACE_ANGSTROM
    surface_percentile: float = DEFAULT_SURFACE_PERCENTILE


StepKind = Literal["intra_residue", "peptide", "tertiary", "interchain"]


@dataclass(frozen=True, slots=True)
class PathStep:
    """One step of a path: an edge of its minimum spanning tree.

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
    """One E-path cluster described as a pathway."""

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
        """Return both endpoint labels in source-to-destination order."""
        return f"{self.start_endpoint.label} -> {self.end_endpoint.label}"


@dataclass(frozen=True, slots=True)
class PathAnalysisResult:
    """Described pathways for one model and the cofactors they were labelled against."""

    reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]


def analyze_paths(
    clusters: Sequence[EpathCluster],
    *,
    scored_atoms: Sequence[EpathAtom],
    cleaned_atoms: Sequence[PreparedAtom] | None = None,
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    options: PathAnalysisOptions | None = None,
) -> PathAnalysisResult:
    """Describe every reported cluster as a pathway.

    Args:
        clusters: Reported clusters, in the order they should be numbered.
        scored_atoms: Every atom that received a BDamage score, used for the
            model's background residue composition and packing distribution.
        cleaned_atoms: The complete cleaned model, which supplies the cofactor
            sites the paths are anchored and their endpoints labelled against.
        distance_cutoff: The connectivity cutoff the clusters were built with,
            reused to decide which residues of a path are in contact.
        options: Threshold overrides for topology and endpoints.

    Returns:
        One report per cluster in cluster order, and the cofactor sites the
        endpoints were labelled against.
    """
    if options is None:
        options = PathAnalysisOptions()

    context = build_endpoint_context(
        scored_atoms=scored_atoms,
        cleaned_atoms=cleaned_atoms,
        cofactor_contact_distance=options.cofactor_contact_distance,
        cofactor_proximity_distance=options.cofactor_proximity_distance,
        interface_distance=options.interface_distance,
        surface_percentile=options.surface_percentile,
    )
    background = class_fractions(
        atom.residue_name for atom in _unique_residue_atoms(scored_atoms)
    )

    return PathAnalysisResult(
        reports=tuple(
            _build_path_report(
                cluster,
                context=context,
                background=background,
                distance_cutoff=distance_cutoff,
                options=options,
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
    options: PathAnalysisOptions,
) -> PathReport:
    residues, residue_indices = _path_residues(cluster)
    composition = analyze_composition(
        (residue.residue_name for residue in residues),
        background=background,
    )
    coordinates = coordinates_for_atoms(cluster.atoms)
    topology = analyze_topology(
        coordinates,
        residue_indices,
        cluster.metrics,
        distance_cutoff=distance_cutoff,
        linear_ratio=options.linear_ratio,
        branched_node_fraction=options.branched_node_fraction,
    )
    steps = _path_steps(cluster)
    first_index, second_index, _ = maximum_pairwise_distance_pair(coordinates)
    start_endpoint = annotate_endpoint(cluster.atoms[first_index].atom, context)
    end_endpoint = annotate_endpoint(cluster.atoms[second_index].atom, context)
    endpoints = (start_endpoint, end_endpoint)

    return PathReport(
        cluster=cluster,
        chains=tuple(dict.fromkeys(residue.chain_id for residue in residues)),
        residues=residues,
        residue_path=_format_residue_path(residues),
        start_endpoint=start_endpoint,
        end_endpoint=end_endpoint,
        anchor=find_path_anchor(
            [scored_atom.atom for scored_atom in cluster.atoms],
            context,
        ),
        topology=topology,
        composition=composition,
        steps=steps,
        structural_descriptor=_structural_descriptor(
            composition=composition,
            topology=topology,
            endpoints=endpoints,
            residue_count=len(residues),
        ),
    )


def _path_residues(
    cluster: EpathCluster,
) -> tuple[tuple[PathResidue, ...], list[int]]:
    """Group a cluster's atoms into residues, in ascending residue order.

    Chains are ordered by the first residue of the path they contribute, and
    residues within a chain by residue number, which is the order the E-path
    table lists them in. Atoms keep the order they were reached in the
    connectivity graph.

    Returns:
        The residues of the path, and the residue each of the cluster's atoms
        belongs to, as indices into that tuple.
    """
    grouped: dict[tuple[str, int | None, str], list[EpathAtom]] = {}
    for scored_atom in cluster.atoms:
        grouped.setdefault(residue_key(scored_atom.atom), []).append(scored_atom.atom)

    first_residue_by_chain: dict[str, int] = {}
    for chain_id, residue_number, _ in grouped:
        number = _sort_number(residue_number)
        current = first_residue_by_chain.get(chain_id)
        if current is None or number < current:
            first_residue_by_chain[chain_id] = number

    chain_order = sorted(
        first_residue_by_chain,
        key=lambda chain_id: (first_residue_by_chain[chain_id], chain_id),
    )
    chain_rank = {chain_id: rank for rank, chain_id in enumerate(chain_order)}

    ordered_keys = sorted(
        grouped,
        key=lambda key: (chain_rank[key[0]], _sort_number(key[1]), key[2]),
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
    residue_indices = [
        position_by_key[residue_key(scored_atom.atom)] for scored_atom in cluster.atoms
    ]

    return residues, residue_indices


def _path_steps(cluster: EpathCluster) -> tuple[PathStep, ...]:
    """Return each step of a path with its length and how it connects residues."""
    atoms = [scored_atom.atom for scored_atom in cluster.atoms]
    edges = minimum_spanning_tree_edges(coordinates_for_atoms(cluster.atoms))

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


def _edge_kind(first: EpathAtom, second: EpathAtom) -> StepKind:
    if residue_key(first) == residue_key(second):
        return "intra_residue"

    if first.chain_id != second.chain_id:
        return "interchain"

    if first.residue_number is None or second.residue_number is None:
        return "tertiary"

    difference = abs(first.residue_number - second.residue_number)
    if difference == 1:
        return "peptide"

    # Residues sharing a number and separated by an insertion code are also
    # sequence neighbours.
    if difference == 0:
        return "peptide"

    return "tertiary"


def _format_residue_path(residues: Sequence[PathResidue]) -> str:
    """Return the residue-and-node listing used by the E-path pathway table."""
    if not residues:
        return ""

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
    dominant = composition.dominant_classes
    if not dominant:
        return "Unassigned cluster"

    adjective = "-".join(_CLASS_ADJECTIVES[name] for name in dominant)
    if residue_count <= 1:
        return f"Single {adjective} hotspot"

    if any(endpoint.kind == "interface" for endpoint in endpoints):
        noun = "interface motif"
    else:
        noun = _TOPOLOGY_NOUNS[topology.topology]

    enriched: tuple[ResidueClass, ...] = tuple(
        name for name in composition.enriched_classes if name not in dominant
    )
    descriptor = f"{adjective.capitalize()} {noun}"
    if not enriched:
        return descriptor

    enrichment = "/".join(_CLASS_NAMES[name] for name in enriched)
    return f"{descriptor} with {enrichment} enrichment"


def _unique_residue_atoms(atoms: Sequence[EpathAtom]) -> list[EpathAtom]:
    seen: dict[tuple[str, int | None, str], EpathAtom] = {}
    for atom in atoms:
        seen.setdefault(residue_key(atom), atom)

    return list(seen.values())


def _sort_number(residue_number: int | None) -> int:
    return residue_number if residue_number is not None else 0
