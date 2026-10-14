"""Describe each E-path cluster as a pathway.

This module turns the geometric clusters produced by connectivity analysis
into the pathway description the E-path analysis reports. A description gives
the residues and protected atoms along the path, the cofactors within reach of
it, and its residue chemistry.

These are measurements, reported as the basis for interpreting a path with
the protein's known chemistry in view.

`epath.analysis.clustering` supplies the clusters and their geometry. The
chemistry comes from `epath.analysis.residues`, and the nearby cofactors from
`epath.analysis.cofactors`. This module gathers them into one report for each
cluster and adds the residue listing and the steps. `epath.pipeline` calls
`analyze_paths` once for a model, and `epath.outputs` and `epath.cli.report`
write the reports out.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.clustering import EpathCluster, coordinates_for_atoms
from epath.analysis.cofactors import (
    CofactorContext,
    CofactorSite,
    NearbyCofactor,
    build_cofactor_context,
    find_nearby_cofactors,
)
from epath.analysis.geometry import minimum_spanning_tree_edges
from epath.analysis.labels import (
    atom_token,
    format_residue_label,
    residue_key,
    residue_label,
)
from epath.analysis.residues import ResidueComposition, analyze_composition
from rabdam.structure.models import PreparedAtom
from spatial import radius_neighbour_indices
from xtal.reader import AtomRecord

StepKind = Literal["intra_residue", "peptide", "tertiary", "interchain"]

# Two residues are joined by a peptide bond when the carbonyl C of one lies
# within this distance of the amide N of the other. The bond is 1.33 A long,
# and a chain break leaves the two atoms well over this apart.
PEPTIDE_BOND_MAXIMUM_ANGSTROM = 2.0

# The chain, residue number and insertion code of a residue, as `residue_key`
# gives them.
type ResidueKey = tuple[str, int | None, str]


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

    A path that only follows the backbone is a rigid stretch of chain, and one
    of the same size whose steps jump between residues that are far apart in
    sequence is not, though geometry alone reports the two alike. Steps are
    counted on the minimum spanning tree, so they are the same steps the
    reported length and mean step describe.

    A peptide step joins two residues that a modelled peptide bond links, so
    a chain break is never one, however the residues on either side are
    numbered.
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

    `nearby_cofactors` are the cofactors within range of any atom of the path,
    the nearest first, and empty when none is. `steps` are the edges of the
    minimum spanning tree.
    """

    cluster: EpathCluster
    chains: tuple[str, ...]
    residues: tuple[PathResidue, ...]
    residue_path: str
    nearby_cofactors: tuple[NearbyCofactor, ...]
    composition: ResidueComposition
    steps: tuple[PathStep, ...]

    @property
    def is_near_cofactor(self) -> bool:
        """Return whether any cofactor lies within range of this path."""
        return bool(self.nearby_cofactors)

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
    def nearby_cofactor_summary(self) -> str:
        """Return each nearby cofactor and its distance, or "none"."""
        if not self.nearby_cofactors:
            return "none"

        return ", ".join(
            f"{cofactor.label} at {cofactor.distance:.1f} A"
            for cofactor in self.nearby_cofactors
        )


@dataclass(frozen=True, slots=True)
class PathAnalysisResult:
    """Described pathways for one model and the cofactors found in it."""

    reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]


def analyze_paths(
    clusters: Sequence[EpathCluster],
    *,
    cleaned_atoms: Sequence[PreparedAtom],
) -> PathAnalysisResult:
    """Describe every reported cluster as a pathway.

    The reports are in the order of `clusters`.
    """
    context = build_cofactor_context(cleaned_atoms)
    peptide_links = _peptide_links(cleaned_atoms)

    return PathAnalysisResult(
        reports=tuple(
            _build_path_report(cluster, context=context, peptide_links=peptide_links)
            for cluster in clusters
        ),
        cofactor_sites=context.sites,
    )


def _peptide_links(
    cleaned_atoms: Sequence[PreparedAtom],
) -> frozenset[frozenset[ResidueKey]]:
    """Return each pair of residues of the model joined by a peptide bond.

    Each pair is the keys of its two residues. A residue whose backbone C or N
    is not modelled is joined to nothing on that side.
    """
    backbone = [
        (atom.record.atom_name.strip().upper(), atom.record)
        for atom in cleaned_atoms
        if atom.is_protein and atom.record.atom_name.strip().upper() in {"C", "N"}
    ]
    if not backbone:
        return frozenset()

    neighbours = radius_neighbour_indices(
        np.asarray([(record.x, record.y, record.z) for _, record in backbone]),
        PEPTIDE_BOND_MAXIMUM_ANGSTROM,
    )
    links: set[frozenset[ResidueKey]] = set()
    for index, (name, record) in enumerate(backbone):
        if name != "C":
            continue

        for neighbour_index in neighbours[index]:
            neighbour_name, neighbour = backbone[neighbour_index]
            if neighbour_name == "N" and residue_key(neighbour) != residue_key(record):
                links.add(frozenset((residue_key(record), residue_key(neighbour))))

    return frozenset(links)


def _build_path_report(
    cluster: EpathCluster,
    *,
    context: CofactorContext,
    peptide_links: frozenset[frozenset[ResidueKey]],
) -> PathReport:
    """Describe one cluster as a pathway.

    `context` and `peptide_links` describe the whole model, not the cluster,
    so `analyze_paths` builds them once and passes them to every cluster.
    """
    atoms = [scored_atom.record for scored_atom in cluster.atoms]
    coordinates = coordinates_for_atoms(cluster.atoms)
    residues = _path_residues(atoms)
    composition = analyze_composition(residue.residue_name for residue in residues)

    return PathReport(
        cluster=cluster,
        chains=tuple(dict.fromkeys(residue.chain_id for residue in residues)),
        residues=residues,
        residue_path=_format_residue_path(residues),
        nearby_cofactors=find_nearby_cofactors(atoms, context),
        composition=composition,
        steps=_path_steps(atoms, coordinates, peptide_links),
    )


def _path_residues(atoms: Sequence[AtomRecord]) -> tuple[PathResidue, ...]:
    """Group a path's atoms into residues, in ascending residue order.

    Chains are ordered by the lowest residue number of the path they
    contribute, with ties broken by chain identifier, and residues within a
    chain by residue number and then insertion code. This is the order the
    E-path table lists them in. The atoms of a residue keep the order they
    have in `atoms`.
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

    return tuple(
        PathResidue(
            chain_id=key[0],
            residue_name=grouped[key][0].residue_name.strip().upper(),
            residue_number=key[1],
            insertion_code=key[2],
            atom_names=tuple(atom.atom_name.strip() for atom in grouped[key]),
        )
        for key in ordered_keys
    )


def _path_steps(
    atoms: Sequence[AtomRecord],
    coordinates: NDArray[np.float64],
    peptide_links: frozenset[frozenset[ResidueKey]],
) -> tuple[PathStep, ...]:
    """Return each step of a path with its length and how it connects residues.

    The steps are numbered from one, in the order the spanning tree was built.
    """
    edges = minimum_spanning_tree_edges(coordinates)

    return tuple(
        PathStep(
            step_index=step_index,
            length=edge.length,
            kind=_edge_kind(
                atoms[edge.first_index], atoms[edge.second_index], peptide_links
            ),
            first_atom=atom_token(atoms[edge.first_index]),
            first_residue=residue_label(atoms[edge.first_index]),
            second_atom=atom_token(atoms[edge.second_index]),
            second_residue=residue_label(atoms[edge.second_index]),
        )
        for step_index, edge in enumerate(edges, start=1)
    )


def _edge_kind(
    first: AtomRecord,
    second: AtomRecord,
    peptide_links: frozenset[frozenset[ResidueKey]],
) -> StepKind:
    """Return how a step between two atoms connects their residues.

    `peptide_links` holds the residue pairs of the model joined by a peptide
    bond, as `_peptide_links` returns them. Residue numbers play no part.
    """
    first_key = residue_key(first)
    second_key = residue_key(second)
    if first_key == second_key:
        return "intra_residue"

    if first.chain_id != second.chain_id:
        return "interchain"

    if frozenset((first_key, second_key)) in peptide_links:
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


def _sort_number(residue_number: int | None) -> int:
    """Return the number a residue sorts by, which is 0 when it has none."""
    return residue_number if residue_number is not None else 0
