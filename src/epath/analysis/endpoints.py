"""Relate an E-path cluster to the cofactors and environments around it.

The E-path pathway table records where each path starts and ends: at a
cofactor, at a chain interface, or at the protein surface. Cofactors are
excluded from the connectivity graph, so a path never runs through one, but
a cofactor within reach of a path is what makes it a candidate electron
conduit rather than an ordinary rigid region.

Endpoints are the two atoms that define the cluster span, which the analysis
already reports as the straight-line end-to-end distance of the path. A path
can also pass a cofactor part-way along its length, so the nearest cofactor to
any of its atoms is reported separately as its anchor.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.atoms import (
    EpathAtom,
    atom_token,
    format_residue_label,
    residue_label,
)
from epath.analysis.geometry import row_norms
from rabdam.structure.models import PreparedAtom

EndpointKind = Literal[
    "cofactor",
    "cofactor_region",
    "interface",
    "surface",
    "buried",
]

# An endpoint this close to a cofactor is in direct contact with it. The
# default matches the connectivity cutoff used to build the path graph, so a
# cofactor at this range would have joined the path had cofactors been scored.
DEFAULT_COFACTOR_CONTACT_ANGSTROM = 4.0

# An endpoint within this range terminates in the cofactor's environment
# without touching it. Beyond roughly this distance a residue no longer shares
# the cofactor's first or second shell.
DEFAULT_COFACTOR_PROXIMITY_ANGSTROM = 8.0

# An endpoint this close to another chain sits in a subunit interface.
DEFAULT_INTERFACE_ANGSTROM = 4.0

# Endpoints in the least densely packed quarter of the scored population are
# reported as surface sites. Packing density is already calculated for every
# scored atom, so no separate solvent-accessibility calculation is needed.
DEFAULT_SURFACE_PERCENTILE = 25.0

# Only redox-capable groups are treated as cofactors. Every other bound group
# (lipids, detergents, sugars, buffer and cryoprotectant molecules, substrates
# and inhibitors, and redox-inert ions such as Na, K, Mg, Ca and Zn) is kept in
# the model and still contributes to packing density, but ending beside one
# does not make a path an electron conduit.
#
# A group qualifies by containing one of these transition metals, which covers
# haems, iron-sulfur clusters, copper centres and molybdenum or tungsten
# cofactors under whatever component name they were deposited.
REDOX_ACTIVE_METALS = frozenset({"V", "MN", "FE", "CO", "NI", "CU", "MO", "W"})

# Organic redox cofactors hold no such metal, so they are listed by chemical
# component code: flavins, nicotinamides, quinones, chlorophylls and
# pheophytins, metal-free and zinc porphyrins, and pterins.
ORGANIC_REDOX_COFACTORS = frozenset(
    {
        "BCB",
        "BCL",
        "BPB",
        "BPH",
        "CHL",
        "CLA",
        "DCQ",
        "F42",
        "FAD",
        "FDA",
        "FMN",
        "H4B",
        "MD1",
        "MGD",
        "MQ7",
        "MQ8",
        "MQ9",
        "MTE",
        "NAD",
        "NAI",
        "NAP",
        "NDP",
        "PHO",
        "PL9",
        "PP9",
        "PQN",
        "PQQ",
        "RBF",
        "U10",
        "UQ1",
        "UQ2",
        "UQ5",
        "UQ6",
        "UQ7",
        "UQ8",
        "UQ9",
        "ZNH",
    }
)


@dataclass(frozen=True, slots=True)
class CofactorSite:
    """One modelled redox cofactor or redox-active metal site."""

    label: str
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_count: int


@dataclass(frozen=True, slots=True)
class PathEndpoint:
    """One end of a path and the environment it terminates in."""

    label: str
    kind: EndpointKind
    atom_token: str
    residue: str
    chain_id: str
    cofactor: CofactorSite | None = None
    cofactor_distance: float | None = None

    @property
    def is_cofactor_anchored(self) -> bool:
        """Return whether this endpoint terminates at or beside a cofactor."""
        return self.kind in ("cofactor", "cofactor_region")


@dataclass(frozen=True, slots=True)
class PathAnchor:
    """The cofactor nearest to any atom of a path, when one is within range."""

    cofactor: CofactorSite
    distance: float
    atom_token: str
    residue: str

    @property
    def label(self) -> str:
        """Return the label of the anchoring cofactor site."""
        return self.cofactor.label


@dataclass(frozen=True, slots=True)
class EndpointContext:
    """Structural context used to label path endpoints."""

    sites: tuple[CofactorSite, ...]
    cofactor_coordinates: NDArray[np.float64]
    cofactor_site_indices: NDArray[np.int_]
    protein_coordinates: NDArray[np.float64]
    protein_chain_indices: NDArray[np.int_]
    chain_ids: tuple[str, ...]
    surface_packing_density: float
    cofactor_contact_distance: float = DEFAULT_COFACTOR_CONTACT_ANGSTROM
    cofactor_proximity_distance: float = DEFAULT_COFACTOR_PROXIMITY_ANGSTROM
    interface_distance: float = DEFAULT_INTERFACE_ANGSTROM


def build_endpoint_context(
    *,
    scored_atoms: Sequence[EpathAtom],
    cleaned_atoms: Sequence[PreparedAtom] | None = None,
    cofactor_contact_distance: float = DEFAULT_COFACTOR_CONTACT_ANGSTROM,
    cofactor_proximity_distance: float = DEFAULT_COFACTOR_PROXIMITY_ANGSTROM,
    interface_distance: float = DEFAULT_INTERFACE_ANGSTROM,
    surface_percentile: float = DEFAULT_SURFACE_PERCENTILE,
) -> EndpointContext:
    """Collect the cofactor, chain, and packing context for endpoint labelling.

    Args:
        scored_atoms: Every atom that received a BDamage score, used for the
            chain map and the surface packing-density threshold.
        cleaned_atoms: The complete cleaned model. Cofactor sites are taken
            from here, because cofactors are never scored. Endpoints are
            labelled without cofactor context when this is omitted.
        cofactor_contact_distance: Contact range for a cofactor endpoint.
        cofactor_proximity_distance: Range for a cofactor-region endpoint.
        interface_distance: Range at which another chain marks an interface.
        surface_percentile: Packing-density percentile below which an endpoint
            is reported as a surface site.

    Returns:
        The context needed to label any endpoint of this model.
    """
    sites, cofactor_coordinates, site_indices = _cofactor_sites(cleaned_atoms)
    protein_coordinates, chain_indices, chain_ids = _protein_chain_map(
        scored_atoms,
        cleaned_atoms,
    )

    return EndpointContext(
        sites=sites,
        cofactor_coordinates=cofactor_coordinates,
        cofactor_site_indices=site_indices,
        protein_coordinates=protein_coordinates,
        protein_chain_indices=chain_indices,
        chain_ids=chain_ids,
        surface_packing_density=_surface_threshold(
            scored_atoms,
            percentile=surface_percentile,
        ),
        cofactor_contact_distance=cofactor_contact_distance,
        cofactor_proximity_distance=cofactor_proximity_distance,
        interface_distance=interface_distance,
    )


def annotate_endpoint(atom: EpathAtom, context: EndpointContext) -> PathEndpoint:
    """Label one endpoint atom by the environment it terminates in.

    Environments are tested in order of mechanistic weight: contact with a
    cofactor, then the environment of a nearby cofactor, then a subunit
    interface, and finally the surface or interior of the chain itself.
    """
    point = np.asarray((atom.x, atom.y, atom.z), dtype=float)
    site, distance = _nearest_cofactor(point, context)
    if site is not None and distance is not None:
        if distance <= context.cofactor_contact_distance:
            return _endpoint(
                atom,
                label=site.label,
                kind="cofactor",
                cofactor=site,
                cofactor_distance=distance,
            )

        if distance <= context.cofactor_proximity_distance:
            return _endpoint(
                atom,
                label=f"{site.label} region",
                kind="cofactor_region",
                cofactor=site,
                cofactor_distance=distance,
            )

    if _is_interface(point, atom.chain_id, context):
        return _endpoint(atom, label="Interface region", kind="interface")

    if atom.packing_density <= context.surface_packing_density:
        return _endpoint(atom, label="Surface region", kind="surface")

    return _endpoint(atom, label="Buried region", kind="buried")


def find_path_anchor(
    atoms: Sequence[EpathAtom],
    context: EndpointContext,
) -> PathAnchor | None:
    """Return the cofactor nearest to any atom of a path, if one is in range.

    The two endpoints only describe where a path stops. A long or branched path
    can touch a cofactor part-way along its length while both of its ends lie
    elsewhere, so every atom is tested, against the same range that places an
    endpoint in a cofactor's environment.

    Args:
        atoms: The atoms of one path.
        context: Cofactor context built by `build_endpoint_context`.

    Returns:
        The nearest cofactor and the path atom closest to it, or None when the
        model has no cofactor within the proximity range of the path.
    """
    if context.cofactor_coordinates.size == 0:
        return None

    best_index = -1
    best_site_index = -1
    best_distance = float("inf")
    for index, atom in enumerate(atoms):
        point = np.asarray((atom.x, atom.y, atom.z), dtype=float)
        distances = row_norms(context.cofactor_coordinates - point)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance < best_distance:
            best_index = index
            best_site_index = int(context.cofactor_site_indices[nearest])
            best_distance = distance

    if best_distance > context.cofactor_proximity_distance:
        return None

    anchor_atom = atoms[best_index]
    return PathAnchor(
        cofactor=context.sites[best_site_index],
        distance=best_distance,
        atom_token=atom_token(anchor_atom),
        residue=residue_label(anchor_atom),
    )


def _endpoint(
    atom: EpathAtom,
    *,
    label: str,
    kind: EndpointKind,
    cofactor: CofactorSite | None = None,
    cofactor_distance: float | None = None,
) -> PathEndpoint:
    return PathEndpoint(
        label=label,
        kind=kind,
        atom_token=atom_token(atom),
        residue=residue_label(atom),
        chain_id=atom.chain_id,
        cofactor=cofactor,
        cofactor_distance=cofactor_distance,
    )


def _nearest_cofactor(
    point: NDArray[np.float64],
    context: EndpointContext,
) -> tuple[CofactorSite | None, float | None]:
    if context.cofactor_coordinates.size == 0:
        return None, None

    differences = context.cofactor_coordinates - point
    distances = row_norms(differences)
    nearest = int(np.argmin(distances))
    distance = float(distances[nearest])
    if distance > context.cofactor_proximity_distance:
        return None, None

    return context.sites[int(context.cofactor_site_indices[nearest])], distance


def _is_interface(
    point: NDArray[np.float64],
    chain_id: str,
    context: EndpointContext,
) -> bool:
    if context.protein_coordinates.size == 0:
        return False

    chain_index = context.chain_ids.index(chain_id)
    differences = context.protein_coordinates - point
    distances = row_norms(differences)
    within_range = distances <= context.interface_distance
    if not bool(np.any(within_range)):
        return False

    return bool(np.any(context.protein_chain_indices[within_range] != chain_index))


def _cofactor_sites(
    cleaned_atoms: Sequence[PreparedAtom] | None,
) -> tuple[tuple[CofactorSite, ...], NDArray[np.float64], NDArray[np.int_]]:
    empty_coordinates = np.zeros((0, 3), dtype=float)
    empty_indices = np.zeros(0, dtype=int)
    if not cleaned_atoms:
        return (), empty_coordinates, empty_indices

    bound_groups: dict[tuple[str, str, int | None, str], list[PreparedAtom]] = {}
    for atom in cleaned_atoms:
        if not _is_bound_group_atom(atom):
            continue

        record = atom.record
        key = (
            record.chain_id,
            record.residue_name.strip().upper(),
            record.residue_number,
            record.insertion_code.strip(),
        )
        bound_groups.setdefault(key, []).append(atom)

    grouped = {
        key: atoms
        for key, atoms in bound_groups.items()
        if _is_redox_cofactor(key[1], atoms)
    }
    if not grouped:
        return (), empty_coordinates, empty_indices

    ordered_keys = sorted(
        grouped,
        key=lambda key: (key[0], key[2] if key[2] is not None else 0, key[3]),
    )
    labels = _site_labels(ordered_keys)

    sites: list[CofactorSite] = []
    coordinates: list[tuple[float, float, float]] = []
    site_indices: list[int] = []
    for site_index, key in enumerate(ordered_keys):
        chain_id, residue_name, residue_number, insertion_code = key
        atoms = grouped[key]
        sites.append(
            CofactorSite(
                label=labels[site_index],
                chain_id=chain_id,
                residue_name=residue_name,
                residue_number=residue_number,
                insertion_code=insertion_code,
                atom_count=len(atoms),
            )
        )
        for atom in atoms:
            record = atom.record
            coordinates.append((record.x, record.y, record.z))
            site_indices.append(site_index)

    return (
        tuple(sites),
        np.asarray(coordinates, dtype=float),
        np.asarray(site_indices, dtype=int),
    )


def _site_labels(
    ordered_keys: Sequence[tuple[str, str, int | None, str]],
) -> list[str]:
    name_counts: dict[str, int] = {}
    for key in ordered_keys:
        name_counts[key[1]] = name_counts.get(key[1], 0) + 1

    seen: dict[str, int] = {}
    labels: list[str] = []
    for key in ordered_keys:
        residue_name = key[1]
        if name_counts[residue_name] == 1:
            labels.append(residue_name)
            continue

        seen[residue_name] = seen.get(residue_name, 0) + 1
        labels.append(f"{residue_name}-{seen[residue_name]}")

    return labels


def _is_bound_group_atom(atom: PreparedAtom) -> bool:
    return not (atom.is_protein or atom.is_solvent or atom.is_nucleic_acid)


def _is_redox_cofactor(residue_name: str, atoms: Sequence[PreparedAtom]) -> bool:
    if residue_name in ORGANIC_REDOX_COFACTORS:
        return True

    return any(
        atom.record.element.strip().upper() in REDOX_ACTIVE_METALS for atom in atoms
    )


def _protein_chain_map(
    scored_atoms: Sequence[EpathAtom],
    cleaned_atoms: Sequence[PreparedAtom] | None,
) -> tuple[NDArray[np.float64], NDArray[np.int_], tuple[str, ...]]:
    coordinates: list[tuple[float, float, float]] = []
    chain_indices: list[int] = []
    chain_ids: list[str] = []
    index_by_chain: dict[str, int] = {}

    def chain_index(chain_id: str) -> int:
        if chain_id not in index_by_chain:
            index_by_chain[chain_id] = len(chain_ids)
            chain_ids.append(chain_id)
        return index_by_chain[chain_id]

    if cleaned_atoms:
        for prepared_atom in cleaned_atoms:
            if not prepared_atom.is_protein:
                continue

            record = prepared_atom.record
            coordinates.append((record.x, record.y, record.z))
            chain_indices.append(chain_index(record.chain_id))
    else:
        for atom in scored_atoms:
            coordinates.append((atom.x, atom.y, atom.z))
            chain_indices.append(chain_index(atom.chain_id))

    for atom in scored_atoms:
        chain_index(atom.chain_id)

    return (
        np.asarray(coordinates, dtype=float).reshape(-1, 3),
        np.asarray(chain_indices, dtype=int),
        tuple(chain_ids),
    )


def _surface_threshold(
    scored_atoms: Iterable[EpathAtom],
    *,
    percentile: float,
) -> float:
    packing_densities = [atom.packing_density for atom in scored_atoms]
    return float(np.percentile(packing_densities, percentile))


def cofactor_site_summary(sites: Sequence[CofactorSite]) -> str:
    """Return a short human-readable list of the modelled cofactor sites."""
    if not sites:
        return "none detected"

    return ", ".join(
        f"{site.label} ("
        f"{format_residue_label(site.residue_name, site.residue_number, site.insertion_code)}"
        f" chain {site.chain_id})"
        for site in sites
    )
