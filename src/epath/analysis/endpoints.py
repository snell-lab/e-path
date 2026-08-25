"""Label the endpoints of an E-path cluster by their structural environment.

The published pathway table records where each path starts and ends: at a
cofactor, at a chain interface, or at the protein surface. Cofactors are
excluded from the connectivity graph, so a path never runs through one, but
the cofactor a path terminates against is what makes it a candidate electron
conduit rather than an ordinary rigid region.

Endpoints are the two atoms that define the cluster span, which the analysis
already reports as the straight-line end-to-end distance of the path.
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

# Common crystallization and cryoprotection additives. These are kept in the
# model and still contribute to packing density, but they are not treated as
# cofactors when labelling an endpoint, because terminating beside a glycerol
# does not make a path an electron conduit.
CRYSTALLIZATION_ADDITIVES = frozenset(
    {
        "ACT",
        "ACY",
        "BME",
        "BR",
        "CAC",
        "CIT",
        "CL",
        "CS",
        "DMS",
        "DTT",
        "EDO",
        "EPE",
        "FLC",
        "FMT",
        "GOL",
        "IMD",
        "IOD",
        "IPA",
        "K",
        "MES",
        "MPD",
        "MRD",
        "NA",
        "NH4",
        "NO3",
        "OXL",
        "P6G",
        "PE4",
        "PEG",
        "PG4",
        "PGE",
        "PO4",
        "RB",
        "SO4",
        "TAR",
        "TLA",
        "TRS",
        "UNX",
    }
)


@dataclass(frozen=True, slots=True)
class CofactorSite:
    """One modelled cofactor, ligand, or metal site."""

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

    try:
        chain_index = context.chain_ids.index(chain_id)
    except ValueError:
        return False

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

    grouped: dict[tuple[str, str, int | None, str], list[PreparedAtom]] = {}
    for atom in cleaned_atoms:
        if not _is_cofactor_atom(atom):
            continue

        record = atom.record
        key = (
            record.chain_id,
            record.residue_name.strip().upper(),
            record.residue_number,
            record.insertion_code.strip(),
        )
        grouped.setdefault(key, []).append(atom)

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
            coordinates.append((float(record.x), float(record.y), float(record.z)))
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


def _is_cofactor_atom(atom: PreparedAtom) -> bool:
    if atom.is_protein or atom.is_solvent or atom.is_nucleic_acid or atom.is_hydrogen:
        return False

    return atom.record.residue_name.strip().upper() not in CRYSTALLIZATION_ADDITIVES


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
            coordinates.append((float(record.x), float(record.y), float(record.z)))
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
    packing_densities = np.asarray(
        [atom.packing_density for atom in scored_atoms],
        dtype=float,
    )
    if packing_densities.size == 0:
        return 0.0

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
