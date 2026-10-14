"""Relate an E-path cluster to the cofactors and environments around it.

The E-path pathway table records where each path starts and ends: at a
cofactor, at a chain interface, or by how densely packed its environment is.
Cofactors are excluded from the connectivity graph, so a path never runs
through one, but a path can still lie within reach of one, and that proximity
is reported.

Endpoints are the two atoms that define the cluster span, which the analysis
already reports as the straight-line end-to-end distance of the path. A path
can also pass a cofactor part-way along its length, so the nearest cofactor to
any of its atoms is reported separately as its anchor.

`pathways` builds one `EndpointContext` per model and labels the endpoints and
the anchor of every path against it. The scored atoms come from `selection`,
and the cofactor sites from the cleaned model that `rabdam.structure`
prepared. Distances are in angstroms.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from epath.analysis.geometry import row_norms
from epath.analysis.labels import atom_token, format_residue_label, residue_label
from epath.analysis.selection import EpathScoredAtom
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord

EndpointKind = Literal[
    "cofactor",
    "cofactor_region",
    "interface",
    "low_packing",
    "typical_packing",
]

# An endpoint this close to a cofactor is in direct contact with it.
COFACTOR_CONTACT_ANGSTROM = 4.0

# An endpoint within this range terminates in the cofactor's environment
# without touching it. Beyond roughly this distance a residue no longer shares
# the cofactor's first or second shell.
COFACTOR_PROXIMITY_ANGSTROM = 8.0

# An endpoint this close to another chain sits in a subunit interface.
INTERFACE_ANGSTROM = 4.0

# Endpoints in the least densely packed quarter of the scored population are
# reported as low-packing sites, and the rest as typical-packing sites. Packing
# density is the count BDamage already calculated for every scored atom.
LOW_PACKING_PERCENTILE = 25.0

# Only redox-capable groups are treated as cofactors. Every other bound group
# (lipids, detergents, sugars, buffer and cryoprotectant molecules, substrates
# and inhibitors, and redox-inert ions such as Na, K, Mg, Ca and Zn) is kept in
# the model and still contributes to packing density, but is not reported as
# a cofactor a path ends beside or lies near.
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
        "CL0",
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
    """One modelled redox cofactor or redox-active metal site.

    A site is one residue of the cleaned model. `label` is its residue name,
    followed by a number, as in SF4-2, when the model holds several sites of
    that name. `residue_name` is in upper case.
    """

    label: str
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str


@dataclass(frozen=True, slots=True)
class PathEndpoint:
    """One end of a path and the environment it terminates in.

    `label` is the reported name of the environment. It is the label of the
    cofactor site for a "cofactor" endpoint, that label followed by " region"
    for a "cofactor_region" endpoint, and "Interface region",
    "Low-packing region" or "Typical-packing region" for the others. `atom_token` and `residue` name the
    endpoint atom, in the forms of `labels.atom_token` and
    `labels.residue_label`.
    """

    label: str
    kind: EndpointKind
    atom_token: str
    residue: str


@dataclass(frozen=True, slots=True)
class PathAnchor:
    """The cofactor nearest to any atom of a path, when one is within range.

    `label` is the label of the cofactor site. `distance` is measured between
    the closest pair of atoms, one of the site and one of the path, and
    `atom_token` and `residue` name the path atom of that pair.
    """

    label: str
    distance: float
    atom_token: str
    residue: str


@dataclass(frozen=True, slots=True)
class EndpointContext:
    """Structural context used to label path endpoints.

    `cofactor_coordinates` holds every atom of every site, with shape
    `(n, 3)`, and `cofactor_site_indices` gives, for each of those rows, the
    position of its site in `sites`. `protein_coordinates` and
    `protein_chain_ids` are aligned in the same way, one row per protein atom.
    `low_packing_density` is the packing density at or below which an
    endpoint is a low-packing site.
    """

    sites: tuple[CofactorSite, ...]
    cofactor_coordinates: NDArray[np.float64]
    cofactor_site_indices: NDArray[np.int_]
    protein_coordinates: NDArray[np.float64]
    protein_chain_ids: NDArray[np.str_]
    low_packing_density: float


def build_endpoint_context(
    *,
    scored_atoms: Sequence[EpathScoredAtom],
    cleaned_atoms: Sequence[PreparedAtom],
) -> EndpointContext:
    """Collect the cofactor, chain, and packing context for endpoint labelling.

    `scored_atoms` is every atom that received a BDamage score. The
    low-packing threshold is the `LOW_PACKING_PERCENTILE` percentile of their packing
    densities, so there must be at least one.

    `cleaned_atoms` is the complete cleaned model. The cofactor sites are taken
    from it, because cofactors are never scored, and its protein atoms are the
    ones that can mark an interface.
    """
    sites, cofactor_coordinates, site_indices = _cofactor_sites(cleaned_atoms)
    protein_coordinates, protein_chain_ids = _protein_chains(cleaned_atoms)

    return EndpointContext(
        sites=sites,
        cofactor_coordinates=cofactor_coordinates,
        cofactor_site_indices=site_indices,
        protein_coordinates=protein_coordinates,
        protein_chain_ids=protein_chain_ids,
        low_packing_density=float(
            np.percentile(
                [atom.bdamage_result.packing_density for atom in scored_atoms],
                LOW_PACKING_PERCENTILE,
            )
        ),
    )


def annotate_endpoint(atom: EpathScoredAtom, context: EndpointContext) -> PathEndpoint:
    """Label one endpoint atom by the environment it terminates in.

    Environments are tested in order of mechanistic weight. Contact with a
    cofactor comes first, then the environment of a nearby cofactor, then a
    subunit interface, and finally how densely packed the endpoint is.

    The cofactor is the one with the atom nearest to the endpoint. An interface
    is a protein atom of another chain within the interface distance, and a
    low-packing site has a packing density no higher than the low-packing
    threshold.
    Every range includes its limit.
    """
    record = atom.record
    point = np.asarray((record.x, record.y, record.z), dtype=float)
    nearest = _nearest_cofactor(point, context)
    if nearest is not None:
        site, distance = nearest
        if distance <= COFACTOR_CONTACT_ANGSTROM:
            return _endpoint(record, label=site.label, kind="cofactor")

        return _endpoint(record, label=f"{site.label} region", kind="cofactor_region")

    if _is_interface(point, record.chain_id, context):
        return _endpoint(record, label="Interface region", kind="interface")

    if atom.bdamage_result.packing_density <= context.low_packing_density:
        return _endpoint(record, label="Low-packing region", kind="low_packing")

    return _endpoint(record, label="Typical-packing region", kind="typical_packing")


def find_path_anchor(
    atoms: Sequence[AtomRecord],
    context: EndpointContext,
) -> PathAnchor | None:
    """Return the cofactor nearest to any atom of a path, if one is in range.

    The two endpoints only describe where a path stops. A long path can
    touch a cofactor part-way along its length while both of its ends lie
    elsewhere, so every atom is tested, against the same range that places an
    endpoint in a cofactor's environment.

    The anchor names the path atom closest to the cofactor, and the first such
    atom in path order when several are equally close. None is returned when
    no cofactor atom lies within the proximity range of any atom of the path.
    """
    anchor: PathAnchor | None = None
    for atom in atoms:
        point = np.asarray((atom.x, atom.y, atom.z), dtype=float)
        nearest = _nearest_cofactor(point, context)
        if nearest is None:
            continue

        site, distance = nearest
        if anchor is None or distance < anchor.distance:
            anchor = PathAnchor(
                label=site.label,
                distance=distance,
                atom_token=atom_token(atom),
                residue=residue_label(atom),
            )

    return anchor


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


def _cofactor_sites(
    cleaned_atoms: Sequence[PreparedAtom],
) -> tuple[tuple[CofactorSite, ...], NDArray[np.float64], NDArray[np.int_]]:
    """Return the cofactor sites, their atom coordinates, and each atom's site.

    The sites are ordered by chain, residue number and insertion code.
    """
    bound_groups: dict[tuple[str, str, int | None, str], list[AtomRecord]] = {}
    for atom in cleaned_atoms:
        # A bound group is whatever is neither polymer nor solvent.
        if atom.is_protein or atom.is_nucleic_acid or atom.is_solvent:
            continue

        record = atom.record
        key = (
            record.chain_id,
            record.residue_name.strip().upper(),
            record.residue_number,
            record.insertion_code.strip(),
        )
        bound_groups.setdefault(key, []).append(record)

    # A residue with no number sorts as residue 0 of its chain.
    ordered_keys = sorted(
        (
            key
            for key, records in bound_groups.items()
            if _is_redox_cofactor(key[1], records)
        ),
        key=lambda key: (key[0], key[2] if key[2] is not None else 0, key[3]),
    )
    labels = _site_labels([key[1] for key in ordered_keys])

    sites: list[CofactorSite] = []
    coordinates: list[tuple[float, float, float]] = []
    site_indices: list[int] = []
    for site_index, key in enumerate(ordered_keys):
        chain_id, residue_name, residue_number, insertion_code = key
        sites.append(
            CofactorSite(
                label=labels[site_index],
                chain_id=chain_id,
                residue_name=residue_name,
                residue_number=residue_number,
                insertion_code=insertion_code,
            )
        )
        for record in bound_groups[key]:
            coordinates.append((record.x, record.y, record.z))
            site_indices.append(site_index)

    # The reshape keeps the `(n, 3)` shape when the model has no cofactor.
    return (
        tuple(sites),
        np.asarray(coordinates, dtype=float).reshape(-1, 3),
        np.asarray(site_indices, dtype=int),
    )


def _is_redox_cofactor(residue_name: str, records: Sequence[AtomRecord]) -> bool:
    if residue_name in ORGANIC_REDOX_COFACTORS:
        return True

    return any(
        record.element.strip().upper() in REDOX_ACTIVE_METALS for record in records
    )


def _site_labels(residue_names: Sequence[str]) -> list[str]:
    """Label each site with its residue name, numbering the names that repeat.

    A name that occurs once is used as it is. The sites of a repeated name are
    numbered from 1 in the order given, as in SF4-1 and SF4-2.
    """
    name_counts = Counter(residue_names)
    seen: Counter[str] = Counter()
    labels: list[str] = []
    for residue_name in residue_names:
        if name_counts[residue_name] == 1:
            labels.append(residue_name)
            continue

        seen[residue_name] += 1
        labels.append(f"{residue_name}-{seen[residue_name]}")

    return labels


def _protein_chains(
    cleaned_atoms: Sequence[PreparedAtom],
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    """Return the position and the chain of each protein atom of the model."""
    records = [atom.record for atom in cleaned_atoms if atom.is_protein]
    # The reshape keeps the `(n, 3)` shape when there is no protein atom.
    return (
        np.asarray(
            [(record.x, record.y, record.z) for record in records], dtype=float
        ).reshape(-1, 3),
        np.asarray([record.chain_id for record in records], dtype=str),
    )


def _nearest_cofactor(
    point: NDArray[np.float64],
    context: EndpointContext,
) -> tuple[CofactorSite, float] | None:
    """Return the site with the atom nearest to a point, and that distance.

    None is returned when the model has no cofactor, or when the nearest one
    is beyond the proximity range.
    """
    if not context.sites:
        return None

    distances = row_norms(context.cofactor_coordinates - point)
    nearest = np.argmin(distances)
    distance = float(distances[nearest])
    if distance > COFACTOR_PROXIMITY_ANGSTROM:
        return None

    return context.sites[int(context.cofactor_site_indices[nearest])], distance


def _endpoint(
    atom: AtomRecord,
    *,
    label: str,
    kind: EndpointKind,
) -> PathEndpoint:
    return PathEndpoint(
        label=label,
        kind=kind,
        atom_token=atom_token(atom),
        residue=residue_label(atom),
    )


def _is_interface(
    point: NDArray[np.float64],
    chain_id: str,
    context: EndpointContext,
) -> bool:
    """Return whether a point lies at an interface with another chain.

    It does when a protein atom of a chain other than `chain_id` is within the
    interface distance of it, the distance itself included.
    """
    distances = row_norms(context.protein_coordinates - point)
    within_range = distances <= INTERFACE_ANGSTROM
    return bool(np.any(context.protein_chain_ids[within_range] != chain_id))
