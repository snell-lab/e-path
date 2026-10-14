"""Find a model's redox cofactors and those within reach of each E-path cluster.

Cofactors are excluded from the connectivity graph, so a path never runs
through one, but a path can still lie within reach of one or more, and every
cofactor within range of a path is reported as near it.

`pathways` builds one `CofactorContext` per model and finds the cofactors near
every path against it. The cofactor sites come from the cleaned model that
`rabdam.structure` prepared. Distances are in angstroms.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from epath.analysis.geometry import row_norms
from epath.analysis.labels import atom_token, format_residue_label, residue_label
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord

# A cofactor within this range of a path atom is reported as near the path.
# The range is a judgement of what is near enough to be worth inspecting. It
# covers the residues around the cofactor, out to a little beyond its second
# shell, while leaving most paths with no cofactor near them.
COFACTOR_PROXIMITY_ANGSTROM = 8.0

# Only redox-capable groups are treated as cofactors. Every other bound group
# (lipids, detergents, sugars, buffer and cryoprotectant molecules, substrates
# and inhibitors, and redox-inert ions such as Na, K, Mg, Ca and Zn) is kept in
# the model and still contributes to packing density, but is not reported as
# a cofactor a path lies near.
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

# These hold a redox-active metal but are not redox cofactors: the free
# vanadate, molybdate and tungstate oxyanions, which bind as phosphate
# analogues or as transported substrates, and cobalt hexammine, a
# crystallization additive.
NON_COFACTOR_METAL_COMPONENTS = frozenset({"MOO", "NCO", "VO4", "WO4"})

# Nickel and cobalt are redox-active inside a cofactor, as in F430, cobalamin
# or a nickel-iron cluster. A lone cobalt ion never is, because it stands in
# for another metal or, as in nitrile hydratase, acts as a Lewis acid. A lone
# nickel ion is redox-active only when sulfur holds it, as the cysteine
# thiolates of nickel superoxide dismutase and nickel-iron hydrogenase and the
# sulfides of the carbon monoxide dehydrogenase cluster do. Nickel held by
# histidine, carboxylates or water, as in urease, glyoxalase I or a His-tag,
# is not. A sulfur atom within the first distance of the nickel counts as a
# ligand, which covers Ni-S bonds of 2.2 to 2.3 A with room for modelling
# error, and the second is how many such ligands the nickel needs.
NICKEL_SULFUR_LIGAND_ANGSTROM = 2.7
NICKEL_MINIMUM_SULFUR_LIGANDS = 2


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
class NearbyCofactor:
    """One cofactor within range of a path.

    `label` is the label of the cofactor site. `distance` is measured between
    the closest pair of atoms, one of the site and one of the path, and
    `atom_token` and `residue` name the path atom of that pair.
    """

    label: str
    distance: float
    atom_token: str
    residue: str


@dataclass(frozen=True, slots=True)
class CofactorContext:
    """The cofactor sites of a model, used to find the cofactors near each path.

    `cofactor_coordinates` holds every atom of every site, with shape
    `(n, 3)`, and `cofactor_site_indices` gives, for each of those rows, the
    position of its site in `sites`.
    """

    sites: tuple[CofactorSite, ...]
    cofactor_coordinates: NDArray[np.float64]
    cofactor_site_indices: NDArray[np.int_]


def build_cofactor_context(cleaned_atoms: Sequence[PreparedAtom]) -> CofactorContext:
    """Collect the cofactor sites of the complete cleaned model.

    The sites are taken from the cleaned model because cofactors are never
    scored.
    """
    sites, cofactor_coordinates, site_indices = _cofactor_sites(cleaned_atoms)

    return CofactorContext(
        sites=sites,
        cofactor_coordinates=cofactor_coordinates,
        cofactor_site_indices=site_indices,
    )


def find_nearby_cofactors(
    atoms: Sequence[AtomRecord],
    context: CofactorContext,
) -> tuple[NearbyCofactor, ...]:
    """Return every cofactor within range of a path, the nearest first.

    A cofactor is in range when one of its atoms lies within the proximity
    range of an atom of the path, the range itself included. Each one names the
    path atom closest to that cofactor, and the first such atom in path
    order when several are equally close. Cofactors equally near are in site
    order. The result is empty when no cofactor is in range.
    """
    if not context.sites or not atoms:
        return ()

    # One row per path atom, one column per site.
    distances = np.vstack(
        [
            _site_distances(np.asarray((atom.x, atom.y, atom.z), dtype=float), context)
            for atom in atoms
        ]
    )
    nearby: list[NearbyCofactor] = []
    for site_index, site in enumerate(context.sites):
        nearest = int(np.argmin(distances[:, site_index]))
        distance = float(distances[nearest, site_index])
        if distance > COFACTOR_PROXIMITY_ANGSTROM:
            continue

        nearby.append(
            NearbyCofactor(
                label=site.label,
                distance=distance,
                atom_token=atom_token(atoms[nearest]),
                residue=residue_label(atoms[nearest]),
            )
        )

    # The sort is stable, so cofactors equally near keep their site order.
    return tuple(sorted(nearby, key=lambda cofactor: cofactor.distance))


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

    The sites are ordered by chain, residue number and insertion code, with
    the chains in the order they first appear in the model.
    """
    chain_order: dict[str, int] = {}
    sulfur_coordinates: list[tuple[float, float, float]] = []
    bound_groups: dict[tuple[str, str, int | None, str], list[AtomRecord]] = {}
    for atom in cleaned_atoms:
        chain_order.setdefault(atom.record.chain_id, len(chain_order))
        if atom.record.element.strip().upper() == "S":
            sulfur_coordinates.append((atom.record.x, atom.record.y, atom.record.z))
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

    # The reshape keeps the `(n, 3)` shape when the model has no sulfur.
    sulfur_array = np.asarray(sulfur_coordinates, dtype=float).reshape(-1, 3)
    # A residue with no number sorts as residue 0 of its chain.
    ordered_keys = sorted(
        (
            key
            for key, records in bound_groups.items()
            if _is_redox_cofactor(
                key[1],
                records,
                sulfur_coordinates=sulfur_array,
            )
        ),
        key=lambda key: (
            chain_order[key[0]],
            key[2] if key[2] is not None else 0,
            key[3],
        ),
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


def _is_redox_cofactor(
    residue_name: str,
    records: Sequence[AtomRecord],
    *,
    sulfur_coordinates: NDArray[np.float64],
) -> bool:
    """Return whether a bound group is a redox cofactor.

    `sulfur_coordinates` holds every sulfur atom of the model, with shape
    `(n, 3)`, which decides whether a lone nickel ion is sulfur-ligated.
    """
    if residue_name in ORGANIC_REDOX_COFACTORS:
        return True

    if residue_name in NON_COFACTOR_METAL_COMPONENTS:
        return False

    elements = [record.element.strip().upper() for record in records]
    if len(elements) == 1 and elements[0] == "CO":
        return False

    if len(elements) == 1 and elements[0] == "NI":
        nickel = records[0]
        distances = row_norms(
            sulfur_coordinates - np.asarray((nickel.x, nickel.y, nickel.z))
        )
        ligands = int(np.count_nonzero(distances <= NICKEL_SULFUR_LIGAND_ANGSTROM))
        return ligands >= NICKEL_MINIMUM_SULFUR_LIGANDS

    return any(element in REDOX_ACTIVE_METALS for element in elements)


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


def _site_distances(
    point: NDArray[np.float64],
    context: CofactorContext,
) -> NDArray[np.float64]:
    """Return the distance from a point to the nearest atom of each site."""
    distances = np.full(len(context.sites), np.inf)
    np.minimum.at(
        distances,
        context.cofactor_site_indices,
        row_norms(context.cofactor_coordinates - point),
    )
    return distances
