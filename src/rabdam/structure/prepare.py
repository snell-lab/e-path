"""Clean, classify, and select the atoms of a parsed structure.

`rabdam.pipeline` runs this on the records `xtal.reader` returns, before any
crystal symmetry is applied. The result types are in `rabdam.structure.models`.
"""

import math
from functools import cache

import gemmi

from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
    StructurePreparationError,
)
from xtal.reader import AtomRecord, StructureData

# Modified nucleotides that Gemmi does not tabulate as nucleic acid.
NUCLEIC_ACID_COMPONENT_OVERRIDES = frozenset(
    {
        "5MC",
        "YYG",
    }
)


def prepare_structure(structure_data: StructureData) -> PreparedStructure:
    """Clean, classify, and select the atoms that receive BDamage scores.

    Each removed atom is counted under the first of these reasons that applies,
    in this order: hydrogen or deuterium, non-finite coordinates, occupancy
    outside (0, 1], and a B-factor that is not positive and finite. The
    highest-occupancy conformer is then kept at each atom site.

    `cleaned_atoms` holds every atom that is left, in input order, and
    `selected_atoms` the protein atoms among them, which are the ones scored.
    A `StructurePreparationError` is raised when no atom is left or none of
    those left is protein.
    """
    hydrogens = 0
    non_finite_coordinates = 0
    invalid_occupancy = 0
    non_positive_b_factor = 0
    usable_atoms: list[AtomRecord] = []

    # Hydrogens and unusable B-factor records cannot contribute to BDamage.
    for atom in structure_data.atoms:
        if atom.element.upper() in {"H", "D"}:
            hydrogens += 1
        elif not (
            math.isfinite(atom.x) and math.isfinite(atom.y) and math.isfinite(atom.z)
        ):
            non_finite_coordinates += 1
        elif not 0.0 < atom.occupancy <= 1.0:
            invalid_occupancy += 1
        # Written as a negation so that a NaN B-factor is removed too. The
        # upper bound removes infinity, which Gemmi returns for a PDB B-factor
        # field that overflows.
        elif not 0.0 < atom.b_factor < math.inf:
            non_positive_b_factor += 1
        else:
            usable_atoms.append(atom)

    retained_atoms = _highest_occupancy_conformers(usable_atoms)
    cleaned_atoms = tuple(classify_atom(atom) for atom in retained_atoms)
    # BDamage compares B-factors between atoms in similar packing environments,
    # and protein, nucleic-acid, and heteroatom B-factors relate to packing
    # differently, so only protein atoms are scored. Modified amino acids are
    # often HETATM records but belong to the polypeptide, so they are kept.
    # Unselected atoms still contribute to the packing density of their
    # neighbours.
    selected_atoms = tuple(atom for atom in cleaned_atoms if atom.is_protein)

    if not cleaned_atoms:
        raise StructurePreparationError(
            "No atoms remain after structure preparation. Every atom was a "
            "hydrogen or had non-finite coordinates, an occupancy outside "
            "(0, 1], or a B-factor that is not positive and finite."
        )

    if not selected_atoms:
        raise StructurePreparationError(
            "No protein atoms were selected for BDamage calculation. Only "
            "protein atoms are scored, and no atom left after cleaning is in "
            "a recognized amino-acid residue."
        )

    return PreparedStructure(
        source_atoms=structure_data.atoms,
        cleaned_atoms=cleaned_atoms,
        selected_atoms=selected_atoms,
        metadata=structure_data.metadata,
        removed=RemovedAtomCounts(
            hydrogens=hydrogens,
            non_finite_coordinates=non_finite_coordinates,
            invalid_occupancy=invalid_occupancy,
            non_positive_b_factor=non_positive_b_factor,
            alternate_conformers=len(usable_atoms) - len(retained_atoms),
        ),
    )


def _highest_occupancy_conformers(
    atoms: list[AtomRecord],
) -> tuple[AtomRecord, ...]:
    """Keep the highest-occupancy conformer at each atom site, in source order."""
    # Sites are resolved independently, even when that mixes altloc labels
    # within one residue. A tie keeps the first record.
    best_by_site: dict[tuple[object, ...], tuple[int, AtomRecord]] = {}
    for index, atom in enumerate(atoms):
        site = (
            atom.chain_id,
            atom.residue_name.strip().upper(),
            atom.residue_number,
            atom.insertion_code.strip(),
            atom.atom_name.strip().upper(),
        )
        best = best_by_site.get(site)
        if best is None or atom.occupancy > best[1].occupancy:
            best_by_site[site] = (index, atom)

    kept_indexes = {index for index, _ in best_by_site.values()}
    return tuple(atom for index, atom in enumerate(atoms) if index in kept_indexes)


def classify_atom(atom: AtomRecord) -> PreparedAtom:
    """Label an atom as protein, nucleic acid, or solvent by its residue name."""
    is_protein, is_nucleic_acid, is_solvent = _classify_component(
        atom.residue_name.strip().upper()
    )
    return PreparedAtom(
        record=atom,
        is_protein=is_protein,
        is_nucleic_acid=is_nucleic_acid,
        is_solvent=is_solvent,
    )


@cache
def _classify_component(component_name: str) -> tuple[bool, bool, bool]:
    residue = gemmi.find_tabulated_residue(component_name)

    return (
        residue.is_amino_acid(),
        component_name in NUCLEIC_ACID_COMPONENT_OVERRIDES or residue.is_nucleic_acid(),
        residue.is_water(),
    )
