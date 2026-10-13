"""Translate symmetry-expanded atoms into neighbouring unit cells."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rabdam.crystal.symmetry import (
    SymmetryExpandedAtom,
    SymmetryExpandedStructure,
)
from xtal.cell import (
    CartesianVector,
    UnitCellParameters,
    UnitCellTranslationVectors,
)

__all__ = (
    "CartesianVector",
    "TranslatedAtom",
    "TranslatedCrystalBlock",
    "UnitCellTranslationVectors",
    "translate_atoms",
    "translate_expanded_unit_cell",
    "translation_vector_for_offsets",
)


@dataclass(frozen=True, slots=True)
class TranslatedAtom:
    """Describe one atom image in a translated unit cell."""

    translated_atom_index: int
    unit_cell_atom_index: int
    source_atom_index: int
    symmetry_operation_index: int
    is_identity_symmetry_operation: bool
    translation_a: int
    translation_b: int
    translation_c: int
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class TranslatedCrystalBlock:
    """Store symmetry-expanded atoms translated across neighbouring cells."""

    atoms: tuple[TranslatedAtom, ...]
    unit_cell: UnitCellParameters
    translation_vectors: UnitCellTranslationVectors
    translation_range: int
    source_unit_cell_atom_count: int


def translate_expanded_unit_cell(
    expanded_structure: SymmetryExpandedStructure,
    *,
    translation_range: int = 1,
) -> TranslatedCrystalBlock:
    """Translate a symmetry-expanded structure across neighbouring cells."""
    return translate_atoms(
        atoms=expanded_structure.atoms,
        unit_cell=expanded_structure.unit_cell,
        translation_range=translation_range,
        translation_vectors=expanded_structure.translation_vectors,
    )


def translate_atoms(
    *,
    atoms: Iterable[SymmetryExpandedAtom],
    unit_cell: UnitCellParameters,
    translation_vectors: UnitCellTranslationVectors,
    translation_range: int = 1,
) -> TranslatedCrystalBlock:
    """Generate translated copies of symmetry-expanded atoms.

    Args:
        atoms: Symmetry-expanded atoms in their source unit cell.
        unit_cell: Unit-cell lengths in angstroms and angles in degrees.
        translation_range: Maximum integer offset along each cell axis. A value
            of one generates 27 cells.
        translation_vectors: Cell axes in the Cartesian frame the atoms use.

    Returns:
        Atom copies ordered by cell offsets and then input atom order.
    """
    atom_tuple = tuple(atoms)
    translated_atoms: list[TranslatedAtom] = []

    for a_offset in range(-translation_range, translation_range + 1):
        for b_offset in range(-translation_range, translation_range + 1):
            for c_offset in range(-translation_range, translation_range + 1):
                shift = translation_vector_for_offsets(
                    translation_vectors,
                    a_offset=a_offset,
                    b_offset=b_offset,
                    c_offset=c_offset,
                )

                for atom in atom_tuple:
                    translated_atoms.append(
                        TranslatedAtom(
                            translated_atom_index=len(translated_atoms) + 1,
                            unit_cell_atom_index=atom.unit_cell_atom_index,
                            source_atom_index=atom.source_atom_index,
                            symmetry_operation_index=atom.symmetry_operation_index,
                            is_identity_symmetry_operation=(
                                atom.is_identity_symmetry_operation
                            ),
                            translation_a=a_offset,
                            translation_b=b_offset,
                            translation_c=c_offset,
                            x=atom.x + shift.x,
                            y=atom.y + shift.y,
                            z=atom.z + shift.z,
                        )
                    )

    return TranslatedCrystalBlock(
        atoms=tuple(translated_atoms),
        unit_cell=unit_cell,
        translation_vectors=translation_vectors,
        translation_range=translation_range,
        source_unit_cell_atom_count=len(atom_tuple),
    )


def translation_vector_for_offsets(
    vectors: UnitCellTranslationVectors,
    *,
    a_offset: int,
    b_offset: int,
    c_offset: int,
) -> CartesianVector:
    """Combine integer unit-cell offsets into a Cartesian displacement."""
    return CartesianVector(
        x=(a_offset * vectors.a.x + b_offset * vectors.b.x + c_offset * vectors.c.x),
        y=(a_offset * vectors.a.y + b_offset * vectors.b.y + c_offset * vectors.c.y),
        z=(a_offset * vectors.a.z + b_offset * vectors.b.z + c_offset * vectors.c.z),
    )
