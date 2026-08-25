"""Translate symmetry-expanded atoms into neighbouring unit cells."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from rabdam.crystal.symmetry import (
    SymmetryExpandedAtom,
    SymmetryExpandedStructure,
)
from xtal.cell import UnitCellParameters


class CrystalTranslationError(ValueError):
    """Report invalid unit-cell translation input or output."""


@dataclass(frozen=True, slots=True)
class CartesianVector:
    """Store a Cartesian displacement vector in angstroms."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class UnitCellTranslationVectors:
    """Store Cartesian displacement vectors for the three unit-cell axes."""

    a: CartesianVector
    b: CartesianVector
    c: CartesianVector


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
    )


def translate_atoms(
    *,
    atoms: Iterable[SymmetryExpandedAtom],
    unit_cell: UnitCellParameters,
    translation_range: int = 1,
) -> TranslatedCrystalBlock:
    """Generate translated copies of symmetry-expanded atoms.

    Args:
        atoms: Symmetry-expanded atoms in their source unit cell.
        unit_cell: Unit-cell lengths in angstroms and angles in degrees.
        translation_range: Maximum integer offset along each cell axis. A value
            of one generates 27 cells.

    Returns:
        Atom copies ordered by cell offsets and then input atom order.

    Raises:
        CrystalTranslationError: If there are no atoms, the range is negative,
            the unit cell is invalid, or the generated count is inconsistent.
    """
    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise CrystalTranslationError(
            "Cannot translate an empty symmetry-expanded atom list."
        )

    if translation_range < 0:
        raise CrystalTranslationError(
            f"translation_range must be non-negative, got {translation_range!r}."
        )

    vectors = unit_cell_translation_vectors(unit_cell)
    translated_atoms: list[TranslatedAtom] = []

    for a_offset in range(-translation_range, translation_range + 1):
        for b_offset in range(-translation_range, translation_range + 1):
            for c_offset in range(-translation_range, translation_range + 1):
                shift = translation_vector_for_offsets(
                    vectors,
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
                            x=float(atom.x + shift.x),
                            y=float(atom.y + shift.y),
                            z=float(atom.z + shift.z),
                        )
                    )

    expected_count = len(atom_tuple) * (2 * translation_range + 1) ** 3
    if len(translated_atoms) != expected_count:
        raise CrystalTranslationError(
            "Failed to generate the expected translated crystal block: "
            f"expected {expected_count} atoms, got {len(translated_atoms)}."
        )

    return TranslatedCrystalBlock(
        atoms=tuple(translated_atoms),
        unit_cell=unit_cell,
        translation_vectors=vectors,
        translation_range=translation_range,
        source_unit_cell_atom_count=len(atom_tuple),
    )


def unit_cell_translation_vectors(
    unit_cell: UnitCellParameters,
) -> UnitCellTranslationVectors:
    """Compute Cartesian unit-cell vectors from lengths and angles.

    Raises:
        CrystalTranslationError: If the unit cell fails validation, or if its
            angles do not describe a valid cell geometry.
    """
    validate_unit_cell_for_translation(unit_cell)

    # Fixing a along x and b in the xy plane gives the standard
    # crystallographic orthogonalization used by the reference implementation.
    alpha = math.radians(unit_cell.alpha)
    beta = math.radians(unit_cell.beta)
    gamma = math.radians(unit_cell.gamma)

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    volume_factor_squared = (
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )

    if volume_factor_squared <= 0:
        raise CrystalTranslationError(
            f"Invalid unit-cell geometry for translation: {unit_cell!r}."
        )

    volume_factor = math.sqrt(volume_factor_squared)

    return UnitCellTranslationVectors(
        a=CartesianVector(unit_cell.a, 0.0, 0.0),
        b=CartesianVector(
            unit_cell.b * cos_gamma,
            unit_cell.b * sin_gamma,
            0.0,
        ),
        c=CartesianVector(
            unit_cell.c * cos_beta,
            unit_cell.c * ((cos_alpha - cos_beta * cos_gamma) / sin_gamma),
            unit_cell.c * (volume_factor / sin_gamma),
        ),
    )


def validate_unit_cell_for_translation(unit_cell: UnitCellParameters) -> None:
    """Validate unit-cell geometry for Cartesian translation.

    Raises:
        CrystalTranslationError: If lengths or angles are outside their valid
            ranges or gamma makes the standard basis singular.
    """
    if unit_cell.a <= 0 or unit_cell.b <= 0 or unit_cell.c <= 0:
        raise CrystalTranslationError(f"Invalid unit-cell lengths: {unit_cell!r}.")

    if not all(
        0 < angle < 180 for angle in (unit_cell.alpha, unit_cell.beta, unit_cell.gamma)
    ):
        raise CrystalTranslationError(f"Invalid unit-cell angles: {unit_cell!r}.")

    if math.isclose(math.sin(math.radians(unit_cell.gamma)), 0.0, abs_tol=1e-12):
        raise CrystalTranslationError(
            f"Invalid unit-cell gamma angle for translation: {unit_cell!r}."
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
