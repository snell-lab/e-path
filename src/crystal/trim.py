"""
Trim translated crystal atoms to the local neighbour box used for BDamage.

After symmetry expansion and unit-cell translation, the crystal environment
contains atoms from a 3x3x3 block of neighbouring unit cells. BDamage only needs
the atoms close enough to the selected asymmetric-unit atoms to contribute to
packing-density calculations.

This module defines an axis-aligned Cartesian box around the selected atoms.
The box is calculated from the selected atoms' minimum and maximum x, y, and z
coordinates, then padded on every side by the packing-density threshold (PDT).

The workflow is:

    selected asymmetric-unit atoms -> Cartesian bounds -> bounds +/- PDT
    translated 3x3x3 block -> atoms inside padded bounds only

The trimmed block is the local neighbour cloud passed to the packing-density
stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from crystal.symmetry import SymmetryExpandedStructure
from crystal.translate import (
    TranslatedAtom,
    TranslatedCrystalBlock,
    UnitCellTranslationVectors,
    translation_vector_for_offsets,
    unit_cell_translation_vectors,
)
from structure.models import PreparedAtom, PreparedStructure


class CrystalTrimError(ValueError):
    """Raised when RABDAM cannot trim a translated crystal block."""


@dataclass(frozen=True, slots=True)
class CartesianBounds:
    """
    Axis-aligned Cartesian coordinate bounds in Angstroms.

    x_min, x_max, y_min, y_max, z_min, z_max are calculated independently from
    the reference asymmetric-unit atoms, then padded by the packing-density
    threshold before filtering the translated crystal block.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True, slots=True)
class TrimmedCrystalBlock:
    """
    Translated crystal block after neighbour-box trimming.

    atoms:
        Translated atoms whose Cartesian coordinates lie inside the padded
        reference-atom bounds.

    reference_bounds:
        Unpadded Cartesian bounds of the reference asymmetric-unit atoms.

    trim_bounds:
        Padded Cartesian bounds used to retain translated atoms.

    padding:
        Padding added to each side of reference_bounds, normally the
        packing-density threshold in Angstroms.

    original_atom_count:
        Number of atoms in the translated block before trimming.
    """

    atoms: tuple[TranslatedAtom, ...]
    reference_bounds: CartesianBounds
    trim_bounds: CartesianBounds
    padding: float
    original_atom_count: int

    @property
    def atom_count(self) -> int:
        """Number of retained translated atoms."""

        return len(self.atoms)


@dataclass(frozen=True, slots=True)
class ArrayTrimmedCrystalBlock:
    """
    NumPy-backed translated crystal block after neighbour-box trimming.

    The full translated 3x3x3 block is never materialized. Coordinates and the
    metadata needed by packing-density self-copy correction are retained only
    for atoms inside the padded reference bounds.
    """

    coordinates: np.ndarray
    source_atom_indices: np.ndarray
    is_identity_symmetry_operation: np.ndarray
    translation_offsets: np.ndarray
    reference_bounds: CartesianBounds
    trim_bounds: CartesianBounds
    padding: float
    original_atom_count: int
    translation_vectors: UnitCellTranslationVectors
    translation_range: int
    source_unit_cell_atom_count: int

    @property
    def atom_count(self) -> int:
        """Number of retained translated atoms."""

        return int(self.coordinates.shape[0])


TrimmedNeighbourBlock = TrimmedCrystalBlock | ArrayTrimmedCrystalBlock


def trim_translated_block_for_bdamage(
    *,
    translated_block: TranslatedCrystalBlock,
    prepared_structure: PreparedStructure,
    padding: float,
) -> TrimmedCrystalBlock:
    """
    Trim a translated crystal block for BDamage packing-density calculation.

    This uses prepared_structure.selected_atoms as the reference atoms.
    """

    return trim_translated_block_to_reference_atoms(
        translated_block=translated_block,
        reference_atoms=prepared_structure.selected_atoms,
        padding=padding,
    )


def trim_expanded_unit_cell_for_bdamage(
    *,
    expanded_structure: SymmetryExpandedStructure,
    prepared_structure: PreparedStructure,
    padding: float,
    translation_range: int = 1,
) -> ArrayTrimmedCrystalBlock:
    """
    Fused translation and trimming for BDamage packing-density calculation.

    This uses prepared_structure.selected_atoms as the reference atoms and
    avoids constructing TranslatedAtom objects for atoms outside the trim box.
    """

    return trim_expanded_unit_cell_to_reference_atoms(
        expanded_structure=expanded_structure,
        reference_atoms=prepared_structure.selected_atoms,
        padding=padding,
        translation_range=translation_range,
    )


def trim_expanded_unit_cell_to_reference_atoms(
    *,
    expanded_structure: SymmetryExpandedStructure,
    reference_atoms: Iterable[PreparedAtom],
    padding: float,
    translation_range: int = 1,
) -> ArrayTrimmedCrystalBlock:
    """
    Translate symmetry-expanded atoms and keep only atoms inside padded bounds.
    """

    if padding < 0:
        raise CrystalTrimError(f"padding must be non-negative, got {padding!r}.")

    if type(translation_range) is not int or translation_range < 0:
        raise CrystalTrimError(
            "translation_range must be a non-negative integer, "
            f"got {translation_range!r}."
        )

    expanded_atoms = tuple(expanded_structure.atoms)
    if not expanded_atoms:
        raise CrystalTrimError(
            "Cannot trim an empty symmetry-expanded atom list."
        )

    reference_bounds = bounds_from_prepared_atoms(reference_atoms)
    trim_bounds = expand_bounds(reference_bounds, padding)
    vectors = unit_cell_translation_vectors(expanded_structure.unit_cell)

    base_coordinates = np.asarray(
        [(atom.x, atom.y, atom.z) for atom in expanded_atoms],
        dtype=np.float64,
    )
    source_atom_indices = np.asarray(
        [atom.source_atom_index for atom in expanded_atoms],
        dtype=np.int64,
    )
    identity_flags = np.asarray(
        [atom.is_identity_symmetry_operation for atom in expanded_atoms],
        dtype=np.bool_,
    )

    coordinate_chunks: list[np.ndarray] = []
    source_index_chunks: list[np.ndarray] = []
    identity_flag_chunks: list[np.ndarray] = []
    offset_chunks: list[np.ndarray] = []

    for a_offset in range(-translation_range, translation_range + 1):
        for b_offset in range(-translation_range, translation_range + 1):
            for c_offset in range(-translation_range, translation_range + 1):
                shift = translation_vector_for_offsets(
                    vectors,
                    a_offset=a_offset,
                    b_offset=b_offset,
                    c_offset=c_offset,
                )
                shifted_coordinates = base_coordinates + np.asarray(
                    (shift.x, shift.y, shift.z),
                    dtype=np.float64,
                )
                inside_bounds = (
                    (trim_bounds.x_min <= shifted_coordinates[:, 0])
                    & (shifted_coordinates[:, 0] <= trim_bounds.x_max)
                    & (trim_bounds.y_min <= shifted_coordinates[:, 1])
                    & (shifted_coordinates[:, 1] <= trim_bounds.y_max)
                    & (trim_bounds.z_min <= shifted_coordinates[:, 2])
                    & (shifted_coordinates[:, 2] <= trim_bounds.z_max)
                )

                retained_count = int(np.count_nonzero(inside_bounds))
                if retained_count == 0:
                    continue

                coordinate_chunks.append(shifted_coordinates[inside_bounds])
                source_index_chunks.append(source_atom_indices[inside_bounds])
                identity_flag_chunks.append(identity_flags[inside_bounds])
                offset_chunks.append(
                    np.full(
                        (retained_count, 3),
                        (a_offset, b_offset, c_offset),
                        dtype=np.int64,
                    )
                )

    if not coordinate_chunks:
        raise CrystalTrimError(
            "Trimmed crystal block contains no atoms. Check unit-cell translation "
            "and packing-density threshold."
        )

    translated_cell_count = (2 * translation_range + 1) ** 3

    return ArrayTrimmedCrystalBlock(
        coordinates=np.concatenate(coordinate_chunks, axis=0),
        source_atom_indices=np.concatenate(source_index_chunks, axis=0),
        is_identity_symmetry_operation=np.concatenate(identity_flag_chunks, axis=0),
        translation_offsets=np.concatenate(offset_chunks, axis=0),
        reference_bounds=reference_bounds,
        trim_bounds=trim_bounds,
        padding=float(padding),
        original_atom_count=len(expanded_atoms) * translated_cell_count,
        translation_vectors=vectors,
        translation_range=translation_range,
        source_unit_cell_atom_count=len(expanded_atoms),
    )


def trim_translated_block_to_reference_atoms(
    *,
    translated_block: TranslatedCrystalBlock,
    reference_atoms: Iterable[PreparedAtom],
    padding: float,
) -> TrimmedCrystalBlock:
    """
    Keep translated atoms inside reference-atom bounds padded by padding.
    """

    if padding < 0:
        raise CrystalTrimError(f"padding must be non-negative, got {padding!r}.")

    translated_atom_tuple = tuple(translated_block.atoms)
    if not translated_atom_tuple:
        raise CrystalTrimError("Cannot trim an empty translated crystal block.")

    reference_bounds = bounds_from_prepared_atoms(reference_atoms)
    trim_bounds = expand_bounds(reference_bounds, padding)

    trimmed_atoms = tuple(
        atom
        for atom in translated_atom_tuple
        if translated_atom_is_inside_bounds(atom, trim_bounds)
    )

    if not trimmed_atoms:
        raise CrystalTrimError(
            "Trimmed crystal block contains no atoms. Check unit-cell translation "
            "and packing-density threshold."
        )

    return TrimmedCrystalBlock(
        atoms=trimmed_atoms,
        reference_bounds=reference_bounds,
        trim_bounds=trim_bounds,
        padding=float(padding),
        original_atom_count=len(translated_atom_tuple),
    )


def bounds_from_prepared_atoms(
    atoms: Iterable[PreparedAtom],
) -> CartesianBounds:
    """
    Return min/max Cartesian bounds for prepared asymmetric-unit atoms.
    """

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise CrystalTrimError("Cannot calculate bounds for an empty atom list.")

    x_values = tuple(atom.record.x for atom in atom_tuple)
    y_values = tuple(atom.record.y for atom in atom_tuple)
    z_values = tuple(atom.record.z for atom in atom_tuple)

    return CartesianBounds(
        x_min=float(min(x_values)),
        x_max=float(max(x_values)),
        y_min=float(min(y_values)),
        y_max=float(max(y_values)),
        z_min=float(min(z_values)),
        z_max=float(max(z_values)),
    )


def expand_bounds(bounds: CartesianBounds, padding: float) -> CartesianBounds:
    """
    Pad Cartesian bounds by padding Angstroms in every direction.
    """

    if padding < 0:
        raise CrystalTrimError(f"padding must be non-negative, got {padding!r}.")

    return CartesianBounds(
        x_min=float(bounds.x_min - padding),
        x_max=float(bounds.x_max + padding),
        y_min=float(bounds.y_min - padding),
        y_max=float(bounds.y_max + padding),
        z_min=float(bounds.z_min - padding),
        z_max=float(bounds.z_max + padding),
    )


def translated_atom_is_inside_bounds(
    atom: TranslatedAtom,
    bounds: CartesianBounds,
) -> bool:
    """
    Return True when a translated atom lies inside inclusive Cartesian bounds.
    """

    return (
        bounds.x_min <= atom.x <= bounds.x_max
        and bounds.y_min <= atom.y <= bounds.y_max
        and bounds.z_min <= atom.z <= bounds.z_max
    )


def trimmed_coordinates_as_tuples(
    trimmed_block: TrimmedNeighbourBlock,
) -> tuple[tuple[float, float, float], ...]:
    """Return only xyz coordinates from a trimmed crystal block."""

    if isinstance(trimmed_block, ArrayTrimmedCrystalBlock):
        return tuple(tuple(row) for row in trimmed_block.coordinates.tolist())

    return tuple((atom.x, atom.y, atom.z) for atom in trimmed_block.atoms)
