"""Trim translated crystal atoms to the BDamage neighbour region."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rabdam.crystal.symmetry import SymmetryExpandedStructure
from rabdam.crystal.translate import (
    TranslatedAtom,
    UnitCellTranslationVectors,
    translation_vector_for_offsets,
    unit_cell_translation_vectors,
)
from rabdam.structure.models import PreparedAtom, PreparedStructure


class CrystalTrimError(ValueError):
    """Report invalid bounds or an unusable trimmed crystal block."""


@dataclass(frozen=True, slots=True)
class CartesianBounds:
    """Store inclusive Cartesian bounds in angstroms."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True, slots=True)
class TrimmedCrystalBlock:
    """Store object-based translated atoms retained inside Cartesian bounds."""

    atoms: tuple[TranslatedAtom, ...]
    reference_bounds: CartesianBounds
    trim_bounds: CartesianBounds
    padding: float
    original_atom_count: int

    @property
    def atom_count(self) -> int:
        """Return the number of retained atoms."""
        return len(self.atoms)


@dataclass(frozen=True, slots=True)
class ArrayTrimmedCrystalBlock:
    """Store a vectorized trimmed crystal block.

    Coordinates have shape `(n, 3)` in angstroms. Source indices and identity
    flags have shape `(n,)`, and integer translation offsets have shape `(n, 3)`.
    Rows across all four arrays describe the same translated atom.
    """

    coordinates: NDArray[np.float64]
    source_atom_indices: NDArray[np.int64]
    is_identity_symmetry_operation: NDArray[np.bool_]
    translation_offsets: NDArray[np.int64]
    reference_bounds: CartesianBounds
    trim_bounds: CartesianBounds
    padding: float
    original_atom_count: int
    translation_vectors: UnitCellTranslationVectors
    translation_range: int
    source_unit_cell_atom_count: int

    @property
    def atom_count(self) -> int:
        """Return the number of retained coordinate rows."""
        return int(self.coordinates.shape[0])


TrimmedNeighbourBlock = TrimmedCrystalBlock | ArrayTrimmedCrystalBlock


def trim_expanded_unit_cell_for_bdamage(
    *,
    expanded_structure: SymmetryExpandedStructure,
    prepared_structure: PreparedStructure,
    padding: float,
    translation_range: int = 1,
) -> ArrayTrimmedCrystalBlock:
    """Translate and trim an expanded cell around selected BDamage atoms."""
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
    """Translate and retain atom images near reference atoms.

    The reference bounding box is expanded by `padding` angstroms on every
    Cartesian face. Translation and trimming are fused to avoid materializing
    every atom copy.

    Args:
        expanded_structure: Symmetry-expanded source unit cell.
        reference_atoms: Atoms whose Cartesian extent defines the trim region.
        padding: Nonnegative extension of each bounding-box face in angstroms.
        translation_range: Maximum integer unit-cell offset along each axis.

    Returns:
        Row-aligned coordinate, source-index, symmetry, and translation arrays
        for retained atom images.

    Raises:
        CrystalTrimError: If an input is empty, a numeric option is invalid, or
            trimming retains no atoms.
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
        raise CrystalTrimError("Cannot trim an empty symmetry-expanded atom list.")

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

    coordinate_chunks: list[NDArray[np.float64]] = []
    source_index_chunks: list[NDArray[np.int64]] = []
    identity_flag_chunks: list[NDArray[np.bool_]] = []
    offset_chunks: list[NDArray[np.int64]] = []

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
                    np.asarray(
                        np.repeat(
                            np.asarray(
                                ((a_offset, b_offset, c_offset),),
                                dtype=np.int64,
                            ),
                            retained_count,
                            axis=0,
                        ),
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


def bounds_from_prepared_atoms(
    atoms: Iterable[PreparedAtom],
) -> CartesianBounds:
    """Compute inclusive Cartesian bounds for prepared atoms.

    Raises:
        CrystalTrimError: If no atoms are provided.
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
    """Expand Cartesian bounds by a nonnegative padding in angstroms."""
    if padding < 0:
        raise CrystalTrimError(f"padding must be non-negative, got {padding!r}.")

    # Cartesian padding cannot exclude a Euclidean neighbour because that atom
    # must be within the cutoff along every individual axis.
    return CartesianBounds(
        x_min=float(bounds.x_min - padding),
        x_max=float(bounds.x_max + padding),
        y_min=float(bounds.y_min - padding),
        y_max=float(bounds.y_max + padding),
        z_min=float(bounds.z_min - padding),
        z_max=float(bounds.z_max + padding),
    )
