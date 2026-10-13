"""Calculate packing-density neighbour counts for selected atoms."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rabdam.crystal.trim import ArrayTrimmedCrystalBlock
from rabdam.structure.models import PreparedAtom, PreparedStructure
from spatial import radius_neighbour_counts


class PackingDensityError(ValueError):
    """Report invalid input to a packing-density calculation."""


@dataclass(frozen=True, slots=True)
class PackingDensityAtomResult:
    """Store the neighbour count and source identity for one selected atom."""

    packing_density_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    neighbour_count: int


@dataclass(frozen=True, slots=True)
class PackingDensityResult:
    """Store per-atom counts and summary values for a neighbour search."""

    atom_results: tuple[PackingDensityAtomResult, ...]
    packing_density_threshold: float
    selected_atom_count: int
    neighbour_atom_count: int


def calculate_bdamage_packing_density(
    *,
    prepared_structure: PreparedStructure,
    trimmed_block: ArrayTrimmedCrystalBlock,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """Calculate packing density for a prepared structure and trimmed block."""
    return calculate_packing_density_from_arrays(
        selected_atoms=prepared_structure.selected_atoms,
        neighbour_coordinates=trimmed_block.coordinates,
        source_atom_indices=trimmed_block.source_atom_indices,
        is_identity_symmetry_operation=trimmed_block.is_identity_symmetry_operation,
        translation_offsets=trimmed_block.translation_offsets,
        packing_density_threshold=packing_density_threshold,
    )


def calculate_packing_density_from_arrays(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_coordinates: NDArray[np.float64],
    source_atom_indices: NDArray[np.int64],
    is_identity_symmetry_operation: NDArray[np.bool_],
    translation_offsets: NDArray[np.int64],
    packing_density_threshold: float,
) -> PackingDensityResult:
    """Count neighbours from row-aligned crystal arrays.

    Args:
        selected_atoms: Atoms that receive packing-density values.
        neighbour_coordinates: Cartesian coordinates with shape `(n, 3)` in
            angstroms.
        source_atom_indices: Source indices with shape `(n,)`.
        is_identity_symmetry_operation: Identity-operation flags with shape
            `(n,)`.
        translation_offsets: Integer unit-cell offsets with shape `(n, 3)`.
        packing_density_threshold: Exclusive neighbour radius in angstroms.

    Returns:
        Counts aligned with selected-atom order, excluding each atom's verified
        central-cell identity copy.

    Raises:
        PackingDensityError: If the arrays are misaligned or a selected atom's
            identity copy cannot be verified.
    """
    selected_atom_tuple = tuple(selected_atoms)

    _validate_neighbour_arrays(
        coordinates=neighbour_coordinates,
        source_atom_indices=source_atom_indices,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        translation_offsets=translation_offsets,
    )

    return _calculate_packing_density_with_ckdtree(
        selected_atoms=selected_atom_tuple,
        neighbour_coordinates=neighbour_coordinates,
        source_atom_indices=source_atom_indices,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        translation_offsets=translation_offsets,
        packing_density_threshold=packing_density_threshold,
    )


def _calculate_packing_density_with_ckdtree(
    *,
    selected_atoms: tuple[PreparedAtom, ...],
    neighbour_coordinates: NDArray[np.float64],
    source_atom_indices: NDArray[np.int64],
    is_identity_symmetry_operation: NDArray[np.bool_],
    translation_offsets: NDArray[np.int64],
    packing_density_threshold: float,
) -> PackingDensityResult:
    threshold_squared = packing_density_threshold**2
    selected_coordinates = _selected_atom_coordinates(selected_atoms)
    _validate_self_copies_are_counted(
        selected_atoms=selected_atoms,
        selected_coordinates=selected_coordinates,
        neighbour_coordinates=neighbour_coordinates,
        source_atom_indices=source_atom_indices,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        translation_offsets=translation_offsets,
        threshold_squared=threshold_squared,
    )

    # A KD-tree keeps the neighbour search near-linear in the size of the
    # neighbour cloud, which is far too large for pairwise distance checks.
    # The radius query counts points at exactly the radius, but packing density
    # counts only atoms strictly inside the threshold, so drop one ULP.
    strict_radius = float(np.nextafter(packing_density_threshold, 0.0))
    # The identity copy is returned by the radius query but is not one of the
    # atom's neighbours.
    neighbour_counts = (
        radius_neighbour_counts(
            neighbour_coordinates,
            selected_coordinates,
            strict_radius,
        )
        - 1
    )

    return PackingDensityResult(
        atom_results=tuple(
            PackingDensityAtomResult(
                packing_density_atom_index=selected_atom_index,
                source_atom_index=selected_atom.record.source_atom_index,
                atom_serial=selected_atom.record.atom_serial,
                neighbour_count=int(neighbour_count),
            )
            for selected_atom_index, (selected_atom, neighbour_count) in enumerate(
                zip(selected_atoms, neighbour_counts, strict=True),
                start=1,
            )
        ),
        packing_density_threshold=packing_density_threshold,
        selected_atom_count=len(selected_atoms),
        neighbour_atom_count=int(neighbour_coordinates.shape[0]),
    )


def _selected_atom_coordinates(
    selected_atoms: tuple[PreparedAtom, ...],
) -> NDArray[np.float64]:
    return np.asarray(
        [(atom.record.x, atom.record.y, atom.record.z) for atom in selected_atoms],
        dtype=np.float64,
    )


def _validate_self_copies_are_counted(
    *,
    selected_atoms: tuple[PreparedAtom, ...],
    selected_coordinates: NDArray[np.float64],
    neighbour_coordinates: NDArray[np.float64],
    source_atom_indices: NDArray[np.int64],
    is_identity_symmetry_operation: NDArray[np.bool_],
    translation_offsets: NDArray[np.int64],
    threshold_squared: float,
) -> None:
    """Verify each selected atom has a counted central-cell identity image."""
    central_cell_identity_indices = np.flatnonzero(
        is_identity_symmetry_operation & np.all(translation_offsets == 0, axis=1)
    )
    candidate_indices_by_source_atom: dict[int, list[int]] = {}
    for neighbour_index in central_cell_identity_indices:
        source_atom_index = int(source_atom_indices[neighbour_index])
        candidate_indices_by_source_atom.setdefault(source_atom_index, []).append(
            int(neighbour_index)
        )

    for selected_atom_index, selected_atom in enumerate(selected_atoms):
        candidate_indices = candidate_indices_by_source_atom.get(
            selected_atom.record.source_atom_index,
            [],
        )
        if not candidate_indices:
            raise PackingDensityError(
                "Cannot subtract the selected atom's central-cell copy from the "
                "packing-density count because that copy was not counted. Check "
                "that the neighbour cloud contains the selected atom's central-cell "
                "image."
            )

        candidate_coordinates = neighbour_coordinates[
            np.asarray(candidate_indices, dtype=np.int64)
        ]
        coordinate_deltas = (
            selected_coordinates[selected_atom_index] - candidate_coordinates
        )
        distances_squared = np.einsum(  # pyright: ignore[reportUnknownMemberType]  # numpy stub gap
            "ij,ij->i",
            coordinate_deltas,
            coordinate_deltas,
        )
        if not bool(np.any(distances_squared < threshold_squared)):
            raise PackingDensityError(
                "Cannot subtract the selected atom's central-cell copy from the "
                "packing-density count because that copy was not counted. Check "
                "that the neighbour cloud contains the selected atom's central-cell "
                "image."
            )


def _validate_neighbour_arrays(
    *,
    coordinates: NDArray[np.float64],
    source_atom_indices: NDArray[np.int64],
    is_identity_symmetry_operation: NDArray[np.bool_],
    translation_offsets: NDArray[np.int64],
) -> None:
    """Validate row counts and shapes across neighbour-cloud arrays."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise PackingDensityError("neighbour_coordinates must have shape (n, 3).")

    neighbour_count = coordinates.shape[0]
    if source_atom_indices.shape != (neighbour_count,):
        raise PackingDensityError(
            "source_atom_indices must have shape (n,) matching coordinates."
        )

    if is_identity_symmetry_operation.shape != (neighbour_count,):
        raise PackingDensityError(
            "is_identity_symmetry_operation must have shape (n,) matching coordinates."
        )

    if translation_offsets.shape != (neighbour_count, 3):
        raise PackingDensityError(
            "translation_offsets must have shape (n, 3) matching coordinates."
        )


def packing_density_counts_as_tuple(
    result: PackingDensityResult,
) -> tuple[int, ...]:
    """Return neighbour counts in selected-atom order."""
    return tuple(atom_result.neighbour_count for atom_result in result.atom_results)
