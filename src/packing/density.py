"""
Packing-density calculation for BDamage.

The trimmed crystal block contains the local neighbour cloud around the selected
asymmetric-unit atoms. For each selected atom, packing density is the number of
trimmed crystal atoms whose Cartesian distance from that selected atom is less
than the packing-density threshold, minus one to remove the selected atom's
central-cell copy.

This module performs the exact distance-counting step after the broader crystal
block has already been reduced by crystal.trim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
import math

import numpy as np

from crystal.translate import TranslatedAtom
from crystal.trim import ArrayTrimmedCrystalBlock, TrimmedNeighbourBlock
from structure.models import PreparedAtom, PreparedStructure


class PackingDensityError(ValueError):
    """Raised when RABDAM cannot calculate packing density."""


@dataclass(frozen=True, slots=True)
class PackingDensityAtomResult:
    """
    Packing-density result for one selected asymmetric-unit atom.

    packing_density_atom_index:
        One-based position of this atom in the packing-density result list.

    source_atom_index:
        Zero-based reader index of the selected asymmetric-unit atom.

    atom_serial:
        Atom serial number from the input structure, when available.

    neighbour_count:
        Number of trimmed crystal atoms within the packing-density threshold of
        this selected atom, after subtracting one for the central-cell copy of
        the atom itself.
    """

    packing_density_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    neighbour_count: int


@dataclass(frozen=True, slots=True)
class PackingDensityResult:
    """
    Packing-density counts for the selected BDamage atoms.

    atom_results:
        One result per selected asymmetric-unit atom, in selected-atom order.

    packing_density_threshold:
        Distance cutoff in Angstroms used for neighbour counting.

    selected_atom_count:
        Number of selected asymmetric-unit atoms that were scored.

    neighbour_atom_count:
        Number of trimmed crystal atoms searched for each selected atom.
    """

    atom_results: tuple[PackingDensityAtomResult, ...]
    packing_density_threshold: float
    selected_atom_count: int
    neighbour_atom_count: int


def calculate_bdamage_packing_density(
    *,
    prepared_structure: PreparedStructure,
    trimmed_block: TrimmedNeighbourBlock,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Calculate packing density for the BDamage-selected atoms.

    This convenience wrapper uses prepared_structure.selected_atoms as the atoms
    that receive packing-density counts and the trimmed block as the local
    crystal neighbour cloud.
    """

    if isinstance(trimmed_block, ArrayTrimmedCrystalBlock):
        return calculate_packing_density_from_arrays(
            selected_atoms=prepared_structure.selected_atoms,
            neighbour_coordinates=trimmed_block.coordinates,
            source_atom_indices=trimmed_block.source_atom_indices,
            is_identity_symmetry_operation=(
                trimmed_block.is_identity_symmetry_operation
            ),
            translation_offsets=trimmed_block.translation_offsets,
            packing_density_threshold=packing_density_threshold,
        )

    return calculate_packing_density(
        selected_atoms=prepared_structure.selected_atoms,
        neighbour_atoms=trimmed_block.atoms,
        packing_density_threshold=packing_density_threshold,
    )


def calculate_packing_density(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_atoms: Iterable[TranslatedAtom],
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbour atoms within packing_density_threshold of each selected atom.
    """

    if (
        not math.isfinite(packing_density_threshold)
        or packing_density_threshold <= 0
    ):
        raise PackingDensityError(
            "packing_density_threshold must be a finite positive number, "
            f"got {packing_density_threshold!r}."
        )

    selected_atom_tuple = tuple(selected_atoms)
    if not selected_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density for an empty selected-atom list."
        )

    neighbour_atom_tuple = tuple(neighbour_atoms)
    if not neighbour_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density with an empty neighbour-atom list."
        )

    return _calculate_packing_density_with_ckdtree(
        selected_atoms=selected_atom_tuple,
        neighbour_coordinates=np.asarray(
            [(atom.x, atom.y, atom.z) for atom in neighbour_atom_tuple],
            dtype=np.float64,
        ),
        source_atom_indices=np.asarray(
            [atom.source_atom_index for atom in neighbour_atom_tuple],
            dtype=np.int64,
        ),
        is_identity_symmetry_operation=np.asarray(
            [atom.is_identity_symmetry_operation for atom in neighbour_atom_tuple],
            dtype=np.bool_,
        ),
        translation_offsets=np.asarray(
            [
                (atom.translation_a, atom.translation_b, atom.translation_c)
                for atom in neighbour_atom_tuple
            ],
            dtype=np.int64,
        ),
        packing_density_threshold=float(packing_density_threshold),
    )


def calculate_packing_density_from_arrays(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbours using array-backed retained translated atoms.
    """

    if (
        not math.isfinite(packing_density_threshold)
        or packing_density_threshold <= 0
    ):
        raise PackingDensityError(
            "packing_density_threshold must be a finite positive number, "
            f"got {packing_density_threshold!r}."
        )

    selected_atom_tuple = tuple(selected_atoms)
    if not selected_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density for an empty selected-atom list."
        )

    coordinates = np.asarray(neighbour_coordinates, dtype=np.float64)
    source_indices = np.asarray(source_atom_indices, dtype=np.int64)
    identity_flags = np.asarray(is_identity_symmetry_operation, dtype=np.bool_)
    offsets = np.asarray(translation_offsets, dtype=np.int64)

    _validate_neighbour_arrays(
        coordinates=coordinates,
        source_atom_indices=source_indices,
        is_identity_symmetry_operation=identity_flags,
        translation_offsets=offsets,
    )

    return _calculate_packing_density_with_ckdtree(
        selected_atoms=selected_atom_tuple,
        neighbour_coordinates=coordinates,
        source_atom_indices=source_indices,
        is_identity_symmetry_operation=identity_flags,
        translation_offsets=offsets,
        packing_density_threshold=float(packing_density_threshold),
    )


def _calculate_packing_density_with_ckdtree(
    *,
    selected_atoms: tuple[PreparedAtom, ...],
    neighbour_coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbours with scipy.spatial.cKDTree.
    """

    threshold = float(packing_density_threshold)
    threshold_squared = threshold**2
    selected_coordinates = _selected_atom_coordinates(selected_atoms)
    _validate_array_self_copies_are_counted(
        selected_atoms=selected_atoms,
        selected_coordinates=selected_coordinates,
        neighbour_coordinates=neighbour_coordinates,
        source_atom_indices=source_atom_indices,
        is_identity_symmetry_operation=is_identity_symmetry_operation,
        translation_offsets=translation_offsets,
        threshold_squared=threshold_squared,
    )

    tree = _build_ckdtree(neighbour_coordinates)
    strict_radius = float(np.nextafter(threshold, 0.0))
    raw_counts = tree.query_ball_point(
        selected_coordinates,
        strict_radius,
        return_length=True,
        workers=-1,
    )
    neighbour_counts = np.asarray(raw_counts, dtype=np.int64).reshape(-1) - 1

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
        packing_density_threshold=threshold,
        selected_atom_count=len(selected_atoms),
        neighbour_atom_count=int(neighbour_coordinates.shape[0]),
    )


def _build_ckdtree(neighbour_coordinates: np.ndarray):
    """Return a SciPy cKDTree for neighbour coordinates."""

    try:
        scipy_spatial = import_module("scipy.spatial")
    except ImportError as error:
        raise PackingDensityError(
            "SciPy is required for packing-density calculation."
        ) from error

    ckdtree_class = getattr(scipy_spatial, "cKDTree", None)
    if ckdtree_class is None:
        raise PackingDensityError(
            "SciPy is installed but scipy.spatial.cKDTree is unavailable."
        )

    return ckdtree_class(neighbour_coordinates)


def _selected_atom_coordinates(
    selected_atoms: tuple[PreparedAtom, ...],
) -> np.ndarray:
    """Return selected-atom coordinates as a float64 array."""

    return np.asarray(
        [
            (atom.record.x, atom.record.y, atom.record.z)
            for atom in selected_atoms
        ],
        dtype=np.float64,
    )


def _validate_array_self_copies_are_counted(
    *,
    selected_atoms: tuple[PreparedAtom, ...],
    selected_coordinates: np.ndarray,
    neighbour_coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
    threshold_squared: float,
) -> None:
    """Verify each selected atom's central-cell self copy can be subtracted."""

    if not math.isfinite(threshold_squared) or threshold_squared < 0:
        raise PackingDensityError(
            "threshold_squared must be a finite non-negative number, "
            f"got {threshold_squared!r}."
        )

    central_cell_identity_indices = np.flatnonzero(
        is_identity_symmetry_operation
        & np.all(translation_offsets == 0, axis=1)
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
        distances_squared = np.einsum(
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
    coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
) -> None:
    """Validate array-backed neighbour-cloud shapes."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise PackingDensityError(
            "neighbour_coordinates must have shape (n, 3)."
        )

    neighbour_count = coordinates.shape[0]
    if neighbour_count == 0:
        raise PackingDensityError(
            "Cannot calculate packing density with an empty neighbour-atom list."
        )

    if source_atom_indices.shape != (neighbour_count,):
        raise PackingDensityError(
            "source_atom_indices must have shape (n,) matching coordinates."
        )

    if is_identity_symmetry_operation.shape != (neighbour_count,):
        raise PackingDensityError(
            "is_identity_symmetry_operation must have shape (n,) matching "
            "coordinates."
        )

    if translation_offsets.shape != (neighbour_count, 3):
        raise PackingDensityError(
            "translation_offsets must have shape (n, 3) matching coordinates."
        )


def squared_distance_to_translated_atom(
    *,
    selected_x: float,
    selected_y: float,
    selected_z: float,
    neighbour_atom: TranslatedAtom,
) -> float:
    """
    Return squared Cartesian distance from a selected atom to a neighbour atom.
    """

    dx = selected_x - neighbour_atom.x
    dy = selected_y - neighbour_atom.y
    dz = selected_z - neighbour_atom.z

    return float(dx * dx + dy * dy + dz * dz)


def packing_density_counts_as_tuple(
    result: PackingDensityResult,
) -> tuple[int, ...]:
    """Return only neighbour counts from a packing-density result."""

    return tuple(atom_result.neighbour_count for atom_result in result.atom_results)
