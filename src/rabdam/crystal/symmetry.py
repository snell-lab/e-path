"""Expand prepared structures by crystallographic symmetry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import gemmi
import numpy as np
from numpy.typing import NDArray

from rabdam.structure.models import PreparedAtom, PreparedStructure
from xtal.cell import UnitCellParameters, UnitCellTranslationVectors
from xtal.validate import ValidatedCrystal, validate_crystal


@dataclass(frozen=True, slots=True)
class SymmetryExpandedAtom:
    """Describe one Cartesian atom image produced by a symmetry operation."""

    unit_cell_atom_index: int
    source_atom_index: int
    symmetry_operation_index: int
    symmetry_operation: str
    is_identity_symmetry_operation: bool
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class SymmetryExpandedStructure:
    """Store all symmetry images in one crystallographic unit cell."""

    atoms: tuple[SymmetryExpandedAtom, ...]
    unit_cell: UnitCellParameters
    space_group_name: str
    operation_count: int
    # Lattice vectors in the same Cartesian frame as the expanded coordinates,
    # so later unit-cell translations stay consistent with the images.
    translation_vectors: UnitCellTranslationVectors


def expand_prepared_structure_by_symmetry(
    prepared_structure: PreparedStructure,
    *,
    crystal: ValidatedCrystal | None = None,
) -> SymmetryExpandedStructure:
    """Expand all cleaned atoms in a prepared structure by symmetry.

    Args:
        prepared_structure: Cleaned atoms and crystallographic metadata.
        crystal: A crystal already validated by `xtal.validate`. When omitted,
            the metadata is validated here, without asymmetric-unit content
            checks.
    """
    if crystal is None:
        crystal = validate_crystal(prepared_structure.metadata)

    # Unselected atoms still contribute to the crystal environment around the
    # atoms receiving BDamage scores.
    return expand_atoms_by_symmetry(
        atoms=prepared_structure.cleaned_atoms,
        crystal=crystal,
    )


def expand_atoms_by_symmetry(
    *,
    atoms: Iterable[PreparedAtom],
    crystal: ValidatedCrystal,
) -> SymmetryExpandedStructure:
    """Apply every space-group operation of a validated crystal to each atom.

    Coordinates are transformed from Cartesian to fractional space, operated
    on without unit-cell wrapping, and returned in Cartesian angstroms.

    Returns:
        Expanded atoms ordered by input atom and then symmetry operation, plus
        the unit cell and normalized space-group name of the validated crystal.
    """
    atom_tuple = tuple(atoms)

    operations = crystal.operations
    expanded_coordinates = _symmetry_expanded_coordinates(
        atoms=atom_tuple,
        gemmi_cell=crystal.gemmi_cell,
        operations=operations,
    )
    operation_triplets = tuple(operation.triplet() for operation in operations)
    operation_identity_flags = tuple(
        operation_triplet.replace(" ", "") == "x,y,z"
        for operation_triplet in operation_triplets
    )

    expanded_atoms: list[SymmetryExpandedAtom] = []
    for atom_index, atom in enumerate(atom_tuple):
        for operation_index, coordinates in enumerate(
            expanded_coordinates[atom_index],
            start=1,
        ):
            expanded_atoms.append(
                SymmetryExpandedAtom(
                    unit_cell_atom_index=len(expanded_atoms) + 1,
                    source_atom_index=atom.record.source_atom_index,
                    symmetry_operation_index=operation_index,
                    symmetry_operation=operation_triplets[operation_index - 1],
                    is_identity_symmetry_operation=(
                        operation_identity_flags[operation_index - 1]
                    ),
                    x=float(coordinates[0]),
                    y=float(coordinates[1]),
                    z=float(coordinates[2]),
                )
            )

    return SymmetryExpandedStructure(
        atoms=tuple(expanded_atoms),
        unit_cell=crystal.unit_cell,
        space_group_name=crystal.space_group_name,
        operation_count=len(operations),
        translation_vectors=crystal.translation_vectors,
    )


def _symmetry_expanded_coordinates(
    *,
    atoms: tuple[PreparedAtom, ...],
    gemmi_cell: gemmi.UnitCell,
    operations: tuple[gemmi.Op, ...],
) -> NDArray[np.float64]:
    """Return unwrapped Cartesian symmetry images with shape `(atoms, ops, 3)`."""
    cartesian_coordinates: NDArray[np.float64] = np.asarray(
        [(atom.record.x, atom.record.y, atom.record.z) for atom in atoms],
        dtype=np.float64,
    )
    fractionalization_matrix = _gemmi_transform_matrix(gemmi_cell.frac)
    fractionalization_vector = _gemmi_transform_vector(gemmi_cell.frac)
    orthogonalization_matrix = _gemmi_transform_matrix(gemmi_cell.orth)
    orthogonalization_vector = _gemmi_transform_vector(gemmi_cell.orth)

    fractional_coordinates: NDArray[np.float64] = np.add(
        np.matmul(
            cartesian_coordinates,
            np.transpose(fractionalization_matrix),
        ),
        fractionalization_vector,
    )
    # Gemmi stores operator rotations and translations as integers scaled by
    # Op.DEN, so divide by DEN to recover the fractional-space values.
    operation_rotations: NDArray[np.float64] = np.asarray(
        [
            np.asarray(operation.rot, dtype=np.float64) / operation.DEN
            for operation in operations
        ],
        dtype=np.float64,
    )
    operation_translations: NDArray[np.float64] = np.asarray(
        [
            np.asarray(operation.tran, dtype=np.float64) / operation.DEN
            for operation in operations
        ],
        dtype=np.float64,
    )

    # Wrapping here would move symmetry images needed by the later neighbour-box
    # trimming step.
    transformed_fractional_coordinates: NDArray[np.float64] = np.asarray(
        np.einsum(  # pyright: ignore[reportUnknownMemberType]  # numpy stub gap
            "ni,mji->nmj", fractional_coordinates, operation_rotations
        )
        + operation_translations[np.newaxis, :, :],
        dtype=np.float64,
    )

    return np.asarray(
        np.add(
            np.matmul(
                transformed_fractional_coordinates,
                np.transpose(orthogonalization_matrix),
            ),
            orthogonalization_vector,
        ),
        dtype=np.float64,
    )


def _gemmi_transform_matrix(transform: gemmi.Transform) -> NDArray[np.float64]:
    return np.asarray(transform.mat.tolist(), dtype=np.float64)


def _gemmi_transform_vector(transform: gemmi.Transform) -> NDArray[np.float64]:
    return np.asarray(
        (transform.vec.x, transform.vec.y, transform.vec.z),
        dtype=np.float64,
    )
