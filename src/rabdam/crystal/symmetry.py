"""Expand prepared structures by crystallographic symmetry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import gemmi
import numpy as np
from numpy.typing import NDArray

from rabdam.input.reader import StructureMetadata
from rabdam.structure.models import PreparedAtom, PreparedStructure


class CrystalSymmetryError(ValueError):
    """Report invalid crystallographic metadata or symmetry expansion."""


@dataclass(frozen=True, slots=True)
class UnitCellParameters:
    """Store unit-cell lengths in angstroms and angles in degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


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


def expand_prepared_structure_by_symmetry(
    prepared_structure: PreparedStructure,
) -> SymmetryExpandedStructure:
    """Expand all cleaned atoms in a prepared structure by symmetry."""
    # Unselected atoms still contribute to the crystal environment around the
    # atoms receiving BDamage scores.
    return expand_atoms_by_symmetry(
        atoms=prepared_structure.cleaned_atoms,
        metadata=prepared_structure.metadata,
    )


def expand_atoms_by_symmetry(
    *,
    atoms: Iterable[PreparedAtom],
    metadata: StructureMetadata,
) -> SymmetryExpandedStructure:
    """Apply every space-group operation to each input atom.

    Coordinates are transformed from Cartesian to fractional space, operated
    on without unit-cell wrapping, and returned in Cartesian angstroms.

    Returns:
        Expanded atoms ordered by input atom and then symmetry operation, plus
        the validated unit cell and normalized space-group metadata.

    Raises:
        CrystalSymmetryError: If atoms are empty or crystallographic metadata is
            missing, invalid, or describes no symmetry operations.
    """
    atom_tuple = tuple(atoms)

    if not atom_tuple:
        raise CrystalSymmetryError("Cannot symmetry-expand an empty atom list.")

    unit_cell = unit_cell_from_metadata(metadata)
    gemmi_cell = make_gemmi_unit_cell(unit_cell)
    space_group = space_group_from_metadata(metadata)
    operations = tuple(space_group.operations())

    if not operations:
        raise CrystalSymmetryError(
            f"Space group {metadata.space_group!r} has no symmetry operations."
        )

    expanded_coordinates = _symmetry_expanded_coordinates(
        atoms=atom_tuple,
        gemmi_cell=gemmi_cell,
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
        unit_cell=unit_cell,
        space_group_name=space_group.hm,
        operation_count=len(operations),
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


def apply_operation_without_wrapping(
    operation: gemmi.Op,
    fractional: gemmi.Fractional,
) -> gemmi.Fractional:
    """Apply a symmetry operation without wrapping fractional coordinates."""
    transformed = operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])

    return gemmi.Fractional(
        float(transformed[0]),
        float(transformed[1]),
        float(transformed[2]),
    )


def unit_cell_from_metadata(metadata: StructureMetadata) -> UnitCellParameters:
    """Validate and extract complete unit-cell parameters from metadata.

    Raises:
        CrystalSymmetryError: If any parameter is absent, nonnumeric, or outside
            its physical range.
    """
    unit_cell_a = metadata.unit_cell_a
    unit_cell_b = metadata.unit_cell_b
    unit_cell_c = metadata.unit_cell_c
    unit_cell_alpha = metadata.unit_cell_alpha
    unit_cell_beta = metadata.unit_cell_beta
    unit_cell_gamma = metadata.unit_cell_gamma

    if (
        unit_cell_a is None
        or unit_cell_b is None
        or unit_cell_c is None
        or unit_cell_alpha is None
        or unit_cell_beta is None
        or unit_cell_gamma is None
    ):
        raise CrystalSymmetryError(
            "Structure metadata is missing complete unit-cell parameters."
        )

    try:
        unit_cell = UnitCellParameters(
            a=float(unit_cell_a),
            b=float(unit_cell_b),
            c=float(unit_cell_c),
            alpha=float(unit_cell_alpha),
            beta=float(unit_cell_beta),
            gamma=float(unit_cell_gamma),
        )
    except (TypeError, ValueError) as error:
        raise CrystalSymmetryError(
            "Structure metadata contains non-numeric unit-cell parameters."
        ) from error

    if unit_cell.a <= 0 or unit_cell.b <= 0 or unit_cell.c <= 0:
        raise CrystalSymmetryError(f"Invalid unit-cell lengths: {unit_cell!r}.")

    angles = (unit_cell.alpha, unit_cell.beta, unit_cell.gamma)
    if not all(0 < angle < 180 for angle in angles):
        raise CrystalSymmetryError(f"Invalid unit-cell angles: {unit_cell!r}.")

    return unit_cell


def make_gemmi_unit_cell(unit_cell: UnitCellParameters) -> gemmi.UnitCell:
    """Build a Gemmi unit cell from validated parameters."""
    return gemmi.UnitCell(
        unit_cell.a,
        unit_cell.b,
        unit_cell.c,
        unit_cell.alpha,
        unit_cell.beta,
        unit_cell.gamma,
    )


def space_group_from_metadata(metadata: StructureMetadata) -> gemmi.SpaceGroup:
    """Resolve a Gemmi space group from structure metadata.

    Raises:
        CrystalSymmetryError: If the space-group name is absent or unrecognized.
    """
    if metadata.space_group is None or not str(metadata.space_group).strip():
        raise CrystalSymmetryError("Structure metadata is missing a space-group name.")

    space_group_name = normalize_space_group_name(metadata.space_group)
    space_group = cast(
        gemmi.SpaceGroup | None,
        gemmi.find_spacegroup_by_name(space_group_name),
    )

    if space_group is None:
        raise CrystalSymmetryError(
            f"Gemmi could not recognize space group {metadata.space_group!r}."
        )

    return space_group


def normalize_space_group_name(space_group_name: str) -> str:
    """Remove quoting and normalize whitespace in a space-group name."""
    cleaned = str(space_group_name).replace("'", "").replace('"', "")
    return " ".join(cleaned.split())


def expanded_coordinates_as_tuples(
    expanded_structure: SymmetryExpandedStructure,
) -> tuple[tuple[float, float, float], ...]:
    """Return expanded Cartesian coordinates in atom order."""
    return tuple((atom.x, atom.y, atom.z) for atom in expanded_structure.atoms)
