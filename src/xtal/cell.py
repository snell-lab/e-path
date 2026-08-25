"""Unit-cell and space-group accessors for crystallographic metadata.

These helpers turn the raw metadata read from a coordinate file into
validated, Gemmi-backed crystallographic objects. They are generic
crystallography rather than part of any particular damage calculation, so
they live here and are consumed by `xtal.validate` and by `rabdam.crystal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gemmi

from xtal.reader import StructureMetadata


class CrystalMetadataError(ValueError):
    """Report missing or invalid unit-cell or space-group metadata."""


@dataclass(frozen=True, slots=True)
class UnitCellParameters:
    """Store unit-cell lengths in angstroms and angles in degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


def unit_cell_from_metadata(metadata: StructureMetadata) -> UnitCellParameters:
    """Validate and extract complete unit-cell parameters from metadata.

    Raises:
        CrystalMetadataError: If any parameter is absent, nonnumeric, or outside
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
        raise CrystalMetadataError(
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
        raise CrystalMetadataError(
            "Structure metadata contains non-numeric unit-cell parameters."
        ) from error

    if unit_cell.a <= 0 or unit_cell.b <= 0 or unit_cell.c <= 0:
        raise CrystalMetadataError(f"Invalid unit-cell lengths: {unit_cell!r}.")

    angles = (unit_cell.alpha, unit_cell.beta, unit_cell.gamma)
    if not all(0 < angle < 180 for angle in angles):
        raise CrystalMetadataError(f"Invalid unit-cell angles: {unit_cell!r}.")

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
        CrystalMetadataError: If the space-group name is absent or unrecognized.
    """
    if metadata.space_group is None or not str(metadata.space_group).strip():
        raise CrystalMetadataError("Structure metadata is missing a space-group name.")

    space_group_name = normalize_space_group_name(metadata.space_group)
    space_group = cast(
        gemmi.SpaceGroup | None,
        gemmi.find_spacegroup_by_name(space_group_name),
    )

    if space_group is None:
        raise CrystalMetadataError(
            f"Gemmi could not recognize space group {metadata.space_group!r}."
        )

    return space_group


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


def translation_vectors_from_gemmi_cell(
    cell: gemmi.UnitCell,
) -> UnitCellTranslationVectors:
    """Return the unit-cell axes in the Cartesian frame of a Gemmi cell.

    The axes are the columns of the cell's orthogonalization matrix. Gemmi
    keeps the file's own frame when a coordinate file carries a non-standard
    `SCALE`/`fract_transf_matrix`, so taking the axes from the same matrix
    that positions the atoms keeps symmetry images and lattice translations
    in one consistent frame.
    """
    matrix = cell.orth.mat.tolist()
    return UnitCellTranslationVectors(
        a=CartesianVector(
            float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0])
        ),
        b=CartesianVector(
            float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1])
        ),
        c=CartesianVector(
            float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])
        ),
    )


def normalize_space_group_name(space_group_name: str) -> str:
    """Remove quoting and normalize whitespace in a space-group name."""
    cleaned = str(space_group_name).replace("'", "").replace('"', "")
    return " ".join(cleaned.split())
