"""Unit cells, space groups, and lattice vectors from crystallographic metadata.

These helpers turn the raw metadata read from a coordinate file into checked
unit-cell parameters, Gemmi cell and space-group objects, and the Cartesian
lattice vectors used to translate atoms between cells. They are generic
crystallography rather than part of any damage calculation, so they live here,
where `xtal.validate` and `rabdam.crystal` both use them.
"""

import math
from dataclasses import dataclass
from typing import cast

import gemmi

from xtal.reader import StructureMetadata


class CrystalMetadataError(ValueError):
    """Report missing or invalid unit-cell or space-group metadata.

    `xtal.validate.CrystalValidationError` derives from this class, so catching
    it also covers a crystal that fails the later volume and content checks.
    """


# A flat cell evaluates to rounding noise of about 1e-16 rather than exactly
# zero, so it is rejected below this floor. Real crystal cells sit many orders
# of magnitude above it. Because the value is also at most sin(gamma)^2, the
# floor keeps the standard orthogonalization clear of dividing by zero.
_MINIMUM_UNIT_VOLUME_SQUARED = 1e-9


@dataclass(frozen=True, slots=True)
class UnitCellParameters:
    """Store unit-cell lengths in angstroms and angles in degrees.

    Constructing one directly checks nothing. `unit_cell_from_metadata`
    returns only cells that pass `validate_unit_cell`.
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True, slots=True)
class CartesianVector:
    """Store a Cartesian displacement vector in angstroms."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class UnitCellTranslationVectors:
    """Store the three unit-cell axes as Cartesian displacements.

    Each axis is the shift that moves an atom one cell along that lattice
    direction, in the same Cartesian frame as the atom coordinates.
    """

    a: CartesianVector
    b: CartesianVector
    c: CartesianVector


def unit_cell_from_metadata(metadata: StructureMetadata) -> UnitCellParameters:
    """Return the unit cell recorded in `metadata`.

    A `CrystalMetadataError` is raised when any of the six parameters is
    missing or the cell fails `validate_unit_cell`.
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
            "Structure metadata is missing complete unit-cell parameters. "
            "E-path needs a crystal structure's unit cell. NMR and cryo-EM "
            "models and biological-assembly downloads have none; for a crystal "
            "structure, use the deposited asymmetric-unit file."
        )

    unit_cell = UnitCellParameters(
        a=unit_cell_a,
        b=unit_cell_b,
        c=unit_cell_c,
        alpha=unit_cell_alpha,
        beta=unit_cell_beta,
        gamma=unit_cell_gamma,
    )
    validate_unit_cell(unit_cell)
    return unit_cell


def validate_unit_cell(unit_cell: UnitCellParameters) -> None:
    """Raise `CrystalMetadataError` unless `unit_cell` describes a real cell.

    Every parameter must be finite, every length positive, and every angle
    strictly between 0 and 180 degrees. The three angles must also close into
    a cell that is not flat, which rules out combinations such as 10, 10 and
    170 degrees that pass the range check one angle at a time.
    """
    lengths = (unit_cell.a, unit_cell.b, unit_cell.c)
    angles = (unit_cell.alpha, unit_cell.beta, unit_cell.gamma)

    if not all(math.isfinite(value) for value in (*lengths, *angles)):
        raise CrystalMetadataError(f"Non-finite unit-cell parameters: {unit_cell!r}.")

    if not all(length > 0 for length in lengths):
        raise CrystalMetadataError(f"Invalid unit-cell lengths: {unit_cell!r}.")

    if not all(0 < angle < 180 for angle in angles):
        raise CrystalMetadataError(f"Invalid unit-cell angles: {unit_cell!r}.")

    # The determinant of the metric tensor divided by (abc)^2 is the squared
    # volume of a cell with unit edges. It falls to zero when one angle reaches
    # the sum of the other two or the three reach 360 degrees.
    cos_alpha, cos_beta, cos_gamma = (math.cos(math.radians(angle)) for angle in angles)
    unit_volume_squared = (
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    if unit_volume_squared <= _MINIMUM_UNIT_VOLUME_SQUARED:
        raise CrystalMetadataError(
            f"Unit-cell angles do not form a valid cell: {unit_cell!r}."
        )


def make_gemmi_unit_cell(unit_cell: UnitCellParameters) -> gemmi.UnitCell:
    """Return a Gemmi cell for `unit_cell` in the standard Cartesian frame.

    The standard frame puts a along x and b in the xy plane. It ignores any
    `SCALE` or `fract_transf_matrix` frame a file carries, so prefer the cell
    Gemmi read from the file, `StructureMetadata.gemmi_cell`, when there is one.
    """
    return gemmi.UnitCell(
        unit_cell.a,
        unit_cell.b,
        unit_cell.c,
        unit_cell.alpha,
        unit_cell.beta,
        unit_cell.gamma,
    )


def space_group_from_metadata(metadata: StructureMetadata) -> gemmi.SpaceGroup:
    """Return the Gemmi space group named in `metadata`.

    The name is normalized with `normalize_space_group_name` before lookup,
    and a name that is missing, empty after normalization, or unrecognized
    raises `CrystalMetadataError`. Files write a bare symbol such as "R 3" for
    both settings of an R-centred group, so the setting, and with it the
    number of operations, is chosen from the cell angles. Without them the
    hexagonal setting is assumed.
    """
    space_group_name = (
        ""
        if metadata.space_group is None
        else normalize_space_group_name(metadata.space_group)
    )
    if not space_group_name:
        raise CrystalMetadataError("Structure metadata is missing a space-group name.")

    alpha = metadata.unit_cell_alpha
    gamma = metadata.unit_cell_gamma
    # Gemmi reads the angles only to choose the setting of an R-centred group
    # and treats zero as unknown. Its stub declares a non-optional return, but
    # the lookup returns None for a name it does not recognize.
    space_group = cast(
        gemmi.SpaceGroup | None,
        gemmi.find_spacegroup_by_name(
            space_group_name,
            alpha=0.0 if alpha is None else alpha,
            gamma=0.0 if gamma is None else gamma,
        ),
    )

    if space_group is None:
        raise CrystalMetadataError(
            f"Gemmi could not recognize space group {metadata.space_group!r}."
        )

    return space_group


def normalize_space_group_name(space_group_name: str) -> str:
    """Remove every quote character and collapse whitespace in a space-group name."""
    cleaned = space_group_name.replace("'", "").replace('"', "")
    return " ".join(cleaned.split())


def translation_vectors_from_gemmi_cell(
    cell: gemmi.UnitCell,
) -> UnitCellTranslationVectors:
    """Return the unit-cell axes in the Cartesian frame of a Gemmi cell.

    The axes are the columns of the cell's orthogonalization matrix. When a
    file carries a non-standard `SCALE` or `fract_transf_matrix`, Gemmi keeps
    that frame for the atoms, so taking the axes from the same matrix keeps
    symmetry images and lattice translations in one consistent frame.
    """
    matrix = cell.orth.mat.tolist()
    return UnitCellTranslationVectors(
        a=CartesianVector(matrix[0][0], matrix[1][0], matrix[2][0]),
        b=CartesianVector(matrix[0][1], matrix[1][1], matrix[2][1]),
        c=CartesianVector(matrix[0][2], matrix[1][2], matrix[2][2]),
    )
