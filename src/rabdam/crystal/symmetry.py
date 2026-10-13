"""Expand prepared atoms into their space-group symmetry images.

This is the first step in rebuilding the crystal around a model. The images
it returns fill one unit cell's worth of asymmetric units, which
`rabdam.crystal.trim` then copies into the neighbouring cells. The cell, the
operations, and the Cartesian frame all come from the `ValidatedCrystal` that
`xtal.validate` builds.
"""

from collections.abc import Iterable

import gemmi
import numpy as np
from numpy.typing import NDArray

from rabdam.structure.models import PreparedAtom
from xtal.validate import ValidatedCrystal


def symmetry_image_coordinates(
    atoms: Iterable[PreparedAtom],
    *,
    crystal: ValidatedCrystal,
) -> NDArray[np.float64]:
    """Return the position of every atom under every space-group operation.

    The result has one row per image and shape `(n, 3)`, in Cartesian angstroms
    in the frame of `crystal.gemmi_cell`. Rows are ordered by input atom and
    then by operation, in Gemmi's order, so each atom's images are contiguous.

    Together the images hold the contents of one unit cell, but they are left
    where the operations put them and so need not lie inside a single cell. No
    atoms give no rows.
    """
    return _symmetry_expanded_coordinates(
        atoms=tuple(atoms),
        gemmi_cell=crystal.gemmi_cell,
        operations=crystal.operations,
    ).reshape(-1, 3)


def _symmetry_expanded_coordinates(
    *,
    atoms: tuple[PreparedAtom, ...],
    gemmi_cell: gemmi.UnitCell,
    operations: tuple[gemmi.Op, ...],
) -> NDArray[np.float64]:
    """Return unwrapped Cartesian symmetry images with shape `(atoms, ops, 3)`."""
    # The reshape keeps an empty atom list as zero rows of three, which the
    # matrix products below turn into zero images instead of rejecting.
    cartesian_coordinates = np.asarray(
        [(atom.record.x, atom.record.y, atom.record.z) for atom in atoms],
        dtype=np.float64,
    ).reshape(-1, 3)
    fractional_coordinates = _apply_transform(gemmi_cell.frac, cartesian_coordinates)

    # Gemmi stores operator rotations and translations as integers scaled by
    # Op.DEN, so divide by DEN to recover the fractional-space values.
    rotations = (
        np.asarray([operation.rot for operation in operations], dtype=np.float64)
        / gemmi.Op.DEN
    )
    translations = (
        np.asarray([operation.tran for operation in operations], dtype=np.float64)
        / gemmi.Op.DEN
    )

    # The images are not wrapped into the cell. The identity image has to stay
    # on its source atom, where the trimming box and the packing-density count
    # both expect to find it.
    transformed_fractional_coordinates: NDArray[np.float64] = (
        np.einsum(  # pyright: ignore[reportUnknownMemberType]  # numpy stub gap
            "ni,mji->nmj", fractional_coordinates, rotations
        )
        + translations
    )

    return _apply_transform(gemmi_cell.orth, transformed_fractional_coordinates)


def _apply_transform(
    transform: gemmi.Transform,
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a Gemmi affine transform to row vectors along the last axis."""
    matrix = np.asarray(transform.mat.tolist(), dtype=np.float64)
    vector = np.asarray(transform.vec.tolist(), dtype=np.float64)
    return coordinates @ matrix.T + vector
