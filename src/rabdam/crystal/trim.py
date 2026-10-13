"""Build the block of crystal atoms that can be neighbours of the scored atoms.

Packing density counts the atoms within a fixed radius of each scored atom, so
only atom images close to the scored atoms are needed. This module copies the
symmetry images from `rabdam.crystal.symmetry` into the neighbouring cells and
keeps the ones inside a padded box around the scored atoms, as an array of
positions for `rabdam.packing`.
"""

import itertools
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from rabdam.structure.models import PreparedAtom


def trim_crystal_to_reference_atoms(
    *,
    image_coordinates: NDArray[np.float64],
    translation_vectors: NDArray[np.float64],
    reference_atoms: Iterable[PreparedAtom],
    padding: float,
    translation_range: int = 1,
) -> NDArray[np.float64]:
    """Copy symmetry images into neighbouring cells and keep those near atoms.

    `image_coordinates` holds the symmetry images of one unit cell, with shape
    `(n, 3)` in angstroms, and `translation_vectors` the a, b and c cell axes
    as the rows of a `(3, 3)` array in the same Cartesian frame. Every image
    is copied into each cell whose integer offsets along a, b and c run from
    `-translation_range` to `translation_range`. A copy is kept when it lies
    inside the axis-aligned Cartesian box around `reference_atoms`, grown by
    `padding` angstroms on every face, with the faces included.

    The kept positions are returned with shape `(m, 3)`, ordered by cell
    offset, a slowest and c fastest, and then by image order. Cells are
    translated and trimmed one at a time, so the full block of images is never
    held in memory.

    At least one reference atom, a nonnegative `padding` and a nonnegative
    `translation_range` are required. The box always holds the reference
    atoms' own images when they are among `image_coordinates`, as the selected
    atoms of a structure are.
    """
    reference_coordinates = np.asarray(
        [(atom.record.x, atom.record.y, atom.record.z) for atom in reference_atoms],
        dtype=np.float64,
    )
    # Cartesian padding cannot exclude a Euclidean neighbour because that atom
    # must be within the cutoff along every individual axis.
    lower_corner = reference_coordinates.min(axis=0) - padding
    upper_corner = reference_coordinates.max(axis=0) + padding
    a_axis, b_axis, c_axis = translation_vectors

    coordinate_chunks: list[NDArray[np.float64]] = []

    cell_offsets = range(-translation_range, translation_range + 1)
    for a_offset, b_offset, c_offset in itertools.product(cell_offsets, repeat=3):
        shifted_coordinates = image_coordinates + (
            a_offset * a_axis + b_offset * b_axis + c_offset * c_axis
        )
        inside_bounds = np.logical_and(
            lower_corner <= shifted_coordinates,
            shifted_coordinates <= upper_corner,
        ).all(axis=1)

        if inside_bounds.any():
            coordinate_chunks.append(shifted_coordinates[inside_bounds])

    return np.concatenate(coordinate_chunks, axis=0)
