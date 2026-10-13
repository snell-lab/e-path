"""Build the block of crystal atoms that can be neighbours of the scored atoms.

Packing density counts the atoms within a fixed radius of each scored atom, so
only atom images close to the scored atoms are needed. This module copies the
symmetry images from `rabdam.crystal.symmetry` into the neighbouring cells and
keeps the ones inside a padded box around the scored atoms, as an array of
positions for `rabdam.packing`.
"""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rabdam.structure.models import PreparedAtom
from xtal.cell import UnitCellTranslationVectors


@dataclass(frozen=True, slots=True)
class CartesianBounds:
    """Store inclusive Cartesian bounds in angstroms."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def trim_crystal_to_reference_atoms(
    *,
    image_coordinates: NDArray[np.float64],
    translation_vectors: UnitCellTranslationVectors,
    reference_atoms: Iterable[PreparedAtom],
    padding: float,
    translation_range: int = 1,
) -> NDArray[np.float64]:
    """Copy symmetry images into neighbouring cells and keep those near atoms.

    `image_coordinates` holds the symmetry images of one unit cell, with shape
    `(n, 3)` in angstroms, and `translation_vectors` the cell axes in the same
    Cartesian frame. Every image is copied into each cell whose integer
    offsets along a, b and c run from `-translation_range` to
    `translation_range`. A copy is kept when it lies inside the axis-aligned
    Cartesian box around `reference_atoms`, grown by `padding` angstroms on
    every face, with the faces included.

    The kept positions are returned with shape `(m, 3)`, ordered by cell
    offset, a slowest and c fastest, and then by image order. Cells are
    translated and trimmed one at a time, so the full block of images is never
    held in memory.

    At least one reference atom, a nonnegative `padding` and a nonnegative
    `translation_range` are required, which
    `run_bdamage_pipeline_from_prepared` checks before this is called. The box
    always holds the reference atoms' own images when they are among
    `image_coordinates`, as the selected atoms of a structure are.
    """
    trim_bounds = expand_bounds(bounds_from_prepared_atoms(reference_atoms), padding)
    lower_corner = np.asarray(
        (trim_bounds.x_min, trim_bounds.y_min, trim_bounds.z_min),
        dtype=np.float64,
    )
    upper_corner = np.asarray(
        (trim_bounds.x_max, trim_bounds.y_max, trim_bounds.z_max),
        dtype=np.float64,
    )

    coordinate_chunks: list[NDArray[np.float64]] = []

    cell_offsets = range(-translation_range, translation_range + 1)
    for a_offset, b_offset, c_offset in itertools.product(cell_offsets, repeat=3):
        shifted_coordinates = image_coordinates + _cell_shift(
            translation_vectors, a_offset, b_offset, c_offset
        )
        inside_bounds = np.logical_and(
            lower_corner <= shifted_coordinates,
            shifted_coordinates <= upper_corner,
        ).all(axis=1)

        if inside_bounds.any():
            coordinate_chunks.append(shifted_coordinates[inside_bounds])

    return np.concatenate(coordinate_chunks, axis=0)


def _cell_shift(
    vectors: UnitCellTranslationVectors,
    a_offset: int,
    b_offset: int,
    c_offset: int,
) -> NDArray[np.float64]:
    """Return the Cartesian displacement of a whole-cell offset."""
    return np.asarray(
        (
            a_offset * vectors.a.x + b_offset * vectors.b.x + c_offset * vectors.c.x,
            a_offset * vectors.a.y + b_offset * vectors.b.y + c_offset * vectors.c.y,
            a_offset * vectors.a.z + b_offset * vectors.b.z + c_offset * vectors.c.z,
        ),
        dtype=np.float64,
    )


def bounds_from_prepared_atoms(
    atoms: Iterable[PreparedAtom],
) -> CartesianBounds:
    """Return the smallest axis-aligned box that holds every atom.

    At least one atom is required.
    """
    atom_tuple = tuple(atoms)
    x_values = tuple(atom.record.x for atom in atom_tuple)
    y_values = tuple(atom.record.y for atom in atom_tuple)
    z_values = tuple(atom.record.z for atom in atom_tuple)

    return CartesianBounds(
        x_min=min(x_values),
        x_max=max(x_values),
        y_min=min(y_values),
        y_max=max(y_values),
        z_min=min(z_values),
        z_max=max(z_values),
    )


def expand_bounds(bounds: CartesianBounds, padding: float) -> CartesianBounds:
    """Return `bounds` moved outward by `padding` angstroms on every face."""
    # Cartesian padding cannot exclude a Euclidean neighbour because that atom
    # must be within the cutoff along every individual axis.
    return CartesianBounds(
        x_min=bounds.x_min - padding,
        x_max=bounds.x_max + padding,
        y_min=bounds.y_min - padding,
        y_max=bounds.y_max + padding,
        z_min=bounds.z_min - padding,
        z_max=bounds.z_max + padding,
    )
