"""Calculate the packing density of each scored atom.

An atom's packing density is the number of atoms within a threshold distance
of it in the crystal, including atoms in neighbouring copies of the model.
`rabdam.crystal` supplies those surrounding atoms. `rabdam.bdamage` uses the
counts to compare B-factors between atoms of similar packing density.
"""

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from rabdam.structure.models import PreparedAtom
from spatial import radius_neighbour_counts


def calculate_packing_density(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_coordinates: NDArray[np.float64],
    packing_density_threshold: float,
) -> tuple[int, ...]:
    """Count the neighbours of each selected atom among the surrounding atoms.

    `neighbour_coordinates` holds the Cartesian position of every atom image
    in the surrounding crystal, in angstroms, with shape `(n, 3)`. It must
    include each selected atom itself, at its own position, and every image
    within the threshold of a selected atom. A block trimmed around the
    selected atoms with a padding of at least the threshold holds both. A
    narrower block is not detected and gives counts that are too low near its
    faces.

    A neighbour is any image strictly closer than `packing_density_threshold`
    angstroms, whichever atom it is an image of, so symmetry and lattice images
    of the selected atom itself are counted. Only the atom itself is left out.
    The counts are returned in selected-atom order.

    Nothing is checked here. The threshold must be finite and positive and the
    selection must not be empty.
    """
    selected_atom_tuple = tuple(selected_atoms)
    selected_coordinates = _selected_atom_coordinates(selected_atom_tuple)

    # The search includes atoms at exactly the radius, but packing density
    # counts only atoms closer than the threshold. Using the largest number
    # below the threshold as the radius leaves them out.
    strict_radius = math.nextafter(packing_density_threshold, 0.0)
    # The search also finds each atom itself, at distance zero, so one is
    # subtracted from every count.
    neighbour_counts = (
        radius_neighbour_counts(
            neighbour_coordinates,
            selected_coordinates,
            strict_radius,
        )
        - 1
    )

    return tuple(int(neighbour_count) for neighbour_count in neighbour_counts)


def _selected_atom_coordinates(
    selected_atoms: tuple[PreparedAtom, ...],
) -> NDArray[np.float64]:
    return np.asarray(
        [(atom.record.x, atom.record.y, atom.record.z) for atom in selected_atoms],
        dtype=np.float64,
    )
