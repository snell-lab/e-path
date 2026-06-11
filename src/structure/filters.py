"""
Basic atom-cleaning filters for RABDAM structure preparation.

This module performs only the first cleaning pass:
    - remove hydrogens/deuteriums
    - remove atoms with non-finite Cartesian coordinates
    - remove atoms with invalid occupancy
    - remove atoms with non-positive B-factors
"""

from collections.abc import Iterable
from dataclasses import dataclass
import math

from input.reader import AtomRecord


@dataclass(frozen=True, slots=True)
class AtomFilterCounts:
    """Counts of atoms removed by the basic cleaning pass."""

    hydrogen: int = 0
    invalid_coordinates: int = 0
    invalid_occupancy: int = 0
    invalid_b_factor: int = 0


@dataclass(frozen=True, slots=True)
class AtomFilterResult:
    """Result of the basic atom-cleaning pass."""

    atoms: tuple[AtomRecord, ...]
    counts: AtomFilterCounts
    warnings: tuple[str, ...] = ()


def filter_clean_atoms(
    atoms: Iterable[AtomRecord],
    *,
    remove_hydrogens: bool = True,
    require_valid_occupancy: bool = True,
    require_positive_b_factor: bool = True,
) -> AtomFilterResult:
    """
    This keeps only atoms that are suitable to pass into altloc handling.
    """

    kept_atoms: list[AtomRecord] = []

    removed_hydrogen_count = 0
    removed_invalid_coordinate_count = 0
    removed_invalid_occupancy_count = 0
    removed_invalid_b_factor_count = 0

    for atom in atoms:
        if remove_hydrogens and is_hydrogen(atom):
            removed_hydrogen_count += 1
            continue

        if not has_finite_coordinates(atom):
            removed_invalid_coordinate_count += 1
            continue

        if require_valid_occupancy and not has_valid_occupancy(atom):
            removed_invalid_occupancy_count += 1
            continue

        if require_positive_b_factor and not has_positive_b_factor(atom):
            removed_invalid_b_factor_count += 1
            continue

        kept_atoms.append(atom)

    counts = AtomFilterCounts(
        hydrogen=removed_hydrogen_count,
        invalid_coordinates=removed_invalid_coordinate_count,
        invalid_occupancy=removed_invalid_occupancy_count,
        invalid_b_factor=removed_invalid_b_factor_count,
    )

    return AtomFilterResult(
        atoms=tuple(kept_atoms),
        counts=counts,
    )


def is_hydrogen(atom: AtomRecord) -> bool:
    """
    Return True if an atom is hydrogen or deuterium.
    """

    element = atom.element.strip().upper()

    if element:
        return element in {"H", "D"}

    atom_name = atom.atom_name.strip().upper()

    return atom_name.startswith(("H", "D"))


def has_finite_coordinates(atom: AtomRecord) -> bool:
    """
    Return True if all Cartesian coordinates are finite numbers.
    """

    return (
        math.isfinite(atom.x)
        and math.isfinite(atom.y)
        and math.isfinite(atom.z)
    )


def has_valid_occupancy(atom: AtomRecord) -> bool:
    """
    Return True if occupancy is valid for RABDAM structure preparation.
    """

    return 0.0 < atom.occupancy <= 1.0


def has_positive_b_factor(atom: AtomRecord) -> bool:
    """
    Return True if an atom has a positive isotropic B-factor.
    """

    return atom.b_factor > 0.0
