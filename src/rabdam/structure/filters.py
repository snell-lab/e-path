"""Remove unusable atoms before structure preparation."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

from xtal.reader import AtomRecord


@dataclass(frozen=True, slots=True)
class AtomFilterCounts:
    """Count atoms removed for each cleaning criterion."""

    hydrogen: int = 0
    invalid_coordinates: int = 0
    invalid_occupancy: int = 0
    invalid_b_factor: int = 0


@dataclass(frozen=True, slots=True)
class AtomFilterResult:
    """Store retained atoms and the associated cleaning report."""

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
    """Remove atoms that cannot contribute valid BDamage scores.

    Coordinates must always be finite. The remaining criteria can be relaxed
    for callers that need to inspect otherwise unusable records.

    Args:
        atoms: Atom records in source order.
        remove_hydrogens: Whether to remove hydrogen and deuterium atoms.
        require_valid_occupancy: Whether to require occupancy in the interval
            `(0, 1]`.
        require_positive_b_factor: Whether to require a positive B-factor.

    Returns:
        Retained atoms in source order and removal counts by criterion.
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

        if not _has_finite_coordinates(atom):
            removed_invalid_coordinate_count += 1
            continue

        if require_valid_occupancy and not _has_valid_occupancy(atom):
            removed_invalid_occupancy_count += 1
            continue

        if require_positive_b_factor and not _has_positive_b_factor(atom):
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
    """Return whether an atom is hydrogen or deuterium."""
    element = atom.element.strip().upper()

    if element:
        return element in {"H", "D"}

    atom_name = atom.atom_name.strip().upper()

    return atom_name.startswith(("H", "D"))


def _has_finite_coordinates(atom: AtomRecord) -> bool:
    return math.isfinite(atom.x) and math.isfinite(atom.y) and math.isfinite(atom.z)


def _has_valid_occupancy(atom: AtomRecord) -> bool:
    return 0.0 < atom.occupancy <= 1.0


def _has_positive_b_factor(atom: AtomRecord) -> bool:
    return atom.b_factor > 0.0
