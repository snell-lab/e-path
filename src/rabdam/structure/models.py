"""Define the prepared-structure data models."""

from dataclasses import dataclass

from xtal.reader import AtomRecord, StructureMetadata


class StructurePreparationError(ValueError):
    """Report a structure that leaves nothing to score after preparation."""


@dataclass(frozen=True, slots=True)
class PreparedAtom:
    """Pair an input atom record with its component classification."""

    record: AtomRecord
    is_protein: bool
    is_nucleic_acid: bool
    is_solvent: bool


@dataclass(frozen=True, slots=True)
class RemovedAtomCounts:
    """Count the atoms removed during preparation, by reason."""

    hydrogens: int
    non_finite_coordinates: int
    invalid_occupancy: int
    non_positive_b_factor: int
    alternate_conformers: int


@dataclass(frozen=True, slots=True)
class PreparedStructure:
    """Store cleaned atoms, the BDamage selection, metadata, and removal counts."""

    cleaned_atoms: tuple[PreparedAtom, ...]
    selected_atoms: tuple[PreparedAtom, ...]
    metadata: StructureMetadata
    removed: RemovedAtomCounts
