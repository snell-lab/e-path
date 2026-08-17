"""Define shared structure-preparation data models and options."""

from dataclasses import dataclass

from rabdam.input.reader import AtomRecord, StructureMetadata


class StructurePreparationError(ValueError):
    """Report invalid input to the structure-preparation pipeline."""


@dataclass(frozen=True, slots=True)
class StructurePreparationOptions:
    """Configure atom cleaning, classification, and BDamage selection."""

    # Hydrogens and unusable B-factor records cannot contribute to the
    # published BDamage metric.
    remove_hydrogens: bool = True
    require_valid_occupancy: bool = True
    require_positive_b_factor: bool = True
    resolve_altlocs: bool = True

    include_hetatm_in_selection: bool = False
    # Modified amino acids are often encoded as HETATM but still belong to the
    # protein selection.
    include_protein_like_hetatm_in_selection: bool = True
    include_nucleic_acid_in_selection: bool = False
    require_protein_selection: bool = True

    remove_atom_serials: frozenset[int] = frozenset()
    add_atom_serials: frozenset[int] = frozenset()

    remove_component_names: frozenset[str] = frozenset()
    add_component_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PreparedAtom:
    """Pair an input atom record with its structural classifications."""

    record: AtomRecord
    is_hydrogen: bool
    is_protein: bool
    is_nucleic_acid: bool
    is_solvent: bool
    is_hetatm: bool


@dataclass(frozen=True, slots=True)
class StructurePreparationReport:
    """Summarize atom counts, removals, and warnings from preparation."""

    input_atom_count: int
    cleaned_atom_count: int
    selected_atom_count: int
    removed_hydrogen_count: int
    removed_invalid_coordinate_count: int
    removed_invalid_occupancy_count: int
    removed_invalid_b_factor_count: int
    removed_altloc_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedStructure:
    """Store cleaned atoms, the BDamage selection, and source metadata."""

    cleaned_atoms: tuple[PreparedAtom, ...]
    selected_atoms: tuple[PreparedAtom, ...]
    metadata: StructureMetadata
    report: StructurePreparationReport
