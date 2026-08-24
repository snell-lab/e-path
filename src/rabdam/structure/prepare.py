"""Run the structure-cleaning and atom-selection pipeline."""

from rabdam.structure.altlocs import select_altlocs
from rabdam.structure.checks import check_prepared_structure
from rabdam.structure.classify import classify_atoms
from rabdam.structure.filters import filter_clean_atoms
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationError,
    StructurePreparationOptions,
    StructurePreparationReport,
)
from rabdam.structure.selection import select_bdamage_atoms
from xtal.reader import StructureData

__all__ = (
    "PreparedAtom",
    "PreparedStructure",
    "StructurePreparationError",
    "StructurePreparationOptions",
    "StructurePreparationReport",
    "prepare_structure",
)


def prepare_structure(
    structure_data: StructureData,
    options: StructurePreparationOptions | None = None,
) -> PreparedStructure:
    """Clean, classify, and select atoms from a parsed structure.

    Args:
        structure_data: Parsed atoms and crystallographic metadata.
        options: Preparation policy. Defaults to the published RABDAM-compatible
            cleaning and protein-selection policy.

    Returns:
        Cleaned atoms, selected BDamage atoms, metadata, and preparation counts.

    Raises:
        StructurePreparationError: If preparation leaves no usable selection.
    """
    if options is None:
        options = StructurePreparationOptions()

    raw_atoms = tuple(structure_data.atoms)

    filter_result = filter_clean_atoms(
        raw_atoms,
        remove_hydrogens=options.remove_hydrogens,
        require_valid_occupancy=options.require_valid_occupancy,
        require_positive_b_factor=options.require_positive_b_factor,
    )

    altloc_result = select_altlocs(
        filter_result.atoms,
        options,
    )

    prepared_atoms = classify_atoms(altloc_result.atoms)

    selected_atoms = select_bdamage_atoms(
        prepared_atoms,
        options,
    )

    check_prepared_structure(
        cleaned_atoms=prepared_atoms,
        selected_atoms=selected_atoms,
        options=options,
    )

    warnings = tuple(filter_result.warnings + altloc_result.warnings)

    report = StructurePreparationReport(
        input_atom_count=len(raw_atoms),
        cleaned_atom_count=len(prepared_atoms),
        selected_atom_count=len(selected_atoms),
        removed_hydrogen_count=filter_result.counts.hydrogen,
        removed_invalid_coordinate_count=filter_result.counts.invalid_coordinates,
        removed_invalid_occupancy_count=filter_result.counts.invalid_occupancy,
        removed_invalid_b_factor_count=filter_result.counts.invalid_b_factor,
        removed_altloc_count=altloc_result.removed_count,
        warnings=warnings,
    )

    return PreparedStructure(
        cleaned_atoms=prepared_atoms,
        selected_atoms=selected_atoms,
        metadata=structure_data.metadata,
        report=report,
    )
