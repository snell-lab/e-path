"""Validate prepared structures before scientific calculations."""

from rabdam.structure.models import (
    PreparedAtom,
    StructurePreparationError,
    StructurePreparationOptions,
)


def check_prepared_structure(
    *,
    cleaned_atoms: tuple[PreparedAtom, ...],
    selected_atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> None:
    """Validate that prepared atoms satisfy workflow requirements.

    Raises:
        StructurePreparationError: If cleaning or selection left no usable
            atoms, or if the configured selection must contain protein but
            does not.
    """
    if not cleaned_atoms:
        raise StructurePreparationError("No atoms remain after structure preparation.")

    if not selected_atoms:
        raise StructurePreparationError(
            "No atoms were selected for BDamage calculation."
        )

    if options.require_protein_selection and not any(
        atom.is_protein for atom in selected_atoms
    ):
        raise StructurePreparationError(
            "No protein atoms were selected for BDamage calculation."
        )
