"""Select atoms that receive BDamage scores."""

from rabdam.structure.models import PreparedAtom, StructurePreparationOptions


def select_bdamage_atoms(
    atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> tuple[PreparedAtom, ...]:
    """Select the protein atoms that should receive BDamage scores.

    Returns:
        Selected atoms in input order.
    """
    return tuple(atom for atom in atoms if is_bdamage_selection(atom, options))


def is_bdamage_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
) -> bool:
    """Return whether an atom is a protein atom eligible for BDamage scoring.

    BDamage compares B-factors between atoms in similar packing density
    environments, and protein, nucleic-acid, and heteroatom B-factors relate to
    packing density differently. Only protein atoms are therefore scored.
    Excluded atoms still contribute to the packing density of nearby atoms.
    """
    if not atom.is_protein:
        return False

    return not atom.is_hetatm or options.include_protein_like_hetatm_in_selection
