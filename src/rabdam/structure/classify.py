"""Classify atoms as protein, nucleic acid, or solvent."""

from collections.abc import Iterable
from functools import cache

import gemmi

from rabdam.structure.filters import is_hydrogen
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord

# Gemmi does not classify several force-field protonation variants and modified
# nucleotides in the way RABDAM needs for atom selection.
PROTEIN_COMPONENT_OVERRIDES = frozenset(
    {
        "ASH",
        "CYM",
        "CYX",
        "GLH",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
    }
)


NUCLEIC_ACID_COMPONENT_OVERRIDES = frozenset(
    {
        "5MC",
        "YYG",
    }
)


SOLVENTS = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
    }
)


def classify_atoms(atoms: Iterable[AtomRecord]) -> tuple[PreparedAtom, ...]:
    """Classify atom records in input order."""
    return tuple(classify_atom(atom) for atom in atoms)


def classify_atom(atom: AtomRecord) -> PreparedAtom:
    """Classify one atom by component type and input record kind."""
    component_name = atom.residue_name.strip().upper()
    is_protein, is_nucleic_acid, is_solvent = classify_component(component_name)

    return PreparedAtom(
        record=atom,
        is_hydrogen=is_hydrogen(atom),
        is_protein=is_protein,
        is_nucleic_acid=is_nucleic_acid,
        is_solvent=is_solvent,
        is_hetatm=atom.is_hetatm,
    )


def classify_component(component_name: str) -> tuple[bool, bool, bool]:
    """Return protein, nucleic-acid, and solvent flags for a component."""
    return _classify_normalized_component(component_name.strip().upper())


@cache
def _classify_normalized_component(component_name: str) -> tuple[bool, bool, bool]:
    residue = gemmi.find_tabulated_residue(component_name)

    return (
        component_name in PROTEIN_COMPONENT_OVERRIDES or residue.is_amino_acid(),
        component_name in NUCLEIC_ACID_COMPONENT_OVERRIDES or residue.is_nucleic_acid(),
        component_name in SOLVENTS or residue.is_water(),
    )
