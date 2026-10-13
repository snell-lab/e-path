"""Prepared atoms, structures, and removal counts for the BDamage calculation.

`rabdam.structure.prepare` builds these from a structure read by `xtal.reader`,
and the later stages of `rabdam` and `epath` read them. They are plain records
with no behaviour of their own.
"""

from dataclasses import dataclass

from xtal.reader import AtomRecord, StructureMetadata


class StructurePreparationError(ValueError):
    """Report a structure that leaves nothing to score after preparation.

    `prepare_structure` raises it when cleaning removes every atom, or when
    none of the atoms that remain is protein.
    """


@dataclass(frozen=True, slots=True)
class PreparedAtom:
    """Pair an input atom record with its component classification.

    The flags come from the residue name alone, not from the ATOM or HETATM
    record type, so a modified amino acid such as MSE is protein.
    `rabdam.structure.prepare` sets at most one of them and leaves all three
    false for ligands, ions, and any other component.
    """

    record: AtomRecord
    is_protein: bool
    is_nucleic_acid: bool
    is_solvent: bool


@dataclass(frozen=True, slots=True)
class RemovedAtomCounts:
    """Count the atoms removed during preparation, by reason.

    Each removed atom is counted once, under the first reason that applies in
    field order, so the five counts and the cleaned atoms add up to the atoms
    read from the file. `hydrogens` includes deuterium. `alternate_conformers`
    counts atoms that passed every other check and lost their site to another
    conformer. `prepare_structure` documents the exact tests.
    """

    hydrogens: int
    non_finite_coordinates: int
    invalid_occupancy: int
    non_positive_b_factor: int
    alternate_conformers: int


@dataclass(frozen=True, slots=True)
class PreparedStructure:
    """Store the atoms as read, cleaned, and selected, with metadata and counts.

    `source_atoms` holds every atom of the file, in file order, so a record's
    `source_atom_index` is its position here. The asymmetric-unit content check
    reads these, because the mass of the model includes the atoms that
    cleaning removes. `cleaned_atoms` holds every atom that survived cleaning,
    in file order. The crystal is rebuilt from all of them, so each one counts
    towards packing density. `selected_atoms` is the subset that receives a
    BDamage score, in the same order. `prepare_structure` selects the protein
    atoms and guarantees at least one. Constructing a `PreparedStructure`
    directly checks nothing.
    """

    source_atoms: tuple[AtomRecord, ...]
    cleaned_atoms: tuple[PreparedAtom, ...]
    selected_atoms: tuple[PreparedAtom, ...]
    metadata: StructureMetadata
    removed: RemovedAtomCounts
