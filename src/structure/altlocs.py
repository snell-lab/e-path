"""
Alternate-conformer handling for RABDAM structure preparation.
"""

from dataclasses import dataclass

from input.reader import AtomRecord
from structure.models import StructurePreparationOptions


@dataclass(frozen=True, slots=True)
class AltlocSelectionResult:
    """Result of alternate-conformer selection."""

    atoms: tuple[AtomRecord, ...]
    removed_count: int
    warnings: tuple[str, ...] = ()


def select_altlocs(
    atoms: tuple[AtomRecord, ...],
    options: StructurePreparationOptions,
) -> AltlocSelectionResult:
    """
    Resolve alternate conformers.
    """

    if not options.resolve_altlocs:
        return AltlocSelectionResult(atoms=atoms, removed_count=0)

    return select_atom_site_altlocs(atoms)


def select_atom_site_altlocs(
    atoms: tuple[AtomRecord, ...],
) -> AltlocSelectionResult:
    """
    Resolve alternate conformers independently for each atom site.

    For atoms that appear to represent the same atom site, keep the highest
    occupancy version. If occupancies tie, keep the one encountered first in
    the input order.
    """

    grouped_atoms: dict[tuple[object, ...], list[tuple[int, AtomRecord]]] = {}

    for index, atom in enumerate(atoms):
        key = atom_site_key(atom)
        grouped_atoms.setdefault(key, []).append((index, atom))

    kept_indexes: set[int] = set()
    removed_count = 0
    residues_with_altlocs: dict[tuple[object, ...], AtomRecord] = {}

    for group in grouped_atoms.values():
        if len(group) == 1:
            kept_indexes.add(group[0][0])
            continue

        selected_index, selected_atom = max(
            group,
            key=lambda item: (item[1].occupancy, -item[0]),
        )

        kept_indexes.add(selected_index)
        removed_count += len(group) - 1
        residues_with_altlocs.setdefault(residue_key(selected_atom), selected_atom)

    kept_atoms = tuple(
        atom
        for index, atom in enumerate(atoms)
        if index in kept_indexes
    )

    warnings = tuple(
        "Alternate conformers detected in "
        f"{format_residue_label(atom)}. "
        "Highest-occupancy atom sites were retained."
        for atom in residues_with_altlocs.values()
    )

    return AltlocSelectionResult(
        atoms=kept_atoms,
        removed_count=removed_count,
        warnings=warnings,
    )


def atom_site_key(atom: AtomRecord) -> tuple[object, ...]:
    """
    Return a key representing a unique atom site before altloc selection.

    Alternate conformers of the same atom site should share this key.
    """

    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
        atom.atom_name.strip().upper(),
    )


def residue_key(atom: AtomRecord) -> tuple[object, ...]:
    """
    Return a key representing a residue.
    """

    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
    )


def format_residue_label(atom: AtomRecord) -> str:
    """
    Return a compact human-readable residue label for warnings.
    """

    chain_id = atom.chain_id.strip() or "?"
    residue_name = atom.residue_name.strip().upper() or "?"
    residue_number = "?" if atom.residue_number is None else str(atom.residue_number)
    insertion_code = atom.insertion_code.strip()

    return f"chain {chain_id} residue {residue_name}{residue_number}{insertion_code}"
