"""Resolve alternate atom conformers for structure preparation."""

from dataclasses import dataclass

from rabdam.input.reader import AtomRecord
from rabdam.structure.models import StructurePreparationOptions


@dataclass(frozen=True, slots=True)
class AltlocSelectionResult:
    """Store retained conformers, removal count, and residue warnings."""

    atoms: tuple[AtomRecord, ...]
    removed_count: int
    warnings: tuple[str, ...] = ()


def select_altlocs(
    atoms: tuple[AtomRecord, ...],
    options: StructurePreparationOptions,
) -> AltlocSelectionResult:
    """Resolve alternate conformers when enabled by preparation options."""
    if not options.resolve_altlocs:
        return AltlocSelectionResult(atoms=atoms, removed_count=0)

    return select_atom_site_altlocs(atoms)


def select_atom_site_altlocs(
    atoms: tuple[AtomRecord, ...],
) -> AltlocSelectionResult:
    """Retain the highest-occupancy conformer at each atom site.

    Ties are resolved by retaining the first source record. Sites are resolved
    independently, so one residue can retain atoms with different altloc labels.

    Returns:
        Retained atoms in source order, the number removed, and one warning per
        residue that contained alternate conformers.
    """
    grouped_atoms: dict[tuple[object, ...], list[tuple[int, AtomRecord]]] = {}

    for index, atom in enumerate(atoms):
        key = _atom_site_key(atom)
        grouped_atoms.setdefault(key, []).append((index, atom))

    kept_indexes: set[int] = set()
    removed_count = 0
    residues_with_altlocs: dict[tuple[object, ...], AtomRecord] = {}

    for group in grouped_atoms.values():
        if len(group) == 1:
            kept_indexes.add(group[0][0])
            continue

        # RABDAM resolves sites independently, even when that mixes altloc
        # labels within one residue.
        selected_index, selected_atom = max(
            group,
            key=lambda item: (item[1].occupancy, -item[0]),
        )

        kept_indexes.add(selected_index)
        removed_count += len(group) - 1
        residues_with_altlocs.setdefault(_residue_key(selected_atom), selected_atom)

    kept_atoms = tuple(
        atom for index, atom in enumerate(atoms) if index in kept_indexes
    )

    warnings = tuple(
        "Alternate conformers detected in "
        f"{_format_residue_label(atom)}. "
        "Highest-occupancy atom sites were retained."
        for atom in residues_with_altlocs.values()
    )

    return AltlocSelectionResult(
        atoms=kept_atoms,
        removed_count=removed_count,
        warnings=warnings,
    )


def _atom_site_key(atom: AtomRecord) -> tuple[object, ...]:
    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
        atom.atom_name.strip().upper(),
    )


def _residue_key(atom: AtomRecord) -> tuple[object, ...]:
    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
    )


def _format_residue_label(atom: AtomRecord) -> str:
    chain_id = atom.chain_id.strip() or "?"
    residue_name = atom.residue_name.strip().upper() or "?"
    residue_number = "?" if atom.residue_number is None else str(atom.residue_number)
    insertion_code = atom.insertion_code.strip()

    return f"chain {chain_id} residue {residue_name}{residue_number}{insertion_code}"
