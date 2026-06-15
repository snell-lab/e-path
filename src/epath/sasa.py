"""FreeSASA solvent-accessibility annotations for E-path atoms."""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from pathlib import Path
from typing import Any

import gemmi

from epath.annotations import (
    AtomAnnotationKey,
    DEFAULT_SASA_ANGSTROM_SQUARED,
    EpathAnnotationError,
    WarningHandler,
    atom_annotation_key,
    existing_structure_path,
    warn_if_requested,
)
from epath.atoms import EpathAtom


def compute_sasa_map(
    structure_path: str | Path,
    atoms: Iterable[EpathAtom] | None = None,
    *,
    disable_sasa: bool = False,
    warn: WarningHandler | None = None,
) -> dict[AtomAnnotationKey, float]:
    """Return FreeSASA surface area by atom key, or an empty map."""

    if disable_sasa:
        warn_if_requested(warn, "SASA disabled by user.")
        return {}

    freesasa = _optional_module("freesasa")
    if freesasa is None:
        warn_if_requested(warn, "FreeSASA not available; skipping SASA annotation.")
        return {}

    path = existing_structure_path(structure_path)
    try:
        file_atom_keys = atom_keys_from_structure_file(path)
    except Exception as error:
        warn_if_requested(
            warn,
            f"Could not read coordinate tokens for SASA mapping: {error}",
        )
        return {}

    if not file_atom_keys:
        warn_if_requested(warn, "No ATOM/HETATM tokens found for SASA mapping.")
        return {}

    try:
        freesasa_structure = freesasa.Structure(str(path))
        result = freesasa.calc(freesasa_structure)
        atom_count = int(result.nAtoms())
    except Exception as error:
        warn_if_requested(warn, f"FreeSASA calculation failed: {error}")
        return {}

    mapped_atom_count = min(atom_count, len(file_atom_keys))
    sasa_by_file_key: dict[AtomAnnotationKey, float] = {}
    for atom_index in range(mapped_atom_count):
        key = file_atom_keys[atom_index]
        if key not in sasa_by_file_key:
            sasa_by_file_key[key] = float(result.atomArea(atom_index))

    if atoms is None:
        return sasa_by_file_key

    return {
        atom_annotation_key(atom): float(
            sasa_by_file_key.get(
                atom_annotation_key(atom),
                DEFAULT_SASA_ANGSTROM_SQUARED,
            )
        )
        for atom in atoms
    }


def atom_keys_from_structure_file(
    structure_path: str | Path,
) -> tuple[AtomAnnotationKey, ...]:
    """Return unique atom annotation keys in file order."""

    path = existing_structure_path(structure_path)
    if _is_pdb_like_path(path):
        return _pdb_atom_keys_from_file(path)

    return _gemmi_atom_keys_from_file(path)


def _pdb_atom_keys_from_file(path: Path) -> tuple[AtomAnnotationKey, ...]:
    atom_keys: list[AtomAnnotationKey] = []
    seen_keys: set[AtomAnnotationKey] = set()
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            if not atom_name or not residue_number:
                continue

            residue_key = f"{residue_number}{insertion_code}"
            atom_key = (chain_id, residue_key, atom_name.upper())
            if atom_key in seen_keys:
                continue

            atom_keys.append(atom_key)
            seen_keys.add(atom_key)

    return tuple(atom_keys)


def _gemmi_atom_keys_from_file(path: Path) -> tuple[AtomAnnotationKey, ...]:
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:
        raise EpathAnnotationError(
            f"Gemmi could not read structure file {path!s}: {error}"
        ) from error

    atom_keys: list[AtomAnnotationKey] = []
    seen_keys: set[AtomAnnotationKey] = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = _gemmi_residue_number_string(residue)
                for atom in residue:
                    atom_key = (
                        chain.name,
                        residue_key,
                        atom.name.strip().upper(),
                    )
                    if atom_key in seen_keys:
                        continue

                    atom_keys.append(atom_key)
                    seen_keys.add(atom_key)

    return tuple(atom_keys)


def _gemmi_residue_number_string(residue: gemmi.Residue) -> str:
    residue_number = residue.seqid.num
    insertion_code = residue.seqid.icode.strip()
    residue_text = "" if residue_number is None else str(residue_number)
    if insertion_code:
        return f"{residue_text}{insertion_code}"

    return residue_text


def _optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _is_pdb_like_path(path: Path) -> bool:
    return path.suffix.lower() in {".pdb", ".ent"}
