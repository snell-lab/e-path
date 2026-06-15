"""DSSP secondary-structure annotations for E-path atoms."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from epath.annotations import (
    ResidueAnnotationKey,
    WarningHandler,
    existing_structure_path,
    warn_if_requested,
)


def compute_secondary_structure_map(
    structure_path: str | Path,
    *,
    warn: WarningHandler | None = None,
) -> dict[ResidueAnnotationKey, str]:
    """Return DSSP secondary structure by residue key, or an empty map."""

    dssp_class = _optional_dssp_class()
    if dssp_class is None:
        warn_if_requested(
            warn,
            "DSSP not available; skipping secondary-structure annotation.",
        )
        return {}

    path = existing_structure_path(structure_path)
    try:
        structure = _parse_biopython_structure(path)
        model = next(structure.get_models())
        dssp = dssp_class(model, str(path))
    except Exception as error:
        warn_if_requested(
            warn,
            f"DSSP secondary-structure assignment skipped: {error}",
        )
        return {}

    return _secondary_structure_map_from_dssp(dssp, warn=warn)


def _secondary_structure_map_from_dssp(
    dssp: Any,
    *,
    warn: WarningHandler | None,
) -> dict[ResidueAnnotationKey, str]:
    if hasattr(dssp, "property_dict"):
        try:
            return {
                residue_key: _normalize_secondary_structure(
                    _secondary_structure_from_dssp_row(row)
                )
                for key, row in dssp.property_dict.items()
                if (residue_key := _residue_key_from_dssp_key(key)) is not None
            }
        except Exception as error:
            warn_if_requested(warn, f"DSSP property_dict parsing failed: {error}")

    try:
        return {
            residue_key: _normalize_secondary_structure(
                _secondary_structure_from_dssp_row(dssp[key])
            )
            for key in dssp.keys()
            if (residue_key := _residue_key_from_dssp_key(key)) is not None
        }
    except Exception as error:
        warn_if_requested(warn, f"DSSP fallback parsing failed: {error}")
        return {}


def _secondary_structure_from_dssp_row(row: Any) -> str:
    if hasattr(row, "get"):
        return str(row.get("SS_DSSP", "C") or "C")

    if isinstance(row, (tuple, list)) and len(row) >= 3:
        return str(row[2] or "C")

    return "C"


def _normalize_secondary_structure(secondary_structure: str) -> str:
    secondary_structure = secondary_structure.strip()
    if not secondary_structure:
        return "C"

    return secondary_structure


def _residue_key_from_dssp_key(key: Any) -> ResidueAnnotationKey | None:
    try:
        chain_id = str(key[0])
        residue_id = key[1]
        residue_number = residue_id[1]
        insertion_code = residue_id[2]
    except (IndexError, TypeError):
        return None

    residue_text = str(residue_number)
    if isinstance(insertion_code, str) and insertion_code.strip():
        residue_text = f"{residue_text}{insertion_code.strip()}"

    return chain_id, residue_text


def _parse_biopython_structure(path: Path) -> Any:
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser_class = getattr(
            importlib.import_module("Bio.PDB.MMCIFParser"),
            "MMCIFParser",
        )
    else:
        parser_class = getattr(importlib.import_module("Bio.PDB.PDBParser"), "PDBParser")

    parser = parser_class(QUIET=True)
    return parser.get_structure("epath", str(path))


def _optional_dssp_class() -> Any | None:
    try:
        return getattr(importlib.import_module("Bio.PDB.DSSP"), "DSSP")
    except Exception:
        return None
