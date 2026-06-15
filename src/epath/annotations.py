"""Shared structural annotation types and helpers for E-path atoms."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from epath.atoms import EpathAtom


ResidueAnnotationKey: TypeAlias = tuple[str, str]
AtomAnnotationKey: TypeAlias = tuple[str, str, str]
WarningHandler: TypeAlias = Callable[[str], None]

DEFAULT_SECONDARY_STRUCTURE = "NA"
DEFAULT_SASA_ANGSTROM_SQUARED = 0.0


class EpathAnnotationError(ValueError):
    """Raised when E-path structural annotations cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class EpathAtomAnnotation:
    """Optional structural annotations for one E-path atom."""

    atom: EpathAtom
    secondary_structure: str
    solvent_accessible_surface_area: float


@dataclass(frozen=True, slots=True)
class EpathAnnotationMaps:
    """Computed annotation maps for E-path atoms."""

    secondary_structure_by_residue: Mapping[ResidueAnnotationKey, str]
    sasa_by_atom: Mapping[AtomAnnotationKey, float]


def compute_annotation_maps(
    structure_path: str | Path,
    atoms: Iterable[EpathAtom],
    *,
    disable_sasa: bool = False,
    warn: WarningHandler | None = None,
) -> EpathAnnotationMaps:
    """Compute optional DSSP and FreeSASA maps for E-path atoms."""

    atom_tuple = tuple(atoms)
    return EpathAnnotationMaps(
        secondary_structure_by_residue=compute_secondary_structure_map(
            structure_path,
            warn=warn,
        ),
        sasa_by_atom=compute_sasa_map(
            structure_path,
            atom_tuple,
            disable_sasa=disable_sasa,
            warn=warn,
        ),
    )


def compute_secondary_structure_map(
    structure_path: str | Path,
    *,
    warn: WarningHandler | None = None,
) -> dict[ResidueAnnotationKey, str]:
    """Return DSSP secondary structure by residue key, or an empty map."""

    from epath.dssp import compute_secondary_structure_map as compute_dssp_map

    return compute_dssp_map(structure_path, warn=warn)


def compute_sasa_map(
    structure_path: str | Path,
    atoms: Iterable[EpathAtom] | None = None,
    *,
    disable_sasa: bool = False,
    warn: WarningHandler | None = None,
) -> dict[AtomAnnotationKey, float]:
    """Return FreeSASA surface area by atom key, or an empty map."""

    from epath.sasa import compute_sasa_map as compute_freesasa_map

    return compute_freesasa_map(
        structure_path,
        atoms,
        disable_sasa=disable_sasa,
        warn=warn,
    )


def annotate_atoms(
    atoms: Iterable[EpathAtom],
    *,
    secondary_structure_map: Mapping[ResidueAnnotationKey, str] | None = None,
    sasa_map: Mapping[AtomAnnotationKey, float] | None = None,
) -> tuple[EpathAtomAnnotation, ...]:
    """Attach optional structural annotations to E-path atoms."""

    secondary_structure_map = secondary_structure_map or {}
    sasa_map = sasa_map or {}

    return tuple(
        EpathAtomAnnotation(
            atom=atom,
            secondary_structure=secondary_structure_map.get(
                residue_annotation_key(atom),
                DEFAULT_SECONDARY_STRUCTURE,
            ),
            solvent_accessible_surface_area=float(
                sasa_map.get(
                    atom_annotation_key(atom),
                    DEFAULT_SASA_ANGSTROM_SQUARED,
                )
            ),
        )
        for atom in atoms
    )


def residue_annotation_key(atom: EpathAtom) -> ResidueAnnotationKey:
    """Return the residue key used by secondary-structure maps."""

    return atom.chain_id, residue_number_string(atom)


def atom_annotation_key(atom: EpathAtom) -> AtomAnnotationKey:
    """Return the atom key used by SASA maps."""

    return (
        atom.chain_id,
        residue_number_string(atom),
        atom.atom_name.strip().upper(),
    )


def residue_number_string(atom: EpathAtom) -> str:
    """Return the residue number plus insertion code as a compact string."""

    residue_number = atom.residue_number
    residue_text = "" if residue_number is None else str(residue_number)
    insertion_code = atom.insertion_code.strip()
    if insertion_code:
        return f"{residue_text}{insertion_code}"

    return residue_text


def existing_structure_path(structure_path: str | Path) -> Path:
    """Return a valid structure file path."""

    path = Path(structure_path)
    if not path.exists():
        raise EpathAnnotationError(f"Structure file does not exist: {path!s}")

    if not path.is_file():
        raise EpathAnnotationError(f"Expected a structure file, got: {path!s}")

    return path


def warn_if_requested(warn: WarningHandler | None, message: str) -> None:
    """Send a warning message to the caller-provided warning handler."""

    if warn is not None:
        warn(message)
