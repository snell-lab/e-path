"""Read structure files into RABDAM data models."""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import gemmi

from xtal.resolver import (
    InputResolutionError,
    ResolvedStructureInput,
    StructureFileFormat,
)

_RESOLUTION_CIF_TAGS = (
    "_refine.ls_d_res_high",
    "_reflns.d_resolution_high",
)
_PDB_RESOLUTION_RE = re.compile(
    r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9]+(?:\.[0-9]*)?)",
    re.IGNORECASE,
)
_MISSING_METADATA_VALUES = frozenset({"", ".", "?", "null", "none", "nan", "na", "n/a"})


class StructureReadError(InputResolutionError):
    """Report a structure file that cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Store one atom record with Cartesian coordinates in angstroms."""

    source_atom_index: int
    model_number: int
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_name: str
    element: str
    altloc: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    atom_serial: int | None
    is_hetatm: bool


@dataclass(frozen=True, slots=True)
class StructureMetadata:
    """Store crystallographic metadata extracted from a structure file.

    Unit-cell lengths and resolution are in angstroms; unit-cell angles are in
    degrees. Missing or invalid metadata values are represented by `None`.
    """

    source_path: Path
    structure_id: str | None
    file_format: StructureFileFormat
    space_group: str | None
    unit_cell_a: float | None
    unit_cell_b: float | None
    unit_cell_c: float | None
    unit_cell_alpha: float | None
    unit_cell_beta: float | None
    unit_cell_gamma: float | None
    resolution_angstrom: float | None = None
    # The cell exactly as Gemmi read it, including any file-specific
    # orthogonalization frame. Excluded from equality so metadata built by
    # hand compares by its parameter values alone.
    gemmi_cell: gemmi.UnitCell | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class StructureData:
    """Pair source-order atom records with crystallographic metadata."""

    atoms: tuple[AtomRecord, ...]
    metadata: StructureMetadata


def read_structure(resolved_input: ResolvedStructureInput) -> StructureData:
    """Read and validate a single-model local structure.

    Returns:
        Source-order atom records and available crystallographic metadata.

    Raises:
        StructureReadError: If the input is not a readable local coordinate
            file, contains anything other than one model, or contains no atoms.
    """
    if not resolved_input.is_local_file:
        raise StructureReadError(
            "read_structure expected a local file input, "
            f"but got {resolved_input.source_type.value!r}."
        )

    if resolved_input.local_path is None:
        raise StructureReadError("Local structure input is missing local_path.")

    path = resolved_input.local_path

    if not path.exists():
        raise StructureReadError(f"Structure file does not exist: {path!s}")

    if not path.is_file():
        raise StructureReadError(f"Expected a structure file, but got: {path!s}")

    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:
        raise StructureReadError(
            f"Gemmi could not read structure file {path!s}: {error}"
        ) from error

    if len(structure) != 1:
        raise StructureReadError(
            f"Expected a single-model structure, but found {len(structure)} models."
        )

    atoms = _extract_atom_records(structure)

    if not atoms:
        raise StructureReadError(
            f"No atom records were found in structure file: {path!s}"
        )

    metadata = _extract_metadata(
        structure=structure,
        resolved_input=resolved_input,
        path=path,
    )

    return StructureData(
        atoms=tuple(atoms),
        metadata=metadata,
    )


def _extract_atom_records(structure: gemmi.Structure) -> list[AtomRecord]:
    atom_records: list[AtomRecord] = []

    for model_index, model in enumerate(structure, start=1):
        for chain in model:
            chain_id = chain.name

            for residue in chain:
                residue_name = residue.name
                residue_number = _get_residue_number(residue)
                insertion_code = _get_insertion_code(residue)
                is_hetatm = _is_hetatm(residue)

                for atom in residue:
                    position = atom.pos

                    atom_records.append(
                        AtomRecord(
                            source_atom_index=len(atom_records),
                            model_number=model_index,
                            chain_id=chain_id,
                            residue_name=residue_name,
                            residue_number=residue_number,
                            insertion_code=insertion_code,
                            atom_name=atom.name.strip(),
                            element=atom.element.name,
                            altloc=_clean_altloc(atom.altloc),
                            x=float(position.x),
                            y=float(position.y),
                            z=float(position.z),
                            occupancy=float(atom.occ),
                            b_factor=float(atom.b_iso),
                            atom_serial=_get_atom_serial(atom),
                            is_hetatm=is_hetatm,
                        )
                    )

    return atom_records


def _extract_metadata(
    *,
    structure: gemmi.Structure,
    resolved_input: ResolvedStructureInput,
    path: Path,
) -> StructureMetadata:
    cell = structure.cell
    space_group = _get_space_group_name(structure)
    # Gemmi represents a missing cell as a 1 A cube rather than zeros, so a
    # file without CRYST1 or _cell must be recognized before its placeholder
    # lengths are mistaken for real ones.
    has_cell = bool(cell.is_crystal())

    return StructureMetadata(
        source_path=path.resolve(),
        structure_id=resolved_input.structure_id,
        file_format=resolved_input.file_format,
        space_group=space_group,
        unit_cell_a=_none_if_zero(cell.a) if has_cell else None,
        unit_cell_b=_none_if_zero(cell.b) if has_cell else None,
        unit_cell_c=_none_if_zero(cell.c) if has_cell else None,
        unit_cell_alpha=_none_if_zero(cell.alpha) if has_cell else None,
        unit_cell_beta=_none_if_zero(cell.beta) if has_cell else None,
        unit_cell_gamma=_none_if_zero(cell.gamma) if has_cell else None,
        resolution_angstrom=_extract_resolution_angstrom(
            structure=structure,
            path=path,
            file_format=resolved_input.file_format,
        ),
        gemmi_cell=cell if has_cell else None,
    )


def _get_space_group_name(structure: gemmi.Structure) -> str | None:
    space_group = getattr(structure, "spacegroup_hm", None)

    if space_group:
        return str(space_group)

    return None


def _extract_resolution_angstrom(
    *,
    structure: gemmi.Structure,
    path: Path,
    file_format: StructureFileFormat,
) -> float | None:
    resolution = _get_structure_resolution_angstrom(structure)
    if resolution is not None:
        return resolution

    # Gemmi does not populate Structure.resolution for every valid file.
    if file_format is StructureFileFormat.MMCIF:
        return _extract_mmcif_resolution_angstrom(path)

    if file_format is StructureFileFormat.PDB:
        return _extract_pdb_resolution_angstrom(path)

    return None


def _get_structure_resolution_angstrom(structure: gemmi.Structure) -> float | None:
    resolution = getattr(structure, "resolution", None)
    return _finite_positive_float_or_none(resolution)


def _extract_mmcif_resolution_angstrom(path: Path) -> float | None:
    try:
        document = gemmi.cif.read_file(str(path))
    except (OSError, RuntimeError, ValueError):
        return None

    if len(document) == 0:
        return None

    block = document[0]
    for tag in _RESOLUTION_CIF_TAGS:
        resolution = _finite_positive_float_or_none(block.find_value(tag))
        if resolution is not None:
            return resolution

    return None


def _extract_pdb_resolution_angstrom(path: Path) -> float | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = _PDB_RESOLUTION_RE.match(line)
                if match is None:
                    continue

                return _finite_positive_float_or_none(match.group(1))
    except OSError:
        return None

    return None


def _finite_positive_float_or_none(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip().strip("'\"")
    if text.casefold() in _MISSING_METADATA_VALUES:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number) or number <= 0.0:
        return None

    return number


def _get_residue_number(residue: gemmi.Residue) -> int | None:
    residue_number = getattr(residue.seqid, "num", None)

    if residue_number is None:
        return None

    return int(residue_number)


def _get_insertion_code(residue: gemmi.Residue) -> str:
    insertion_code = getattr(residue.seqid, "icode", "")

    if insertion_code is None:
        return ""

    return str(insertion_code).strip()


def _get_atom_serial(atom: gemmi.Atom) -> int | None:
    atom_serial = getattr(atom, "serial", None)

    if atom_serial is None:
        return None

    return int(atom_serial)


def _clean_altloc(altloc: str | None) -> str:
    if altloc is None:
        return ""

    cleaned = str(altloc).replace("\x00", "").strip()

    if cleaned in {".", "?"}:
        return ""

    return cleaned


def _is_hetatm(residue: gemmi.Residue) -> bool:
    return str(getattr(residue, "het_flag", "")).strip().upper() == "H"


def _none_if_zero(value: float) -> float | None:
    value = float(value)

    if value == 0.0:
        return None

    return value
