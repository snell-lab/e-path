"""Read a coordinate file into flat atom records and crystallographic metadata.

Gemmi does the parsing. This module flattens its model-chain-residue-atom
hierarchy into immutable records in file order and recovers the resolution
where Gemmi leaves it unset.
"""

import copy
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import gemmi

from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureInputError,
)


class StructureReadError(StructureInputError):
    """Report a structure file that cannot be read or validated."""


# Tried in order, so the refinement resolution wins over the high-resolution
# limit of the merged reflection data.
_RESOLUTION_CIF_TAGS = (
    "_refine.ls_d_res_high",
    "_reflns.d_resolution_high",
)
_PDB_RESOLUTION_RE = re.compile(
    r"^REMARK\s+2\s+RESOLUTION\.\s+([0-9]+(?:\.[0-9]*)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Store one atom's identifiers, position, and isotropic B-factor.

    Coordinates are Cartesian, in angstroms. `source_atom_index` counts atoms
    from 0 in the order they appear in the file, for PDB and mmCIF alike. A
    chain whose atoms are split in the file, typically because its ligands and
    waters follow a later chain, keeps that split, so atoms with one
    `chain_id` need not be contiguous.

    Absent alternate-location and insertion codes are empty strings.
    `residue_number` is None when the file gives no number, and `atom_serial`
    is None when the serial is missing or unparseable. Gemmi
    reports an unrecognised element as "X". `model_number` counts models from
    1 rather than repeating the file's own MODEL serial.
    """

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

    Unit-cell lengths and resolution are in angstroms and unit-cell angles in
    degrees. A value the file does not provide is None. Cell values that are
    present are passed through without range checks, so consumers go through
    `xtal.cell`, which validates them. The resolution, by contrast, is only
    ever a positive finite number or None.
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
    # A copy of the cell exactly as Gemmi read it, so a file-specific
    # orthogonalization frame from SCALE or _atom_sites.fract_transf_matrix
    # survives into symmetry expansion. It is excluded from equality so that
    # metadata built by hand compares by its parameter values alone.
    gemmi_cell: gemmi.UnitCell | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class StructureData:
    """Pair the atom records of one model with the file's crystallographic metadata."""

    atoms: tuple[AtomRecord, ...]
    metadata: StructureMetadata


def read_structure(resolved_input: ResolvedStructureInput) -> StructureData:
    """Read a local coordinate file that holds exactly one model.

    Gemmi chooses the parser from the file extension. Every atom is kept,
    including hydrogens, waters, and all alternate conformers, so filtering is
    left to the caller. Raise StructureReadError if the input is not an
    existing local file, Gemmi cannot parse it, or it does not hold exactly
    one model with at least one atom.
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

    # Gemmi fills the document only for mmCIF input. Keeping it lets the
    # resolution fallback reuse the parsed file instead of reading it again.
    document = gemmi.cif.Document()
    try:
        # Merging would move the later parts of a split chain up to its first
        # part, so atom order would no longer match the file.
        structure = gemmi.read_structure(
            str(path), merge_chain_parts=False, save_doc=document
        )
    except Exception as error:
        raise StructureReadError(
            f"Gemmi could not read structure file {path!s}: {error}"
        ) from error

    if len(structure) > 1:
        raise StructureReadError(
            f"Expected a single-model structure, but found {len(structure)} models."
        )

    # An mmCIF file without atoms yields no model at all, while a PDB file
    # without atoms yields one empty model. Both end up here.
    atoms = _extract_atom_records(structure)

    if not atoms:
        raise StructureReadError(
            f"No atom records were found in structure file: {path!s}"
        )

    metadata = _extract_metadata(
        structure=structure,
        document=document,
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
                residue_number = residue.seqid.num
                insertion_code = residue.seqid.icode.strip()
                # Gemmi keeps the record type per residue, not per atom.
                is_hetatm = residue.het_flag == "H"

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
                            # Gemmi reports a blank alternate location as NUL.
                            altloc=atom.altloc.replace("\x00", "").strip(),
                            x=float(position.x),
                            y=float(position.y),
                            z=float(position.z),
                            occupancy=float(atom.occ),
                            b_factor=float(atom.b_iso),
                            # Gemmi reports a missing or unparseable serial as 0.
                            atom_serial=atom.serial or None,
                            is_hetatm=is_hetatm,
                        )
                    )

    return atom_records


def _extract_metadata(
    *,
    structure: gemmi.Structure,
    document: gemmi.cif.Document,
    resolved_input: ResolvedStructureInput,
    path: Path,
) -> StructureMetadata:
    # Gemmi represents a missing cell, and an all-zero CRYST1, as a 1 A cube.
    # is_crystal() recognizes that placeholder so its lengths are not
    # mistaken for a real cell. EM and NMR files often carry the same cube
    # explicitly and are treated as having no cell. The cell is copied because
    # Structure.cell is a view that would keep the whole parsed model alive.
    cell = copy.copy(structure.cell) if structure.cell.is_crystal() else None

    return StructureMetadata(
        source_path=path.resolve(),
        structure_id=resolved_input.structure_id,
        file_format=resolved_input.file_format,
        space_group=structure.spacegroup_hm or None,
        unit_cell_a=cell.a if cell is not None else None,
        unit_cell_b=cell.b if cell is not None else None,
        unit_cell_c=cell.c if cell is not None else None,
        unit_cell_alpha=cell.alpha if cell is not None else None,
        unit_cell_beta=cell.beta if cell is not None else None,
        unit_cell_gamma=cell.gamma if cell is not None else None,
        resolution_angstrom=_extract_resolution_angstrom(structure, document),
        gemmi_cell=cell,
    )


def _extract_resolution_angstrom(
    structure: gemmi.Structure, document: gemmi.cif.Document
) -> float | None:
    """Return the resolution, falling back to looser parsing than Gemmi's.

    Gemmi leaves Structure.resolution at 0 for some files it otherwise reads
    correctly. Examples are an mmCIF whose _refine category lacks
    pdbx_refine_id, and a PDB REMARK 2 line that is not column-aligned or
    omits ANGSTROMS. The mmCIF fallback reads the first data block, as Gemmi's
    structure reader does.
    """
    resolution = _positive_finite_or_none(structure.resolution)
    if resolution is not None:
        return resolution

    if len(document) > 0:
        block = document[0]
        for tag in _RESOLUTION_CIF_TAGS:
            # find_values covers both a single pair and a looped category,
            # such as a joint X-ray and neutron refinement.
            resolution = _first_positive_finite(
                gemmi.cif.as_string(value) for value in block.find_values(tag)
            )
            if resolution is not None:
                return resolution
        return None

    for line in structure.raw_remarks:
        match = _PDB_RESOLUTION_RE.match(line)
        if match is not None:
            return _positive_finite_or_none(match.group(1))

    return None


def _positive_finite_or_none(value: str | float) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number) or number <= 0.0:
        return None

    return number


def _first_positive_finite(values: Iterable[str]) -> float | None:
    for value in values:
        number = _positive_finite_or_none(value)
        if number is not None:
            return number

    return None
