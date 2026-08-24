"""Structure-coordinate outputs for E-path atom selections."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from epath.atoms import EpathAtom
from epath.selection import EpathScoredAtom
from rabdam.structure.models import PreparedAtom


class EpathStructureOutputError(ValueError):
    """Raised when E-path structure output cannot be written."""


# BDamage is a ratio of B-factors and is always greater than zero, so zero is
# an unambiguous marker for an atom that was not scored.
UNSCORED_BDAMAGE_PLACEHOLDER = 0.0


@dataclass(frozen=True, slots=True)
class EpathStructureOutputFiles:
    """Coordinate outputs for an E-path run."""

    selected_pdb: Path
    model_pdb: Path | None = None


def write_structure_outputs(
    output_prefix: str | Path,
    selected_atoms: Iterable[EpathScoredAtom],
    *,
    cleaned_atoms: Sequence[PreparedAtom] | None = None,
    bdamage_by_source_index: Mapping[int, float] | None = None,
    bdamage_threshold: float | None = None,
) -> EpathStructureOutputFiles:
    """Write the E-path coordinate outputs.

    Always writes the selected-atom subset. When the cleaned model is supplied,
    also writes the complete cleaned model annotated with BDamage, so that
    solvent, ions, and cofactors can be inspected alongside the E-path results
    even though they were not scored.
    """

    prefix = Path(output_prefix)
    selected_pdb = prefix.with_name(f"{prefix.name}_selected.pdb")
    write_selected_atoms_pdb(
        selected_pdb,
        selected_atoms,
        title="E-path selected atoms",
    )

    model_pdb: Path | None = None
    if cleaned_atoms is not None:
        model_pdb = prefix.with_name(f"{prefix.name}_model.pdb")
        write_model_pdb(
            model_pdb,
            cleaned_atoms,
            bdamage_by_source_index=bdamage_by_source_index or {},
            bdamage_threshold=bdamage_threshold,
        )

    return EpathStructureOutputFiles(
        selected_pdb=selected_pdb,
        model_pdb=model_pdb,
    )


def write_model_pdb(
    output_path: str | Path,
    cleaned_atoms: Sequence[PreparedAtom],
    *,
    bdamage_by_source_index: Mapping[int, float],
    bdamage_threshold: float | None = None,
    title: str = "E-path cleaned model with BDamage",
) -> Path:
    """Write the complete cleaned model with BDamage in the B-factor column.

    Every atom that survived structure preparation is written, including the
    solvent, ions, ligands, and cofactors that are excluded from the BDamage
    calculation but still contribute to the packing density of nearby atoms.
    Excluded atoms carry a B-factor of zero.
    """

    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    atom_tuple = tuple(cleaned_atoms)
    if len(atom_tuple) > 99999:
        raise EpathStructureOutputError(
            "PDB output can contain at most 99999 atoms when serials are "
            f"renumbered, got {len(atom_tuple)}."
        )

    scored_count = sum(
        1
        for atom in atom_tuple
        if atom.record.source_atom_index in bdamage_by_source_index
    )
    # PDB REMARK text is limited to 69 characters, so keep each line short.
    remarks = [
        "E-path cleaned model.",
        "Hydrogens removed, alternate sites resolved by occupancy.",
        "B-factor column stores BDamage.",
        f"{scored_count} of {len(atom_tuple)} atoms were scored.",
        "Unscored atoms have B-factor 0.00.",
        "Unscored atoms still add to the packing density of scored atoms.",
    ]
    if bdamage_threshold is not None and math.isfinite(bdamage_threshold):
        remarks.append(
            f"Selected atoms satisfy 0.00 < B-factor <= {bdamage_threshold:.4f}."
        )

    with output_file.open("w", encoding="utf-8", newline="\n") as file:
        file.write(_pdb_title_line(title))
        for remark in remarks:
            file.write(_pdb_remark_line(remark))

        for output_serial, prepared_atom in enumerate(atom_tuple, start=1):
            record = prepared_atom.record
            file.write(
                _pdb_atom_line_from_fields(
                    record_type="HETATM" if record.is_hetatm else "ATOM",
                    atom_name=record.atom_name,
                    residue_name=record.residue_name,
                    chain_id=record.chain_id,
                    residue_number=record.residue_number,
                    insertion_code=record.insertion_code,
                    element=record.element,
                    x=float(record.x),
                    y=float(record.y),
                    z=float(record.z),
                    occupancy=float(record.occupancy),
                    b_factor=bdamage_by_source_index.get(
                        record.source_atom_index,
                        UNSCORED_BDAMAGE_PLACEHOLDER,
                    ),
                    output_serial=output_serial,
                )
            )

        file.write("END\n")

    return output_file


def write_selected_atoms_pdb(
    output_path: str | Path,
    atoms: Iterable[EpathScoredAtom | EpathAtom],
    *,
    title: str = "E-path selected atoms",
) -> Path:
    """Write selected atoms as a PDB subset."""

    return write_atom_subset_pdb(
        output_path,
        atoms,
        title=title,
    )


def write_atom_subset_pdb(
    output_path: str | Path,
    atoms: Iterable[EpathScoredAtom | EpathAtom],
    *,
    title: str = "E-path atom subset",
    remarks: Iterable[str] = (),
) -> Path:
    """Write E-path atoms to a simple PDB coordinate subset."""

    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    atom_tuple = tuple(atoms)
    if len(atom_tuple) > 99999:
        raise EpathStructureOutputError(
            "PDB output can contain at most 99999 atoms when serials are "
            f"renumbered, got {len(atom_tuple)}."
        )

    with output_file.open("w", encoding="utf-8", newline="\n") as file:
        file.write(_pdb_title_line(title))
        file.write(_pdb_remark_line("E-path coordinate subset."))
        file.write(_pdb_remark_line("B-factor column stores BDamage."))
        for remark in remarks:
            file.write(_pdb_remark_line(remark))

        for output_index, atom_record in enumerate(atom_tuple, start=1):
            file.write(_pdb_atom_line(atom_record, output_serial=output_index))

        file.write("END\n")

    return output_file


def _pdb_atom_line(
    atom_record: EpathScoredAtom | EpathAtom,
    *,
    output_serial: int,
) -> str:
    atom = _atom_from_record(atom_record)
    return _pdb_atom_line_from_fields(
        record_type=atom.record_type,
        atom_name=atom.atom_name,
        residue_name=atom.residue_name,
        chain_id=atom.chain_id,
        residue_number=atom.residue_number,
        insertion_code=atom.insertion_code,
        element=atom.element,
        x=atom.x,
        y=atom.y,
        z=atom.z,
        occupancy=atom.occupancy,
        b_factor=atom.bdamage,
        output_serial=output_serial,
    )


def _pdb_atom_line_from_fields(
    *,
    record_type: str,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int | None,
    insertion_code: str,
    element: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    b_factor: float,
    output_serial: int,
) -> str:
    record_type = record_type if record_type in {"ATOM", "HETATM"} else "ATOM"
    residue_number_field = _pdb_residue_number(residue_number)
    return (
        f"{record_type:<6}"
        f"{output_serial:5d} "
        f"{_pdb_atom_name(atom_name, element)}"
        f" "
        f"{_pdb_residue_name(residue_name)} "
        f"{_pdb_chain_id(chain_id)}"
        f"{residue_number_field:4d}"
        f"{_pdb_insertion_code(insertion_code)}"
        f"   "
        f"{_pdb_float(x, width=8, precision=3, name='x')}"
        f"{_pdb_float(y, width=8, precision=3, name='y')}"
        f"{_pdb_float(z, width=8, precision=3, name='z')}"
        f"{_pdb_float(occupancy, width=6, precision=2, name='occupancy')}"
        f"{_pdb_float(b_factor, width=6, precision=2, name='b_factor')}"
        f"          "
        f"{_pdb_element(element)}"
        f"\n"
    )


def _atom_from_record(atom_record: EpathScoredAtom | EpathAtom) -> EpathAtom:
    if isinstance(atom_record, EpathScoredAtom):
        return atom_record.atom

    return atom_record


def _pdb_title_line(title: str) -> str:
    return f"TITLE     {_pdb_text(title, width=70)}\n"


def _pdb_remark_line(text: str) -> str:
    return f"REMARK 250 {_pdb_text(text, width=69)}\n"


def _pdb_text(text: str, *, width: int) -> str:
    return str(text).replace("\n", " ")[:width]


def _pdb_atom_name(atom_name: str, element: str) -> str:
    clean_name = atom_name.strip()
    if not clean_name:
        clean_name = element.strip() or "X"

    clean_name = clean_name[:4]
    clean_element = element.strip()
    if len(clean_name) < 4 and len(clean_element) == 1 and not clean_name[0].isdigit():
        return f" {clean_name:<3}"

    return f"{clean_name:<4}"


def _pdb_residue_name(residue_name: str) -> str:
    clean_name = residue_name.strip() or "UNK"
    return f"{clean_name[:3]:>3}"


def _pdb_chain_id(chain_id: str) -> str:
    clean_chain = chain_id.strip()
    return clean_chain[:1] if clean_chain else " "


def _pdb_residue_number(residue_number: int | None) -> int:
    number = 0 if residue_number is None else int(residue_number)
    if number < -999 or number > 9999:
        raise EpathStructureOutputError(
            f"Residue number cannot be represented in PDB format: {number!r}."
        )

    return number


def _pdb_insertion_code(insertion_code: str) -> str:
    clean_code = insertion_code.strip()
    return clean_code[:1] if clean_code else " "


def _pdb_element(element: str) -> str:
    clean_element = element.strip().upper()
    if len(clean_element) > 2:
        clean_element = clean_element[:2]

    return f"{clean_element:>2}"


def _pdb_float(value: float, *, width: int, precision: int, name: str) -> str:
    number = _finite_float(value, name=name)
    text = f"{number:{width}.{precision}f}"
    if len(text) > width:
        raise EpathStructureOutputError(
            f"{name} cannot be represented in PDB format: {number!r}."
        )

    return text


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathStructureOutputError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathStructureOutputError(
            f"{name} must be numeric, got {value!r}."
        ) from error

    if not math.isfinite(number):
        raise EpathStructureOutputError(f"{name} must be finite, got {number!r}.")

    return number
