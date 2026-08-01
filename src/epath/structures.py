"""Structure-coordinate outputs for E-path atom selections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

from epath.atoms import EpathAtom
from epath.selection import EpathScoredAtom


PdbBFactorMode = Literal["bdamage", "b_factor", "z_score"]


class EpathStructureOutputError(ValueError):
    """Raised when E-path structure output cannot be written."""


@dataclass(frozen=True, slots=True)
class EpathStructureOutputFiles:
    """Coordinate outputs for selected E-path atoms."""

    selected_pdb: Path


def write_structure_outputs(
    output_prefix: str | Path,
    selected_atoms: Iterable[EpathScoredAtom],
    *,
    b_factor_mode: PdbBFactorMode = "bdamage",
) -> EpathStructureOutputFiles:
    """Write the selected-atom PDB coordinate subset."""

    prefix = Path(output_prefix)
    selected_pdb = prefix.with_name(f"{prefix.name}_selected.pdb")
    write_selected_atoms_pdb(
        selected_pdb,
        selected_atoms,
        b_factor_mode=b_factor_mode,
        title="E-path selected atoms",
    )

    return EpathStructureOutputFiles(selected_pdb=selected_pdb)


def write_selected_atoms_pdb(
    output_path: str | Path,
    atoms: Iterable[EpathScoredAtom | EpathAtom],
    *,
    b_factor_mode: PdbBFactorMode = "bdamage",
    title: str = "E-path selected atoms",
) -> Path:
    """Write selected atoms as a PDB subset."""

    return write_atom_subset_pdb(
        output_path,
        atoms,
        b_factor_mode=b_factor_mode,
        title=title,
    )


def write_atom_subset_pdb(
    output_path: str | Path,
    atoms: Iterable[EpathScoredAtom | EpathAtom],
    *,
    b_factor_mode: PdbBFactorMode = "bdamage",
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
        file.write(_pdb_remark_line(f"B-factor column stores {b_factor_mode}."))
        for remark in remarks:
            file.write(_pdb_remark_line(remark))

        for output_index, atom_record in enumerate(atom_tuple, start=1):
            file.write(
                _pdb_atom_line(
                    atom_record,
                    output_serial=output_index,
                    b_factor_mode=b_factor_mode,
                )
            )

        file.write("END\n")

    return output_file


def _pdb_atom_line(
    atom_record: EpathScoredAtom | EpathAtom,
    *,
    output_serial: int,
    b_factor_mode: PdbBFactorMode,
) -> str:
    atom = _atom_from_record(atom_record)
    record_type = atom.record_type if atom.record_type in {"ATOM", "HETATM"} else "ATOM"
    residue_number = _pdb_residue_number(atom.residue_number)
    return (
        f"{record_type:<6}"
        f"{output_serial:5d} "
        f"{_pdb_atom_name(atom.atom_name, atom.element)}"
        f" "
        f"{_pdb_residue_name(atom.residue_name)} "
        f"{_pdb_chain_id(atom.chain_id)}"
        f"{residue_number:4d}"
        f"{_pdb_insertion_code(atom.insertion_code)}"
        f"   "
        f"{_pdb_float(atom.x, width=8, precision=3, name='x')}"
        f"{_pdb_float(atom.y, width=8, precision=3, name='y')}"
        f"{_pdb_float(atom.z, width=8, precision=3, name='z')}"
        f"{_pdb_float(atom.occupancy, width=6, precision=2, name='occupancy')}"
        f"{_pdb_float(_b_factor_value(atom_record, b_factor_mode), width=6, precision=2, name='b_factor')}"
        f"          "
        f"{_pdb_element(atom.element)}"
        f"\n"
    )


def _b_factor_value(
    atom_record: EpathScoredAtom | EpathAtom,
    b_factor_mode: PdbBFactorMode,
) -> float:
    atom = _atom_from_record(atom_record)
    if b_factor_mode == "bdamage":
        return atom.bdamage

    if b_factor_mode == "b_factor":
        return atom.b_factor

    if b_factor_mode == "z_score":
        if isinstance(atom_record, EpathScoredAtom):
            return atom_record.bdamage_z_score

        raise EpathStructureOutputError(
            "b_factor_mode='z_score' requires EpathScoredAtom records."
        )

    raise EpathStructureOutputError(f"Unknown B-factor mode: {b_factor_mode!r}.")


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
