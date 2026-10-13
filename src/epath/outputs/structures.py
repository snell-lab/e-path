"""Structure-coordinate outputs for E-path atom selections.

Every coordinate file is written as mmCIF through Gemmi, so the outputs keep
multi-character chain identifiers, large atom counts, and residue numbers
outside the fixed-width PDB columns exactly as they were read.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import gemmi

from epath.analysis.pathways import PathReport
from epath.analysis.selection import EpathScoredAtom
from rabdam.structure.models import PreparedAtom


class EpathStructureOutputError(ValueError):
    """Raised when E-path structure output cannot be written."""


# BDamage is a ratio of B-factors and is always greater than zero, so zero is
# an unambiguous marker for an atom that was not scored.
UNSCORED_BDAMAGE_PLACEHOLDER = 0.0


@dataclass(frozen=True, slots=True)
class EpathStructureOutputFiles:
    """Coordinate outputs for an E-path run."""

    selected_cif: Path
    model_cif: Path | None = None
    path_cifs: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _CoordinateAtom:
    """The fields of one atom that reach the coordinate file."""

    is_hetatm: bool
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int | None
    insertion_code: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float


def write_structure_outputs(
    output_prefix: str | Path,
    selected_atoms: Iterable[EpathScoredAtom],
    *,
    bdamage_by_source_index: Mapping[int, float],
    path_reports: Sequence[PathReport] = (),
    cleaned_atoms: Sequence[PreparedAtom] | None = None,
    bdamage_threshold: float | None = None,
    display_chains: Sequence[str] | None = None,
) -> EpathStructureOutputFiles:
    """Write the E-path coordinate outputs.

    Always writes the selected-atom subset, and one subset per reported
    pathway. When the cleaned model is supplied, also writes the complete
    cleaned model annotated with BDamage, so that solvent, ions, and cofactors
    can be inspected alongside the E-path results even though they were not
    scored.

    `display_chains` restricts every coordinate file to those chains. It is a
    display filter only: the atoms and pathways were selected over the whole
    model, and a pathway that has no atoms in the kept chains is skipped.

    Raises:
        EpathStructureOutputError: If `display_chains` names a chain that is
            not in the selected atoms or the cleaned model.
    """
    prefix = Path(output_prefix)
    selected_atom_tuple = tuple(selected_atoms)
    chain_remarks: tuple[str, ...] = ()
    if display_chains is not None:
        kept_chains = tuple(dict.fromkeys(display_chains))
        _check_display_chains(kept_chains, selected_atom_tuple, cleaned_atoms)
        chain_remarks = (f"Coordinates limited to chains {', '.join(kept_chains)}.",)
        selected_atom_tuple = tuple(
            atom for atom in selected_atom_tuple if atom.chain_id in kept_chains
        )
        if cleaned_atoms is not None:
            cleaned_atoms = tuple(
                atom for atom in cleaned_atoms if atom.record.chain_id in kept_chains
            )
    else:
        kept_chains = None

    selected_cif = prefix.with_name(f"{prefix.name}_selected.cif")
    write_atom_subset_cif(
        selected_cif,
        selected_atom_tuple,
        title="E-path selected atoms",
        remarks=chain_remarks,
    )

    model_cif: Path | None = None
    if cleaned_atoms is not None:
        model_cif = prefix.with_name(f"{prefix.name}_model.cif")
        write_model_cif(
            model_cif,
            cleaned_atoms,
            bdamage_by_source_index=bdamage_by_source_index,
            bdamage_threshold=bdamage_threshold,
            remarks=chain_remarks,
        )

    return EpathStructureOutputFiles(
        selected_cif=selected_cif,
        model_cif=model_cif,
        path_cifs=write_path_subset_cifs(
            prefix, path_reports, display_chains=kept_chains
        ),
    )


def _check_display_chains(
    display_chains: Sequence[str],
    selected_atoms: Sequence[EpathScoredAtom],
    cleaned_atoms: Sequence[PreparedAtom] | None,
) -> None:
    present = {atom.chain_id for atom in selected_atoms}
    if cleaned_atoms is not None:
        present.update(atom.record.chain_id for atom in cleaned_atoms)
    unknown = [chain for chain in display_chains if chain not in present]
    if unknown:
        raise EpathStructureOutputError(
            f"--chains names chains not in the model: {', '.join(unknown)}. "
            f"Available chains: {', '.join(sorted(present)) or 'none'}."
        )


def write_model_cif(
    output_path: str | Path,
    cleaned_atoms: Sequence[PreparedAtom],
    *,
    bdamage_by_source_index: Mapping[int, float],
    bdamage_threshold: float | None = None,
    title: str = "E-path cleaned model with BDamage",
    remarks: Iterable[str] = (),
) -> Path:
    """Write the complete cleaned model with BDamage in the B-factor column.

    Every atom that survived structure preparation is written, including the
    solvent, ions, ligands, and cofactors that are excluded from the BDamage
    calculation but still contribute to the packing density of nearby atoms.
    Excluded atoms carry a B-factor of zero.

    Raises:
        EpathStructureOutputError: If a coordinate, occupancy, or BDamage value
            is not a finite number.
    """
    atom_tuple = tuple(cleaned_atoms)
    scored_count = sum(
        1
        for atom in atom_tuple
        if atom.record.source_atom_index in bdamage_by_source_index
    )
    remark_lines = [
        "E-path cleaned model.",
        "Hydrogens removed, alternate sites resolved by occupancy.",
        "B_iso_or_equiv stores BDamage.",
        f"{scored_count} of {len(atom_tuple)} atoms were scored.",
        "Unscored atoms have B_iso_or_equiv 0.00.",
        "Unscored atoms still add to the packing density of scored atoms.",
    ]
    values: dict[str, str] = {
        "atom_count": str(len(atom_tuple)),
        "scored_atom_count": str(scored_count),
    }
    if bdamage_threshold is not None and math.isfinite(bdamage_threshold):
        remark_lines.append(
            f"Selected atoms satisfy 0.00 < B_iso_or_equiv <= {bdamage_threshold:.4f}."
        )
        values["bdamage_threshold"] = f"{bdamage_threshold:.4f}"
    remark_lines.extend(remarks)

    coordinate_atoms = tuple(
        _CoordinateAtom(
            is_hetatm=prepared.record.is_hetatm,
            atom_name=prepared.record.atom_name,
            residue_name=prepared.record.residue_name,
            chain_id=prepared.record.chain_id,
            residue_number=prepared.record.residue_number,
            insertion_code=prepared.record.insertion_code,
            element=prepared.record.element,
            x=prepared.record.x,
            y=prepared.record.y,
            z=prepared.record.z,
            occupancy=prepared.record.occupancy,
            b_factor=bdamage_by_source_index.get(
                prepared.record.source_atom_index, UNSCORED_BDAMAGE_PLACEHOLDER
            ),
        )
        for prepared in atom_tuple
    )
    return _write_mmcif(
        output_path, coordinate_atoms, title=title, remarks=remark_lines, values=values
    )


def write_path_subset_cifs(
    output_prefix: str | Path,
    path_reports: Sequence[PathReport],
    *,
    display_chains: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Write one mmCIF subset per reported pathway.

    The combined selected-atom file holds every selected atom, including those
    in components too small to report. These files hold one pathway each, so a
    viewer can load and colour them separately without a per-atom label being
    encoded into the occupancy or B-factor column, which both carry real
    values.

    Path numbers are zero padded so the files sort in path order, and keep
    their numbering when `display_chains` drops a pathway with no atoms in the
    kept chains.
    """
    if not path_reports:
        return ()

    prefix = Path(output_prefix)
    width = max(2, len(str(max(report.path_id for report in path_reports))))
    written: list[Path] = []
    for report in path_reports:
        atoms = report.cluster.atoms
        remarks = [
            f"Pathway {report.path_id}.",
            f"{report.structural_descriptor}.",
            (
                f"{report.residue_count} residues, {report.node_count} nodes, "
                f"{report.total_length:.2f} A."
            ),
            f"Start: {report.start_endpoint.label}.",
            f"End: {report.end_endpoint.label}.",
            f"Nearest cofactor: {report.anchor_summary}.",
        ]
        if display_chains is not None:
            atoms = tuple(atom for atom in atoms if atom.chain_id in display_chains)
            if not atoms:
                continue
            remarks.append(
                f"Coordinates limited to chains {', '.join(display_chains)}."
            )
        written.append(
            write_atom_subset_cif(
                prefix.with_name(f"{prefix.name}_path{report.path_id:0{width}d}.cif"),
                atoms,
                title=f"E-path pathway {report.path_id}",
                remarks=remarks,
                values={
                    "path_id": str(report.path_id),
                    "anchor_cofactor": (
                        report.anchor.label if report.anchor is not None else "none"
                    ),
                },
            )
        )

    return tuple(written)


def write_atom_subset_cif(
    output_path: str | Path,
    atoms: Iterable[EpathScoredAtom],
    *,
    title: str = "E-path atom subset",
    remarks: Iterable[str] = (),
    values: Mapping[str, str] | None = None,
) -> Path:
    """Write E-path atoms to an mmCIF coordinate subset.

    BDamage is stored in `_atom_site.B_iso_or_equiv`, so colouring by
    B-factor in a viewer shows the BDamage landscape.

    Raises:
        EpathStructureOutputError: If a coordinate, occupancy, or BDamage value
            is not a finite number.
    """
    coordinate_atoms = tuple(_coordinate_atom(atom) for atom in atoms)
    remark_lines = [
        "E-path coordinate subset.",
        "B_iso_or_equiv stores BDamage.",
        *remarks,
    ]
    return _write_mmcif(
        output_path,
        coordinate_atoms,
        title=title,
        remarks=remark_lines,
        values=values or {},
    )


def _coordinate_atom(scored_atom: EpathScoredAtom) -> _CoordinateAtom:
    atom = scored_atom.atom
    return _CoordinateAtom(
        is_hetatm=atom.record_type == "HETATM",
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
    )


def _write_mmcif(
    output_path: str | Path,
    atoms: Sequence[_CoordinateAtom],
    *,
    title: str,
    remarks: Sequence[str],
    values: Mapping[str, str],
) -> Path:
    """Build a Gemmi structure from the atoms and write it as mmCIF.

    Atoms are renumbered from one in output order. Consecutive atoms that
    share a chain, residue number, insertion code, and residue name are
    grouped into one residue; a repeated residue elsewhere in the sequence
    becomes a second residue block, mirroring the input order.

    The run metadata goes into `_struct.title`, an `_epath` category of
    machine-readable values, and an `_epath_remark` loop of human-readable
    notes, all of which viewers ignore.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    structure = gemmi.Structure()
    structure.name = _block_name(output_file)
    model = gemmi.Model(1)
    chain: gemmi.Chain | None = None
    residue: gemmi.Residue | None = None
    residue_key: tuple[str, int | None, str, str] | None = None
    for serial, atom in enumerate(atoms, start=1):
        chain_id = atom.chain_id.strip()
        key = (chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        if residue is not None and chain is not None and key != residue_key:
            # Gemmi copies on add, so a residue is only added once complete.
            chain.add_residue(residue)
            residue = None
        if chain is not None and chain.name != chain_id:
            model.add_chain(chain)
            chain = None
        if chain is None:
            chain = gemmi.Chain(chain_id)
        if residue is None:
            residue = _make_residue(atom)
            residue_key = key
        residue.add_atom(_make_atom(atom, serial=serial))
    if residue is not None and chain is not None:
        chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    document = structure.make_mmcif_document()
    block = document.sole_block()
    block.set_pair("_struct.entry_id", gemmi.cif.quote(structure.name))
    block.set_pair("_struct.title", gemmi.cif.quote(title))
    for name, value in values.items():
        block.set_pair(f"_epath.{name}", value)
    loop = block.init_loop("_epath_remark.", ["id", "text"])
    for index, remark in enumerate(remarks, start=1):
        loop.add_row([str(index), gemmi.cif.quote(remark)])
    document.write_file(str(output_file))

    return output_file


def _make_residue(atom: _CoordinateAtom) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = atom.residue_name.strip() or "UNK"
    number = 0 if atom.residue_number is None else atom.residue_number
    insertion = atom.insertion_code.strip()[:1] or " "
    residue.seqid = gemmi.SeqId(number, insertion)
    residue.het_flag = "H" if atom.is_hetatm else "A"
    return residue


def _make_atom(atom: _CoordinateAtom, *, serial: int) -> gemmi.Atom:
    element = atom.element.strip()
    name = atom.atom_name.strip() or element or "X"
    output = gemmi.Atom()
    output.name = name
    output.element = gemmi.Element(element) if element else gemmi.Element("X")
    output.serial = serial
    output.pos = gemmi.Position(
        _finite_float(atom.x, name="x"),
        _finite_float(atom.y, name="y"),
        _finite_float(atom.z, name="z"),
    )
    output.occ = _finite_float(atom.occupancy, name="occupancy")
    output.b_iso = _finite_float(atom.b_factor, name="b_factor")
    return output


def _block_name(output_file: Path) -> str:
    """Return a CIF data-block name: the file stem without whitespace."""
    stem = "".join(output_file.stem.split())
    return stem or "epath"


def _finite_float(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise EpathStructureOutputError(f"{name} must be finite, got {value!r}.")

    return value
