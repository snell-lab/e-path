"""Write the mmCIF coordinate files of an E-path run.

`epath.outputs.write` calls `write_structure_outputs`, which writes the
selected atoms, each reported pathway, and the cleaned model. Every file
carries BDamage in the B-factor column, so colouring by B-factor in a viewer
shows the BDamage landscape.

Every coordinate file is written as mmCIF through Gemmi, so the outputs keep
multi-character chain identifiers, large atom counts, and residue numbers
outside the fixed-width PDB columns exactly as they were read.

The values are written as given. Structure preparation in `rabdam` has already
removed every atom whose coordinates, occupancy, or B-factor is unusable, so
the coordinates, occupancies, and BDamage values are finite and are not
checked again here.
"""

from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby
from pathlib import Path

import gemmi

from epath.analysis.pathways import PathReport
from epath.analysis.selection import EpathScoredAtom
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord

# BDamage is a ratio of B-factors and is always greater than zero, so zero is
# an unambiguous marker for an atom that was not scored.
UNSCORED_BDAMAGE_PLACEHOLDER = 0.0


def write_structure_outputs(
    output_prefix: str | Path,
    selected_atoms: Iterable[EpathScoredAtom],
    *,
    bdamage_by_source_index: Mapping[int, float],
    path_reports: Sequence[PathReport],
    cleaned_atoms: Sequence[PreparedAtom],
    bdamage_threshold: float,
    display_chains: Sequence[str] | None,
) -> tuple[Path, Path, dict[int, Path]]:
    """Write the E-path coordinate outputs and return their paths.

    Three kinds of file are written: the selected-atom subset, one subset per
    reported pathway, and the complete cleaned model annotated with BDamage,
    so that solvent, ions, and cofactors can be inspected alongside the E-path
    results even though they were not scored. The file names are
    `output_prefix` followed by `_selected.cif`, `_pathNN.cif`, and
    `_model.cif`. The paths come back as the selected-atom file, the model
    file, and the file of each pathway that was written, by path number and
    in pathway order.

    `display_chains` restricts every coordinate file to those chains, and None
    keeps them all. It is a display filter only: the atoms and pathways were
    selected over the whole model, and a pathway that has no atoms in the kept
    chains is skipped. `epath.pipeline` has checked that the chains are in the
    model, and one that is not simply matches no atoms here.
    """
    prefix = Path(output_prefix)
    selected_atom_tuple = tuple(selected_atoms)
    chain_remarks: tuple[str, ...] = ()
    if display_chains is not None:
        chain_remarks = (f"Coordinates limited to chains {', '.join(display_chains)}.",)
        selected_atom_tuple = tuple(
            atom
            for atom in selected_atom_tuple
            if atom.record.chain_id in display_chains
        )
        cleaned_atoms = tuple(
            atom for atom in cleaned_atoms if atom.record.chain_id in display_chains
        )

    return (
        write_atom_subset_cif(
            prefix.with_name(f"{prefix.name}_selected.cif"),
            selected_atom_tuple,
            title="E-path selected atoms",
            remarks=chain_remarks,
        ),
        write_model_cif(
            prefix.with_name(f"{prefix.name}_model.cif"),
            cleaned_atoms,
            bdamage_by_source_index=bdamage_by_source_index,
            bdamage_threshold=bdamage_threshold,
            remarks=chain_remarks,
        ),
        write_path_subset_cifs(prefix, path_reports, display_chains=display_chains),
    )


def write_model_cif(
    output_path: str | Path,
    cleaned_atoms: Sequence[PreparedAtom],
    *,
    bdamage_by_source_index: Mapping[int, float],
    bdamage_threshold: float,
    remarks: Iterable[str] = (),
) -> Path:
    """Write the complete cleaned model with BDamage in the B-factor column.

    Every atom that survived structure preparation is written, including the
    solvent, ions, ligands, and cofactors that are excluded from the BDamage
    calculation but still contribute to the packing density of nearby atoms.
    An atom whose source index is not in `bdamage_by_source_index` was not
    scored and carries a B-factor of zero.

    `bdamage_threshold` is the threshold of the selection and is recorded
    with the run metadata. Every selected atom is at or below it, but so can a
    few unselected atoms be, because the selection is cut to a whole number of
    atoms. The remark written here says only the first.
    """
    scored_count = sum(
        1
        for atom in cleaned_atoms
        if atom.record.source_atom_index in bdamage_by_source_index
    )
    remark_lines = [
        "E-path cleaned model.",
        "Hydrogens removed, alternate sites resolved by occupancy.",
        "B_iso_or_equiv stores BDamage.",
        f"{scored_count} of {len(cleaned_atoms)} atoms were scored.",
        "Unscored atoms have B_iso_or_equiv 0.00.",
        "Unscored atoms still add to the packing density of scored atoms.",
        (
            "Selected atoms are among those with "
            f"0.00 < B_iso_or_equiv <= {bdamage_threshold:.4f}."
        ),
        *remarks,
    ]
    values = {
        "atom_count": str(len(cleaned_atoms)),
        "scored_atom_count": str(scored_count),
        "bdamage_threshold": f"{bdamage_threshold:.4f}",
    }

    atoms = [
        (
            atom.record,
            bdamage_by_source_index.get(
                atom.record.source_atom_index, UNSCORED_BDAMAGE_PLACEHOLDER
            ),
        )
        for atom in cleaned_atoms
    ]
    return _write_mmcif(
        output_path,
        atoms,
        title="E-path cleaned model with BDamage",
        remarks=remark_lines,
        values=values,
    )


def write_path_subset_cifs(
    output_prefix: str | Path,
    path_reports: Sequence[PathReport],
    *,
    display_chains: Sequence[str] | None = None,
) -> dict[int, Path]:
    """Write one mmCIF subset per reported pathway and return them by path number.

    The combined selected-atom file holds every selected atom, including those
    in components too small to report. These files hold one pathway each, so a
    viewer can load and colour them separately without a per-atom label being
    encoded into the occupancy or B-factor column, which both carry real
    values.

    Path numbers are zero padded to at least two digits so the files sort in
    path order, and keep their numbering when `display_chains` drops a pathway
    with no atoms in the kept chains. The files written are returned in the
    order of `path_reports`.
    """
    if not path_reports:
        return {}

    prefix = Path(output_prefix)
    width = max(2, len(str(max(report.path_id for report in path_reports))))
    written: dict[int, Path] = {}
    for report in path_reports:
        atoms = report.cluster.atoms
        remarks = [
            f"Pathway {report.path_id}.",
            (
                f"{report.residue_count} residues, {report.node_count} nodes, "
                f"{report.total_length:.2f} A."
            ),
            f"Start: {report.start_endpoint.label}.",
            f"End: {report.end_endpoint.label}.",
            f"Nearest cofactor: {report.anchor_summary}.",
        ]
        if display_chains is not None:
            atoms = tuple(
                atom for atom in atoms if atom.record.chain_id in display_chains
            )
            if not atoms:
                continue
            remarks.append(
                f"Coordinates limited to chains {', '.join(display_chains)}."
            )
        written[report.path_id] = write_atom_subset_cif(
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

    return written


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
    B-factor in a viewer shows the BDamage landscape. The atoms are written in
    the order given, and no atoms gives a file with no atom sites.
    """
    remark_lines = [
        "E-path coordinate subset.",
        "B_iso_or_equiv stores BDamage.",
        *remarks,
    ]
    return _write_mmcif(
        output_path,
        [(atom.record, atom.bdamage_result.bdamage) for atom in atoms],
        title=title,
        remarks=remark_lines,
        values=values or {},
    )


def _write_mmcif(
    output_path: str | Path,
    atoms: Sequence[tuple[AtomRecord, float]],
    *,
    title: str,
    remarks: Sequence[str],
    values: Mapping[str, str],
) -> Path:
    """Build a Gemmi structure from the atoms and write it as mmCIF.

    Each atom is its record and the value to write in its B-factor column.
    Everything else about it is written as the record gives it. Atoms are
    renumbered from one in output order. Consecutive atoms that
    share a chain, residue number, insertion code, and residue name are
    grouped into one residue; a repeated residue elsewhere in the sequence
    becomes a second residue block, mirroring the input order.

    The run metadata goes into `_struct.title`, an `_epath` category of
    machine-readable values, and an `_epath_remark` loop of human-readable
    notes, all of which viewers ignore. The data block is named after the file
    stem. The output directory is created if it is missing.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model = gemmi.Model(1)
    for chain_id, chain_atoms in groupby(atoms, key=lambda atom: atom[0].chain_id):
        chain = gemmi.Chain(chain_id)
        for _, group in groupby(
            chain_atoms,
            key=lambda atom: (
                atom[0].residue_number,
                atom[0].insertion_code,
                atom[0].residue_name,
            ),
        ):
            residue_atoms = tuple(group)
            residue = _make_residue(residue_atoms[0][0])
            for record, b_factor in residue_atoms:
                residue.add_atom(_make_atom(record, b_factor))
            # Gemmi copies on add, so a residue is only added once complete.
            chain.add_residue(residue)
        model.add_chain(chain)

    structure = gemmi.Structure()
    # A CIF data-block name cannot hold whitespace.
    structure.name = "".join(output_file.stem.split())
    structure.add_model(model)
    structure.assign_serial_numbers()
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


def _make_residue(record: AtomRecord) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = record.residue_name or "UNK"
    # Gemmi needs a number, and takes a space as the absent insertion code.
    residue.seqid = gemmi.SeqId(
        record.residue_number or 0, record.insertion_code or " "
    )
    residue.het_flag = "H" if record.is_hetatm else "A"
    return residue


def _make_atom(record: AtomRecord, b_factor: float) -> gemmi.Atom:
    atom = gemmi.Atom()
    atom.name = record.atom_name or record.element
    atom.element = gemmi.Element(record.element)
    atom.pos = gemmi.Position(record.x, record.y, record.z)
    atom.occ = record.occupancy
    atom.b_iso = b_factor
    return atom
