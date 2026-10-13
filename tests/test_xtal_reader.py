"""Structure reading: atom records, metadata, and rejection of bad files."""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
from support import (
    SYNTHETIC_CELL,
    SYNTHETIC_SPACE_GROUP,
    approx,
    format_pdb_atom,
    synthetic_atoms,
    synthetic_pdb_text,
    write_synthetic_pdb,
)

from xtal.cell import CrystalMetadataError, unit_cell_from_metadata
from xtal.reader import StructureData, StructureReadError, read_structure
from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
    resolve_structure_input,
)


def _read(path: Path) -> StructureData:
    """Read a local structure file through the input resolver."""
    return read_structure(resolve_structure_input(str(path)))


def test_pdb_atoms_are_read_completely_in_file_order(synthetic_pdb: Path) -> None:
    data = _read(synthetic_pdb)
    source_atoms = synthetic_atoms()

    assert len(data.atoms) == len(source_atoms) == 88
    for index, (record, source) in enumerate(
        zip(data.atoms, source_atoms, strict=True)
    ):
        assert record.source_atom_index == index
        assert record.atom_serial == source.serial
        assert record.model_number == 1
        assert (record.chain_id, record.residue_name, record.residue_number) == (
            source.chain_id,
            source.residue_name,
            source.residue_number,
        )
        assert record.atom_name == source.name
        assert record.element.upper() == source.element.upper()
        assert record.altloc == source.altloc
        assert (record.x, record.y, record.z) == approx(
            (source.x, source.y, source.z), abs=5e-4
        )
        assert record.occupancy == approx(source.occupancy)
        assert record.b_factor == approx(source.b_factor)
        assert record.is_hetatm is (source.record == "HETATM")
        assert record.insertion_code == ""


def test_a_split_chain_is_not_merged(synthetic_pdb: Path) -> None:
    # The synthetic file lists chain A's heteroatoms after chain B. Gemmi's
    # default would move them up to the rest of chain A.
    chains = [record.chain_id for record in _read(synthetic_pdb).atoms]

    assert "A" in chains[chains.index("B") :]


def test_pdb_metadata(synthetic_pdb: Path) -> None:
    metadata = _read(synthetic_pdb).metadata

    assert metadata.source_path == synthetic_pdb.resolve()
    assert metadata.structure_id is None
    assert metadata.file_format is StructureFileFormat.PDB
    assert metadata.space_group == SYNTHETIC_SPACE_GROUP
    cell = (
        metadata.unit_cell_a,
        metadata.unit_cell_b,
        metadata.unit_cell_c,
        metadata.unit_cell_alpha,
        metadata.unit_cell_beta,
        metadata.unit_cell_gamma,
    )
    assert cell == approx(SYNTHETIC_CELL)
    assert metadata.resolution_angstrom == approx(1.8)
    assert metadata.gemmi_cell is not None


def _write_file_order_mmcif(path: Path, pdb_path: Path) -> Path:
    """Convert a PDB file to mmCIF, keeping the PDB file's atom order."""
    structure = gemmi.read_structure(str(pdb_path), merge_chain_parts=False)
    structure.setup_entities()
    document = structure.make_mmcif_document()
    document.sole_block().set_pair("_refine.ls_d_res_high", "1.80")
    document.write_file(str(path))
    return path


def test_mmcif_reads_the_same_model(synthetic_pdb: Path, tmp_path: Path) -> None:
    pdb = _read(synthetic_pdb)
    cif = _read(_write_file_order_mmcif(tmp_path / "model.cif", synthetic_pdb))

    assert cif.metadata.file_format is StructureFileFormat.MMCIF
    assert len(cif.atoms) == len(pdb.atoms)
    assert cif.metadata.space_group == pdb.metadata.space_group
    assert cif.metadata.unit_cell_beta == approx(95.0)
    assert cif.metadata.resolution_angstrom == approx(1.8)
    for a, b in zip(cif.atoms, pdb.atoms, strict=True):
        assert (
            a.source_atom_index,
            a.chain_id,
            a.residue_name,
            a.residue_number,
            a.atom_name,
            a.altloc,
            a.is_hetatm,
        ) == (
            b.source_atom_index,
            b.chain_id,
            b.residue_name,
            b.residue_number,
            b.atom_name,
            b.altloc,
            b.is_hetatm,
        )
        assert (a.x, a.y, a.z) == approx((b.x, b.y, b.z), abs=1e-3)


def test_structure_id_is_carried_from_a_downloaded_input(synthetic_mmcif: Path) -> None:
    resolved = ResolvedStructureInput(
        original_input="1abc",
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=synthetic_mmcif,
        structure_id="1ABC",
    )
    assert read_structure(resolved).metadata.structure_id == "1ABC"


@pytest.mark.parametrize(
    ("options", "space_group"),
    [
        ({"include_cryst1": False}, None),
        # EM and NMR entries carry Gemmi's placeholder cube explicitly.
        ({"cell": (1.0, 1.0, 1.0, 90.0, 90.0, 90.0), "space_group": "P 1"}, "P 1"),
    ],
)
def test_a_missing_or_placeholder_cell_is_undefined(
    tmp_path: Path, options: dict[str, object], space_group: str | None
) -> None:
    # Gemmi represents an absent cell as a 1 A cube, which must not be
    # reported as a real cell.
    path = write_synthetic_pdb(tmp_path / "nocell.pdb", **options)

    metadata = _read(path).metadata

    assert metadata.space_group == space_group
    assert metadata.unit_cell_a is None and metadata.unit_cell_gamma is None
    assert metadata.gemmi_cell is None
    with pytest.raises(CrystalMetadataError, match="unit-cell"):
        unit_cell_from_metadata(metadata)


@pytest.mark.parametrize(
    ("resolution_line", "expected"),
    [
        (None, None),
        ("REMARK   2 RESOLUTION. NOT APPLICABLE.", None),
        ("REMARK   2 RESOLUTION.    2.35 ANGSTROMS.", 2.35),
        # Gemmi rejects these two, so they exercise the reader's fallback.
        ("REMARK 2 RESOLUTION. 2.35", 2.35),
        ("REMARK   2 RESOLUTION.  2.35 A", 2.35),
    ],
)
def test_pdb_resolution_parsing(
    tmp_path: Path, resolution_line: str | None, expected: float | None
) -> None:
    text = synthetic_pdb_text(resolution=None)
    if resolution_line is not None:
        text = text.replace(
            "HEADER    SYNTHETIC TEST STRUCTURE\n",
            f"HEADER    SYNTHETIC TEST STRUCTURE\n{resolution_line}\n",
        )
    path = tmp_path / "res.pdb"
    path.write_text(text)

    assert _read(path).metadata.resolution_angstrom == expected


def test_multi_model_files_are_rejected(tmp_path: Path) -> None:
    atom = format_pdb_atom(synthetic_atoms()[1])
    path = tmp_path / "nmr.pdb"
    path.write_text(
        f"MODEL        1\n{atom}\nENDMDL\nMODEL        2\n{atom}\nENDMDL\nEND\n"
    )

    with pytest.raises(StructureReadError, match="found 2 models"):
        _read(path)


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("empty.pdb", "HEADER    EMPTY\nEND\n"),
        # Gemmi builds no model at all from an mmCIF file without atoms.
        ("empty.cif", "data_empty\n_cell.length_a 10.0\n"),
    ],
)
def test_files_without_atoms_are_rejected(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text)

    with pytest.raises(StructureReadError, match="No atom records"):
        _read(path)


def test_unreadable_inputs_are_rejected(tmp_path: Path) -> None:
    rcsb_input = ResolvedStructureInput(
        "1abc",
        StructureSourceType.RCSB_ID,
        StructureFileFormat.MMCIF,
        structure_id="1ABC",
    )
    with pytest.raises(StructureReadError, match="expected a local file"):
        read_structure(rcsb_input)

    missing = ResolvedStructureInput(
        "m.pdb",
        StructureSourceType.LOCAL_FILE,
        StructureFileFormat.PDB,
        local_path=tmp_path / "m.pdb",
    )
    with pytest.raises(StructureReadError, match="does not exist"):
        read_structure(missing)

    garbage = tmp_path / "garbage.cif"
    garbage.write_text("this is not a cif file\n")
    with pytest.raises(StructureReadError):
        read_structure(resolve_structure_input(str(garbage)))


_PAIR_RESOLUTION = "_refine.ls_d_res_high 1.80\n"


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        # Without pdbx_refine_id Gemmi leaves the resolution unset.
        (_PAIR_RESOLUTION, 1.8),
        ("loop_\n_refine.ls_d_res_high\n1.90\n2.20\n", 1.9),
        ("loop_\n_refine.ls_d_res_high\n?\n2.20\n", 2.2),
        ("_reflns.d_resolution_high 2.50\n", 2.5),
        ("_refine.ls_d_res_high ?\n", None),
        ("", None),
    ],
)
def test_mmcif_resolution_fallback(
    tmp_path: Path, synthetic_mmcif: Path, replacement: str, expected: float | None
) -> None:
    text = synthetic_mmcif.read_text()
    assert _PAIR_RESOLUTION in text
    path = tmp_path / "res.cif"
    path.write_text(text.replace(_PAIR_RESOLUTION, replacement))

    resolution = _read(path).metadata.resolution_angstrom

    assert resolution == (None if expected is None else approx(expected))


def test_a_missing_atom_serial_is_none(tmp_path: Path) -> None:
    atoms = synthetic_atoms()[:3]
    lines = [format_pdb_atom(atom) for atom in atoms]
    # Blank the serial field, columns 7 to 11, of the middle atom. Gemmi
    # reports it as 0, which is not a serial.
    lines[1] = f"{lines[1][:6]}     {lines[1][11:]}"
    path = tmp_path / "noserial.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    serials = [record.atom_serial for record in _read(path).atoms]

    assert serials == [atoms[0].serial, None, atoms[2].serial]
