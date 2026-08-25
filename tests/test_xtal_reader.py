"""Structure reading: atom records, metadata, and rejection of bad files."""

from __future__ import annotations

from pathlib import Path

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


def test_pdb_atoms_are_read_completely(synthetic_pdb: Path) -> None:
    data = _read(synthetic_pdb)
    expected = {atom.serial: atom for atom in synthetic_atoms()}

    assert len(data.atoms) == len(expected) == 88
    assert [record.source_atom_index for record in data.atoms] == list(range(88))
    # Gemmi groups the trailing chain-A heteroatoms with chain A, so records
    # are matched by serial rather than by file position.
    serials = [record.atom_serial for record in data.atoms]
    assert all(serial is not None for serial in serials)
    assert sorted(serial or 0 for serial in serials) == sorted(expected)
    for record in data.atoms:
        assert record.atom_serial is not None
        source = expected[record.atom_serial]
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


def test_mmcif_reads_the_same_model(synthetic_pdb: Path, synthetic_mmcif: Path) -> None:
    pdb = _read(synthetic_pdb)
    cif = _read(synthetic_mmcif)

    assert cif.metadata.file_format is StructureFileFormat.MMCIF
    assert len(cif.atoms) == len(pdb.atoms)
    assert cif.metadata.space_group == pdb.metadata.space_group
    assert cif.metadata.unit_cell_beta == approx(95.0)
    assert cif.metadata.resolution_angstrom == approx(1.8)
    for a, b in zip(cif.atoms, pdb.atoms, strict=True):
        assert (
            a.chain_id,
            a.residue_name,
            a.residue_number,
            a.atom_name,
            a.altloc,
            a.is_hetatm,
        ) == (
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


def test_missing_cryst1_leaves_the_unit_cell_undefined(tmp_path: Path) -> None:
    # Gemmi represents an absent cell as a 1 A cube; it must not be reported
    # as a real cell.
    path = write_synthetic_pdb(tmp_path / "nocell.pdb", include_cryst1=False)

    metadata = _read(path).metadata

    assert metadata.space_group is None
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


def test_files_without_atoms_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdb"
    path.write_text("HEADER    EMPTY\nEND\n")

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
