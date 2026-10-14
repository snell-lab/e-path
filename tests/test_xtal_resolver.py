"""Input resolution: local files, their formats, and PDB identifiers."""

import gzip
import sys
from pathlib import Path
from typing import NoReturn

import pytest
from support import approx, synthetic_pdb_text

from xtal.reader import read_structure
from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureInputError,
    StructureSourceType,
    detect_structure_format_from_path,
    resolve_structure_input,
    split_structure_file_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.pdb", StructureFileFormat.PDB),
        ("model.cif", StructureFileFormat.MMCIF),
        ("m.MMCIF", StructureFileFormat.MMCIF),
        ("pdb1abc.ENT", StructureFileFormat.PDB),
        ("m.Cif.GZ", StructureFileFormat.MMCIF),
        ("m.mmCIF.gz", StructureFileFormat.MMCIF),
        ("m.PDB.Gz", StructureFileFormat.PDB),
        ("pdb1abc.Ent.gZ", StructureFileFormat.PDB),
    ],
)
def test_local_file_resolves_with_its_format(
    tmp_path: Path, name: str, expected: StructureFileFormat
) -> None:
    path = tmp_path / name
    path.write_text("x")

    resolved = resolve_structure_input(str(path))

    assert resolved.source_type is StructureSourceType.LOCAL_FILE
    assert resolved.file_format is expected
    assert resolved.local_path == path.resolve()
    assert not resolved.needs_download
    assert resolved.structure_id is None


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("model.txt", "Unsupported structure file extension"),
        ("model", "no extension"),
        ("model.gz", "must name its format before .gz"),
        ("model.GZ", "must name its format before .gz"),
        (".gz", "must name its format before .gz"),
        ("model.txt.gz", "Unsupported structure file extension"),
        ("model.pdb.gz.gz", "Unsupported structure file extension"),
    ],
)
def test_unsupported_or_missing_extension_is_rejected(
    tmp_path: Path, name: str, message: str
) -> None:
    path = tmp_path / name
    path.write_text("x")

    with pytest.raises(StructureInputError, match=message):
        resolve_structure_input(str(path))
    with pytest.raises(StructureInputError, match=message):
        detect_structure_format_from_path(path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.v2.pdb", StructureFileFormat.PDB),
        ("model.final.CIF", StructureFileFormat.MMCIF),
        ("model.v2.pdb.gz", StructureFileFormat.PDB),
        ("....pdb", StructureFileFormat.PDB),
    ],
)
def test_only_the_final_suffix_decides_the_format(
    tmp_path: Path, name: str, expected: StructureFileFormat
) -> None:
    path = tmp_path / name
    path.write_text("x")

    assert resolve_structure_input(str(path)).file_format is expected


def test_error_messages_list_the_supported_extensions(tmp_path: Path) -> None:
    path = tmp_path / "model.txt"
    path.write_text("x")

    with pytest.raises(StructureInputError, match=r"\.cif, \.mmcif, \.pdb, or \.ent"):
        detect_structure_format_from_path(path)
    with pytest.raises(StructureInputError, match=r"\.cif, \.mmcif, \.pdb, or \.ent"):
        resolve_structure_input(str(tmp_path / "missing.pdb"))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("1ABC.cif.GZ", ("1ABC", ".cif")),
        ("x.mmcif", ("x", ".mmcif")),
        ("model.v2.pdb.gz", ("model.v2", ".pdb")),
        ("pdb1abc.ent.gz", ("pdb1abc", ".ent")),
        ("....pdb", ("...", ".pdb")),
        ("model.gz", ("model.gz", "")),
        ("model.txt", ("model.txt", "")),
    ],
)
def test_split_structure_file_name(name: str, expected: tuple[str, str]) -> None:
    assert split_structure_file_name(name) == expected


_FALLBACK_REMARK = "REMARK 2 RESOLUTION. 2.35"


@pytest.mark.parametrize(
    ("name", "resolution"),
    [
        ("m.pdb.gz", 2.35),
        ("pdb1abc.ent", 2.35),
        ("pdb1abc.ent.gz", 2.35),
        ("m.cif.gz", 1.8),
        ("m.mmcif.gz", 1.8),
    ],
)
def test_gzipped_and_ent_files_read_like_plain_ones(
    tmp_path: Path, synthetic_mmcif: Path, name: str, resolution: float
) -> None:
    # Both resolutions are written so that Gemmi leaves them unset and the
    # reader's fallback, which reads remarks or the saved mmCIF document, has
    # to find them in the decompressed file.
    if ".cif" in name or ".mmcif" in name:
        plain = synthetic_mmcif
    else:
        plain = tmp_path / "plain.pdb"
        plain.write_text(
            synthetic_pdb_text(resolution=None).replace(
                "HEADER    SYNTHETIC TEST STRUCTURE\n",
                f"HEADER    SYNTHETIC TEST STRUCTURE\n{_FALLBACK_REMARK}\n",
            )
        )
    path = tmp_path / name
    data = plain.read_bytes()
    path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)

    expected = read_structure(resolve_structure_input(str(plain)))
    actual = read_structure(resolve_structure_input(str(path)))

    assert actual.atoms == expected.atoms
    assert actual.metadata.resolution_angstrom == approx(resolution)
    assert actual.metadata.structure_id is None


def test_directory_is_not_a_structure_file(tmp_path: Path) -> None:
    with pytest.raises(StructureInputError, match="directory"):
        resolve_structure_input(str(tmp_path))


def test_pdb_id_resolves_to_rcsb_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = resolve_structure_input("  1lyz ")

    assert resolved.source_type is StructureSourceType.RCSB_ID
    assert resolved.structure_id == "1LYZ"
    assert resolved.file_format is StructureFileFormat.MMCIF
    assert resolved.needs_download
    assert resolved.original_input == "  1lyz "


@pytest.mark.parametrize("value", ["12345", "ABCD", "1AB", "missing.pdb"])
def test_unresolvable_inputs_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(StructureInputError):
        resolve_structure_input(value)


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_input_is_reported_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # Without the check, blank text would become Path("."), the working directory.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(StructureInputError, match="cannot be empty"):
        resolve_structure_input(value)


@pytest.mark.parametrize("kind", ["directory", "extensionless file"])
def test_pdb_id_is_not_shadowed_by_a_local_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.chdir(tmp_path)
    shadow = tmp_path / "1LYZ"
    if kind == "directory":
        shadow.mkdir()
    else:
        shadow.write_text("x")

    resolved = resolve_structure_input("1lyz")

    assert resolved.needs_download
    assert resolved.structure_id == "1LYZ"


def test_filesystem_error_is_a_resolution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path) -> NoReturn:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", denied)

    with pytest.raises(StructureInputError, match="Permission denied") as info:
        resolve_structure_input(str(tmp_path / "model.pdb"))
    assert isinstance(info.value.__cause__, PermissionError)


# Windows does not fail on an unknown user. It guesses a home directory beside
# the current user's, so the input is reported as a missing file instead.
@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX home lookup")
def test_unknown_home_directory_is_an_input_error() -> None:
    with pytest.raises(StructureInputError, match="home directory"):
        resolve_structure_input("~no_such_epath_test_user/model.pdb")


def test_tilde_is_expanded_but_environment_variables_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Windows reads the home directory from USERPROFILE rather than HOME.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("EPATH_TEST_DIR", str(tmp_path))
    path = tmp_path / "model.pdb"
    path.write_text("x")

    assert resolve_structure_input("~/model.pdb").local_path == path.resolve()
    with pytest.raises(StructureInputError, match="Could not resolve"):
        resolve_structure_input("$EPATH_TEST_DIR/model.pdb")
    # So a file whose name holds a dollar sign is found as written.
    literal = tmp_path / "x$HOME.pdb"
    literal.write_text("x")
    assert resolve_structure_input(str(literal)).local_path == literal.resolve()


@pytest.mark.parametrize(
    ("source_type", "local_path", "structure_id", "message"),
    [
        (StructureSourceType.LOCAL_FILE, None, None, "needs a local_path"),
        (StructureSourceType.RCSB_ID, None, None, "needs a structure_id"),
        (StructureSourceType.RCSB_ID, None, "", "needs a structure_id"),
        (
            StructureSourceType.RCSB_ID,
            Path("1ABC.cif"),
            "1ABC",
            "cannot have a local_path",
        ),
    ],
)
def test_resolved_input_rejects_mismatched_fields(
    source_type: StructureSourceType,
    local_path: Path | None,
    structure_id: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ResolvedStructureInput(
            original_input="x",
            source_type=source_type,
            file_format=StructureFileFormat.MMCIF,
            local_path=local_path,
            structure_id=structure_id,
        )
