"""Input resolution and RCSB download handling."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import NoReturn
from urllib.error import HTTPError, URLError

import pytest
from support import approx, synthetic_pdb_text

import xtal.rcsb as rcsb
from xtal.reader import read_structure
from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureInputError,
    StructureSourceType,
    detect_structure_format_from_path,
    is_valid_classic_pdb_id,
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
    assert resolved.is_local_file and not resolved.needs_download
    assert resolved.structure_id is None


def _skip_unless_case_sensitive(tmp_path: Path) -> None:
    """Skip a test that relies on a case-sensitive filesystem.

    macOS and Windows normally use case-insensitive filesystems, where a
    differently-cased name already resolves to the file directly and the
    fallback matching in the resolver never runs.
    """
    (tmp_path / "CaseProbe").write_text("x")
    if (tmp_path / "caseprobe").exists():
        pytest.skip("requires a case-sensitive filesystem")


def test_local_file_name_matches_case_insensitively(tmp_path: Path) -> None:
    _skip_unless_case_sensitive(tmp_path)
    (tmp_path / "Model.PDB").write_text("x")

    resolved = resolve_structure_input(str(tmp_path / "model.pdb"))

    assert resolved.local_path == (tmp_path / "Model.PDB").resolve()


def test_ambiguous_case_insensitive_match_is_rejected(tmp_path: Path) -> None:
    _skip_unless_case_sensitive(tmp_path)
    (tmp_path / "a.pdb").write_text("x")
    (tmp_path / "A.pdb").write_text("x")

    with pytest.raises(StructureInputError, match="ambiguous"):
        resolve_structure_input(str(tmp_path / "a.PDB"))


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
    assert actual.metadata.file_format is expected.metadata.file_format
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
    assert resolved.needs_download and not resolved.is_local_file
    assert resolved.original_input == "  1lyz "


@pytest.mark.parametrize("value", ["", "   ", "12345", "ABCD", "1AB", "missing.pdb"])
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


@pytest.mark.parametrize(
    ("value", "valid"),
    [("1LYZ", True), ("7xyz", True), ("0000", True), ("ABCD", False), ("1LY", False)],
)
def test_is_valid_classic_pdb_id(value: str, valid: bool) -> None:
    assert is_valid_classic_pdb_id(value) is valid


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


def test_case_insensitive_match_that_is_a_directory_is_rejected(
    tmp_path: Path,
) -> None:
    _skip_unless_case_sensitive(tmp_path)
    (tmp_path / "Model.PDB").mkdir()

    with pytest.raises(StructureInputError, match="directory"):
        resolve_structure_input(str(tmp_path / "model.pdb"))


def test_unlistable_parent_is_a_resolution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(self: Path) -> NoReturn:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", denied)

    with pytest.raises(StructureInputError, match="Permission denied") as info:
        resolve_structure_input(str(tmp_path / "model.pdb"))
    assert isinstance(info.value.__cause__, PermissionError)


def test_unknown_home_directory_is_a_resolution_error() -> None:
    with pytest.raises(StructureInputError, match="home directory"):
        resolve_structure_input("~no_such_epath_test_user/model.pdb")


def test_tilde_is_expanded_but_environment_variables_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EPATH_TEST_DIR", str(tmp_path))
    path = tmp_path / "model.pdb"
    path.write_text("x")

    assert resolve_structure_input("~/model.pdb").local_path == path.resolve()
    with pytest.raises(StructureInputError, match="Could not resolve"):
        resolve_structure_input("$EPATH_TEST_DIR/model.pdb")


def test_filename_with_a_literal_dollar_sign_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/elsewhere")
    path = tmp_path / "x$HOME.pdb"
    path.write_text("x")

    assert resolve_structure_input(str(path)).local_path == path.resolve()


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


def test_downloaded_input_may_keep_its_structure_id(tmp_path: Path) -> None:
    resolved = ResolvedStructureInput(
        original_input="1abc",
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=tmp_path / "1ABC.cif",
        structure_id="1ABC",
    )

    assert resolved.is_local_file and resolved.structure_id == "1ABC"


class _FakeResponse(io.BytesIO):
    """Stand in for the context manager returned by `urlopen`."""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _rcsb_input(structure_id: str = "1ABC") -> ResolvedStructureInput:
    """Return a resolved RCSB identifier input for `structure_id`."""
    return ResolvedStructureInput(
        original_input=structure_id.lower(),
        source_type=StructureSourceType.RCSB_ID,
        file_format=StructureFileFormat.MMCIF,
        structure_id=structure_id,
    )


def test_download_writes_cache_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(b"data_1ABC\n")

    monkeypatch.setattr(rcsb, "urlopen", fake_urlopen)

    local = rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path / "cache")

    assert calls == ["https://files.rcsb.org/download/1abc.cif"]
    assert local.source_type is StructureSourceType.LOCAL_FILE
    assert local.structure_id == "1ABC"
    assert local.local_path == (tmp_path / "cache" / "1ABC.cif").resolve()
    assert local.local_path is not None
    assert local.local_path.read_bytes() == b"data_1ABC\n"

    again = rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path / "cache")
    assert len(calls) == 1, "a cached file must not be downloaded again"
    assert again.local_path == local.local_path

    rcsb.download_rcsb_mmcif(
        _rcsb_input(), download_dir=tmp_path / "cache", overwrite=True
    )
    assert len(calls) == 2


def test_download_errors_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found(url: str, timeout: float) -> _FakeResponse:
        raise HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(rcsb, "urlopen", not_found)
    with pytest.raises(
        rcsb.RcsbDownloadError, match="Could not find RCSB/PDB entry '1ABC'"
    ):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    def server_error(url: str, timeout: float) -> _FakeResponse:
        raise HTTPError(url, 503, "Unavailable", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(rcsb, "urlopen", server_error)
    with pytest.raises(rcsb.RcsbDownloadError, match="HTTP status 503"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    def offline(url: str, timeout: float) -> _FakeResponse:
        raise URLError("no network")

    monkeypatch.setattr(rcsb, "urlopen", offline)
    with pytest.raises(rcsb.RcsbDownloadError, match="Could not connect"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    def empty(url: str, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"")

    monkeypatch.setattr(rcsb, "urlopen", empty)
    with pytest.raises(rcsb.RcsbDownloadError, match="empty file"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)
    assert not (tmp_path / "1ABC.cif").exists()


def test_a_local_input_is_not_downloaded(tmp_path: Path) -> None:
    path = tmp_path / "m.pdb"
    path.write_text("x")
    local = resolve_structure_input(str(path))

    assert not local.needs_download
    with pytest.raises(rcsb.RcsbDownloadError, match="expected an RCSB_ID input"):
        rcsb.download_rcsb_mmcif(local)
