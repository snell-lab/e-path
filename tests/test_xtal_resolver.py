"""Input resolution and RCSB download handling."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import xtal.rcsb as rcsb
from xtal.resolver import (
    InputResolutionError,
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
    detect_structure_format_from_path,
    is_valid_classic_pdb_id,
    resolve_structure_input,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.pdb", StructureFileFormat.PDB),
        ("model.cif", StructureFileFormat.MMCIF),
        ("m.MMCIF", StructureFileFormat.MMCIF),
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

    with pytest.raises(InputResolutionError, match="ambiguous"):
        resolve_structure_input(str(tmp_path / "a.PDB"))


@pytest.mark.parametrize("name", ["model.txt", "model"])
def test_unsupported_or_missing_extension_is_rejected(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.write_text("x")

    with pytest.raises(InputResolutionError):
        resolve_structure_input(str(path))
    with pytest.raises(InputResolutionError):
        detect_structure_format_from_path(path)


def test_directory_is_not_a_structure_file(tmp_path: Path) -> None:
    with pytest.raises(InputResolutionError, match="directory"):
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
    with pytest.raises(InputResolutionError):
        resolve_structure_input(value)


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(InputResolutionError, match="must be a string"):
        resolve_structure_input(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "valid"),
    [("1LYZ", True), ("7xyz", True), ("0000", True), ("ABCD", False), ("1LY", False)],
)
def test_is_valid_classic_pdb_id(value: str, valid: bool) -> None:
    assert is_valid_classic_pdb_id(value) is valid


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

    again = rcsb.ensure_local_structure_file(
        _rcsb_input(), download_dir=tmp_path / "cache"
    )
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


def test_local_input_passes_through_and_other_kinds_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m.pdb"
    path.write_text("x")
    local = resolve_structure_input(str(path))

    assert rcsb.ensure_local_structure_file(local) is local
    with pytest.raises(rcsb.RcsbDownloadError, match="expected an RCSB_ID input"):
        rcsb.download_rcsb_mmcif(local)
