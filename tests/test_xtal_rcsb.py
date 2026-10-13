"""RCSB download caching, validation and failure handling."""

from __future__ import annotations

import errno
import io
from collections.abc import Callable
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError

import pytest

import xtal.rcsb as rcsb
from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)

VALID_CIF = b"data_1ABC\n_entry.id 1ABC\n"


class _FakeResponse(io.BytesIO):
    """Stand in for the context manager returned by `urlopen`."""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _FailingResponse(_FakeResponse):
    """Fail while the body is being read, after the request succeeded."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def read(self, size: int | None = -1, /) -> bytes:
        raise self.error


def _rcsb_input(structure_id: str = "1ABC") -> ResolvedStructureInput:
    return ResolvedStructureInput(
        original_input=structure_id,
        source_type=StructureSourceType.RCSB_ID,
        file_format=StructureFileFormat.MMCIF,
        structure_id=structure_id,
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[], _FakeResponse],
) -> list[tuple[str, float]]:
    """Route `urlopen` to `respond` and return the calls it receives."""
    calls: list[tuple[str, float]] = []

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        calls.append((url, timeout))
        return respond()

    monkeypatch.setattr(rcsb, "urlopen", fake_urlopen)
    return calls


def _leftover_temp_files(directory: Path) -> list[Path]:
    return sorted(directory.glob(".*.part"))


def test_download_normalises_id_and_uses_timeout_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _serve(monkeypatch, lambda: _FakeResponse(VALID_CIF))

    local = rcsb.download_rcsb_mmcif(_rcsb_input(" 1abc "), download_dir=tmp_path)

    assert calls == [
        (
            "https://files.rcsb.org/download/1abc.cif",
            rcsb.RCSB_DOWNLOAD_TIMEOUT_SECONDS,
        )
    ]
    assert local.structure_id == "1ABC"
    assert local.original_input == " 1abc "
    assert local.local_path == (tmp_path / "1ABC.cif").resolve()
    assert (tmp_path / "1ABC.cif").read_bytes() == VALID_CIF
    assert _leftover_temp_files(tmp_path) == []


def test_empty_cache_file_is_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "1ABC.cif").write_bytes(b"")
    calls = _serve(monkeypatch, lambda: _FakeResponse(VALID_CIF))

    rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    assert len(calls) == 1
    assert (tmp_path / "1ABC.cif").read_bytes() == VALID_CIF


def test_failed_overwrite_download_keeps_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / "1ABC.cif"
    cached.write_bytes(b"data_OLD\n")

    def unavailable() -> _FakeResponse:
        raise HTTPError("url", 503, "Unavailable", None, None)  # type: ignore[arg-type]

    _serve(monkeypatch, unavailable)

    with pytest.raises(rcsb.RcsbDownloadError, match="HTTP status 503"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path, overwrite=True)

    assert cached.read_bytes() == b"data_OLD\n"


@pytest.mark.parametrize("failing_call", ["fsync", "replace"])
def test_failed_write_keeps_cached_file_and_removes_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    cached = tmp_path / "1ABC.cif"
    cached.write_bytes(b"data_OLD\n")
    _serve(monkeypatch, lambda: _FakeResponse(VALID_CIF))

    def disk_full(*args: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(f"os.{failing_call}", disk_full)

    with pytest.raises(OSError, match="No space left"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path, overwrite=True)

    assert cached.read_bytes() == b"data_OLD\n"
    assert _leftover_temp_files(tmp_path) == []


def test_failed_first_write_leaves_no_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, lambda: _FakeResponse(VALID_CIF))

    def disk_full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("os.fsync", disk_full)

    with pytest.raises(OSError):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [
        IncompleteRead(b"data_1A", 100),
        TimeoutError("timed out"),
        TimeoutError(),
        ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
    ],
    ids=["truncated", "timeout", "timeout-without-message", "reset"],
)
def test_failure_while_reading_is_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    _serve(monkeypatch, lambda: _FailingResponse(error))

    with pytest.raises(rcsb.RcsbDownloadError, match="was interrupted") as caught:
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    assert caught.value.__cause__ is error
    assert not (tmp_path / "1ABC.cif").exists()
    assert _leftover_temp_files(tmp_path) == []


def test_reset_before_response_is_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # urllib wraps only failures while sending the request, so a server that
    # drops the connection before answering surfaces from urlopen unwrapped.
    def reset() -> _FakeResponse:
        raise ConnectionResetError(errno.ECONNRESET, "Remote end closed connection")

    _serve(monkeypatch, reset)

    with pytest.raises(rcsb.RcsbDownloadError, match="was interrupted"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        b"<!DOCTYPE html><html><body>Sign in to the network</body></html>",
        b"HEADER    HYDROLASE                               01-JAN-00   1ABC\n",
        b"# only a comment\n",
        b"   \n\t\n",
    ],
    ids=["html", "pdb-format", "comment-only", "whitespace-only"],
)
def test_non_mmcif_body_is_rejected_and_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    _serve(monkeypatch, lambda: _FakeResponse(body))

    with pytest.raises(rcsb.RcsbDownloadError, match="not an mmCIF file"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_non_mmcif_overwrite_keeps_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / "1ABC.cif"
    cached.write_bytes(b"data_OLD\n")
    _serve(monkeypatch, lambda: _FakeResponse(b"<html></html>"))

    with pytest.raises(rcsb.RcsbDownloadError, match="not an mmCIF file"):
        rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path, overwrite=True)

    assert cached.read_bytes() == b"data_OLD\n"


@pytest.mark.parametrize(
    "body",
    [
        b"\n  \r\ndata_1ABC\n",
        b"# written by a tool\n#\n  data_1ABC\n",
        b"DATA_1ABC\n",
    ],
    ids=["leading-whitespace", "leading-comments", "uppercase-keyword"],
)
def test_mmcif_header_after_whitespace_or_comments_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    _serve(monkeypatch, lambda: _FakeResponse(body))

    local = rcsb.download_rcsb_mmcif(_rcsb_input(), download_dir=tmp_path)

    assert local.local_path is not None
    assert local.local_path.read_bytes() == body


@pytest.mark.parametrize(
    "structure_id",
    ["ABCD", "1AB", "1ABCD", "../1A", "1A/C", "1ABC/../../x"],
)
def test_bad_structure_id_is_rejected_before_any_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, structure_id: str
) -> None:
    calls = _serve(monkeypatch, lambda: _FakeResponse(VALID_CIF))
    download_dir = tmp_path / "cache"

    with pytest.raises(rcsb.RcsbDownloadError, match="invalid classic PDB ID"):
        rcsb.download_rcsb_mmcif(_rcsb_input(structure_id), download_dir=download_dir)

    assert calls == []
    assert not download_dir.exists()
