"""Download and cache mmCIF structures from RCSB.

Each entry is cached as `<ID>.cif`, with the ID in upper case, so a structure
is fetched once and later runs read it from disk. A cache file only ever
appears whole, because downloads are written to a temporary file beside it and
renamed into place.
"""

import os
import uuid
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from xtal.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureInputError,
    StructureSourceType,
    is_valid_classic_pdb_id,
)


class RcsbDownloadError(StructureInputError):
    """Report a structure that cannot be downloaded from RCSB.

    It subclasses `StructureInputError` so that callers treat a failed
    download like any other structure input that cannot be used.
    """


RCSB_MMCIF_DOWNLOAD_URL = "https://files.rcsb.org/download/{structure_id}.cif"
DEFAULT_RCSB_DOWNLOAD_DIR = Path("downloads") / "rcsb"

RCSB_DOWNLOAD_TIMEOUT_SECONDS = 30.0

_CIF_WHITESPACE = b" \t\r\n\f\v"


def download_rcsb_mmcif(
    resolved_input: ResolvedStructureInput,
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> ResolvedStructureInput:
    """Return a local mmCIF copy of an RCSB entry, downloading it unless cached.

    `download_dir` is created if it does not exist. A nonempty cache file is
    reused without checking its contents unless `overwrite` is true, and an
    empty one is downloaded again. A failed download or overwrite leaves any
    existing cache file as it was. The returned input names the cached file
    and keeps the original input text, with the PDB ID normalised to upper
    case.

    Every failure to obtain a usable file from RCSB raises `RcsbDownloadError`.
    That covers an input that is not a valid classic PDB ID, a missing entry,
    any other HTTP error, a connection that fails or stalls, a truncated
    response, and a body that is empty or does not start with an mmCIF
    `data_` block. Failing to create the directory or write the cache file
    raises `OSError`.
    """
    structure_id = _normalised_structure_id(resolved_input)
    output_dir = Path(download_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{structure_id}.cif"

    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return _as_downloaded_local_input(
            resolved_input=resolved_input,
            local_path=output_path,
            structure_id=structure_id,
        )

    cif_bytes = _fetch_rcsb_mmcif(structure_id)
    _write_atomically(output_path, cif_bytes)

    return _as_downloaded_local_input(
        resolved_input=resolved_input,
        local_path=output_path,
        structure_id=structure_id,
    )


def _normalised_structure_id(resolved_input: ResolvedStructureInput) -> str:
    """Return the input's PDB ID in upper case, rejecting anything else.

    The resolver already validates IDs, but this check keeps a hand-built input
    from naming a cache file outside the download directory or an arbitrary
    URL path.
    """
    if resolved_input.source_type != StructureSourceType.RCSB_ID:
        raise RcsbDownloadError(
            "download_rcsb_mmcif expected an RCSB_ID input, "
            f"but got {resolved_input.source_type.value!r}."
        )

    if resolved_input.structure_id is None:
        raise RcsbDownloadError("RCSB input is missing a structure ID.")

    structure_id = resolved_input.structure_id.strip().upper()
    if not is_valid_classic_pdb_id(structure_id):
        raise RcsbDownloadError(
            f"RCSB input has an invalid classic PDB ID {resolved_input.structure_id!r}."
        )

    return structure_id


def _as_downloaded_local_input(
    *,
    resolved_input: ResolvedStructureInput,
    local_path: Path,
    structure_id: str,
) -> ResolvedStructureInput:
    """Describe a cached RCSB download as a local mmCIF input."""
    # structure_id is carried over so later stages can still report which RCSB
    # entry the downloaded file came from.
    return ResolvedStructureInput(
        original_input=resolved_input.original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=local_path.resolve(),
        structure_id=structure_id,
    )


def _fetch_rcsb_mmcif(structure_id: str) -> bytes:
    """Return the mmCIF body RCSB serves for a normalised PDB ID."""
    url = RCSB_MMCIF_DOWNLOAD_URL.format(structure_id=structure_id.lower())

    # The order matters, since HTTPError is a URLError and both are OSErrors.
    # urllib wraps only failures while sending the request in URLError, so a
    # stall, reset or truncation while the response arrives reaches the last
    # clause as a bare OSError or HTTPException.
    try:
        with urlopen(url, timeout=RCSB_DOWNLOAD_TIMEOUT_SECONDS) as response:
            cif_bytes: bytes = response.read()
    except HTTPError as error:
        if error.code == 404:
            raise RcsbDownloadError(
                f"Could not find RCSB/PDB entry {structure_id!r}."
            ) from error

        raise RcsbDownloadError(
            f"RCSB download failed for {structure_id!r} with HTTP status {error.code}."
        ) from error

    except URLError as error:
        raise RcsbDownloadError(
            f"Could not connect to RCSB to download {structure_id!r}: {error.reason}"
        ) from error

    except (HTTPException, OSError) as error:
        reason = str(error) or type(error).__name__
        raise RcsbDownloadError(
            f"RCSB download for {structure_id!r} was interrupted ({reason})."
        ) from error

    if not cif_bytes:
        raise RcsbDownloadError(
            f"RCSB download for {structure_id!r} returned an empty file."
        )

    if not _starts_with_data_block(cif_bytes):
        raise RcsbDownloadError(
            f"RCSB download for {structure_id!r} is not an mmCIF file, "
            "because it does not start with a data_ block."
        )

    return cif_bytes


def _starts_with_data_block(cif_bytes: bytes) -> bool:
    """Return whether the first content after whitespace and comments is `data_`.

    This is a cheap guard against caching an HTML error or proxy page served
    with status 200, not a check that the file parses. CIF reserved words are
    case-insensitive, so `DATA_` is accepted too.
    """
    position = 0
    while True:
        while position < len(cif_bytes) and cif_bytes[position] in _CIF_WHITESPACE:
            position += 1

        if not cif_bytes.startswith(b"#", position):
            break

        line_end = cif_bytes.find(b"\n", position)
        if line_end == -1:
            return False
        position = line_end + 1

    return cif_bytes[position : position + 5].lower() == b"data_"


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace `path` with `data` so readers see the old file or the new one.

    The temporary file sits in the same directory, because `os.replace` is only
    atomic within one filesystem. Its name is unique per call so concurrent
    runs fetching the same entry do not write into each other's files.
    """
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        # Opening with "xb" rather than through tempfile.mkstemp keeps the
        # usual umask permissions instead of forcing 0600 on the cache file.
        with temp_path.open("xb") as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
