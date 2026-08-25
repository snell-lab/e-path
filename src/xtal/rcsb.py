"""Download and cache mmCIF structures from RCSB."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from xtal.resolver import (
    InputResolutionError,
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)

RCSB_MMCIF_DOWNLOAD_URL = "https://files.rcsb.org/download/{structure_id}.cif"
DEFAULT_RCSB_DOWNLOAD_DIR = Path("downloads") / "rcsb"


class RcsbDownloadError(InputResolutionError):
    """Report a structure that cannot be downloaded from RCSB."""


class RcsbBatchDownloadError(RcsbDownloadError):
    """Collect RCSB download failures from a batch of inputs."""

    def __init__(
        self,
        errors: list[tuple[ResolvedStructureInput, RcsbDownloadError]],
    ):
        """Initialize the error from resolved-input-and-failure pairs."""
        self.errors = errors
        summary = "\n".join(
            f"- {resolved_input.original_input!r}: {error}"
            for resolved_input, error in errors
        )
        super().__init__(
            "One or more RCSB/PDB inputs could not be downloaded:\n" + summary
        )


def download_rcsb_mmcif(
    resolved_input: ResolvedStructureInput,
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> ResolvedStructureInput:
    """Download and cache an RCSB entry as an mmCIF file.

    Args:
        resolved_input: RCSB identifier produced by the input resolver.
        download_dir: Directory for cached mmCIF files.
        overwrite: Whether to replace an existing nonempty cache file.

    Returns:
        A local-file input that retains the original RCSB structure ID.

    Raises:
        RcsbDownloadError: If the input is not an RCSB ID or the download fails,
            returns no data, or identifies no RCSB entry.
    """
    if resolved_input.source_type != StructureSourceType.RCSB_ID:
        raise RcsbDownloadError(
            "download_rcsb_mmcif expected an RCSB_ID input, "
            f"but got {resolved_input.source_type.value!r}."
        )

    if resolved_input.structure_id is None:
        raise RcsbDownloadError("RCSB input is missing a structure ID.")

    structure_id = resolved_input.structure_id.upper()
    output_dir = Path(download_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{structure_id}.cif"

    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return _as_downloaded_local_input(
            resolved_input=resolved_input,
            local_path=output_path,
        )

    url = RCSB_MMCIF_DOWNLOAD_URL.format(structure_id=structure_id.lower())

    try:
        with urlopen(url, timeout=30) as response:
            cif_bytes = response.read()
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

    if not cif_bytes:
        raise RcsbDownloadError(
            f"RCSB download for {structure_id!r} returned an empty file."
        )

    output_path.write_bytes(cif_bytes)

    return _as_downloaded_local_input(
        resolved_input=resolved_input,
        local_path=output_path,
    )


def ensure_local_structure_file(
    resolved_input: ResolvedStructureInput,
    *,
    download_dir: Path | str = DEFAULT_RCSB_DOWNLOAD_DIR,
    overwrite: bool = False,
) -> ResolvedStructureInput:
    """Return a local input, downloading an RCSB entry when necessary.

    Raises:
        RcsbDownloadError: If the resolved input has a source type that cannot
            be turned into a local file, or if the RCSB download fails.
    """
    if resolved_input.is_local_file:
        return resolved_input

    if resolved_input.needs_download:
        return download_rcsb_mmcif(
            resolved_input,
            download_dir=download_dir,
            overwrite=overwrite,
        )

    raise RcsbDownloadError(
        f"Cannot handle structure input source type: {resolved_input.source_type.value!r}"
    )


def _as_downloaded_local_input(
    *,
    resolved_input: ResolvedStructureInput,
    local_path: Path,
) -> ResolvedStructureInput:
    # structure_id is carried over so later stages can still report which RCSB
    # entry the downloaded file came from.
    return ResolvedStructureInput(
        original_input=resolved_input.original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=StructureFileFormat.MMCIF,
        local_path=local_path.resolve(),
        structure_id=resolved_input.structure_id,
    )
