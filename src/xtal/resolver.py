"""Resolve structure inputs as local files or RCSB identifiers.

Nothing here touches the network; `xtal.rcsb` does the download.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class StructureInputError(ValueError):
    """Report a structure input that cannot be found, downloaded, or read.

    It derives from `ValueError` so the command line reports it with other
    bad-input errors. The download and reading errors in `xtal` derive from it
    in turn.
    """


class StructureSourceType(StrEnum):
    """Say whether an input is a file on disk or an RCSB entry to download.

    A downloaded entry becomes `LOCAL_FILE` and keeps its `structure_id`, so
    this records where the coordinates are now, not where they came from.
    """

    LOCAL_FILE = "local_file"
    RCSB_ID = "rcsb_id"


class StructureFileFormat(StrEnum):
    """Identify a supported coordinate-file format."""

    MMCIF = "mmcif"
    PDB = "pdb"


_FORMAT_BY_SUFFIX = {
    ".cif": StructureFileFormat.MMCIF,
    ".mmcif": StructureFileFormat.MMCIF,
    ".pdb": StructureFileFormat.PDB,
    ".ent": StructureFileFormat.PDB,
}
SUPPORTED_STRUCTURE_SUFFIXES = tuple(_FORMAT_BY_SUFFIX)

# Any of the format extensions may be followed by this one. Gemmi decompresses
# gzipped files itself, so the format is the same as without it.
COMPRESSED_SUFFIX = ".gz"


@dataclass(frozen=True, slots=True)
class ResolvedStructureInput:
    """Describe a validated local file or an RCSB entry still to be downloaded.

    `local_path` is set exactly when the source is `LOCAL_FILE`, and the
    resolver makes it absolute and symlink-resolved. `structure_id` is required
    for an RCSB entry. It survives the download so later stages can still name
    the entry, and it is `None` for a file the user supplied. `original_input`
    is the text exactly as given, before stripping, for use in messages.

    Construction raises `ValueError` when the source type and the path or ID
    fields disagree. Whether the file exists is not checked.
    """

    original_input: str
    source_type: StructureSourceType
    file_format: StructureFileFormat
    local_path: Path | None = None
    structure_id: str | None = None

    def __post_init__(self) -> None:
        """Reject a source type whose path or ID field is missing or misplaced."""
        if self.source_type == StructureSourceType.LOCAL_FILE:
            if self.local_path is None:
                raise ValueError("A local-file structure input needs a local_path.")
        else:
            if not self.structure_id:
                raise ValueError("An RCSB structure input needs a structure_id.")
            if self.local_path is not None:
                raise ValueError(
                    "An RCSB structure input cannot have a local_path; "
                    "a downloaded entry is a local-file input."
                )

    @property
    def needs_download(self) -> bool:
        """Return whether the structure must be downloaded before reading."""
        return self.source_type == StructureSourceType.RCSB_ID


def resolve_structure_input(raw_input: str) -> ResolvedStructureInput:
    """Resolve one input as a local coordinate file or a classic PDB ID.

    Text shaped like a classic PDB ID is always taken as one. Such text has no
    extension, so a local file or directory of the same name could never be
    read as a structure and does not shadow the RCSB entry.

    Anything else is a path, with surrounding whitespace stripped and a leading
    `~` expanded. Environment variables are not expanded, because the shell
    has already done that and a `$` left in the text belongs to the filename.
    The path must exist and end in one of `SUPPORTED_STRUCTURE_SUFFIXES`,
    optionally followed by `.gz`.

    Every failure raises `StructureInputError`, including a directory, an
    unsupported extension, an unknown `~user`, and a filesystem error while
    looking for the file.
    """
    cleaned_input = raw_input.strip()
    if not cleaned_input:
        raise StructureInputError("Structure input cannot be empty.")

    if is_valid_classic_pdb_id(cleaned_input):
        return ResolvedStructureInput(
            original_input=raw_input,
            source_type=StructureSourceType.RCSB_ID,
            file_format=StructureFileFormat.MMCIF,
            structure_id=cleaned_input.upper(),
        )

    try:
        local_path = Path(cleaned_input).expanduser()
        if local_path.exists():
            return _resolve_local_file(local_path, original_input=raw_input)
    except (OSError, RuntimeError) as error:
        # RuntimeError is what `expanduser` raises for an unknown `~user`.
        raise StructureInputError(
            f"Could not inspect structure input {raw_input!r}: {error}"
        ) from error

    raise StructureInputError(
        f"Could not resolve structure input {raw_input!r}.\n"
        f"Expected an existing local {_describe_suffixes()} file, optionally "
        f"gzipped ({COMPRESSED_SUFFIX}), "
        "or a classic 4-character PDB ID such as 1ABC."
    )


def is_valid_classic_pdb_id(value: str) -> bool:
    """Return whether a value is a classic four-character PDB identifier.

    That is a digit followed by three letters or digits in either case, with
    surrounding whitespace ignored. Extended IDs such as `pdb_00001abc` are not
    accepted.
    """
    return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", value.strip()))


def _resolve_local_file(
    path: Path,
    *,
    original_input: str,
) -> ResolvedStructureInput:
    if not path.is_file():
        raise StructureInputError(
            f"Expected a structure file, but got a directory or non-file path: {path!s}"
        )

    return ResolvedStructureInput(
        original_input=original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=detect_structure_format_from_path(path),
        local_path=path.resolve(),
    )


def detect_structure_format_from_path(path: Path) -> StructureFileFormat:
    """Return the coordinate format implied by a path's extension.

    The extension is found as in `split_structure_file_name`, so `model.v2.PDB`
    and `pdb1abc.ent.gz` are PDB and `model.cif.gz` is mmCIF. The file is not
    opened. A missing or unsupported extension, including a `.gz` with no
    format extension before it, raises `StructureInputError`.
    """
    _, suffix = split_structure_file_name(path.name)
    if suffix:
        return _FORMAT_BY_SUFFIX[suffix]

    uncompressed, compressed = _strip_compression(path.name)
    # A leading dot marks a hidden file, not an extension.
    has_inner_extension = "." in uncompressed.lstrip(".")

    if compressed and not has_inner_extension:
        raise StructureInputError(
            f"Could not detect structure format for {path!s}: a gzipped file "
            f"must name its format before {COMPRESSED_SUFFIX}, as in model.cif.gz."
        )

    if not has_inner_extension:
        raise StructureInputError(
            f"Could not detect structure format for {path!s}: file has no extension."
        )

    raise StructureInputError(
        f"Unsupported structure file extension for {path!s}.\n"
        f"Supported extensions are {_describe_suffixes()}, each optionally "
        f"followed by {COMPRESSED_SUFFIX}."
    )


def split_structure_file_name(name: str) -> tuple[str, str]:
    """Split a file name into its stem and its lower-case format extension.

    A trailing `.gz` in any case is looked past and dropped, so `1ABC.cif.GZ`
    gives `("1ABC", ".cif")`. A name without a supported format extension
    comes back whole, with an empty extension.
    """
    uncompressed, _ = _strip_compression(name)
    lowered = uncompressed.lower()

    for suffix in _FORMAT_BY_SUFFIX:
        if lowered.endswith(suffix):
            return uncompressed[: -len(suffix)], suffix

    return name, ""


def _strip_compression(name: str) -> tuple[str, bool]:
    if name.lower().endswith(COMPRESSED_SUFFIX):
        return name[: -len(COMPRESSED_SUFFIX)], True
    return name, False


def _describe_suffixes() -> str:
    *leading, last = SUPPORTED_STRUCTURE_SUFFIXES
    return f"{', '.join(leading)}, or {last}"
