"""Resolve structure inputs as local files or RCSB identifiers."""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class InputResolutionError(ValueError):
    """Report a structure input that cannot be resolved."""


class InputBatchResolutionError(InputResolutionError):
    """Collect resolution failures from a batch of structure inputs."""

    def __init__(self, errors: list[tuple[str, InputResolutionError]]):
        """Initialize the error from input-and-failure pairs."""
        self.errors = errors
        summary = "\n".join(f"- {raw_input!r}: {error}" for raw_input, error in errors)
        super().__init__(
            "One or more structure inputs could not be resolved:\n" + summary
        )


class StructureSourceType(str, Enum):
    """Identify whether a structure originated locally or from RCSB."""

    LOCAL_FILE = "local_file"
    RCSB_ID = "rcsb_id"


class StructureFileFormat(str, Enum):
    """Identify a supported coordinate-file format."""

    MMCIF = "mmcif"
    PDB = "pdb"


@dataclass(frozen=True, slots=True)
class ResolvedStructureInput:
    """Describe a validated local file or downloadable RCSB entry."""

    original_input: str
    source_type: StructureSourceType
    file_format: StructureFileFormat
    local_path: Path | None = None
    structure_id: str | None = None

    @property
    def needs_download(self) -> bool:
        """Return whether the structure must be downloaded before reading."""
        return self.source_type == StructureSourceType.RCSB_ID

    @property
    def is_local_file(self) -> bool:
        """Return whether the resolved input names an existing local file."""
        return self.source_type == StructureSourceType.LOCAL_FILE


def resolve_structure_input(raw_input: str) -> ResolvedStructureInput:
    """Resolve one input as a local coordinate file or classic PDB ID.

    Raises:
        InputResolutionError: If the input is empty, ambiguous, unsupported, or
            neither an existing file nor a valid four-character PDB ID.
    """
    cleaned_input = _clean_raw_input(raw_input)
    possible_path = _expand_path(cleaned_input)

    if possible_path.exists():
        return _resolve_local_file(
            cleaned_input,
            original_input=raw_input,
        )

    case_insensitive_path = _find_case_insensitive_local_path(possible_path)
    if case_insensitive_path is not None:
        return _resolve_local_file(
            str(case_insensitive_path),
            original_input=raw_input,
        )

    if is_valid_classic_pdb_id(cleaned_input):
        return _resolve_rcsb_id(
            cleaned_input,
            original_input=raw_input,
        )

    raise InputResolutionError(
        f"Could not resolve structure input {raw_input!r}.\n"
        "Expected an existing local .cif, .mmcif, or .pdb file "
        "or a classic 4-character PDB ID such as 1LYZ."
    )


def is_valid_classic_pdb_id(value: str) -> bool:
    """Return whether a value is a classic four-character PDB identifier."""
    return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", value.strip()))


def detect_structure_format_from_path(path: Path) -> StructureFileFormat:
    """Detect a supported structure format from a filename extension.

    Raises:
        InputResolutionError: If the path has no extension or uses an
            unsupported extension.
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]

    if not suffixes:
        raise InputResolutionError(
            f"Could not detect structure format for {path!s}: file has no extension."
        )

    final_suffix = suffixes[-1]

    if final_suffix in {".cif", ".mmcif"}:
        return StructureFileFormat.MMCIF

    if final_suffix == ".pdb":
        return StructureFileFormat.PDB

    raise InputResolutionError(
        f"Unsupported structure file extension for {path!s}.\n"
        "Supported extensions are: .cif, .mmcif, and .pdb"
    )


def _resolve_local_file(
    path_text: str,
    *,
    original_input: str,
) -> ResolvedStructureInput:
    path = _expand_path(path_text)

    if not path.exists():
        raise InputResolutionError(f"Local structure file does not exist: {path!s}")

    if not path.is_file():
        raise InputResolutionError(
            f"Expected a structure file, but got a directory or non-file path: {path!s}"
        )

    file_format = detect_structure_format_from_path(path)

    return ResolvedStructureInput(
        original_input=original_input,
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=file_format,
        local_path=path.resolve(),
        structure_id=None,
    )


def _resolve_rcsb_id(
    structure_id: str,
    *,
    original_input: str,
) -> ResolvedStructureInput:
    cleaned_id = structure_id.strip()

    if not is_valid_classic_pdb_id(cleaned_id):
        raise InputResolutionError(
            f"Invalid classic PDB ID: {structure_id!r}.\n"
            "Expected a 4-character ID such as 1LYZ, 1ABC, or 7XYZ."
        )

    return ResolvedStructureInput(
        original_input=original_input,
        source_type=StructureSourceType.RCSB_ID,
        file_format=StructureFileFormat.MMCIF,
        local_path=None,
        structure_id=cleaned_id.upper(),
    )


def _clean_raw_input(raw_input: object) -> str:
    if not isinstance(raw_input, str):
        raise InputResolutionError(
            f"Structure input must be a string, not {type(raw_input).__name__}."
        )

    cleaned = raw_input.strip()

    if not cleaned:
        raise InputResolutionError("Structure input cannot be empty.")

    return cleaned


def _expand_path(path_text: str) -> Path:
    expanded = os.path.expandvars(path_text)
    return Path(expanded).expanduser()


def _find_case_insensitive_local_path(path: Path) -> Path | None:
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        return None

    requested_name = path.name.lower()
    matches = [
        candidate
        for candidate in parent.iterdir()
        if candidate.name.lower() == requested_name
    ]

    if not matches:
        return None

    if len(matches) > 1:
        matching_names = ", ".join(sorted(candidate.name for candidate in matches))
        raise InputResolutionError(
            f"Local structure input {path!s} is ambiguous because multiple files "
            f"match case-insensitively: {matching_names}."
        )

    return matches[0]
