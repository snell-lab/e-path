"""Load and validate Bnet reference databases."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_REFERENCE_SCHEMA_VERSION = "1.0"
DEFAULT_REFERENCE_METRIC_KIND = "protein_cryo_asp_glu_bnet"
DEFAULT_REFERENCE_DATABASE_FILENAME = "database.csv"
DEFAULT_REFERENCE_DATABASE_DIR = Path(__file__).resolve().parent / "database"
DEFAULT_REFERENCE_DATABASE_CSV_PATH = (
    DEFAULT_REFERENCE_DATABASE_DIR / DEFAULT_REFERENCE_DATABASE_FILENAME
)

_PDB_ID_COLUMN_ALIASES = frozenset(
    {
        "pdbid",
        "pdbcode",
        "pdb",
        "code",
        "entryid",
        "entry",
    }
)
_RESOLUTION_COLUMN_ALIASES = frozenset(
    {
        "resolutionangstrom",
        "resolutionangstroms",
        "resolutiona",
        "resolution",
        "resolutionang",
    }
)
_BNET_COLUMN_ALIASES = frozenset(
    {
        "bnet",
        "bnetvalue",
        "bnetscore",
    }
)


class BnetReferenceError(ValueError):
    """Report invalid Bnet reference data or metadata."""


@dataclass(frozen=True, slots=True)
class BnetReferenceEntry:
    """Store one structure's resolution in angstroms and raw Bnet value."""

    pdb_id: str
    resolution_angstrom: float
    bnet: float

    def __post_init__(self) -> None:
        """Validate numeric reference values."""
        resolution_angstrom = _finite_float(
            self.resolution_angstrom,
            name="resolution_angstrom",
        )
        if resolution_angstrom <= 0.0:
            raise BnetReferenceError(
                "resolution_angstrom must be greater than zero, "
                f"got {resolution_angstrom!r}."
            )

        bnet = _finite_float(self.bnet, name="bnet")
        if bnet < 0.0:
            raise BnetReferenceError(
                f"bnet must be greater than or equal to zero, got {bnet!r}."
            )

        object.__setattr__(self, "resolution_angstrom", resolution_angstrom)
        object.__setattr__(self, "bnet", bnet)


@dataclass(frozen=True, slots=True)
class BnetReferenceMetadata:
    """Describe the provenance and schema of a Bnet reference database."""

    database_id: str = "unknown_bnet_reference"
    schema_version: str = DEFAULT_REFERENCE_SCHEMA_VERSION
    metric_kind: str = DEFAULT_REFERENCE_METRIC_KIND
    source: str | None = None
    pdb_redo_snapshot: str | None = None
    eligibility_policy_id: str | None = None
    source_path: str | None = None
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate required metadata strings."""
        database_id = self.database_id.strip()
        if not database_id:
            raise BnetReferenceError("metadata.database_id must not be empty.")

        schema_version = self.schema_version.strip()
        if not schema_version:
            raise BnetReferenceError("metadata.schema_version must not be empty.")

        metric_kind = self.metric_kind.strip()
        if not metric_kind:
            raise BnetReferenceError("metadata.metric_kind must not be empty.")

        object.__setattr__(self, "database_id", database_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "metric_kind", metric_kind)


@dataclass(frozen=True, slots=True)
class BnetReferenceDatabase:
    """Store validated reference entries and immutable numeric columns.

    The resolution and Bnet arrays have shape `(n,)`, align with `entries`, and
    are read-only so cached column views cannot diverge from the records.
    """

    entries: Sequence[BnetReferenceEntry]
    metadata: BnetReferenceMetadata = field(default_factory=BnetReferenceMetadata)
    _pdb_ids: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _resolution_values: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _bnet_values: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate entries and build immutable aligned column views."""
        entries = tuple(self.entries)
        if not entries:
            raise BnetReferenceError("Bnet reference database must contain entries.")

        seen_pdb_ids: set[str] = set()
        for entry in entries:
            duplicate_key = entry.pdb_id.casefold()
            if duplicate_key in seen_pdb_ids:
                raise BnetReferenceError(
                    f"Duplicate pdb_id in Bnet reference database: {entry.pdb_id!r}."
                )
            seen_pdb_ids.add(duplicate_key)

        pdb_ids = tuple(entry.pdb_id for entry in entries)
        resolution_values = np.asarray(
            [entry.resolution_angstrom for entry in entries],
            dtype=float,
        )
        bnet_values = np.asarray([entry.bnet for entry in entries], dtype=float)

        resolution_values.setflags(write=False)
        bnet_values.setflags(write=False)

        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_pdb_ids", pdb_ids)
        object.__setattr__(self, "_resolution_values", resolution_values)
        object.__setattr__(self, "_bnet_values", bnet_values)

    def __len__(self) -> int:
        """Return the number of reference entries."""
        return self.entry_count

    @property
    def entry_count(self) -> int:
        """Return the number of reference entries."""
        return len(self.entries)

    @property
    def pdb_ids(self) -> tuple[str, ...]:
        """Return structure identifiers in entry order."""
        return self._pdb_ids

    @property
    def resolution_values(self) -> NDArray[np.float64]:
        """Return read-only resolutions in entry order."""
        return self._resolution_values

    @property
    def bnet_values(self) -> NDArray[np.float64]:
        """Return read-only raw Bnet values in entry order."""
        return self._bnet_values


def load_bnet_reference_database(path: str | Path) -> BnetReferenceDatabase:
    """Load a Bnet reference CSV and its optional JSON manifest.

    The CSV holds structure ID, resolution, and Bnet columns. The manifest is
    the sibling file ending in `.manifest.json`, when present.

    Raises:
        OSError: If a file cannot be read.
        BnetReferenceError: If files cannot be parsed, required columns are
            absent or ambiguous, or reference values are invalid.
    """
    csv_path = Path(path).expanduser()
    entries = _read_reference_entries_csv(csv_path)
    candidate = csv_path.with_suffix(".manifest.json")
    manifest_path = candidate if candidate.is_file() else None
    manifest = _read_manifest_json(manifest_path)

    metadata = _metadata_from_manifest(
        manifest,
        source_path=csv_path,
        manifest_path=manifest_path,
    )

    return BnetReferenceDatabase(entries=entries, metadata=metadata)


def load_default_bnet_reference_database() -> BnetReferenceDatabase:
    """Load the reference database bundled with the package."""
    return load_bnet_reference_database(DEFAULT_REFERENCE_DATABASE_CSV_PATH)


def _read_reference_entries_csv(path: Path) -> tuple[BnetReferenceEntry, ...]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BnetReferenceError(f"Bnet reference CSV has no header: {path}.")

        pdb_id_column = _find_column(
            reader.fieldnames,
            aliases=_PDB_ID_COLUMN_ALIASES,
            required_name="pdb_id",
        )
        resolution_column = _find_column(
            reader.fieldnames,
            aliases=_RESOLUTION_COLUMN_ALIASES,
            required_name="resolution_angstrom",
        )
        bnet_column = _find_column(
            reader.fieldnames,
            aliases=_BNET_COLUMN_ALIASES,
            required_name="bnet",
        )

        entries: list[BnetReferenceEntry] = []
        for row_number, row in enumerate(reader, start=2):
            if _row_is_empty(row):
                continue

            try:
                entries.append(
                    BnetReferenceEntry(
                        pdb_id=_required_text(
                            row.get(pdb_id_column),
                            column=pdb_id_column,
                            row_number=row_number,
                        ),
                        resolution_angstrom=_required_float(
                            row.get(resolution_column),
                            column=resolution_column,
                            row_number=row_number,
                        ),
                        bnet=_required_float(
                            row.get(bnet_column),
                            column=bnet_column,
                            row_number=row_number,
                        ),
                    )
                )
            except BnetReferenceError as error:
                raise BnetReferenceError(
                    f"Invalid Bnet reference row {row_number}: {error}"
                ) from error

    return tuple(entries)


def _read_manifest_json(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        try:
            value = json.load(handle)
        except json.JSONDecodeError as error:
            raise BnetReferenceError(
                f"Could not parse Bnet manifest JSON: {path}."
            ) from error

    if not isinstance(value, Mapping):
        raise BnetReferenceError(f"Bnet manifest must contain a JSON object: {path}.")

    return cast(Mapping[str, Any], value)


def _metadata_from_manifest(
    manifest: Mapping[str, Any],
    *,
    source_path: Path,
    manifest_path: Path | None,
) -> BnetReferenceMetadata:
    return BnetReferenceMetadata(
        database_id=str(manifest.get("database_id") or source_path.stem),
        schema_version=str(
            manifest.get("schema_version") or DEFAULT_REFERENCE_SCHEMA_VERSION
        ),
        metric_kind=str(manifest.get("metric_kind") or DEFAULT_REFERENCE_METRIC_KIND),
        source=_optional_string(manifest.get("source")),
        pdb_redo_snapshot=_optional_string(manifest.get("pdb_redo_snapshot")),
        eligibility_policy_id=_optional_string(manifest.get("eligibility_policy_id")),
        source_path=str(source_path),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


def _find_column(
    fieldnames: Sequence[str],
    *,
    aliases: frozenset[str],
    required_name: str,
) -> str:
    matching_columns: list[str] = []
    for fieldname in fieldnames:
        normalized = _normalize_column_name(fieldname)
        if normalized in aliases:
            matching_columns.append(fieldname)

    if len(matching_columns) == 1:
        return matching_columns[0]

    if len(matching_columns) > 1:
        matches = ", ".join(repr(name) for name in matching_columns)
        raise BnetReferenceError(
            f"Bnet reference CSV has ambiguous {required_name!r} columns: "
            f"{matches}. Use exactly one."
        )

    available = ", ".join(repr(name) for name in fieldnames)
    raise BnetReferenceError(
        f"Bnet reference CSV is missing required {required_name!r} column. "
        f"Available columns: {available}."
    )


def _normalize_column_name(value: str) -> str:
    normalized = value.casefold().replace("å", "a")
    return "".join(character for character in normalized if character.isalnum())


def _required_text(value: str | None, *, column: str, row_number: int) -> str:
    text = "" if value is None else value.strip()
    if not text:
        raise BnetReferenceError(f"column {column!r} is empty at row {row_number}.")
    return text


def _required_float(value: str | None, *, column: str, row_number: int) -> float:
    text = "" if value is None else value.strip()
    if not text:
        raise BnetReferenceError(f"column {column!r} is empty at row {row_number}.")

    try:
        return float(text)
    except ValueError as error:
        raise BnetReferenceError(
            f"column {column!r} contains a non-numeric value at row {row_number}: "
            f"{text!r}."
        ) from error


def _row_is_empty(row: Mapping[str, Any]) -> bool:
    return all(value is None or str(value).strip() == "" for value in row.values())


def _finite_float(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise BnetReferenceError(f"{name} must be finite, got {number!r}.")

    return number


def _optional_string(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_REFERENCE_DATABASE_CSV_PATH",
    "DEFAULT_REFERENCE_DATABASE_DIR",
    "DEFAULT_REFERENCE_DATABASE_FILENAME",
    "DEFAULT_REFERENCE_METRIC_KIND",
    "DEFAULT_REFERENCE_SCHEMA_VERSION",
    "BnetReferenceDatabase",
    "BnetReferenceEntry",
    "BnetReferenceError",
    "BnetReferenceMetadata",
    "load_bnet_reference_database",
    "load_default_bnet_reference_database",
]
