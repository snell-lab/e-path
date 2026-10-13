"""Load the reference database that Bnet percentiles are ranked against.

A reference database is a table of PDB entries, each with its resolution and
Bnet, and an ID naming the snapshot it was built from. One built from
PDB-REDO is bundled with the package under `database/`. `rabdam.bnet.pipeline`
loads it, and `rabdam.bnet.percentile` ranks a structure's Bnet against the
entries nearest to it in resolution.
"""

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_REFERENCE_DATABASE_CSV_PATH = (
    Path(__file__).resolve().parent / "database" / "database.csv"
)

# The column headers of the reference CSV.
_PDB_ID_COLUMN = "PDB code"
_RESOLUTION_COLUMN = "Resolution (A)"
_BNET_COLUMN = "Bnet"


class BnetReferenceError(ValueError):
    """Report invalid Bnet reference data."""


@dataclass(frozen=True, slots=True)
class BnetReferenceEntry:
    """Store one structure's resolution in angstroms and Bnet value.

    Construction raises `BnetReferenceError` unless the resolution is finite
    and positive and Bnet is finite and not negative. `pdb_id` is not checked.
    """

    pdb_id: str
    resolution_angstrom: float
    bnet: float

    def __post_init__(self) -> None:
        """Reject a resolution or Bnet value outside its valid range."""
        resolution_angstrom = self.resolution_angstrom
        if not (math.isfinite(resolution_angstrom) and resolution_angstrom > 0.0):
            raise BnetReferenceError(
                "resolution_angstrom must be finite and greater than zero, "
                f"got {resolution_angstrom!r}."
            )

        if not (math.isfinite(self.bnet) and self.bnet >= 0.0):
            raise BnetReferenceError(
                "bnet must be finite and greater than or equal to zero, "
                f"got {self.bnet!r}."
            )


@dataclass(frozen=True, slots=True)
class BnetReferenceDatabase:
    """Store validated reference entries and immutable numeric columns.

    `database_id` names the database in reports. `entries` is kept as a tuple. The resolution and Bnet arrays have shape
    `(n,)`, align with `entries`, and are read-only so cached column views
    cannot diverge from the records.

    Construction raises `BnetReferenceError` when there are no entries or two
    share a `pdb_id`, ignoring case.
    """

    entries: Sequence[BnetReferenceEntry]
    database_id: str
    _resolution_values: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _bnet_values: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject empty or duplicated entries and build the column arrays."""
        entries = tuple(self.entries)
        if not entries:
            raise BnetReferenceError("Bnet reference database must contain entries.")

        # PDB IDs are case-insensitive, so 1abc and 1ABC are one entry.
        seen_pdb_ids: set[str] = set()
        for entry in entries:
            folded_pdb_id = entry.pdb_id.casefold()
            if folded_pdb_id in seen_pdb_ids:
                raise BnetReferenceError(
                    f"Duplicate pdb_id in Bnet reference database: {entry.pdb_id!r}."
                )
            seen_pdb_ids.add(folded_pdb_id)

        resolution_values = np.asarray(
            [entry.resolution_angstrom for entry in entries],
            dtype=float,
        )
        bnet_values = np.asarray([entry.bnet for entry in entries], dtype=float)

        resolution_values.setflags(write=False)
        bnet_values.setflags(write=False)

        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_resolution_values", resolution_values)
        object.__setattr__(self, "_bnet_values", bnet_values)

    @property
    def entry_count(self) -> int:
        """Return the number of reference entries."""
        return len(self.entries)

    @property
    def resolution_values(self) -> NDArray[np.float64]:
        """Return read-only resolutions in entry order."""
        return self._resolution_values

    @property
    def bnet_values(self) -> NDArray[np.float64]:
        """Return read-only Bnet values in entry order."""
        return self._bnet_values


def load_bnet_reference_database(path: str | Path) -> BnetReferenceDatabase:
    """Load a Bnet reference CSV and the JSON manifest beside it, if any.

    The CSV needs the columns `PDB code`, `Resolution (A)` and `Bnet`, named
    exactly so in its header. Other columns are ignored and lines holding no
    values are skipped.

    The manifest is the sibling file whose name ends in `.manifest.json` in
    place of the CSV's extension. It supplies the database ID, which is
    otherwise the CSV file's stem. A manifest that records an `entry_count` or
    a `sha256` for the CSV is also checked against it, so a truncated or edited
    CSV does not load as if whole.

    `BnetReferenceError` is raised when either file cannot be parsed, a
    required column is missing, a value is missing, not a number or out of
    range, a PDB ID repeats, there are no entries, or the CSV does not match
    its manifest. A file that cannot be
    opened raises `OSError`.
    """
    csv_path = Path(path)
    try:
        entries = _read_reference_entries_csv(csv_path)
    except (csv.Error, UnicodeDecodeError) as error:
        raise BnetReferenceError(
            f"Could not parse Bnet reference CSV {csv_path}: {error}"
        ) from error

    candidate = csv_path.with_suffix(".manifest.json")
    manifest_path = candidate if candidate.is_file() else None
    manifest = _read_manifest_json(manifest_path)
    _check_csv_against_manifest(manifest, csv_path=csv_path, entry_count=len(entries))

    database_id = str(manifest.get("database_id") or "").strip() or csv_path.stem

    return BnetReferenceDatabase(entries=entries, database_id=database_id)


def load_default_bnet_reference_database() -> BnetReferenceDatabase:
    """Load the reference database bundled with the package.

    The parsed database is kept for the life of the process, so a batch run
    reads the file once. It raises as `load_bnet_reference_database` does.
    """
    return _cached_reference_database(DEFAULT_REFERENCE_DATABASE_CSV_PATH)


@cache
def _cached_reference_database(path: Path) -> BnetReferenceDatabase:
    return load_bnet_reference_database(path)


def _read_reference_entries_csv(path: Path) -> tuple[BnetReferenceEntry, ...]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or ()
        missing_columns = [
            column
            for column in (_PDB_ID_COLUMN, _RESOLUTION_COLUMN, _BNET_COLUMN)
            if column not in fieldnames
        ]
        if missing_columns:
            raise BnetReferenceError(
                f"Bnet reference CSV {path} is missing the column "
                f"{', '.join(repr(column) for column in missing_columns)}."
            )

        entries: list[BnetReferenceEntry] = []
        for row in reader:
            if _row_is_empty(row):
                continue

            try:
                entries.append(
                    BnetReferenceEntry(
                        pdb_id=_required_text(row, _PDB_ID_COLUMN),
                        resolution_angstrom=_required_float(row, _RESOLUTION_COLUMN),
                        bnet=_required_float(row, _BNET_COLUMN),
                    )
                )
            except BnetReferenceError as error:
                # The reader drops blank lines without yielding them, so only
                # its own line counter still matches the file.
                raise BnetReferenceError(
                    f"Invalid Bnet reference CSV line {reader.line_num}: {error}"
                ) from error

    return tuple(entries)


def _read_manifest_json(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as handle:
        try:
            value = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BnetReferenceError(
                f"Could not parse Bnet manifest JSON {path}: {error}"
            ) from error

    if not isinstance(value, dict):
        raise BnetReferenceError(f"Bnet manifest must contain a JSON object: {path}.")

    return cast(dict[str, object], value)


def _check_csv_against_manifest(
    manifest: Mapping[str, object], *, csv_path: Path, entry_count: int
) -> None:
    expected_entry_count = manifest.get("entry_count")
    if expected_entry_count is not None and expected_entry_count != entry_count:
        raise BnetReferenceError(
            f"Bnet reference CSV {csv_path} holds {entry_count} entries, but "
            f"its manifest records {expected_entry_count!r}."
        )

    expected_sha256 = manifest.get("sha256")
    if expected_sha256 is not None:
        # The digest is of the file's bytes as stored, line endings included.
        with csv_path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if sha256 != expected_sha256:
            raise BnetReferenceError(
                f"Bnet reference CSV {csv_path} does not match the SHA-256 "
                "recorded in its manifest, so it has been changed since the "
                "manifest was written."
            )


def _row_is_empty(row: Mapping[str, object]) -> bool:
    return all(value is None or str(value).strip() == "" for value in row.values())


def _required_text(row: Mapping[str, str | None], column: str) -> str:
    text = (row.get(column) or "").strip()
    if not text:
        raise BnetReferenceError(f"column {column!r} is empty.")
    return text


def _required_float(row: Mapping[str, str | None], column: str) -> float:
    text = _required_text(row, column)
    try:
        return float(text)
    except ValueError as error:
        raise BnetReferenceError(
            f"column {column!r} contains a non-numeric value: {text!r}."
        ) from error
