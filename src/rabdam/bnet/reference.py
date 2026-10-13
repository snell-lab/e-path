"""Load the reference database that Bnet percentiles are ranked against.

A reference database is a table of PDB entries, each with its resolution and
Bnet, and an ID naming the snapshot it was built from. One built from
PDB-REDO is bundled with the package under `database/`. `rabdam.bnet.pipeline`
loads it, and `rabdam.bnet.percentile` ranks a structure's Bnet against the
entries nearest to it in resolution.
"""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_REFERENCE_DATABASE_CSV_PATH = (
    Path(__file__).resolve().parent / "database" / "database.csv"
)

# The column headers of the reference CSV.
_RESOLUTION_COLUMN = "Resolution (A)"
_BNET_COLUMN = "Bnet"


class BnetReferenceError(ValueError):
    """Report a reference CSV that does not match its manifest."""


@dataclass(frozen=True, slots=True, eq=False)
class BnetReferenceDatabase:
    """Store the resolution and Bnet of every reference entry.

    `database_id` names the database in reports. The two arrays have shape
    `(n,)` and are aligned, one element per entry, with resolutions in
    angstroms.
    """

    database_id: str
    resolution_values: NDArray[np.float64]
    bnet_values: NDArray[np.float64]

    @property
    def entry_count(self) -> int:
        """Return the number of reference entries."""
        return len(self.bnet_values)


def load_bnet_reference_database(csv_path: Path) -> BnetReferenceDatabase:
    """Load a Bnet reference CSV and the JSON manifest beside it.

    The manifest is the sibling file whose name ends in `.manifest.json` in
    place of the CSV's extension. It supplies the database ID and the SHA-256
    of the CSV, and a CSV that does not match that digest raises
    `BnetReferenceError`, so a truncated or edited file does not load as if
    whole. The values of a matching file are trusted as they are.

    The resolutions and Bnet values are read from the columns named
    `Resolution (A)` and `Bnet`. A file that cannot be opened raises `OSError`.
    """
    manifest = json.loads(
        csv_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    # The digest is of the file's bytes as stored, line endings included.
    csv_bytes = csv_path.read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != manifest["sha256"]:
        raise BnetReferenceError(
            f"Bnet reference CSV {csv_path} does not match the SHA-256 "
            "recorded in its manifest, so it has been changed since the "
            "manifest was written."
        )

    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"), newline="")))
    return BnetReferenceDatabase(
        database_id=str(manifest["database_id"]),
        resolution_values=np.asarray(
            [float(row[_RESOLUTION_COLUMN]) for row in rows], dtype=np.float64
        ),
        bnet_values=np.asarray(
            [float(row[_BNET_COLUMN]) for row in rows], dtype=np.float64
        ),
    )


@cache
def load_default_bnet_reference_database() -> BnetReferenceDatabase:
    """Load the reference database bundled with the package.

    The parsed database is kept for the life of the process, so a batch run
    reads the file once. It raises as `load_bnet_reference_database` does.
    """
    return load_bnet_reference_database(DEFAULT_REFERENCE_DATABASE_CSV_PATH)
