"""Builders and reference calculations shared by the E-path test suite.

The synthetic crystal below is small enough to run the whole pipeline in well
under a second, yet exercises every cleaning and classification rule: two
chains, a modified residue deposited as HETATM, Asp/Glu carboxyl oxygens for
Bnet, a cofactor, a crystallization additive, waters, a hydrogen, alternate
conformers, and records with unusable occupancy or B-factor.

The reference calculations here are deliberately naive. They use Gemmi's own
symmetry and cell handling and brute-force distance loops, so they share no
code with the implementation they check.
"""

import csv
import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest

from epath.analysis.selection import EpathScoredAtom
from rabdam.bdamage import BDamageAtomResult
from rabdam.pipeline import (
    BDamagePipelineOptions,
    BDamagePipelineResult,
    validate_crystal_for_prepared_structure,
)
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
)
from rabdam.structure.prepare import classify_atom
from xtal.reader import AtomRecord, StructureMetadata
from xtal.validate import MacromoleculeAtom, MacromoleculeKind

# A monoclinic cell so that symmetry expansion and lattice translation are
# both non-trivial. The volume gives a Matthews coefficient of about 2 for
# the synthetic protein below.
SYNTHETIC_CELL = (15.0, 16.0, 17.0, 90.0, 95.0, 90.0)
SYNTHETIC_SPACE_GROUP = "P 1 21 1"
SYNTHETIC_RESOLUTION = 1.8

# Chain A: twelve residues on a helix, one per chemistry class at least.
# Residue 1 is selenomethionine deposited as HETATM.
CHAIN_A_SEQUENCE = (
    "MSE",
    "ALA",
    "TRP",
    "GLU",
    "SER",
    "CYS",
    "LYS",
    "PHE",
    "GLY",
    "ASP",
    "TYR",
    "LEU",
)
# Chain B: two residues packed against chain A.
CHAIN_B_SEQUENCE = ("GLY", "ALA")

# Backbone and CB offsets from the residue anchor, in angstroms.
_BACKBONE_OFFSETS = (
    ("N", "N", (-1.2, 0.4, -0.5)),
    ("CA", "C", (0.0, 0.0, 0.0)),
    ("C", "C", (1.2, 0.5, 0.4)),
    ("O", "O", (1.5, 1.6, 0.8)),
    ("CB", "C", (0.2, -1.3, 0.6)),
)
_SIDE_CHAIN_OFFSETS = {
    "MSE": (("SE", "SE", (0.6, -2.6, 1.4)),),
    "GLU": (
        ("CG", "C", (0.4, -2.6, 1.2)),
        ("CD", "C", (0.7, -3.9, 1.8)),
        ("OE1", "O", (0.3, -4.9, 1.3)),
        ("OE2", "O", (1.4, -4.0, 2.8)),
    ),
    "ASP": (
        ("CG", "C", (0.4, -2.6, 1.2)),
        ("OD1", "O", (0.0, -3.6, 0.7)),
        ("OD2", "O", (1.1, -2.7, 2.2)),
    ),
}


@dataclass(frozen=True)
class PdbAtom:
    """One fixed-format PDB record to be written to a test file."""

    record: str
    serial: int
    name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str


def format_pdb_atom(atom: PdbAtom) -> str:
    """Format one PDB ATOM/HETATM line with standard column layout."""
    name = atom.name
    name = f" {name:<3}" if len(name) < 4 and len(atom.element) == 1 else f"{name:<4}"
    return (
        f"{atom.record:<6}{atom.serial:5d} {name}{atom.altloc or ' '}"
        f"{atom.residue_name:>3} {atom.chain_id}{atom.residue_number:4d}    "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{atom.occupancy:6.2f}{atom.b_factor:6.2f}          {atom.element:>2}"
    )


def _residue_anchor(
    index: int, chain_offset: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the CA position of residue `index` on the synthetic helix."""
    angle = math.radians(100.0 * index)
    return (
        chain_offset[0] + 2.3 * math.cos(angle),
        chain_offset[1] + 2.3 * math.sin(angle),
        chain_offset[2] + 1.3 * index,
    )


def _b_factor(serial: int) -> float:
    # Deterministic, positive, and varied enough that windows differ.
    """Return a positive B-factor that varies smoothly with atom serial."""
    return round(12.0 + 6.0 * math.sin(0.7 * serial) + 0.35 * (serial % 7), 2)


def synthetic_atoms() -> list[PdbAtom]:
    """Return the synthetic crystal's atom records in deposition order."""
    atoms: list[PdbAtom] = []
    serial = 0

    def add(**fields: Any) -> None:
        nonlocal serial
        serial += 1
        defaults: dict[str, Any] = {
            "record": "ATOM",
            "serial": serial,
            "altloc": "",
            "occupancy": 1.0,
            "b_factor": _b_factor(serial),
        }
        atoms.append(PdbAtom(**{**defaults, **fields}))

    for chain_id, sequence, offset in (
        ("A", CHAIN_A_SEQUENCE, (5.0, 5.0, 1.0)),
        ("B", CHAIN_B_SEQUENCE, (11.5, 9.5, 6.0)),
    ):
        for index, residue_name in enumerate(sequence, start=1):
            anchor = _residue_anchor(index, offset)
            record = "HETATM" if residue_name == "MSE" else "ATOM"
            side_chain = _SIDE_CHAIN_OFFSETS.get(residue_name, ())
            for name, element, delta in (*_BACKBONE_OFFSETS, *side_chain):
                if residue_name == "GLY" and name == "CB":
                    continue
                add(
                    record=record,
                    name=name,
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=index,
                    x=anchor[0] + delta[0],
                    y=anchor[1] + delta[1],
                    z=anchor[2] + delta[2],
                    element=element,
                )
            if residue_name == "SER":
                # Alternate conformers of OG: A has the higher occupancy.
                for altloc, occupancy, delta in (
                    ("A", 0.6, (0.9, -2.4, 0.9)),
                    ("B", 0.4, (-0.6, -2.5, 1.1)),
                ):
                    add(
                        name="OG",
                        altloc=altloc,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_number=index,
                        x=anchor[0] + delta[0],
                        y=anchor[1] + delta[1],
                        z=anchor[2] + delta[2],
                        occupancy=occupancy,
                        element="O",
                    )
            if residue_name == "ALA" and chain_id == "A":
                # A hydrogen, which must be removed before scoring.
                add(
                    name="HB1",
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=index,
                    x=anchor[0] + 0.5,
                    y=anchor[1] - 2.1,
                    z=anchor[2] + 1.2,
                    element="H",
                )
            if residue_name == "LYS":
                # Unusable records: zero occupancy and zero B-factor.
                add(
                    name="CG",
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=index,
                    x=anchor[0] + 0.4,
                    y=anchor[1] - 2.6,
                    z=anchor[2] + 1.2,
                    occupancy=0.0,
                    element="C",
                )
                add(
                    name="CD",
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=index,
                    x=anchor[0] + 0.7,
                    y=anchor[1] - 3.9,
                    z=anchor[2] + 1.8,
                    b_factor=0.0,
                    element="C",
                )

    # A cofactor near the chain A helix axis, an additive, and two waters.
    for name, element, position in (
        ("N1", "N", (5.0, 5.0, 8.4)),
        ("C2", "C", (5.6, 5.9, 9.2)),
        ("O2", "O", (6.4, 6.6, 9.0)),
    ):
        add(
            record="HETATM",
            name=name,
            residue_name="FMN",
            chain_id="A",
            residue_number=201,
            x=position[0],
            y=position[1],
            z=position[2],
            element=element,
        )
    for name, position in (("C1", (12.5, 3.0, 12.0)), ("O1", (13.2, 2.4, 12.9))):
        add(
            record="HETATM",
            name=name,
            residue_name="GOL",
            chain_id="A",
            residue_number=301,
            x=position[0],
            y=position[1],
            z=position[2],
            element=name[0],
        )
    for number, position in ((401, (2.0, 10.5, 3.0)), (402, (10.0, 1.5, 14.5))):
        add(
            record="HETATM",
            name="O",
            residue_name="HOH",
            chain_id="A",
            residue_number=number,
            x=position[0],
            y=position[1],
            z=position[2],
            element="O",
        )

    return atoms


def synthetic_pdb_text(
    *,
    cell: tuple[float, float, float, float, float, float] = SYNTHETIC_CELL,
    space_group: str = SYNTHETIC_SPACE_GROUP,
    resolution: float | None = SYNTHETIC_RESOLUTION,
    include_cryst1: bool = True,
    scale_lines: Sequence[str] = (),
    atoms: Iterable[PdbAtom] | None = None,
) -> str:
    """Render the synthetic crystal, or a variant of it, as PDB text."""
    lines = ["HEADER    SYNTHETIC TEST STRUCTURE"]
    if resolution is not None:
        lines.append(f"REMARK   2 RESOLUTION.    {resolution:.2f} ANGSTROMS.")
    if include_cryst1:
        a, b, c, alpha, beta, gamma = cell
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {space_group:<11}1"
        )
    lines.extend(scale_lines)
    lines.extend(
        format_pdb_atom(atom)
        for atom in (synthetic_atoms() if atoms is None else atoms)
    )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_synthetic_pdb(path: Path, **options: object) -> Path:
    """Write the synthetic crystal to `path` and return it."""
    path.write_text(synthetic_pdb_text(**options), encoding="utf-8")  # type: ignore[arg-type]
    return path


def write_synthetic_mmcif(path: Path, pdb_path: Path) -> Path:
    """Convert a PDB test file to mmCIF with Gemmi."""
    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    document = structure.make_mmcif_document()
    if structure.resolution > 0.0:
        # Gemmi's writer does not emit the resolution, so record it the way a
        # deposited entry does.
        document.sole_block().set_pair(
            "_refine.ls_d_res_high", f"{structure.resolution:.2f}"
        )
    document.write_file(str(path))
    return path


def approx(
    expected: object, *, rel: float | None = None, abs: float | None = None
) -> Any:
    """Return `pytest.approx` with a fully known type for the type checkers."""
    return pytest.approx(expected, rel=rel, abs=abs)  # pyright: ignore[reportUnknownMemberType]


class StageRecorder:
    """Run pipeline stages directly, recording each stage label in order."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def __call__[T](self, label: str, callback: Callable[[], T]) -> T:
        self.labels.append(label)
        return callback()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into one dict per row."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_atom_record(**overrides: object) -> AtomRecord:
    """Return an AtomRecord with sensible defaults, overridden by keyword."""
    fields: dict[str, object] = {
        "source_atom_index": 0,
        "chain_id": "A",
        "residue_name": "ALA",
        "residue_number": 1,
        "insertion_code": "",
        "atom_name": "CA",
        "element": "C",
        "altloc": "",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "occupancy": 1.0,
        "b_factor": 10.0,
        "atom_serial": 1,
        "is_hetatm": False,
    }
    fields.update(overrides)
    return AtomRecord(**fields)  # type: ignore[arg-type]


def make_prepared_atom(**overrides: object) -> PreparedAtom:
    """Return a classified PreparedAtom built from make_atom_record."""
    return classify_atom(make_atom_record(**overrides))


def make_metadata(
    cell: tuple[float, float, float, float, float, float] | None = SYNTHETIC_CELL,
    space_group: str | None = SYNTHETIC_SPACE_GROUP,
    resolution: float | None = SYNTHETIC_RESOLUTION,
) -> StructureMetadata:
    """Return structure metadata for a synthetic crystal."""
    a, b, c, alpha, beta, gamma = cell if cell is not None else (None,) * 6
    return StructureMetadata(
        source_path=Path("synthetic.pdb"),
        structure_id=None,
        space_group=space_group,
        unit_cell_a=a,
        unit_cell_b=b,
        unit_cell_c=c,
        unit_cell_alpha=alpha,
        unit_cell_beta=beta,
        unit_cell_gamma=gamma,
        resolution_angstrom=resolution,
        gemmi_cell=gemmi.UnitCell(*cell) if cell is not None else None,
    )


def make_macromolecule_atoms(
    elements: Sequence[str],
    *,
    kind: MacromoleculeKind = MacromoleculeKind.PROTEIN,
    occupancy: float = 1.0,
) -> list[MacromoleculeAtom]:
    """Return one-atom, untabulated residues with the given element symbols.

    Each atom gets the per-heavy-atom implicit-hydrogen fallback, so tests
    that need exact masses should compute them with `macromolecule_mass`.
    """
    return [
        MacromoleculeAtom(
            element=element,
            occupancy=occupancy,
            kind=kind,
            residue_name="",
            residue_id=("A", index, ""),
            atom_name=element,
        )
        for index, element in enumerate(elements, start=1)
    ]


def make_scored_atom(z_score: float = 0.0, **overrides: object) -> EpathScoredAtom:
    """Return a scored atom with sensible defaults, overridden by keyword.

    A keyword that names a field of the BDamage result sets it there, and any
    other one goes to the atom record, as for `make_atom_record`.
    """
    result_fields: dict[str, object] = {
        "bdamage_atom_index": 1,
        "packing_density": 50,
        "average_b_factor": 10.0,
        "bdamage": 1.0,
    }
    record_fields = {
        name: value for name, value in overrides.items() if name not in result_fields
    }
    result_fields.update(
        (name, value) for name, value in overrides.items() if name in result_fields
    )
    record = make_atom_record(**record_fields)
    return EpathScoredAtom(
        record=record,
        bdamage_result=BDamageAtomResult(
            source_atom_index=record.source_atom_index,
            **result_fields,  # type: ignore[arg-type]
        ),
        bdamage_z_score=z_score,
    )


def bdamage_result_from_atoms(
    atoms: Sequence[EpathScoredAtom],
) -> BDamagePipelineResult:
    """Return a BDamage result whose scored atoms are exactly `atoms`.

    The E-path pipeline starts from a BDamage result, so a test that needs
    chosen positions and BDamage values wraps them in one. The model holds
    nothing but these atoms, which must have protein residue names, and it
    sits in the synthetic cell whatever its coordinates are.
    """
    prepared_atoms = tuple(classify_atom(atom.record) for atom in atoms)
    prepared_structure = PreparedStructure(
        source_atoms=tuple(atom.record for atom in atoms),
        cleaned_atoms=prepared_atoms,
        selected_atoms=prepared_atoms,
        metadata=make_metadata(),
        removed=RemovedAtomCounts(0, 0, 0, 0, 0),
    )
    return BDamagePipelineResult(
        prepared_structure=prepared_structure,
        crystal=validate_crystal_for_prepared_structure(prepared_structure),
        bdamage_results=tuple(atom.bdamage_result for atom in atoms),
        options=BDamagePipelineOptions(),
        window_size=BDamagePipelineOptions().minimum_window_size,
    )


def scored_atoms_from_coordinates(
    coordinates: Sequence[tuple[float, float, float]],
    *,
    residue_numbers: Sequence[int] | None = None,
    residue_names: Sequence[str] | None = None,
    chain_ids: Sequence[str] | None = None,
    atom_names: Sequence[str] | None = None,
) -> tuple[EpathScoredAtom, ...]:
    """Build scored atoms at the given coordinates, one residue per atom by default."""
    count = len(coordinates)
    residue_numbers = residue_numbers or list(range(1, count + 1))
    residue_names = residue_names or ["ALA"] * count
    chain_ids = chain_ids or ["A"] * count
    atom_names = atom_names or ["CA"] * count
    return tuple(
        make_scored_atom(
            source_atom_index=index,
            bdamage_atom_index=index + 1,
            atom_serial=index + 1,
            chain_id=chain_ids[index],
            residue_name=residue_names[index],
            residue_number=residue_numbers[index],
            atom_name=atom_names[index],
            x=x,
            y=y,
            z=z,
        )
        for index, (x, y, z) in enumerate(coordinates)
    )


def brute_force_packing_density(
    *,
    cleaned_atoms: Sequence[PreparedAtom],
    selected_atoms: Sequence[PreparedAtom],
    cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
    threshold: float = 7.0,
    translation_range: int = 1,
) -> list[int]:
    """Count crystal neighbours strictly within `threshold` using Gemmi only.

    Every symmetry image of every cleaned atom is generated in every cell of
    the translation block, and each selected atom's own image in the central
    cell under the identity operation is excluded.
    """
    operations = list(space_group.operations())
    offsets: list[tuple[int, int, int]] = [
        (a, b, c)
        for a, b, c in itertools.product(
            range(-translation_range, translation_range + 1), repeat=3
        )
    ]
    images: list[tuple[float, float, float]] = []
    self_keys: list[tuple[int, int, tuple[int, int, int]]] = []
    for atom in cleaned_atoms:
        record = atom.record
        fractional = cell.fractionalize(gemmi.Position(record.x, record.y, record.z))
        for operation_index, operation in enumerate(operations):
            moved = operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])
            for offset in offsets:
                position = cell.orthogonalize(
                    gemmi.Fractional(
                        moved[0] + offset[0], moved[1] + offset[1], moved[2] + offset[2]
                    )
                )
                images.append((position.x, position.y, position.z))
                self_keys.append((record.source_atom_index, operation_index, offset))

    image_array = np.asarray(images, dtype=float)
    identity_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.triplet().replace(" ", "") == "x,y,z"
    )
    counts: list[int] = []
    for atom in selected_atoms:
        record = atom.record
        distances = np.sqrt(
            ((image_array - (record.x, record.y, record.z)) ** 2).sum(axis=1)
        )
        count = 0
        for index, distance in enumerate(distances):
            if distance >= threshold:
                continue
            if self_keys[index] == (
                record.source_atom_index,
                identity_index,
                (0, 0, 0),
            ):
                continue
            count += 1
        counts.append(count)
    return counts


def reference_bdamage(
    *,
    b_factors: Sequence[float],
    packing_densities: Sequence[int],
    serials: Sequence[int],
    window_size: int,
) -> list[float]:
    """Return BDamage values following the published sliding-window rule.

    Atoms are sorted by packing density, ties broken by serial. Each atom is
    divided by the mean B-factor of the `window_size` atoms centred on it; the
    first and last complete windows are reused at the ends of the ordering.
    """
    count = len(b_factors)
    order = sorted(
        range(count), key=lambda index: (packing_densities[index], serials[index])
    )
    sorted_b = [b_factors[index] for index in order]
    half = window_size // 2
    result = [0.0] * count
    for position, index in enumerate(order):
        if position < half:
            start = 0
        elif position >= count - half:
            start = count - window_size
        else:
            start = position - half
        result[index] = b_factors[index] / (
            sum(sorted_b[start : start + window_size]) / window_size
        )
    return result
