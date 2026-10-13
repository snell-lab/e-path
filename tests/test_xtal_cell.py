"""Unit-cell validation and space-group settings."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from support import make_metadata, write_synthetic_pdb

from xtal.cell import (
    CrystalMetadataError,
    UnitCellParameters,
    space_group_from_metadata,
    unit_cell_from_metadata,
    validate_unit_cell,
)
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input
from xtal.validate import validate_crystal

CellTuple = tuple[float, float, float, float, float, float]


@pytest.mark.parametrize(
    ("cell", "space_group", "expected_setting", "expected_operations"),
    [
        ((50, 50, 50, 80, 80, 80), "R 3", "R 3:R", 3),
        ((50, 50, 70, 90, 90, 120), "R 3", "R 3:H", 9),
        ((50, 50, 50, 80, 80, 80), "R 3 2", "R 3 2:R", 6),
        ((50, 50, 70, 90, 90, 120), "H 3", "R 3:H", 9),
    ],
)
def test_r_centred_setting_follows_the_cell(
    tmp_path: Path,
    cell: CellTuple,
    space_group: str,
    expected_setting: str,
    expected_operations: int,
) -> None:
    path = write_synthetic_pdb(tmp_path / "r.pdb", cell=cell, space_group=space_group)
    metadata = read_structure(resolve_structure_input(str(path))).metadata

    crystal = validate_crystal(metadata)

    assert crystal.space_group.xhm() == expected_setting
    assert crystal.operation_count == expected_operations


def test_space_group_without_a_cell_defaults_to_hexagonal() -> None:
    space_group = space_group_from_metadata(make_metadata(cell=None, space_group="R 3"))
    assert space_group.xhm() == "R 3:H"


@pytest.mark.parametrize("name", ["''", ' " " ', "   "])
def test_quote_only_space_group_names_are_missing(name: str) -> None:
    with pytest.raises(CrystalMetadataError, match="missing a space-group"):
        space_group_from_metadata(make_metadata(space_group=name))


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        ((math.nan, 20, 30, 90, 90, 90), "Non-finite"),
        ((math.inf, 20, 30, 90, 90, 90), "Non-finite"),
        ((10, 20, 30, 90, math.nan, 90), "Non-finite"),
        ((10, -20, 30, 90, 90, 90), "lengths"),
        ((10, 20, 30, 90, 90, 180), "angles"),
        # One angle at least the sum of the other two.
        ((10, 20, 30, 10, 10, 170), "do not form"),
        ((10, 20, 30, 30, 40, 70), "do not form"),
        # Angles summing to 360 degrees give a flat cell, which rounding
        # leaves very slightly positive rather than exactly zero.
        ((10, 20, 30, 120, 120, 120), "do not form"),
        # A vanishing gamma would make the standard basis divide by zero.
        ((10, 20, 30, 90, 90, 1e-13), "do not form"),
    ],
)
def test_invalid_cells_are_rejected(cell: CellTuple, message: str) -> None:
    with pytest.raises(CrystalMetadataError, match=message):
        validate_unit_cell(UnitCellParameters(*cell))
    with pytest.raises(CrystalMetadataError, match=message):
        unit_cell_from_metadata(make_metadata(cell=cell))


@pytest.mark.parametrize(
    "cell",
    [
        (10, 20, 30, 90, 90, 90),
        (50, 50, 50, 80, 80, 80),
        (25, 26, 27, 80, 85, 95),
        (10, 20, 30, 30, 40, 69),
    ],
)
def test_valid_cells_are_accepted(cell: CellTuple) -> None:
    validate_unit_cell(UnitCellParameters(*cell))
    assert unit_cell_from_metadata(make_metadata(cell=cell)) == UnitCellParameters(
        *cell
    )
