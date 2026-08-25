"""Unit-cell accessors, space groups, and asymmetric-unit validation."""

from __future__ import annotations

import dataclasses
import math

import gemmi
import pytest
from support import approx, make_metadata

from rabdam.crystal.translate import unit_cell_translation_vectors
from xtal.cell import (
    CrystalMetadataError,
    UnitCellParameters,
    make_gemmi_unit_cell,
    normalize_space_group_name,
    space_group_from_metadata,
    translation_vectors_from_gemmi_cell,
    unit_cell_from_metadata,
)
from xtal.validate import (
    MATTHEWS_PROTEIN_CONSTANT,
    CrystalValidationError,
    macromolecule_mass_dalton,
    validate_crystal,
)


def test_unit_cell_from_metadata_round_trips() -> None:
    cell = unit_cell_from_metadata(make_metadata(cell=(10, 20, 30, 90, 95, 120)))
    assert cell == UnitCellParameters(10, 20, 30, 90, 95, 120)


@pytest.mark.parametrize(
    "cell",
    [None, (0, 20, 30, 90, 90, 90), (10, 20, 30, 0, 90, 90), (10, 20, 30, 90, 180, 90)],
)
def test_invalid_unit_cells_are_rejected(
    cell: tuple[float, float, float, float, float, float] | None,
) -> None:
    with pytest.raises(CrystalMetadataError):
        unit_cell_from_metadata(make_metadata(cell=cell))


def test_incomplete_unit_cells_are_rejected() -> None:
    partial = dataclasses.replace(make_metadata(), unit_cell_c=None)
    with pytest.raises(CrystalMetadataError, match="missing complete"):
        unit_cell_from_metadata(partial)


def test_space_group_names_are_normalized_and_resolved() -> None:
    assert normalize_space_group_name("'P  21 21  21'") == "P 21 21 21"
    assert (
        space_group_from_metadata(make_metadata(space_group='"P 1 21 1"')).hm
        == "P 1 21 1"
    )
    with pytest.raises(CrystalMetadataError, match="missing a space-group"):
        space_group_from_metadata(make_metadata(space_group=None))
    with pytest.raises(CrystalMetadataError, match="could not recognize"):
        space_group_from_metadata(make_metadata(space_group="Q 99"))


def test_validate_crystal_without_content_check() -> None:
    crystal = validate_crystal(make_metadata())

    assert crystal.space_group_name == "P 1 21 1"
    assert crystal.operation_count == 2
    assert crystal.content is None and crystal.warnings == ()
    expected_volume = 15 * 16 * 17 * math.sin(math.radians(95))
    assert crystal.unit_cell_volume_angstrom3 == approx(expected_volume, rel=1e-9)


def test_matthews_coefficient_and_solvent_fraction() -> None:
    elements = ["C"] * 100 + ["N"] * 30 + ["O"] * 30 + ["S"] * 2
    crystal = validate_crystal(
        make_metadata(cell=(20, 20, 20, 90, 90, 90), space_group="P 1"),
        macromolecule_elements=elements,
    )

    content = crystal.content
    assert content is not None
    mass = sum(gemmi.Element(e).weight for e in elements)
    assert content.macromolecule_mass_dalton == approx(mass)
    assert content.macromolecule_atom_count == len(elements)
    assert content.matthews_coefficient == approx(8000 / mass)
    assert content.solvent_fraction == approx(
        1 - MATTHEWS_PROTEIN_CONSTANT / (8000 / mass)
    )
    assert content.hydrogens_excluded is True
    assert crystal.warnings == ()


def test_content_warnings_and_rejection() -> None:
    # Ten carbon atoms in a 30 A cube: far too little protein for the cell.
    sparse = validate_crystal(
        make_metadata(cell=(30, 30, 30, 90, 90, 90), space_group="P 1"),
        macromolecule_elements=["C"] * 10,
    )
    assert len(sparse.warnings) == 1 and "higher than expected" in sparse.warnings[0]

    # A 10 A cube holding 60 carbon atoms is dense but still fits.
    dense = validate_crystal(
        make_metadata(cell=(10, 10, 10, 90, 90, 90), space_group="P 1"),
        macromolecule_elements=["C"] * 60,
    )
    assert len(dense.warnings) == 1 and "lower than expected" in dense.warnings[0]

    with pytest.raises(CrystalValidationError, match="cannot fit"):
        validate_crystal(
            make_metadata(cell=(10, 10, 10, 90, 90, 90), space_group="P 1"),
            macromolecule_elements=["C"] * 200,
        )


def test_unknown_elements_contribute_no_mass() -> None:
    assert macromolecule_mass_dalton(["C"]) == approx(gemmi.Element("C").weight)
    assert macromolecule_mass_dalton(["Xx", ""]) == 0.0
    crystal = validate_crystal(make_metadata(), macromolecule_elements=["Xx"])
    assert crystal.content is None


@pytest.mark.parametrize(
    "cell",
    [
        (10, 20, 30, 90, 90, 90),
        (31.2, 44.5, 52.9, 90, 101.3, 90),
        (25, 26, 27, 80, 85, 95),
    ],
)
def test_gemmi_frame_vectors_match_the_standard_orthogonalization(
    cell: tuple[float, float, float, float, float, float],
) -> None:
    parameters = UnitCellParameters(*cell)
    standard = unit_cell_translation_vectors(parameters)
    from_gemmi = translation_vectors_from_gemmi_cell(make_gemmi_unit_cell(parameters))

    for axis in "abc":
        expected, actual = getattr(standard, axis), getattr(from_gemmi, axis)
        assert (actual.x, actual.y, actual.z) == approx(
            (expected.x, expected.y, expected.z), abs=1e-9
        )

    crystal = validate_crystal(make_metadata(cell=cell, space_group="P 1"))
    assert crystal.translation_vectors == from_gemmi
