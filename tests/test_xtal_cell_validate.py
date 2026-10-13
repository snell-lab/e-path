"""Unit-cell accessors, space groups, and asymmetric-unit validation."""

import dataclasses
import math

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray
from support import approx, make_macromolecule_atoms, make_metadata

from xtal.cell import (
    CrystalMetadataError,
    UnitCellParameters,
    space_group_from_metadata,
    translation_vectors_from_gemmi_cell,
    unit_cell_from_metadata,
)
from xtal.validate import (
    MATTHEWS_PROTEIN_CONSTANT,
    CrystalValidationError,
    macromolecule_mass,
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
    # Gemmi refuses to build some of these cells itself, so the parameters
    # are set on metadata that already holds a valid Gemmi cell.
    metadata = make_metadata(cell=None)
    if cell is not None:
        a, b, c, alpha, beta, gamma = cell
        metadata = dataclasses.replace(
            make_metadata(),
            unit_cell_a=a,
            unit_cell_b=b,
            unit_cell_c=c,
            unit_cell_alpha=alpha,
            unit_cell_beta=beta,
            unit_cell_gamma=gamma,
        )

    with pytest.raises(CrystalMetadataError):
        unit_cell_from_metadata(metadata)


def test_incomplete_unit_cells_are_rejected() -> None:
    partial = dataclasses.replace(make_metadata(), unit_cell_c=None)
    with pytest.raises(CrystalMetadataError, match="missing complete") as error:
        unit_cell_from_metadata(partial)
    assert "biological-assembly downloads" in str(error.value)


def test_space_group_names_are_resolved() -> None:
    assert (
        space_group_from_metadata(make_metadata(space_group="P 1 21 1")).hm
        == "P 1 21 1"
    )
    with pytest.raises(CrystalMetadataError, match="missing a space-group"):
        space_group_from_metadata(make_metadata(space_group=None))
    with pytest.raises(CrystalMetadataError, match="could not recognize"):
        space_group_from_metadata(make_metadata(space_group="Q 99"))


def test_validate_crystal_without_macromolecule_atoms() -> None:
    crystal = validate_crystal(make_metadata(), macromolecule_atoms=())

    assert crystal.space_group_name == "P 1 21 1"
    assert crystal.operation_count == 2
    assert crystal.content is None and crystal.warnings == ()
    expected_volume = 15 * 16 * 17 * math.sin(math.radians(95))
    assert crystal.unit_cell_volume_angstrom3 == approx(expected_volume, rel=1e-9)


def test_matthews_coefficient_and_solvent_fraction() -> None:
    atoms = make_macromolecule_atoms(["C"] * 100 + ["N"] * 30 + ["O"] * 30 + ["S"] * 2)
    crystal = validate_crystal(
        make_metadata(cell=(20, 20, 20, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=atoms,
    )

    content = crystal.content
    assert content is not None
    mass = macromolecule_mass(atoms).total_dalton
    assert content.macromolecule_mass_dalton == approx(mass)
    assert content.macromolecule_atom_count == len(atoms)
    assert content.matthews_coefficient == approx(8000 / mass)
    assert content.solvent_fraction == approx(
        1 - MATTHEWS_PROTEIN_CONSTANT / (8000 / mass)
    )
    assert crystal.warnings == ()


def test_content_warnings_and_rejection() -> None:
    # Ten carbon atoms in a 30 A cube: far too little protein for the cell.
    sparse = validate_crystal(
        make_metadata(cell=(30, 30, 30, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=make_macromolecule_atoms(["C"] * 10),
    )
    assert len(sparse.warnings) == 1 and "higher than expected" in sparse.warnings[0]

    # A 10 A cube holding 60 carbon atoms is dense but still fits.
    dense = validate_crystal(
        make_metadata(cell=(10, 10, 10, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=make_macromolecule_atoms(["C"] * 60),
    )
    assert len(dense.warnings) == 1 and "lower than expected" in dense.warnings[0]

    with pytest.raises(CrystalValidationError, match="cannot fit"):
        validate_crystal(
            make_metadata(cell=(10, 10, 10, 90, 90, 90), space_group="P 1"),
            macromolecule_atoms=make_macromolecule_atoms(["C"] * 200),
        )


def test_unknown_elements_contribute_no_mass() -> None:
    assert macromolecule_mass(make_macromolecule_atoms(["Xx", ""])).total_dalton == 0.0
    crystal = validate_crystal(
        make_metadata(), macromolecule_atoms=make_macromolecule_atoms(["Xx"])
    )
    assert crystal.content is None


def _standard_orthogonalization(cell: UnitCellParameters) -> NDArray[np.float64]:
    """Build cell vectors, one per row, with a along x and b in the xy plane.

    This is an independent reference for the vectors read back from gemmi.
    """
    cos_alpha, cos_beta, cos_gamma = (
        math.cos(math.radians(angle)) for angle in (cell.alpha, cell.beta, cell.gamma)
    )
    sin_gamma = math.sin(math.radians(cell.gamma))
    volume_factor = math.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    return np.asarray(
        (
            (cell.a, 0.0, 0.0),
            (cell.b * cos_gamma, cell.b * sin_gamma, 0.0),
            (
                cell.c * cos_beta,
                cell.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                cell.c * volume_factor / sin_gamma,
            ),
        ),
        dtype=np.float64,
    )


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
    standard = _standard_orthogonalization(UnitCellParameters(*cell))
    from_gemmi = translation_vectors_from_gemmi_cell(gemmi.UnitCell(*cell))

    assert np.allclose(from_gemmi, standard, atol=1e-9)

    crystal = validate_crystal(
        make_metadata(cell=cell, space_group="P 1"), macromolecule_atoms=()
    )
    assert np.array_equal(crystal.translation_vectors, from_gemmi)
