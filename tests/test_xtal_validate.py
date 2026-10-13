"""Asymmetric-unit content checks and the derived fields of a validated crystal."""

import dataclasses
import math
from collections.abc import Callable

import gemmi
import pytest
from support import approx, make_macromolecule_atoms, make_metadata

from xtal.validate import (
    MATTHEWS_NUCLEIC_ACID_CONSTANT,
    MATTHEWS_PROTEIN_CONSTANT,
    MAXIMUM_PLAUSIBLE_MATTHEWS,
    MINIMUM_PLAUSIBLE_MATTHEWS,
    NUCLEIC_ACID_HYDROGENS_PER_HEAVY_ATOM,
    PROTEIN_HYDROGENS_PER_HEAVY_ATOM,
    CrystalValidationError,
    MacromoleculeAtom,
    MacromoleculeKind,
    macromolecule_mass,
    validate_crystal,
)

PROTEIN = MacromoleculeKind.PROTEIN
NUCLEIC_ACID = MacromoleculeKind.NUCLEIC_ACID
HYDROGEN = gemmi.Element("H").weight
CARBON = gemmi.Element("C").weight
NITROGEN = gemmi.Element("N").weight
OXYGEN = gemmi.Element("O").weight
ALANINE_HEAVY_ATOMS = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))
# Alanine inside a chain is C3H5NO.
ALANINE_IN_CHAIN = 3 * CARBON + 5 * HYDROGEN + NITROGEN + OXYGEN

type Cell = tuple[float, float, float, float, float, float]


def _residue(
    name: str,
    atoms: tuple[tuple[str, str], ...],
    *,
    kind: MacromoleculeKind = PROTEIN,
    occupancy: float = 1.0,
    number: int = 1,
) -> list[MacromoleculeAtom]:
    return [
        MacromoleculeAtom(
            element=element,
            occupancy=occupancy,
            kind=kind,
            residue_name=name,
            residue_id=("A", number, ""),
            atom_name=atom_name,
        )
        for atom_name, element in atoms
    ]


def _exact_case(
    target: float, lands_on_target: Callable[[float, float], bool]
) -> tuple[Cell, list[MacromoleculeAtom]]:
    """Return a P 1 cell and carbon atoms for which floating point hits `target`.

    The boundary tests are only meaningful when the compared quantity lands
    exactly on the limit, so the atom count is searched until it does.
    """
    for count in range(50, 500):
        atoms = make_macromolecule_atoms(["C"] * count)
        mass = macromolecule_mass(atoms).total_dalton
        cell = (target * mass / 100.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        volume = float(gemmi.UnitCell(*cell).volume)
        if lands_on_target(volume, mass):
            return cell, atoms
    pytest.fail(f"No carbon count lands exactly on {target}.")


@pytest.mark.parametrize(
    "target", [MINIMUM_PLAUSIBLE_MATTHEWS, MAXIMUM_PLAUSIBLE_MATTHEWS]
)
def test_advisory_band_limits_are_inclusive(target: float) -> None:
    cell, atoms = _exact_case(target, lambda volume, mass: volume / mass == target)

    crystal = validate_crystal(
        make_metadata(cell=cell, space_group="P 1"), macromolecule_atoms=atoms
    )

    assert crystal.content is not None
    assert crystal.content.matthews_coefficient == target
    assert crystal.warnings == ()


def test_zero_solvent_is_rejected() -> None:
    cell, atoms = _exact_case(
        MATTHEWS_PROTEIN_CONSTANT,
        lambda volume, mass: 1.0 - MATTHEWS_PROTEIN_CONSTANT * mass / volume == 0.0,
    )

    with pytest.raises(CrystalValidationError, match="cannot fit"):
        validate_crystal(
            make_metadata(cell=cell, space_group="P 1"), macromolecule_atoms=atoms
        )


def test_alternate_conformers_sum_to_one_full_site() -> None:
    full = _residue("ALA", ALANINE_HEAVY_ATOMS)
    split = _residue("ALA", ALANINE_HEAVY_ATOMS[:4]) + [
        dataclasses.replace(atom, occupancy=0.5)
        for atom in _residue("ALA", (("CB", "C"), ("CB", "C")))
    ]

    assert macromolecule_mass(split).total_dalton == approx(
        macromolecule_mass(full).total_dalton
    )
    assert macromolecule_mass(split).atom_count == 6


def test_alanine_gains_its_five_chain_hydrogens() -> None:
    mass = macromolecule_mass(_residue("ALA", ALANINE_HEAVY_ATOMS))

    assert mass.implicit_hydrogen_dalton == approx(5 * HYDROGEN)
    assert mass.total_dalton == approx(ALANINE_IN_CHAIN)
    # Gemmi tabulates free alanine, one water heavier than the chain residue.
    water = 2 * HYDROGEN + OXYGEN
    alanine = gemmi.find_tabulated_residue("ALA")
    assert mass.total_dalton == approx(alanine.weight - water, abs=1e-3)


def test_implicit_hydrogens_follow_residue_occupancy() -> None:
    mass = macromolecule_mass(_residue("ALA", ALANINE_HEAVY_ATOMS, occupancy=0.5))

    assert mass.implicit_hydrogen_dalton == approx(2.5 * HYDROGEN)
    assert mass.total_dalton == approx(ALANINE_IN_CHAIN / 2)


def test_modelled_hydrogens_are_not_added_twice() -> None:
    hydrogens = tuple((name, "H") for name in ("H", "HA", "HB1", "HB2", "HB3"))
    protonated = macromolecule_mass(_residue("ALA", ALANINE_HEAVY_ATOMS + hydrogens))
    polar_only = macromolecule_mass(_residue("ALA", (*ALANINE_HEAVY_ATOMS, ("H", "H"))))

    assert protonated.implicit_hydrogen_dalton == 0.0
    assert protonated.total_dalton == approx(ALANINE_IN_CHAIN)
    assert polar_only.implicit_hydrogen_dalton == approx(4 * HYDROGEN)
    assert polar_only.total_dalton == approx(ALANINE_IN_CHAIN)


def test_untabulated_residues_use_the_per_heavy_atom_average() -> None:
    histidine = macromolecule_mass(
        _residue("HID", (("N", "N"), ("CA", "C"), ("C", "C")))
    )
    cytosine = macromolecule_mass(
        _residue("5MC", (("P", "P"), ("OP1", "O")), kind=NUCLEIC_ACID)
    )

    assert gemmi.find_tabulated_residue("HID").found() is False
    assert histidine.implicit_hydrogen_dalton == approx(
        3 * PROTEIN_HYDROGENS_PER_HEAVY_ATOM * HYDROGEN
    )
    assert cytosine.implicit_hydrogen_dalton == approx(
        2 * NUCLEIC_ACID_HYDROGENS_PER_HEAVY_ATOM * HYDROGEN
    )
    assert cytosine.protein_dalton == 0.0


def test_out_of_range_occupancies_are_clamped_with_a_warning() -> None:
    atoms = [
        dataclasses.replace(atom, occupancy=occupancy)
        for atom, occupancy in zip(
            make_macromolecule_atoms(["C"] * 4),
            (1.5, -0.2, math.nan, 1.0),
            strict=True,
        )
    ]
    mass = macromolecule_mass(atoms)
    full = macromolecule_mass(make_macromolecule_atoms(["C"] * 2))

    assert mass.adjusted_occupancy_count == 3
    assert mass.total_dalton == approx(full.total_dalton)

    crystal = validate_crystal(
        make_metadata(cell=(5, 5, 5, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=atoms,
    )
    assert any("clamped" in warning for warning in crystal.warnings)


def test_zero_occupancy_skips_the_content_check_with_a_warning() -> None:
    crystal = validate_crystal(
        make_metadata(),
        macromolecule_atoms=make_macromolecule_atoms(["C"] * 3, occupancy=0.0),
    )

    assert crystal.content is None
    assert len(crystal.warnings) == 1 and "zero occupancy" in crystal.warnings[0]


def test_nucleic_acid_uses_its_own_partial_specific_volume() -> None:
    atoms = make_macromolecule_atoms(["C"] * 100, kind=NUCLEIC_ACID)
    mass = macromolecule_mass(atoms).total_dalton
    # A Matthews coefficient of 1.1 leaves no room for protein, but nucleic
    # acid at 0.83 A^3/Da still leaves about a quarter of the cell for solvent.
    volume = 1.1 * mass
    crystal = validate_crystal(
        make_metadata(cell=(volume / 100.0, 10, 10, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=atoms,
    )

    content = crystal.content
    assert content is not None
    assert content.nucleic_acid_mass_dalton == approx(mass)
    assert content.protein_mass_dalton == 0.0
    assert content.matthews_coefficient == approx(1.1)
    assert content.solvent_fraction == approx(
        1.0 - MATTHEWS_NUCLEIC_ACID_CONSTANT / 1.1
    )


def test_mixed_models_weight_each_partial_specific_volume() -> None:
    protein = make_macromolecule_atoms(["C"] * 300)
    nucleic_acid = [
        dataclasses.replace(atom, kind=NUCLEIC_ACID, residue_id=("B", index, ""))
        for index, atom in enumerate(make_macromolecule_atoms(["C"] * 100))
    ]
    protein_mass = macromolecule_mass(protein).total_dalton
    nucleic_acid_mass = macromolecule_mass(nucleic_acid).total_dalton
    volume = 40.0 * 30.0 * 20.0

    crystal = validate_crystal(
        make_metadata(cell=(40, 30, 20, 90, 90, 90), space_group="P 1"),
        macromolecule_atoms=protein + nucleic_acid,
    )

    content = crystal.content
    assert content is not None
    assert content.protein_mass_dalton == approx(protein_mass)
    assert content.nucleic_acid_mass_dalton == approx(nucleic_acid_mass)
    assert content.matthews_coefficient == approx(
        volume / (protein_mass + nucleic_acid_mass)
    )
    assert content.solvent_fraction == approx(
        1.0
        - (
            MATTHEWS_PROTEIN_CONSTANT * protein_mass
            + MATTHEWS_NUCLEIC_ACID_CONSTANT * nucleic_acid_mass
        )
        / volume
    )


def test_unrecognized_elements_are_dropped_with_a_warning() -> None:
    metadata = make_metadata(cell=(30, 30, 30, 90, 90, 90), space_group="P 1")
    known = make_macromolecule_atoms(["C"] * 1000)
    unknown = make_macromolecule_atoms(["Xx", ""])

    crystal = validate_crystal(metadata, macromolecule_atoms=known + unknown)

    assert crystal.content is not None
    assert crystal.content.macromolecule_atom_count == len(known)
    assert crystal.content.macromolecule_mass_dalton == approx(
        macromolecule_mass(known).total_dalton
    )
    assert len(crystal.warnings) == 1
    assert "2 of 1002 macromolecule atoms" in crystal.warnings[0]
    assert "'Xx'" in crystal.warnings[0]


def test_all_unrecognized_elements_skip_the_content_check_with_a_warning() -> None:
    crystal = validate_crystal(
        make_metadata(), macromolecule_atoms=make_macromolecule_atoms(["Xx", "Q"])
    )

    assert crystal.content is None
    assert len(crystal.warnings) == 1
    assert "None of the 2 macromolecule element symbols" in crystal.warnings[0]


def test_no_atoms_skip_the_content_check_silently() -> None:
    crystal = validate_crystal(make_metadata(), macromolecule_atoms=[])
    assert crystal.content is None and crystal.warnings == ()


def test_centering_counts_toward_the_asymmetric_units() -> None:
    atoms = make_macromolecule_atoms(["C"] * 100)

    crystal = validate_crystal(
        make_metadata(cell=(40.0, 30.0, 20.0, 90.0, 90.0, 90.0), space_group="C 1 2 1"),
        macromolecule_atoms=atoms,
    )

    assert crystal.operation_count == 4
    assert crystal.content is not None
    assert crystal.content.matthews_coefficient == approx(
        40 * 30 * 20 / (4 * macromolecule_mass(atoms).total_dalton)
    )


def test_reported_name_shows_the_setting() -> None:
    crystal = validate_crystal(
        make_metadata(cell=(50, 50, 60, 90, 90, 120), space_group="R 3"),
        macromolecule_atoms=(),
    )

    assert crystal.space_group_name == "R 3:H"
    assert crystal.operation_count == 9


def test_derived_fields_follow_the_gemmi_cell() -> None:
    crystal = validate_crystal(
        make_metadata(cell=(10, 20, 30, 90, 90, 90)), macromolecule_atoms=()
    )

    assert crystal.unit_cell_volume_angstrom3 == approx(6000.0)
    assert crystal.translation_vectors[2, 2] == approx(30.0)
    assert len(crystal.operations) == crystal.operation_count


def test_a_file_cell_with_no_physical_volume_is_rejected() -> None:
    # The file's own cell skips the parameter checks, which see only the valid
    # numbers in the metadata, so the volume check is what catches it.
    metadata = dataclasses.replace(
        make_metadata(cell=(10, 10, 10, 90, 90, 90), space_group="P 1"),
        gemmi_cell=gemmi.UnitCell(10, 10, 10, 60, 60, 170),
    )

    with pytest.raises(CrystalValidationError, match="non-physical volume"):
        validate_crystal(metadata, macromolecule_atoms=())
