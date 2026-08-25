"""Atom cleaning, alternate-conformer resolution, classification, selection."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from support import make_atom_record, make_metadata, make_prepared_atom

from rabdam.structure.altlocs import select_atom_site_altlocs
from rabdam.structure.classify import classify_component
from rabdam.structure.filters import filter_clean_atoms, is_hydrogen
from rabdam.structure.models import (
    StructurePreparationError,
    StructurePreparationOptions,
)
from rabdam.structure.prepare import prepare_structure
from rabdam.structure.selection import select_bdamage_atoms
from xtal.reader import StructureData, read_structure
from xtal.resolver import resolve_structure_input


def test_filter_removes_each_kind_of_unusable_atom() -> None:
    atoms = (
        make_atom_record(source_atom_index=0),
        make_atom_record(source_atom_index=1, atom_name="HB1", element="H"),
        make_atom_record(source_atom_index=2, atom_name="DG", element="D"),
        make_atom_record(source_atom_index=3, x=math.nan),
        make_atom_record(source_atom_index=4, occupancy=0.0),
        make_atom_record(source_atom_index=5, occupancy=1.5),
        make_atom_record(source_atom_index=6, b_factor=0.0),
        make_atom_record(source_atom_index=7, b_factor=-3.0),
    )

    result = filter_clean_atoms(atoms)

    assert [atom.source_atom_index for atom in result.atoms] == [0]
    assert (
        result.counts.hydrogen,
        result.counts.invalid_coordinates,
        result.counts.invalid_occupancy,
        result.counts.invalid_b_factor,
    ) == (2, 1, 2, 2)

    relaxed = filter_clean_atoms(
        atoms,
        remove_hydrogens=False,
        require_valid_occupancy=False,
        require_positive_b_factor=False,
    )
    assert [atom.source_atom_index for atom in relaxed.atoms] == [0, 1, 2, 4, 5, 6, 7]


def test_hydrogen_detection_falls_back_to_the_atom_name() -> None:
    assert is_hydrogen(make_atom_record(element="H"))
    assert is_hydrogen(make_atom_record(element="", atom_name="HD21"))
    assert not is_hydrogen(make_atom_record(element="HG", atom_name="HG"))  # mercury
    assert not is_hydrogen(make_atom_record(element="", atom_name="CA"))


def test_altlocs_keep_the_highest_occupancy_site_and_warn_once_per_residue() -> None:
    atoms = (
        make_atom_record(source_atom_index=0, atom_name="CA"),
        make_atom_record(
            source_atom_index=1, atom_name="OG", altloc="A", occupancy=0.4
        ),
        make_atom_record(
            source_atom_index=2, atom_name="OG", altloc="B", occupancy=0.6
        ),
        # A tie is resolved in favour of the first record.
        make_atom_record(
            source_atom_index=3, atom_name="CB", altloc="A", occupancy=0.5
        ),
        make_atom_record(
            source_atom_index=4, atom_name="CB", altloc="B", occupancy=0.5
        ),
        # A second residue with conformers, and one without.
        make_atom_record(
            source_atom_index=5,
            residue_number=2,
            atom_name="N",
            altloc="A",
            occupancy=0.7,
        ),
        make_atom_record(
            source_atom_index=6,
            residue_number=2,
            atom_name="N",
            altloc="B",
            occupancy=0.3,
        ),
        make_atom_record(source_atom_index=7, residue_number=3, atom_name="N"),
    )

    result = select_atom_site_altlocs(atoms)

    assert [atom.source_atom_index for atom in result.atoms] == [0, 2, 3, 5, 7]
    assert result.removed_count == 3
    assert result.warnings == (
        "Alternate conformers detected in chain A residue ALA1. Highest-occupancy atom sites were retained.",
        "Alternate conformers detected in chain A residue ALA2. Highest-occupancy atom sites were retained.",
    )


def test_altloc_resolution_can_be_disabled() -> None:
    atoms = (
        make_atom_record(altloc="A", occupancy=0.5),
        make_atom_record(source_atom_index=1, altloc="B", occupancy=0.5),
    )
    from rabdam.structure.altlocs import select_altlocs

    result = select_altlocs(atoms, StructurePreparationOptions(resolve_altlocs=False))
    assert result.atoms == atoms and result.removed_count == 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ALA", (True, False, False)),
        ("mse ", (True, False, False)),
        ("HID", (True, False, False)),
        ("DA", (False, True, False)),
        ("5MC", (False, True, False)),
        ("HOH", (False, False, True)),
        ("DOD", (False, False, True)),
        ("FMN", (False, False, False)),
        ("ZN", (False, False, False)),
        ("XYZ", (False, False, False)),
    ],
)
def test_component_classification(name: str, expected: tuple[bool, bool, bool]) -> None:
    assert classify_component(name) == expected


def test_only_protein_atoms_are_selected() -> None:
    atoms = (
        make_prepared_atom(source_atom_index=0, residue_name="ALA"),
        make_prepared_atom(source_atom_index=1, residue_name="MSE", is_hetatm=True),
        make_prepared_atom(source_atom_index=2, residue_name="HOH", is_hetatm=True),
        make_prepared_atom(source_atom_index=3, residue_name="DA"),
        make_prepared_atom(source_atom_index=4, residue_name="FMN", is_hetatm=True),
    )

    default = select_bdamage_atoms(atoms, StructurePreparationOptions())
    assert [atom.record.source_atom_index for atom in default] == [0, 1]

    strict = select_bdamage_atoms(
        atoms,
        StructurePreparationOptions(include_protein_like_hetatm_in_selection=False),
    )
    assert [atom.record.source_atom_index for atom in strict] == [0]


def test_prepare_structure_reports_every_cleaning_step(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))

    prepared = prepare_structure(data)
    report = prepared.report

    assert report.input_atom_count == 88
    assert report.removed_hydrogen_count == 1
    assert report.removed_invalid_occupancy_count == 1
    assert report.removed_invalid_b_factor_count == 1
    assert report.removed_altloc_count == 1
    assert report.removed_invalid_coordinate_count == 0
    assert report.cleaned_atom_count == len(prepared.cleaned_atoms) == 84
    assert report.selected_atom_count == len(prepared.selected_atoms) == 77
    assert report.warnings == (
        "Alternate conformers detected in chain A residue SER5. Highest-occupancy atom sites were retained.",
    )
    assert prepared.metadata is data.metadata

    kept_og = [a for a in prepared.cleaned_atoms if a.record.atom_name == "OG"]
    assert len(kept_og) == 1 and kept_og[0].record.altloc == "A"
    assert all(atom.is_protein for atom in prepared.selected_atoms)
    assert any(atom.record.residue_name == "MSE" for atom in prepared.selected_atoms)
    unselected = {a.record.residue_name for a in prepared.cleaned_atoms} - {
        a.record.residue_name for a in prepared.selected_atoms
    }
    assert unselected == {"FMN", "GOL", "HOH"}


def test_prepare_structure_requires_protein() -> None:
    water_only = StructureData(
        atoms=(make_atom_record(residue_name="HOH", is_hetatm=True),),
        metadata=make_metadata(),
    )
    with pytest.raises(StructurePreparationError, match="No protein atoms"):
        prepare_structure(water_only)

    nothing_left = StructureData(
        atoms=(make_atom_record(b_factor=0.0),), metadata=make_metadata()
    )
    with pytest.raises(StructurePreparationError, match="No atoms remain"):
        prepare_structure(nothing_left)
