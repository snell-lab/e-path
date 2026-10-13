"""Structure preparation: cleaning, conformer resolution, classification, selection."""

import math
from pathlib import Path

import pytest
from support import make_atom_record, make_metadata

from rabdam.structure.models import (
    PreparedStructure,
    RemovedAtomCounts,
    StructurePreparationError,
)
from rabdam.structure.prepare import prepare_structure
from xtal.reader import AtomRecord, StructureData, read_structure
from xtal.resolver import resolve_structure_input


def _prepare(*atoms: AtomRecord) -> PreparedStructure:
    return prepare_structure(StructureData(atoms=atoms, metadata=make_metadata()))


def _source_indices(prepared: PreparedStructure) -> list[int]:
    return [atom.record.source_atom_index for atom in prepared.cleaned_atoms]


def test_each_kind_of_unusable_atom_is_removed_and_counted() -> None:
    prepared = _prepare(
        make_atom_record(source_atom_index=0),
        make_atom_record(source_atom_index=1, atom_name="HB1", element="H"),
        make_atom_record(source_atom_index=2, atom_name="DG", element="D"),
        # A hydrogen is counted as a hydrogen even when it fails another test.
        make_atom_record(source_atom_index=3, atom_name="HA", element="H", x=math.nan),
        make_atom_record(source_atom_index=4, atom_name="CB", x=math.nan),
        make_atom_record(source_atom_index=5, atom_name="N", occupancy=0.0),
        make_atom_record(source_atom_index=6, atom_name="C", occupancy=1.5),
        make_atom_record(source_atom_index=7, atom_name="O", b_factor=0.0),
        make_atom_record(source_atom_index=8, atom_name="OXT", b_factor=-3.0),
        make_atom_record(source_atom_index=9, atom_name="CG", b_factor=math.nan),
        # Mercury is not hydrogen, even though its symbol starts with H.
        make_atom_record(
            source_atom_index=10,
            residue_name="HG",
            residue_number=2,
            atom_name="HG",
            element="HG",
            is_hetatm=True,
        ),
    )

    assert _source_indices(prepared) == [0, 10]
    assert prepared.removed == RemovedAtomCounts(
        hydrogens=3,
        non_finite_coordinates=1,
        invalid_occupancy=2,
        non_positive_b_factor=3,
        alternate_conformers=0,
    )


def test_the_highest_occupancy_conformer_is_kept_at_each_site() -> None:
    prepared = _prepare(
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

    assert _source_indices(prepared) == [0, 2, 3, 5, 7]
    assert prepared.removed.alternate_conformers == 3


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ALA", (True, False, False)),
        ("mse ", (True, False, False)),
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
    prepared = _prepare(
        make_atom_record(source_atom_index=0),
        make_atom_record(source_atom_index=1, residue_name=name, residue_number=2),
    )

    atom = prepared.cleaned_atoms[1]
    assert (atom.is_protein, atom.is_nucleic_acid, atom.is_solvent) == expected


def test_only_protein_atoms_are_selected() -> None:
    prepared = _prepare(
        make_atom_record(source_atom_index=0, residue_name="ALA"),
        make_atom_record(source_atom_index=1, residue_name="MSE", is_hetatm=True),
        make_atom_record(source_atom_index=2, residue_name="HOH", is_hetatm=True),
        make_atom_record(source_atom_index=3, residue_name="DA"),
        make_atom_record(source_atom_index=4, residue_name="FMN", is_hetatm=True),
    )

    assert _source_indices(prepared) == [0, 1, 2, 3, 4]
    assert [atom.record.source_atom_index for atom in prepared.selected_atoms] == [
        0,
        1,
    ]


def test_prepare_structure_counts_every_cleaning_step(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))

    prepared = prepare_structure(data)

    assert len(data.atoms) == 88
    assert prepared.removed == RemovedAtomCounts(
        hydrogens=1,
        non_finite_coordinates=0,
        invalid_occupancy=1,
        non_positive_b_factor=1,
        alternate_conformers=1,
    )
    assert len(prepared.cleaned_atoms) == 84
    assert len(prepared.selected_atoms) == 77
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
    with pytest.raises(StructurePreparationError, match="No protein atoms"):
        _prepare(make_atom_record(residue_name="HOH", is_hetatm=True))

    with pytest.raises(StructurePreparationError, match="No atoms remain"):
        _prepare(make_atom_record(b_factor=0.0))
