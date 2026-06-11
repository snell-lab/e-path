import math
import unittest
from pathlib import Path

from input.reader import AtomRecord, StructureData, StructureMetadata
from input.resolver import StructureFileFormat
from structure.filters import filter_clean_atoms, has_finite_coordinates
from structure.prepare import prepare_structure


def make_atom(
    *,
    source_atom_index: int,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    atom_name: str = "CA",
    element: str = "C",
    residue_name: str = "ALA",
) -> AtomRecord:
    return AtomRecord(
        source_atom_index=source_atom_index,
        model_number=1,
        chain_id="A",
        residue_name=residue_name,
        residue_number=1,
        insertion_code="",
        atom_name=atom_name,
        element=element,
        altloc="",
        x=x,
        y=y,
        z=z,
        occupancy=1.0,
        b_factor=10.0,
        atom_serial=source_atom_index + 1,
        record_type="ATOM",
    )


def make_structure_data(atoms: tuple[AtomRecord, ...]) -> StructureData:
    return StructureData(
        atoms=atoms,
        metadata=StructureMetadata(
            source_path=Path("test.pdb"),
            structure_id=None,
            file_format=StructureFileFormat.PDB,
            space_group="P 1",
            unit_cell_a=10.0,
            unit_cell_b=20.0,
            unit_cell_c=30.0,
            unit_cell_alpha=90.0,
            unit_cell_beta=90.0,
            unit_cell_gamma=90.0,
        ),
    )


class StructureFilterTests(unittest.TestCase):
    def test_has_finite_coordinates_rejects_nan_and_infinities(self) -> None:
        self.assertTrue(has_finite_coordinates(make_atom(source_atom_index=0)))

        for coordinate in (math.nan, math.inf, -math.inf):
            with self.subTest(coordinate=coordinate):
                self.assertFalse(
                    has_finite_coordinates(
                        make_atom(source_atom_index=1, x=coordinate)
                    )
                )

    def test_filter_clean_atoms_removes_atoms_with_non_finite_coordinates(self) -> None:
        atoms = (
            make_atom(source_atom_index=0),
            make_atom(source_atom_index=1, x=math.nan),
            make_atom(source_atom_index=2, y=math.inf),
            make_atom(source_atom_index=3, z=-math.inf),
        )

        result = filter_clean_atoms(atoms)

        self.assertEqual(tuple(atom.source_atom_index for atom in result.atoms), (0,))
        self.assertEqual(result.counts.invalid_coordinates, 3)

    def test_prepare_structure_reports_removed_invalid_coordinates(self) -> None:
        structure_data = make_structure_data(
            (
                make_atom(source_atom_index=0),
                make_atom(source_atom_index=1, x=math.nan),
            )
        )

        prepared = prepare_structure(structure_data)

        self.assertEqual(prepared.report.cleaned_atom_count, 1)
        self.assertEqual(prepared.report.selected_atom_count, 1)
        self.assertEqual(prepared.report.removed_invalid_coordinate_count, 1)


if __name__ == "__main__":
    unittest.main()
