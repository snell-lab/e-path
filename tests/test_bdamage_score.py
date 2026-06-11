from pathlib import Path
import unittest

from bdamage.score import (
    BDamageAtomInput,
    BDamageScoreError,
    bdamage_input_from_prepared_atom,
    bdamage_scores_as_tuple,
    calculate_bdamage_scores,
    calculate_bdamage_scores_for_structure,
    centered_window_average_b_factors,
    sort_atom_inputs_for_bdamage,
)
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from packing.density import PackingDensityAtomResult, PackingDensityResult
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationReport,
)


def make_atom_input(
    *,
    bdamage_atom_index: int,
    source_atom_index: int,
    atom_serial: int | None,
    b_factor: float,
    packing_density: int,
) -> BDamageAtomInput:
    return BDamageAtomInput(
        bdamage_atom_index=bdamage_atom_index,
        source_atom_index=source_atom_index,
        atom_serial=atom_serial,
        b_factor=b_factor,
        packing_density=packing_density,
    )


def make_prepared_atom(
    *,
    source_atom_index: int,
    atom_serial: int | None,
    b_factor: float,
) -> PreparedAtom:
    record = AtomRecord(
        source_atom_index=source_atom_index,
        model_number=1,
        chain_id="A",
        residue_name="ALA",
        residue_number=1,
        insertion_code="",
        atom_name="CA",
        element="C",
        altloc="",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        b_factor=b_factor,
        atom_serial=atom_serial,
        record_type="ATOM",
    )
    return PreparedAtom(
        record=record,
        is_hydrogen=False,
        is_protein=True,
        is_nucleic_acid=False,
        is_solvent=False,
        is_hetatm=False,
    )


def make_prepared_structure(atoms: tuple[PreparedAtom, ...]) -> PreparedStructure:
    return PreparedStructure(
        cleaned_atoms=atoms,
        selected_atoms=atoms,
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
        report=StructurePreparationReport(
            input_atom_count=len(atoms),
            cleaned_atom_count=len(atoms),
            selected_atom_count=len(atoms),
            removed_hydrogen_count=0,
            removed_invalid_coordinate_count=0,
            removed_invalid_occupancy_count=0,
            removed_invalid_b_factor_count=0,
            removed_altloc_count=0,
        ),
    )


class BDamageScoreTests(unittest.TestCase):
    def test_sort_atom_inputs_by_packing_density_then_atom_serial(self) -> None:
        inputs = (
            make_atom_input(bdamage_atom_index=1, source_atom_index=0, atom_serial=30, b_factor=10.0, packing_density=5),
            make_atom_input(bdamage_atom_index=2, source_atom_index=1, atom_serial=10, b_factor=20.0, packing_density=3),
            make_atom_input(bdamage_atom_index=3, source_atom_index=2, atom_serial=20, b_factor=30.0, packing_density=3),
        )

        sorted_inputs = sort_atom_inputs_for_bdamage(inputs)

        self.assertEqual([atom.atom_serial for atom in sorted_inputs], [10, 20, 30])

    def test_centered_window_average_b_factors_uses_edge_windows(self) -> None:
        inputs = tuple(
            make_atom_input(
                bdamage_atom_index=index,
                source_atom_index=index - 1,
                atom_serial=index,
                b_factor=float(index * 10),
                packing_density=index,
            )
            for index in range(1, 6)
        )

        averages = centered_window_average_b_factors(inputs, window_size=3)

        self.assertEqual(averages, (20.0, 20.0, 30.0, 40.0, 40.0))

    def test_centered_window_average_b_factors_uses_prefix_sum_window_ranges(self) -> None:
        inputs = tuple(
            make_atom_input(
                bdamage_atom_index=index,
                source_atom_index=index - 1,
                atom_serial=index,
                b_factor=float(index),
                packing_density=index,
            )
            for index in range(1, 10)
        )

        averages = centered_window_average_b_factors(inputs, window_size=5)

        self.assertEqual(
            averages,
            (
                3.0,
                3.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                7.0,
                7.0,
            ),
        )

    def test_even_window_size_raises(self) -> None:
        inputs = tuple(
            make_atom_input(
                bdamage_atom_index=index,
                source_atom_index=index - 1,
                atom_serial=index,
                b_factor=float(index),
                packing_density=index,
            )
            for index in range(1, 7)
        )

        with self.assertRaises(BDamageScoreError):
            centered_window_average_b_factors(inputs, window_size=4)

    def test_calculate_bdamage_scores_returns_original_selected_atom_order(self) -> None:
        inputs = (
            make_atom_input(bdamage_atom_index=1, source_atom_index=0, atom_serial=30, b_factor=40.0, packing_density=5),
            make_atom_input(bdamage_atom_index=2, source_atom_index=1, atom_serial=10, b_factor=10.0, packing_density=3),
            make_atom_input(bdamage_atom_index=3, source_atom_index=2, atom_serial=20, b_factor=20.0, packing_density=3),
            make_atom_input(bdamage_atom_index=4, source_atom_index=3, atom_serial=40, b_factor=80.0, packing_density=6),
        )

        result = calculate_bdamage_scores(atom_inputs=inputs, window_size=3)

        self.assertEqual(result.selected_atom_count, 4)
        self.assertEqual(result.window_size, 3)
        self.assertEqual([atom.atom_serial for atom in result.atom_results], [30, 10, 20, 40])
        self.assertEqual(
            [atom.sorted_packing_density_index for atom in result.atom_results],
            [3, 1, 2, 4],
        )
        self.assertEqual(
            bdamage_scores_as_tuple(result),
            (40.0 / (140.0 / 3.0), 10.0 / (70.0 / 3.0), 20.0 / (70.0 / 3.0), 80.0 / (140.0 / 3.0)),
        )

    def test_bdamage_input_from_prepared_atom_validates_identity(self) -> None:
        atom = make_prepared_atom(source_atom_index=5, atom_serial=100, b_factor=22.0)
        packing_result = PackingDensityAtomResult(
            packing_density_atom_index=1,
            source_atom_index=5,
            atom_serial=100,
            neighbour_count=12,
        )

        atom_input = bdamage_input_from_prepared_atom(
            bdamage_atom_index=1,
            atom=atom,
            packing_density_atom_result=packing_result,
        )

        self.assertEqual(atom_input.b_factor, 22.0)
        self.assertEqual(atom_input.packing_density, 12)

    def test_calculate_bdamage_scores_for_structure_joins_selected_atoms_and_packing_density(self) -> None:
        atoms = (
            make_prepared_atom(source_atom_index=0, atom_serial=1, b_factor=10.0),
            make_prepared_atom(source_atom_index=1, atom_serial=2, b_factor=20.0),
            make_prepared_atom(source_atom_index=2, atom_serial=3, b_factor=40.0),
        )
        prepared_structure = make_prepared_structure(atoms)
        packing_density_result = PackingDensityResult(
            atom_results=(
                PackingDensityAtomResult(1, 0, 1, 2),
                PackingDensityAtomResult(2, 1, 2, 4),
                PackingDensityAtomResult(3, 2, 3, 6),
            ),
            packing_density_threshold=7.5,
            selected_atom_count=3,
            neighbour_atom_count=100,
        )

        result = calculate_bdamage_scores_for_structure(
            prepared_structure=prepared_structure,
            packing_density_result=packing_density_result,
            window_size=3,
        )

        self.assertEqual([atom.packing_density for atom in result.atom_results], [2, 4, 6])
        self.assertEqual(
            bdamage_scores_as_tuple(result),
            (10.0 / (70.0 / 3.0), 20.0 / (70.0 / 3.0), 40.0 / (70.0 / 3.0)),
        )

    def test_mismatched_source_atom_raises(self) -> None:
        atom = make_prepared_atom(source_atom_index=5, atom_serial=100, b_factor=22.0)
        packing_result = PackingDensityAtomResult(
            packing_density_atom_index=1,
            source_atom_index=6,
            atom_serial=100,
            neighbour_count=12,
        )

        with self.assertRaises(BDamageScoreError):
            bdamage_input_from_prepared_atom(
                bdamage_atom_index=1,
                atom=atom,
                packing_density_atom_result=packing_result,
            )

    def test_invalid_window_size_raises(self) -> None:
        inputs = (
            make_atom_input(bdamage_atom_index=1, source_atom_index=0, atom_serial=1, b_factor=10.0, packing_density=1),
        )

        with self.assertRaises(BDamageScoreError):
            calculate_bdamage_scores(atom_inputs=inputs, window_size=2)

    def test_boolean_window_size_raises(self) -> None:
        inputs = (
            make_atom_input(bdamage_atom_index=1, source_atom_index=0, atom_serial=1, b_factor=10.0, packing_density=1),
        )

        with self.assertRaises(BDamageScoreError):
            calculate_bdamage_scores(atom_inputs=inputs, window_size=True)

    def test_non_positive_b_factor_raises(self) -> None:
        inputs = (
            make_atom_input(bdamage_atom_index=1, source_atom_index=0, atom_serial=1, b_factor=0.0, packing_density=1),
        )

        with self.assertRaises(BDamageScoreError):
            calculate_bdamage_scores(atom_inputs=inputs, window_size=1)


if __name__ == "__main__":
    unittest.main()
