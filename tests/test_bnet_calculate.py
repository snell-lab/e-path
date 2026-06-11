from pathlib import Path
import unittest
from unittest.mock import patch

from bdamage.score import BDamageAtomResult, BDamageScoreResult
from bnet.calculate import (
    ProteinBnetCalculationError,
    ProteinBnetResult,
    calculate_protein_bnet,
)
from bnet.metric import BnetResult
from bnet.sites import BnetSiteSelectionError
from input.reader import AtomRecord, StructureMetadata
from input.resolver import StructureFileFormat
from structure.models import (
    PreparedAtom,
    PreparedStructure,
    StructurePreparationReport,
)


def make_prepared_atom(
    *,
    source_atom_index: int,
    atom_serial: int,
    residue_name: str,
    atom_name: str,
) -> PreparedAtom:
    record = AtomRecord(
        source_atom_index=source_atom_index,
        model_number=1,
        chain_id="A",
        residue_name=residue_name,
        residue_number=source_atom_index + 1,
        insertion_code="",
        atom_name=atom_name,
        element="O" if atom_name.startswith("O") else "C",
        altloc="",
        x=0.0,
        y=0.0,
        z=0.0,
        occupancy=1.0,
        b_factor=10.0,
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
            source_path=Path("test.cif"),
            structure_id=None,
            file_format=StructureFileFormat.MMCIF,
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


def make_bdamage_result(
    atoms: tuple[PreparedAtom, ...],
    bdamage_values: tuple[float, ...],
) -> BDamageScoreResult:
    return BDamageScoreResult(
        atom_results=tuple(
            BDamageAtomResult(
                bdamage_atom_index=index,
                source_atom_index=atom.record.source_atom_index,
                atom_serial=atom.record.atom_serial,
                b_factor=10.0,
                packing_density=5,
                average_b_factor=10.0,
                bdamage=bdamage,
                sorted_packing_density_index=index,
            )
            for index, (atom, bdamage) in enumerate(
                zip(atoms, bdamage_values, strict=True),
                start=1,
            )
        ),
        window_size=1,
        selected_atom_count=len(atoms),
    )


class ProteinBnetCalculationTests(unittest.TestCase):
    def test_calculate_protein_bnet_returns_metric_and_sites(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
            make_prepared_atom(
                source_atom_index=1,
                atom_serial=11,
                residue_name="ALA",
                atom_name="CA",
            ),
            make_prepared_atom(
                source_atom_index=2,
                atom_serial=12,
                residue_name="GLU",
                atom_name="OE2",
            ),
            make_prepared_atom(
                source_atom_index=3,
                atom_serial=13,
                residue_name="SER",
                atom_name="CA",
            ),
        )
        bdamage_result = make_bdamage_result(atoms, (0.7, 0.9, 2.1, 1.2))

        result = calculate_protein_bnet(
            prepared_structure=make_prepared_structure(atoms),
            bdamage_score_result=bdamage_result,
        )

        self.assertIsInstance(result, ProteinBnetResult)
        self.assertEqual(result.site_count, 2)
        self.assertEqual([site.atom_serial for site in result.sites], [10, 12])
        self.assertEqual(result.metric.site_count, 2)
        self.assertEqual(result.bnet, result.metric.bnet)
        self.assertEqual(result.median_bdamage, result.metric.median_bdamage)
        self.assertEqual(result.left_area, result.metric.left_area)
        self.assertEqual(result.right_area, result.metric.right_area)

    def test_calculate_protein_bnet_raises_when_no_sites_are_selected(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ALA",
                atom_name="CA",
            ),
        )
        bdamage_result = make_bdamage_result(atoms, (1.0,))

        with self.assertRaisesRegex(
            ProteinBnetCalculationError,
            "no Asp/Glu carboxyl oxygen Bnet sites",
        ):
            calculate_protein_bnet(
                prepared_structure=make_prepared_structure(atoms),
                bdamage_score_result=bdamage_result,
            )

    def test_metric_calculation_errors_are_wrapped(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        bdamage_result = make_bdamage_result(atoms, (1.0,))

        with self.assertRaises(ProteinBnetCalculationError) as context:
            calculate_protein_bnet(
                prepared_structure=make_prepared_structure(atoms),
                bdamage_score_result=bdamage_result,
            )

        self.assertIsInstance(context.exception.__cause__, ValueError)
        self.assertIn("fewer than two", str(context.exception))

    def test_site_selection_errors_are_wrapped_with_original_cause(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        bdamage_result = make_bdamage_result(atoms, (float("nan"),))

        with self.assertRaises(ProteinBnetCalculationError) as context:
            calculate_protein_bnet(
                prepared_structure=make_prepared_structure(atoms),
                bdamage_score_result=bdamage_result,
            )

        self.assertIsInstance(context.exception.__cause__, BnetSiteSelectionError)
        self.assertIn("non-finite BDamage value", str(context.exception))

    def test_metric_site_count_must_match_selected_site_count(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
            make_prepared_atom(
                source_atom_index=1,
                atom_serial=11,
                residue_name="GLU",
                atom_name="OE2",
            ),
        )
        bdamage_result = make_bdamage_result(atoms, (0.7, 2.1))
        mismatched_metric = BnetResult(
            bnet=1.0,
            median_bdamage=1.4,
            left_area=0.5,
            right_area=0.5,
            site_count=1,
        )

        with patch(
            "bnet.metric.calculate_bnet",
            return_value=mismatched_metric,
        ):
            with self.assertRaisesRegex(
                ProteinBnetCalculationError,
                "site count does not match metric site count",
            ):
                calculate_protein_bnet(
                    prepared_structure=make_prepared_structure(atoms),
                    bdamage_score_result=bdamage_result,
                )
