from pathlib import Path
import unittest

from bdamage.score import BDamageAtomResult, BDamageScoreResult
from bnet.sites import (
    BnetSite,
    BnetSiteSelectionError,
    ProteinBnetSiteSelection,
    all_selected_bdamage_values,
    is_protein_bnet_site,
    select_protein_bnet_sites,
)
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
    atom_serial: int | None,
    residue_name: str,
    atom_name: str,
    is_protein: bool = True,
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
        is_protein=is_protein,
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


class BnetSiteSelectionTests(unittest.TestCase):
    def test_is_protein_bnet_site_selects_expected_atoms(self) -> None:
        true_cases = (
            ("ASP", "OD1"),
            ("ASP", "OD2"),
            ("GLU", "OE1"),
            ("GLU", "OE2"),
            ("DAS", "OD1"),
            ("DGL", "OE2"),
            (" asp ", " od1 "),
        )

        for residue_name, atom_name in true_cases:
            with self.subTest(residue_name=residue_name, atom_name=atom_name):
                self.assertTrue(
                    is_protein_bnet_site(
                        residue_name=residue_name,
                        atom_name=atom_name,
                    )
                )

    def test_is_protein_bnet_site_rejects_non_sites(self) -> None:
        false_cases = (
            ("ASP", "CG"),
            ("ASP", "OE1"),
            ("ASP", "OE2"),
            ("GLU", "CD"),
            ("GLU", "OD1"),
            ("GLU", "OD2"),
            ("ASH", "OD1"),
            ("GLH", "OE1"),
            ("ASN", "OD1"),
        )

        for residue_name, atom_name in false_cases:
            with self.subTest(residue_name=residue_name, atom_name=atom_name):
                self.assertFalse(
                    is_protein_bnet_site(
                        residue_name=residue_name,
                        atom_name=atom_name,
                    )
                )

    def test_select_protein_bnet_sites_returns_site_metadata_and_values(self) -> None:
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
        )
        prepared_structure = make_prepared_structure(atoms)
        bdamage_result = make_bdamage_result(atoms, (1.2, 0.9, 1.8))

        selection = select_protein_bnet_sites(
            prepared_structure=prepared_structure,
            bdamage_score_result=bdamage_result,
        )

        self.assertEqual(selection.bdamage_values, (1.2, 1.8))
        self.assertEqual([site.atom_serial for site in selection.sites], [10, 12])
        self.assertEqual([site.residue_name for site in selection.sites], ["ASP", "GLU"])
        self.assertEqual([site.atom_name for site in selection.sites], ["OD1", "OE2"])

    def test_selection_bdamage_values_are_computed_from_sites(self) -> None:
        site = BnetSite(
            source_atom_index=0,
            bdamage_atom_index=1,
            atom_serial=10,
            chain_id="A",
            residue_name="ASP",
            residue_number=1,
            insertion_code="",
            atom_name="OD1",
            bdamage=1.2,
        )

        selection = ProteinBnetSiteSelection(sites=(site,))

        self.assertEqual(selection.bdamage_values, (1.2,))

    def test_select_protein_bnet_sites_validates_result_count(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        prepared_structure = make_prepared_structure(atoms)
        bdamage_result = BDamageScoreResult(
            atom_results=(),
            window_size=1,
            selected_atom_count=0,
        )

        with self.assertRaises(BnetSiteSelectionError):
            select_protein_bnet_sites(
                prepared_structure=prepared_structure,
                bdamage_score_result=bdamage_result,
            )

    def test_select_protein_bnet_sites_validates_atom_identity(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        prepared_structure = make_prepared_structure(atoms)
        bdamage_result = BDamageScoreResult(
            atom_results=(
                BDamageAtomResult(1, 99, 10, 10.0, 5, 10.0, 1.2, 1),
            ),
            window_size=1,
            selected_atom_count=1,
        )

        with self.assertRaises(BnetSiteSelectionError):
            select_protein_bnet_sites(
                prepared_structure=prepared_structure,
                bdamage_score_result=bdamage_result,
            )

    def test_select_protein_bnet_sites_validates_atom_serial(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        prepared_structure = make_prepared_structure(atoms)
        bdamage_result = BDamageScoreResult(
            atom_results=(
                BDamageAtomResult(1, 0, 99, 10.0, 5, 10.0, 1.2, 1),
            ),
            window_size=1,
            selected_atom_count=1,
        )

        with self.assertRaises(BnetSiteSelectionError):
            select_protein_bnet_sites(
                prepared_structure=prepared_structure,
                bdamage_score_result=bdamage_result,
            )

    def test_select_protein_bnet_sites_rejects_non_finite_site_bdamage(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )
        prepared_structure = make_prepared_structure(atoms)
        bdamage_result = make_bdamage_result(atoms, (float("nan"),))

        with self.assertRaises(BnetSiteSelectionError):
            select_protein_bnet_sites(
                prepared_structure=prepared_structure,
                bdamage_score_result=bdamage_result,
            )

    def test_all_selected_bdamage_values_returns_all_values(self) -> None:
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
        )
        bdamage_result = make_bdamage_result(atoms, (1.2, 0.9))

        self.assertEqual(all_selected_bdamage_values(bdamage_result), (1.2, 0.9))

    def test_all_selected_bdamage_values_validates_values_and_empty_results(self) -> None:
        atoms = (
            make_prepared_atom(
                source_atom_index=0,
                atom_serial=10,
                residue_name="ASP",
                atom_name="OD1",
            ),
        )

        with self.assertRaises(BnetSiteSelectionError):
            all_selected_bdamage_values(make_bdamage_result(atoms, (float("inf"),)))

        with self.assertRaises(BnetSiteSelectionError):
            all_selected_bdamage_values(
                BDamageScoreResult(
                    atom_results=(),
                    window_size=1,
                    selected_atom_count=0,
                )
            )
