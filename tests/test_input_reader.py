from pathlib import Path
import tempfile
import unittest

from input.reader import read_structure
from input.resolver import (
    ResolvedStructureInput,
    StructureFileFormat,
    StructureSourceType,
)


def resolved_local(path: Path, file_format: StructureFileFormat) -> ResolvedStructureInput:
    return ResolvedStructureInput(
        original_input=str(path),
        source_type=StructureSourceType.LOCAL_FILE,
        file_format=file_format,
        local_path=path,
    )


def mmcif_text(*, resolution_tag: str, resolution: str) -> str:
    return (
        "data_1abc\n"
        f"{resolution_tag} {resolution}\n"
        "_cell.length_a 10.0\n"
        "_cell.length_b 10.0\n"
        "_cell.length_c 10.0\n"
        "_cell.angle_alpha 90.0\n"
        "_cell.angle_beta 90.0\n"
        "_cell.angle_gamma 90.0\n"
        "_symmetry.space_group_name_H-M 'P 1'\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM 1 C CA . ALA A 1 1 ? 1.0 2.0 3.0 1.00 10.00 1 ALA A CA 1\n"
        "#\n"
    )


def pdb_text(*, resolution: str) -> str:
    return (
        "HEADER    TEST\n"
        f"REMARK   2 RESOLUTION.    {resolution} ANGSTROMS.\n"
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\n"
        "END\n"
    )


class StructureReaderTests(unittest.TestCase):
    def test_reads_mmcif_resolution_from_common_tags(self) -> None:
        cases = (
            ("_refine.ls_d_res_high", 1.5),
            ("_reflns.d_resolution_high", 1.8),
        )

        for tag, expected_resolution in cases:
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "model.cif"
                path.write_text(
                    mmcif_text(
                        resolution_tag=tag,
                        resolution=str(expected_resolution),
                    ),
                    encoding="utf-8",
                )

                structure_data = read_structure(
                    resolved_local(path, StructureFileFormat.MMCIF)
                )

            self.assertEqual(
                structure_data.metadata.resolution_angstrom,
                expected_resolution,
            )

    def test_reads_pdb_resolution_from_remark_2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pdb"
            path.write_text(pdb_text(resolution="2.10"), encoding="utf-8")

            structure_data = read_structure(resolved_local(path, StructureFileFormat.PDB))

        self.assertEqual(structure_data.metadata.resolution_angstrom, 2.1)


if __name__ == "__main__":
    unittest.main()
