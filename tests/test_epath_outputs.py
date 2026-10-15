"""Output files: CSV tables, coordinate subsets, the plot, and the summary."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest
from support import (
    SYNTHETIC_CELL,
    approx,
    make_crystal,
    make_scored_atom,
    read_csv_rows,
)

from epath import __version__
from epath.outputs.files import EpathOutputPipelineOptions, EpathPipelineOutputFiles
from epath.outputs.plots import write_z_score_falloff
from epath.outputs.structures import (
    write_atom_subset_cif,
    write_model_cif,
    write_path_subset_cifs,
)
from epath.outputs.summary import (
    output_file_entries,
    removed_atoms_line,
    write_summary_txt,
)
from epath.outputs.tables import (
    CLUSTER_FIELDNAMES,
    COMBINED_ATOM_FIELDNAMES,
    RUN_FIELDNAMES,
    STEP_FIELDNAMES,
    write_cluster_csv,
    write_combined_atom_csv,
    write_run_csv,
    write_step_csv,
)
from epath.outputs.write import write_epath_outputs
from epath.pipeline import (
    EpathClusterPipelineOptions,
    EpathPipelineOptions,
    EpathPipelineResult,
    EpathSelectionPipelineOptions,
    run_epath_pipeline,
)
from rabdam.pipeline import run_bdamage_pipeline
from rabdam.structure.models import RemovedAtomCounts
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input


@pytest.fixture(scope="module")
def result(synthetic_pdb: Path) -> EpathPipelineResult:
    """Return an analysed synthetic crystal with reportable pathways."""
    bdamage = run_bdamage_pipeline(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )
    options = EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(percentile=30.0),
        clustering=EpathClusterPipelineOptions(minimum_cluster_size=3),
    )
    pipeline_result, files = run_epath_pipeline(bdamage, options=options)
    assert files == EpathPipelineOutputFiles()
    assert len(pipeline_result.path_reports) >= 1
    return pipeline_result


def test_combined_atom_csv(tmp_path: Path, result: EpathPipelineResult) -> None:
    path = write_combined_atom_csv(
        tmp_path / "nested" / "x_epath.csv",
        result.selection_result,
        clusters=result.clusters,
    )
    rows = read_csv_rows(path)

    assert list(rows[0]) == COMBINED_ATOM_FIELDNAMES
    assert len(rows) == len(result.selection_result.scored_atoms) == 77
    selected = {
        a.record.source_atom_index for a in result.selection_result.selected_atoms
    }
    clustered = {
        a.record.source_atom_index: c.cluster_id
        for c in result.clusters
        for a in c.atoms
    }
    for row in rows:
        index = int(row["source_atom_index"])
        assert row["selected"] == ("true" if index in selected else "false")
        assert row["cluster_id"] == (
            str(clustered[index]) if index in clustered else ""
        )
        assert (
            row["atom_token"]
            == f"{row['chain_id']}:{row['residue_number']}:{row['atom_name']}"
        )
        assert float(row["bdamage"]) == approx(
            float(row["b_factor"]) / float(row["average_b_factor"])
        )
    assert sum(row["selected"] == "true" for row in rows) == 23


def test_cluster_step_and_run_csvs(tmp_path: Path, result: EpathPipelineResult) -> None:
    clusters = read_csv_rows(
        write_cluster_csv(tmp_path / "x_clusters.csv", result.path_reports)
    )
    assert list(clusters[0]) == CLUSTER_FIELDNAMES
    assert len(clusters) == len(result.path_reports)
    first = clusters[0]
    report = result.path_reports[0]
    assert first["residue_path"] == report.residue_path
    for row, path_report in zip(clusters, result.path_reports, strict=True):
        nearby = path_report.nearby_cofactors
        assert row["nearby_cofactors"] == "/".join(a.label for a in nearby)
        assert row["nearby_cofactor_atoms"] == "/".join(a.atom_token for a in nearby)
        distances = row["nearby_cofactor_distances"].split("/") if nearby else []
        assert [float(d) for d in distances] == approx([a.distance for a in nearby])
    assert int(first["atom_count"]) == report.node_count
    assert float(first["mst_total_length"]) == approx(report.total_length)
    assert first["chains"] == "/".join(report.chains)
    fractions = [
        float(first[f"{name}_fraction"])
        for name in ("aromatic", "sulfur", "charged", "polar", "hydrophobic")
    ]
    assert sum(fractions) <= 1.0 + 1e-9

    # A path that never leaves one residue has no peptide share to report.
    within_one_residue = replace(report, steps=report.steps[:0])
    rows = read_csv_rows(
        write_cluster_csv(tmp_path / "y_clusters.csv", (within_one_residue,))
    )
    assert rows[0]["peptide_edge_fraction"] == ""

    steps = read_csv_rows(write_step_csv(tmp_path / "x_steps.csv", result.path_reports))
    assert list(steps[0]) == STEP_FIELDNAMES
    assert (
        len(steps)
        == sum(len(r.steps) for r in result.path_reports)
        == sum(r.node_count - 1 for r in result.path_reports)
    )
    assert {row["kind"] for row in steps} <= {
        "intra_residue",
        "peptide",
        "tertiary",
        "interchain",
    }

    runs = read_csv_rows(write_run_csv(tmp_path / "x_run.csv", (result.run_record,)))
    assert list(runs[0]) == RUN_FIELDNAMES and len(runs) == 1
    assert runs[0]["structure"] == "synthetic"
    assert runs[0]["total_atoms"] == "77" and runs[0]["atoms_selected"] == "23"
    assert runs[0]["largest_component_reported"] == "true"
    assert float(runs[0]["bnet"]) > 0 and 0 <= float(runs[0]["bnet_percentile"]) <= 100
    assert runs[0]["clusters_reported"] == str(len(result.path_reports))
    assert runs[0]["epath_version"] == __version__
    assert runs[0]["cluster_distance"] == "4"
    assert runs[0]["cofactor_range"] == "8"


def test_atom_subset_cif_round_trips_through_gemmi(tmp_path: Path) -> None:
    atoms = (
        make_scored_atom(
            chain_id="A",
            residue_name="TRP",
            residue_number=76,
            atom_name="CA",
            x=1.234,
            y=-5.678,
            z=90.123,
            occupancy=0.5,
            bdamage=0.4321,
        ),
        make_scored_atom(
            source_atom_index=1,
            atom_serial=9,
            is_hetatm=True,
            chain_id="B",
            residue_name="MSE",
            residue_number=3,
            atom_name="SE",
            element="SE",
            bdamage=1.5,
        ),
    )
    path = write_atom_subset_cif(
        tmp_path / "subset.cif",
        atoms,
        crystal=make_crystal(),
        remarks=("Custom remark.",),
    )
    block = gemmi.cif.read_file(str(path)).sole_block()
    assert block.name == "subset"
    assert gemmi.cif.as_string(block.find_value("_struct.title")) == (
        "E-path atom subset"
    )
    remarks = [
        gemmi.cif.as_string(r[1]) for r in block.find("_epath_remark.", ["id", "text"])
    ]
    assert remarks == [
        "E-path coordinate subset.",
        "B_iso_or_equiv stores BDamage.",
        "Custom remark.",
    ]

    structure = gemmi.read_structure(str(path))
    model = structure[0]
    residue_a = model["A"][0]
    assert residue_a.name == "TRP" and residue_a.seqid.num == 76
    atom_a = residue_a[0]
    assert atom_a.name == "CA" and atom_a.serial == 1
    assert (atom_a.pos.x, atom_a.pos.y, atom_a.pos.z) == approx((1.234, -5.678, 90.123))
    assert atom_a.occ == approx(0.5) and atom_a.b_iso == approx(0.4321)
    residue_b = model["B"][0]
    assert (
        residue_b.het_flag == "H"
        and residue_b[0].name == "SE"
        and residue_b[0].element.name == "Se"
    )
    assert residue_b[0].serial == 2


def test_atom_subset_cif_keeps_fields_that_do_not_fit_pdb_columns(
    tmp_path: Path,
) -> None:
    atoms = (
        make_scored_atom(chain_id="AB", residue_number=12345, atom_name="CA"),
        make_scored_atom(chain_id="AB", residue_number=12346, atom_name="CA"),
        make_scored_atom(chain_id="C", residue_number=-1200, atom_name="CA"),
    )
    path = write_atom_subset_cif(tmp_path / "wide.cif", atoms, crystal=make_crystal())
    model = gemmi.read_structure(str(path))[0]
    assert [chain.name for chain in model] == ["AB", "C"]
    assert [res.seqid.num for res in model["AB"]] == [12345, 12346]
    assert model["C"][0].seqid.num == -1200

    empty = write_atom_subset_cif(tmp_path / "empty.cif", (), crystal=make_crystal())
    block = gemmi.cif.read_file(str(empty)).sole_block()
    assert len(block.find("_atom_site.", ["id"])) == 0


def test_model_cif_marks_unscored_atoms_with_zero_bdamage(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    cleaned = result.bdamage_pipeline_result.prepared_structure.cleaned_atoms
    scores = {
        a.record.source_atom_index: a.bdamage_result.bdamage
        for a in result.selection_result.scored_atoms
    }
    # An atom exactly at a threshold that four decimals would round down.
    threshold = 0.77344449
    at_threshold = next(iter(scores))
    scores[at_threshold] = threshold

    path = write_model_cif(
        tmp_path / "model.cif",
        cleaned,
        crystal=result.bdamage_pipeline_result.crystal,
        bdamage_by_source_index=scores,
        bdamage_threshold=threshold,
    )
    block = gemmi.cif.read_file(str(path)).sole_block()
    assert block.find_value("_epath.atom_count") == str(len(cleaned))
    assert block.find_value("_epath.scored_atom_count") == str(len(scores))
    # Written as the B-factor column writes it, single precision to six
    # significant figures, so the atom at the threshold reads as at it.
    assert block.find_value("_epath.bdamage_threshold") == "0.773444"
    remarks = [
        gemmi.cif.as_string(r[1]) for r in block.find("_epath_remark.", ["id", "text"])
    ]
    assert f"{len(scores)} of {len(cleaned)} atoms were scored." in remarks
    assert (
        "Selected atoms are among those with 0.00 < B_iso_or_equiv <= 0.773444."
        in remarks
    )

    atom_sites = block.find(
        "_atom_site.", ["group_PDB", "id", "label_comp_id", "B_iso_or_equiv"]
    )
    rows = [tuple(row) for row in atom_sites]
    assert len(rows) == len(cleaned) == 84
    unscored = [row for row in rows if float(row[3]) == 0.0]
    assert len(unscored) == len(cleaned) - len(scores) == 7
    assert {row[2] for row in unscored} == {"FMN", "GOL", "HOH"}
    assert [int(row[1]) for row in rows] == list(range(1, 85))
    assert all(row[0] == "HETATM" for row in rows if row[2] in {"MSE", "FMN", "HOH"})
    source_indices = [atom.record.source_atom_index for atom in cleaned]
    assert rows[source_indices.index(at_threshold)][3] == "0.773444"


def test_path_subset_cifs_are_zero_padded(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    crystal = result.bdamage_pipeline_result.crystal
    paths = write_path_subset_cifs(
        tmp_path / "run", result.path_reports, crystal=crystal, cluster_distance=3.5
    )

    assert paths == {
        r.path_id: tmp_path / "pathways" / f"run_path{r.path_id:02d}.cif"
        for r in result.path_reports
    }
    block = gemmi.cif.read_file(str(paths[1])).sole_block()
    assert block.find_value("_epath.path_id") == "1"
    assert block.find_value("_epath.cluster_distance") == "3.5"
    remarks = [
        gemmi.cif.as_string(r[1]) for r in block.find("_epath_remark.", ["id", "text"])
    ]
    first = result.path_reports[0]
    assert "Pathway 1." in remarks
    assert f"Cofactors in range: {first.nearby_cofactor_summary}." in remarks
    assert "Atoms connected within 3.5 A." in remarks
    assert block.find_value("_epath.nearby_cofactors") == (
        "/".join(cofactor.label for cofactor in first.nearby_cofactors) or "none"
    )
    assert len(block.find("_atom_site.", ["id"])) == result.path_reports[0].node_count
    assert (
        write_path_subset_cifs(
            tmp_path / "run", (), crystal=crystal, cluster_distance=3.5
        )
        == {}
    )


def test_path_subset_cifs_replace_those_of_an_earlier_run(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    crystal = result.bdamage_pipeline_result.crystal
    directory = tmp_path / "pathways"
    directory.mkdir()
    earlier = ["run_path01.cif", "run_path07.cif", "run_path123.cif"]
    unrelated = ["run_path01.cif.bak", "run_path01_notes.cif", "other_path01.cif"]
    for name in earlier + unrelated:
        (directory / name).write_text("old")

    paths = write_path_subset_cifs(
        tmp_path / "run", result.path_reports[:1], crystal=crystal, cluster_distance=4.0
    )

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        ["run_path01.cif", *unrelated]
    )
    assert paths[1].read_text() != "old"

    # A run that reports no pathway leaves none behind.
    write_path_subset_cifs(tmp_path / "run", (), crystal=crystal, cluster_distance=4.0)
    assert sorted(path.name for path in directory.iterdir()) == sorted(unrelated)

    # Nor an empty directory, once nothing else is in it.
    for name in unrelated:
        (directory / name).unlink()
    write_path_subset_cifs(
        tmp_path / "run", result.path_reports[:1], crystal=crystal, cluster_distance=4.0
    )
    write_path_subset_cifs(tmp_path / "run", (), crystal=crystal, cluster_distance=4.0)
    assert not directory.exists()


def test_path_subset_cifs_write_atoms_in_input_order(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    # Reversed, the search order splits every residue of more than one atom.
    report = result.path_reports[0]
    atoms = report.cluster.atoms
    scrambled = replace(report, cluster=replace(report.cluster, atoms=atoms[::-1]))
    paths = write_path_subset_cifs(
        tmp_path / "run",
        (scrambled,),
        crystal=result.bdamage_pipeline_result.crystal,
        cluster_distance=4.0,
    )

    block = gemmi.cif.read_file(str(paths[report.path_id])).sole_block()
    rows = [
        tuple(row)
        for row in block.find(
            "_atom_site.", ["auth_asym_id", "auth_seq_id", "label_atom_id"]
        )
    ]
    in_order = sorted(atoms, key=lambda atom: atom.record.source_atom_index)
    assert rows == [
        (a.record.chain_id, str(a.record.residue_number), a.record.atom_name)
        for a in in_order
    ]
    asym_ids = list(block.find_values("_struct_asym.id"))
    assert len(asym_ids) == len(set(asym_ids))


def test_coordinate_files_carry_the_epath_version(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    files = write_epath_outputs(
        result, EpathOutputPipelineOptions(output_prefix=tmp_path / "run")
    )
    paths = [files.selected_cif, files.model_cif, *files.path_cifs.values()]
    for path in paths:
        assert path is not None
        block = gemmi.cif.read_file(str(path)).sole_block()
        assert block.find_value("_epath.version") == __version__
    for path in files.path_cifs.values():
        block = gemmi.cif.read_file(str(path)).sole_block()
        assert block.find_value("_epath.cluster_distance") == "4"


def test_coordinate_files_carry_the_input_cell_and_space_group(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    files = write_epath_outputs(
        result, EpathOutputPipelineOptions(output_prefix=tmp_path / "run")
    )
    paths = [files.selected_cif, files.model_cif, *files.path_cifs.values()]
    assert len(paths) == 2 + len(result.path_reports)
    for path in paths:
        assert path is not None
        structure = gemmi.read_structure(str(path))
        cell = structure.cell
        assert (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma) == approx(
            SYNTHETIC_CELL
        )
        assert structure.spacegroup_hm == "P 1 21 1"
        assert structure.find_spacegroup().number == 4
        block = gemmi.cif.read_file(str(path)).sole_block()
        assert block.find_value("_cell.Z_PDB") == "?"


@pytest.mark.parametrize(
    ("cell", "written", "extended"),
    [
        ((80.0, 80.0, 60.0, 90.0, 90.0, 120.0), "H 3", "R 3:H"),
        ((50.0, 50.0, 50.0, 80.0, 80.0, 80.0), "R 3", "R 3:R"),
    ],
)
def test_coordinate_files_name_the_setting_of_an_r_centred_group(
    tmp_path: Path,
    cell: tuple[float, float, float, float, float, float],
    written: str,
    extended: str,
) -> None:
    path = write_atom_subset_cif(
        tmp_path / "r3.cif", (make_scored_atom(),), crystal=make_crystal(cell, "R 3")
    )
    block = gemmi.cif.read_file(str(path)).sole_block()
    assert gemmi.cif.as_string(block.find_value("_symmetry.space_group_name_H-M")) == (
        written
    )
    assert block.find_value("_symmetry.Int_Tables_number") == "146"
    space_group = gemmi.read_structure(str(path)).find_spacegroup()
    assert space_group is not None and space_group.xhm() == extended


def test_z_score_plot_is_written_as_png(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    path = write_z_score_falloff(tmp_path / "plots" / "z.png", result.selection_result)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_summary_report_lists_results_and_files(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    # Pathway 1 has no file, as when a chain filter leaves it without atoms.
    files = EpathPipelineOutputFiles(
        combined_csv=tmp_path / "x_epath.csv",
        path_cifs={2: tmp_path / "x_path02.cif"},
    )
    path = write_summary_txt(tmp_path / "summary.txt", result, output_files=files)
    text = path.read_text()

    assert f"\nE-path version: {__version__}\n" in text
    assert "Selected atoms: 23 / 77" in text
    assert "Cluster distance: 4 A" in text
    assert (
        "Atoms removed before scoring: 1 hydrogen, 1 with invalid occupancy, "
        "1 without a usable B-factor, 1 alternate conformer"
    ) in text
    assert "Window size used: 11 atoms" in text
    assert "Space group: P 1 21 1" in text
    assert "Protein Bnet sites: 4" in text
    assert "Cofactor sites: FMN (Fmn201 chain A)" in text
    assert f"Path count: {len(result.path_reports)}" in text
    assert "\nPath 1\n" in text and result.path_reports[0].residue_path in text
    assert "  Cofactors in range: " in text
    assert "Combined atom CSV: " in text and "Pathway mmCIF 2: " in text
    assert "Cluster CSV" not in text and "Pathway mmCIF 1" not in text
    # No coordinate file of the whole selection is listed, so no chains are.
    assert "Chains in coordinate files" not in text
    assert "RABDAM Bnet results\n-------------------\n" in text
    assert output_file_entries(files) == (
        ("Combined atom CSV", files.combined_csv),
        ("Pathway mmCIF 2", files.path_cifs[2]),
    )


def test_removed_atoms_line_lists_only_non_zero_counts() -> None:
    assert removed_atoms_line(RemovedAtomCounts(1204, 0, 3, 0, 87)) == (
        "Atoms removed before scoring: 1204 hydrogens, 3 with invalid occupancy, "
        "87 alternate conformers"
    )
    assert removed_atoms_line(RemovedAtomCounts(0, 2, 0, 5, 0)) == (
        "Atoms removed before scoring: 2 with non-finite coordinates, "
        "5 without a usable B-factor"
    )
    assert removed_atoms_line(RemovedAtomCounts(0, 0, 0, 0, 0)) == (
        "Atoms removed before scoring: none"
    )


def test_write_epath_outputs_honours_every_toggle(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    assert (
        write_epath_outputs(result, EpathOutputPipelineOptions())
        == EpathPipelineOutputFiles()
    )

    everything = write_epath_outputs(
        result, EpathOutputPipelineOptions(output_prefix=tmp_path / "all" / "run")
    )
    for label, path in output_file_entries(everything):
        assert path.is_file(), label
    assert (
        everything.summary_report is not None
        and "Files written" in everything.summary_report.read_text()
    )
    assert len(everything.path_cifs) == len(result.path_reports)

    minimal = write_epath_outputs(
        result,
        EpathOutputPipelineOptions(
            output_prefix=tmp_path / "min" / "run",
            write_combined_csv=False,
            write_structure_files=False,
            write_z_score_plot=False,
            write_reports=False,
        ),
    )
    assert minimal == EpathPipelineOutputFiles()
    assert not (tmp_path / "min").exists()


def test_display_chains_only_trim_the_coordinate_files(
    tmp_path: Path, result: EpathPipelineResult
) -> None:
    assert result.analyzed_chains == ("A", "B")
    everything = write_epath_outputs(
        result, EpathOutputPipelineOptions(output_prefix=tmp_path / "all" / "run")
    )
    filtered = write_epath_outputs(
        result,
        EpathOutputPipelineOptions(
            output_prefix=tmp_path / "b" / "run", display_chains=("B",)
        ),
    )

    def chains_in(path: Path) -> set[str]:
        block = gemmi.cif.read_file(str(path)).sole_block()
        return {row[0] for row in block.find("_atom_site.", ["auth_asym_id"])}

    assert (
        everything.selected_cif is not None
        and filtered.selected_cif is not None
        and filtered.model_cif is not None
    )
    assert chains_in(everything.selected_cif) == {"A", "B"}
    assert chains_in(filtered.selected_cif) == {"B"}
    assert chains_in(filtered.model_cif) == {"B"}
    assert "Coordinates limited to chains B." in filtered.selected_cif.read_text()
    for path_cif in filtered.path_cifs.values():
        assert chains_in(path_cif) == {"B"}
    # A pathway keeps its number when an earlier one has no atoms in chain B.
    assert list(filtered.path_cifs) == [
        report.path_id
        for report in result.path_reports
        if any(atom.record.chain_id == "B" for atom in report.cluster.atoms)
    ]

    # Analysis outputs are untouched by the display filter.
    assert everything.combined_csv is not None and filtered.combined_csv is not None
    assert everything.combined_csv.read_text() == filtered.combined_csv.read_text()
    assert filtered.summary_report is not None
    summary = filtered.summary_report.read_text()
    assert "Chains analyzed: A, B" in summary
    assert "Chains in coordinate files: B (display filter only)" in summary
    assert everything.summary_report is not None
    assert "Chains in coordinate files: all" in everything.summary_report.read_text()
