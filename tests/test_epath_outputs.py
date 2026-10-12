"""Output files: CSV tables, coordinate subsets, the plot, and the summary."""

from __future__ import annotations

import csv
from pathlib import Path

import gemmi
import pytest
from support import approx, make_epath_atom, make_scored_atom

from epath.outputs.files import EpathOutputWorkflowOptions, EpathWorkflowOutputFiles
from epath.outputs.plots import write_z_score_falloff
from epath.outputs.structures import (
    EpathStructureOutputError,
    write_atom_subset_cif,
    write_model_cif,
    write_path_subset_cifs,
)
from epath.outputs.summary import output_file_entries, write_summary_txt
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
from epath.workflow import (
    EpathClusterWorkflowOptions,
    EpathSelectionWorkflowOptions,
    EpathWorkflowOptions,
    EpathWorkflowResult,
    run_epath_workflow,
)
from rabdam.workflow import calculate_bdamage_for_structure_data
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input


@pytest.fixture(scope="module")
def result(synthetic_pdb: Path) -> EpathWorkflowResult:
    """Return an analysed synthetic crystal with reportable pathways."""
    bdamage = calculate_bdamage_for_structure_data(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )
    options = EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(percentile=30.0),
        clustering=EpathClusterWorkflowOptions(minimum_cluster_size=3),
    )
    workflow_result, files = run_epath_workflow(bdamage, options=options)
    assert files == EpathWorkflowOutputFiles()
    assert len(workflow_result.path_reports) >= 1
    return workflow_result


def _rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into one dict per row."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_combined_atom_csv(tmp_path: Path, result: EpathWorkflowResult) -> None:
    path = write_combined_atom_csv(
        tmp_path / "nested" / "x_epath.csv",
        result.selection_result,
        clusters=result.clusters,
    )
    rows = _rows(path)

    assert list(rows[0]) == COMBINED_ATOM_FIELDNAMES
    assert len(rows) == len(result.selection_result.scored_atoms) == 77
    selected = {
        a.atom.source_atom_index for a in result.selection_result.selected_atoms
    }
    clustered = {
        a.atom.source_atom_index: c.cluster_id for c in result.clusters for a in c.atoms
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


def test_cluster_step_and_run_csvs(tmp_path: Path, result: EpathWorkflowResult) -> None:
    clusters = _rows(
        write_cluster_csv(tmp_path / "x_clusters.csv", result.path_reports)
    )
    assert list(clusters[0]) == CLUSTER_FIELDNAMES
    assert len(clusters) == len(result.path_reports)
    first = clusters[0]
    report = result.path_reports[0]
    assert first["residue_path"] == report.residue_path
    for row, path_report in zip(clusters, result.path_reports, strict=True):
        anchor = path_report.anchor
        if anchor is None:
            assert (row["anchor_cofactor"], row["anchor_distance"]) == ("", "")
            assert row["anchor_atom"] == ""
        else:
            assert row["anchor_cofactor"] == anchor.label
            assert float(row["anchor_distance"]) == approx(anchor.distance)
            assert row["anchor_atom"] == anchor.atom_token
    assert int(first["atom_count"]) == report.node_count
    assert float(first["mst_total_length"]) == approx(report.total_length)
    assert first["chains"] == "/".join(report.chains)
    fractions = [
        float(first[f"{name}_fraction"])
        for name in ("aromatic", "sulfur", "charged", "polar", "hydrophobic")
    ]
    assert sum(fractions) <= 1.0 + 1e-9

    steps = _rows(write_step_csv(tmp_path / "x_steps.csv", result.path_reports))
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

    runs = _rows(write_run_csv(tmp_path / "x_run.csv", (result.run_record,)))
    assert list(runs[0]) == RUN_FIELDNAMES and len(runs) == 1
    assert runs[0]["structure"] == "synthetic"
    assert runs[0]["total_atoms"] == "77" and runs[0]["atoms_selected"] == "23"
    assert runs[0]["largest_component_reported"] == "true"
    assert float(runs[0]["bnet"]) > 0 and 0 <= float(runs[0]["bnet_percentile"]) <= 100
    assert runs[0]["clusters_reported"] == str(len(result.path_reports))


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
        make_epath_atom(
            source_atom_index=1,
            atom_serial=9,
            record_type="HETATM",
            chain_id="B",
            residue_name="MSE",
            residue_number=3,
            atom_name="SE",
            element="SE",
            bdamage=1.5,
        ),
    )
    path = write_atom_subset_cif(
        tmp_path / "subset.cif", atoms, remarks=("Custom remark.",)
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

    with pytest.raises(EpathStructureOutputError, match="finite"):
        write_atom_subset_cif(
            tmp_path / "bad.cif", (make_epath_atom(bdamage=float("nan")),)
        )


def test_atom_subset_cif_keeps_fields_that_do_not_fit_pdb_columns(
    tmp_path: Path,
) -> None:
    atoms = (
        make_epath_atom(chain_id="AB", residue_number=12345, atom_name="CA"),
        make_epath_atom(chain_id="AB", residue_number=12346, atom_name="CA"),
        make_epath_atom(chain_id="C", residue_number=-1200, atom_name="CA"),
    )
    path = write_atom_subset_cif(tmp_path / "wide.cif", atoms)
    model = gemmi.read_structure(str(path))[0]
    assert [chain.name for chain in model] == ["AB", "C"]
    assert [res.seqid.num for res in model["AB"]] == [12345, 12346]
    assert model["C"][0].seqid.num == -1200

    empty = write_atom_subset_cif(tmp_path / "empty.cif", ())
    block = gemmi.cif.read_file(str(empty)).sole_block()
    assert len(block.find("_atom_site.", ["id"])) == 0


def test_model_cif_marks_unscored_atoms_with_zero_bdamage(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    cleaned = result.cleaned_atoms
    assert cleaned is not None
    scores = {
        a.atom.source_atom_index: a.atom.bdamage
        for a in result.selection_result.scored_atoms
    }

    path = write_model_cif(
        tmp_path / "model.cif",
        cleaned,
        bdamage_by_source_index=scores,
        bdamage_threshold=0.7734,
    )
    block = gemmi.cif.read_file(str(path)).sole_block()
    assert block.find_value("_epath.atom_count") == str(len(cleaned))
    assert block.find_value("_epath.scored_atom_count") == str(len(scores))
    assert block.find_value("_epath.bdamage_threshold") == "0.7734"
    remarks = [
        gemmi.cif.as_string(r[1]) for r in block.find("_epath_remark.", ["id", "text"])
    ]
    assert f"{len(scores)} of {len(cleaned)} atoms were scored." in remarks
    assert "Selected atoms satisfy 0.00 < B_iso_or_equiv <= 0.7734." in remarks

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


def test_path_subset_cifs_are_zero_padded(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    paths = write_path_subset_cifs(tmp_path / "run", result.path_reports)

    assert [p.name for p in paths] == [
        f"run_path{r.path_id:02d}.cif" for r in result.path_reports
    ]
    block = gemmi.cif.read_file(str(paths[0])).sole_block()
    assert block.find_value("_epath.path_id") == "1"
    remarks = [
        gemmi.cif.as_string(r[1]) for r in block.find("_epath_remark.", ["id", "text"])
    ]
    first = result.path_reports[0]
    assert "Pathway 1." in remarks
    assert f"Nearest cofactor: {first.anchor_summary}." in remarks
    assert block.find_value("_epath.anchor_cofactor") == (
        first.anchor.label if first.anchor is not None else "none"
    )
    assert len(block.find("_atom_site.", ["id"])) == result.path_reports[0].node_count
    assert write_path_subset_cifs(tmp_path / "run", ()) == ()


def test_z_score_plot_is_written_as_png(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    path = write_z_score_falloff(
        tmp_path / "plots" / "z.png", result.selection_result, dpi=50
    )
    assert path is not None and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    from epath.analysis.selection import select_atoms_by_percentile

    empty = select_atoms_by_percentile((), percentile=2.0)
    assert write_z_score_falloff(tmp_path / "none.png", empty) is None

    # No threshold: nothing to put in a legend, but still a plot.
    unthresholded = select_atoms_by_percentile(
        [a.atom for a in result.selection_result.scored_atoms], percentile=100.0
    )
    assert (
        write_z_score_falloff(tmp_path / "full.png", unthresholded, dpi=50) is not None
    )


def test_summary_report_lists_results_and_files(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    files = EpathWorkflowOutputFiles(
        combined_csv=tmp_path / "x_epath.csv", path_cifs=(tmp_path / "x_path01.cif",)
    )
    path = write_summary_txt(tmp_path / "summary.txt", result, output_files=files)
    text = path.read_text()

    assert "Selected atoms: 23 / 77" in text
    assert "Window size used: 11 atoms" in text
    assert "Space group: P 1 21 1" in text
    assert "Protein Bnet sites: 4" in text
    assert "Cofactor sites: FMN (Fmn201 chain A)" in text
    assert f"Path count: {len(result.path_reports)}" in text
    assert "Path 1 " in text and result.path_reports[0].residue_path in text
    assert "  Nearest cofactor: " in text
    assert "Combined atom CSV: " in text and "Pathway mmCIF 1: " in text
    assert "Cluster CSV" not in text
    assert output_file_entries(files) == (
        ("Combined atom CSV", files.combined_csv),
        ("Pathway mmCIF 1", files.path_cifs[0]),
    )


def test_write_epath_outputs_honours_every_toggle(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    assert (
        write_epath_outputs(result, EpathOutputWorkflowOptions())
        == EpathWorkflowOutputFiles()
    )

    everything = write_epath_outputs(
        result, EpathOutputWorkflowOptions(output_prefix=tmp_path / "all" / "run")
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
        EpathOutputWorkflowOptions(
            output_prefix=tmp_path / "min" / "run",
            write_combined_csv=False,
            write_structure_files=False,
            write_z_score_plot=False,
            write_cluster_csv=False,
            write_step_csv=False,
            write_run_csv=False,
            write_summary_report=False,
        ),
    )
    assert minimal == EpathWorkflowOutputFiles()
    assert not (tmp_path / "min").exists()


def test_display_chains_only_trim_the_coordinate_files(
    tmp_path: Path, result: EpathWorkflowResult
) -> None:
    from epath.outputs.structures import EpathStructureOutputError

    assert result.analyzed_chains == ("A", "B")
    everything = write_epath_outputs(
        result, EpathOutputWorkflowOptions(output_prefix=tmp_path / "all" / "run")
    )
    filtered = write_epath_outputs(
        result,
        EpathOutputWorkflowOptions(
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
    for path_cif in filtered.path_cifs:
        assert chains_in(path_cif) == {"B"}
    assert len(filtered.path_cifs) == sum(
        any(atom.chain_id == "B" for atom in report.cluster.atoms)
        for report in result.path_reports
    )

    # Analysis outputs are untouched by the display filter.
    assert everything.combined_csv is not None and filtered.combined_csv is not None
    assert everything.combined_csv.read_text() == filtered.combined_csv.read_text()
    assert filtered.summary_report is not None
    summary = filtered.summary_report.read_text()
    assert "Chains analyzed: A, B" in summary
    assert "Chains in coordinate files: B (display filter only)" in summary

    with pytest.raises(EpathStructureOutputError, match="not in the model: Z"):
        write_epath_outputs(
            result,
            EpathOutputWorkflowOptions(
                output_prefix=tmp_path / "z" / "run", display_chains=("Z",)
            ),
        )
