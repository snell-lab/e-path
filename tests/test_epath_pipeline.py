"""The E-path analysis pipeline, from a BDamage result to its run record."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from support import approx, bdamage_result_from_atoms, make_scored_atom

from epath.analysis.pathways import PathAnalysisOptions
from epath.analysis.selection import EpathScoredAtom
from epath.outputs.files import EpathOutputPipelineOptions
from epath.pipeline import (
    EpathClusterPipelineOptions,
    EpathPipelineError,
    EpathPipelineOptions,
    EpathSelectionPipelineOptions,
    run_epath_pipeline,
)
from rabdam.pipeline import run_bdamage_pipeline
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input

EPATH_STAGES = [
    "Selecting E-path atoms",
    "Clustering selected atoms",
    "Analyzing pathways",
    "Calculating Bnet and Bnet percentile",
    "Writing E-path outputs",
]


def _grid_atoms() -> tuple[EpathScoredAtom, ...]:
    """Ten atoms: a compact group of six, a pair, and two isolated atoms.

    BDamage rises with index, so a percentile cutoff selects a prefix.
    """
    coordinates = [
        (0, 0, 0),
        (2, 0, 0),
        (4, 0, 0),
        (0, 2, 0),
        (2, 2, 0),
        (4, 2, 0),
        (30, 0, 0),
        (33, 0, 0),
        (60, 0, 0),
        (90, 0, 0),
    ]
    return tuple(
        make_scored_atom(
            source_atom_index=i,
            bdamage_atom_index=i + 1,
            atom_serial=i + 1,
            residue_number=i + 1,
            residue_name="TRP" if i % 2 else "ALA",
            x=float(x),
            y=float(y),
            z=float(z),
            bdamage=0.5 + 0.1 * i,
        )
        for i, (x, y, z) in enumerate(coordinates)
    )


def test_components_are_found_filtered_and_recorded() -> None:
    options = EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(percentile=90.0),
        clustering=EpathClusterPipelineOptions(
            distance_cutoff=3.0, minimum_cluster_size=3
        ),
    )

    result, files = run_epath_pipeline(
        bdamage_result_from_atoms(_grid_atoms()), options=options
    )

    # The 90% cutoff drops the highest-BDamage atom, the isolated one at x=90.
    assert len(result.selection_result.selected_atoms) == 9
    assert result.component_statistics.total_count == 3
    assert result.component_statistics.largest_atom_count == 6
    assert [c.atom_count for c in result.clusters] == [6]
    assert len(result.path_reports) == 1
    report = result.path_reports[0]
    assert report.start_endpoint.kind in {"surface", "buried"}
    assert report.end_endpoint.kind in {"surface", "buried"}
    assert result.cofactor_sites == ()
    # Bnet needs Asp/Glu carboxyl oxygens, and this model has none.
    assert result.bnet_pipeline_result.bnet_result is None
    assert result.analyzed_chains == ("A",)
    record = result.run_record
    assert (record.structure, record.input_path) == ("synthetic", "synthetic.pdb")
    assert (
        record.total_atoms,
        record.atoms_selected,
        record.total_components,
        record.clusters_reported,
    ) == (10, 9, 3, 1)
    assert (
        record.largest_component_atoms == 6
        and record.largest_component_reported is True
    )
    assert record.bnet is None and record.bnet_percentile is None
    assert record.cutoff_percent == 90.0 and record.minimum_cluster_size == 3
    assert files.combined_csv is None


def test_largest_component_is_recorded_even_when_nothing_is_reported() -> None:
    options = EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(percentile=90.0),
        clustering=EpathClusterPipelineOptions(
            distance_cutoff=3.0, minimum_cluster_size=8
        ),
    )
    result, _ = run_epath_pipeline(
        bdamage_result_from_atoms(_grid_atoms()), options=options
    )

    assert result.clusters == () and result.path_reports == ()
    assert result.run_record.largest_component_atoms == 6
    assert result.run_record.largest_component_reported is False
    largest = result.component_statistics.largest_metrics
    assert largest is not None
    assert result.run_record.largest_component_span == approx(largest.span)


def test_an_empty_selection_still_writes_every_output(tmp_path: Path) -> None:
    options = EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(percentile=0.0),
        output=EpathOutputPipelineOptions(output_prefix=tmp_path / "run"),
    )
    result, files = run_epath_pipeline(
        bdamage_result_from_atoms(_grid_atoms()), options=options
    )

    assert result.selection_result.selected_atoms == ()
    assert result.component_statistics.total_count == 0
    assert files.z_score_falloff is not None and files.z_score_falloff.is_file()
    assert files.path_cifs == {}
    # Nothing is reported as a missing number: what has no value is left out.
    assert files.summary_report is not None and files.run_csv is not None
    summary = files.summary_report.read_text()
    assert "No atoms selected." in summary and "BDamage threshold: 0.5" in summary
    assert "nan" not in summary and "nan" not in files.run_csv.read_text()


def test_options_are_checked_before_any_stage_runs() -> None:
    bdamage = bdamage_result_from_atoms(_grid_atoms())
    stages: list[str] = []

    def stage_runner(label: str, callback: Callable[[], Any]) -> Any:
        stages.append(label)
        return callback()

    with pytest.raises(EpathPipelineError, match="minimum_cluster_size"):
        run_epath_pipeline(
            bdamage,
            options=EpathPipelineOptions(
                clustering=EpathClusterPipelineOptions(minimum_cluster_size=0)
            ),
            stage_runner=stage_runner,
        )
    # Chain A is in the model and chain Z is not.
    with pytest.raises(
        EpathPipelineError, match=r"not in the model: Z\. The model has chains A\."
    ):
        run_epath_pipeline(
            bdamage,
            options=EpathPipelineOptions(
                output=EpathOutputPipelineOptions(display_chains=("A", "Z"))
            ),
            stage_runner=stage_runner,
        )
    assert stages == []


def test_pipeline_from_a_bdamage_result(synthetic_pdb: Path) -> None:
    bdamage = run_bdamage_pipeline(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )
    stages: list[str] = []

    def stage_runner(label: str, callback: Callable[[], Any]) -> Any:
        stages.append(label)
        return callback()

    options = EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(percentile=30.0),
        clustering=EpathClusterPipelineOptions(minimum_cluster_size=3),
        paths=PathAnalysisOptions(cofactor_proximity_distance=6.0),
    )
    result, files = run_epath_pipeline(
        bdamage, options=options, stage_runner=stage_runner
    )

    assert stages == EPATH_STAGES
    assert files.combined_csv is None
    assert result.analyzed_chains == ("A", "B")
    assert result.selection_result.method == "percentile"
    assert result.bdamage_pipeline_result is bdamage
    assert [site.label for site in result.cofactor_sites] == ["FMN"]
    assert result.bnet_pipeline_result.bnet_result is not None
    assert result.run_record.bnet == approx(
        result.bnet_pipeline_result.bnet_result.bnet
    )
    assert result.run_record.structure == "synthetic"
    assert result.run_record.input_path == str(synthetic_pdb.resolve())
    assert all(report.node_count >= 3 for report in result.path_reports)
