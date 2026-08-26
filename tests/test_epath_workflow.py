"""The E-path analysis workflow, with and without a BDamage result behind it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from support import approx, make_epath_atom

from epath.analysis.atoms import EpathAtom
from epath.analysis.pathways import PathAnalysisOptions
from epath.workflow import (
    EpathClusterWorkflowOptions,
    EpathSelectionWorkflowOptions,
    EpathWorkflowError,
    EpathWorkflowOptions,
    analyze_epath_atoms,
    run_epath_workflow,
)
from rabdam.workflow import calculate_bdamage_for_structure_data
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input

EPATH_STAGES = [
    "Selecting E-path atoms",
    "Clustering selected atoms",
    "Analyzing pathways",
    "Calculating Bnet and Bnet percentile",
    "Writing E-path outputs",
]


def _grid_atoms() -> tuple[EpathAtom, ...]:
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
        make_epath_atom(
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


def test_analysis_without_a_bdamage_result_degrades_gracefully() -> None:
    options = EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(percentile=90.0),
        clustering=EpathClusterWorkflowOptions(
            distance_cutoff=3.0, minimum_cluster_size=3
        ),
    )

    result, files = analyze_epath_atoms(
        _grid_atoms(), structure_path="models/demo.cif", options=options
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
    assert (
        result.bnet_workflow_result is None
        and result.crystal is None
        and result.cleaned_atoms is None
    )
    assert result.analyzed_chains == ("A",)
    assert result.structure_path == Path("models/demo.cif")
    record = result.run_record
    # input_path is stored in the platform's native form (backslashes on Windows).
    assert (record.structure, record.input_path) == (
        "demo",
        str(Path("models/demo.cif")),
    )
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
    options = EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(percentile=90.0),
        clustering=EpathClusterWorkflowOptions(
            distance_cutoff=3.0, minimum_cluster_size=8
        ),
    )
    result, _ = analyze_epath_atoms(_grid_atoms(), options=options)

    assert result.clusters == () and result.path_reports == ()
    assert result.run_record.largest_component_atoms == 6
    assert result.run_record.largest_component_reported is False
    largest = result.component_statistics.largest_metrics
    assert largest is not None
    assert result.run_record.largest_component_span == approx(largest.span)
    assert result.structure_path is None and result.run_record.structure == ""


def test_workflow_validation() -> None:
    with pytest.raises(EpathWorkflowError, match="at least one atom"):
        analyze_epath_atoms(())
    with pytest.raises(EpathWorkflowError, match="minimum_cluster_size"):
        analyze_epath_atoms(
            _grid_atoms(),
            options=EpathWorkflowOptions(
                clustering=EpathClusterWorkflowOptions(minimum_cluster_size=0)
            ),
        )
    with pytest.raises(EpathWorkflowError, match="Unknown selection method"):
        analyze_epath_atoms(
            _grid_atoms(),
            options=EpathWorkflowOptions(
                selection=EpathSelectionWorkflowOptions(
                    method="median"  # type: ignore[arg-type]
                )
            ),
        )


def test_workflow_from_a_bdamage_result(synthetic_pdb: Path) -> None:
    bdamage = calculate_bdamage_for_structure_data(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )
    stages: list[str] = []

    def stage_runner(label: str, callback: Callable[[], Any]) -> Any:
        stages.append(label)
        return callback()

    options = EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(
            method="percentile_per_chain", percentile=30.0
        ),
        clustering=EpathClusterWorkflowOptions(minimum_cluster_size=3),
        paths=PathAnalysisOptions(cofactor_proximity_distance=6.0),
    )
    result, files = run_epath_workflow(
        bdamage, options=options, stage_runner=stage_runner
    )

    assert stages == EPATH_STAGES
    assert files.combined_csv is None
    assert result.analyzed_chains == ("A", "B")
    assert result.selection_result.method == "percentile_per_chain"
    assert [t.chain_id for t in result.selection_result.chain_thresholds] == ["A", "B"]
    assert result.bdamage_workflow_result is bdamage
    assert result.crystal is bdamage.crystal
    assert result.cleaned_atoms is not None and len(result.cleaned_atoms) == 84
    assert [site.label for site in result.cofactor_sites] == ["FMN"]
    assert (
        result.bnet_workflow_result is not None
        and result.bnet_workflow_result.bnet_result is not None
    )
    assert result.run_record.bnet == approx(
        result.bnet_workflow_result.bnet_result.bnet
    )
    assert result.run_record.structure == "synthetic"
    assert result.run_record.input_path == str(synthetic_pdb.resolve())
    assert all(report.node_count >= 3 for report in result.path_reports)
