"""Coordinate the output files written for an E-path run."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from epath.outputs.files import (
    EpathOutputWorkflowOptions,
    EpathWorkflowOutputFiles,
)
from epath.outputs.plots import plot_z_falloff
from epath.outputs.structures import write_structure_outputs
from epath.outputs.summary import write_summary_txt
from epath.outputs.tables import (
    write_cluster_csv,
    write_combined_atom_csv,
    write_run_csv,
    write_step_csv,
)

if TYPE_CHECKING:
    from epath.workflow import EpathWorkflowResult


def write_epath_outputs(
    result: EpathWorkflowResult,
    options: EpathOutputWorkflowOptions,
) -> EpathWorkflowOutputFiles:
    """Write enabled E-path artifacts for a result and return their paths."""

    if options.output_prefix is None:
        return EpathWorkflowOutputFiles()

    prefix = Path(options.output_prefix)
    selection_result = result.selection_result
    path_reports = result.path_reports
    combined_csv = (
        write_combined_atom_csv(
            _output_path(prefix, "epath.csv"),
            selection_result,
            clusters=result.clusters,
        )
        if options.write_combined_csv
        else None
    )
    cluster_csv = (
        write_cluster_csv(_output_path(prefix, "clusters.csv"), path_reports)
        if options.write_cluster_csv
        else None
    )
    step_csv = (
        write_step_csv(_output_path(prefix, "steps.csv"), path_reports)
        if options.write_step_csv
        else None
    )
    run_csv = (
        write_run_csv(_output_path(prefix, "run.csv"), (result.run_record,))
        if options.write_run_csv
        else None
    )

    selected_pdb: Path | None = None
    model_pdb: Path | None = None
    path_pdbs: tuple[Path, ...] = ()
    if options.write_structure_files:
        structure_files = write_structure_outputs(
            prefix,
            selection_result.selected_atoms,
            path_reports=path_reports,
            cleaned_atoms=result.cleaned_atoms,
            bdamage_by_source_index={
                scored_atom.atom.source_atom_index: scored_atom.atom.bdamage
                for scored_atom in selection_result.scored_atoms
            },
            bdamage_threshold=selection_result.bdamage_threshold,
        )
        selected_pdb = structure_files.selected_pdb
        model_pdb = structure_files.model_pdb
        path_pdbs = structure_files.path_pdbs

    z_score_falloff = (
        plot_z_falloff(prefix, selection_result) if options.write_z_score_plot else None
    )
    summary_report = (
        _output_path(prefix, "summary.txt") if options.write_summary_report else None
    )
    output_files = EpathWorkflowOutputFiles(
        combined_csv=combined_csv,
        cluster_csv=cluster_csv,
        step_csv=step_csv,
        run_csv=run_csv,
        selected_pdb=selected_pdb,
        model_pdb=model_pdb,
        path_pdbs=path_pdbs,
        z_score_falloff=z_score_falloff,
        summary_report=summary_report,
    )

    if summary_report is not None:
        write_summary_txt(summary_report, result, output_files=output_files)

    return output_files


def _output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{suffix}")
