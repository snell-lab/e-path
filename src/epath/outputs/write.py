"""Write the output files of an E-path run.

`write_epath_outputs` decides which files are enabled and hands each one to
the module that writes it: `tables` for the CSVs, `structures` for the mmCIF
coordinate files, `plots` for the Z-score plot and `summary` for the text
report. `epath.pipeline` calls it as its last stage.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from epath.outputs.files import (
    EpathOutputPipelineOptions,
    EpathPipelineOutputFiles,
)
from epath.outputs.plots import write_z_score_falloff
from epath.outputs.structures import write_structure_outputs
from epath.outputs.summary import write_summary_txt
from epath.outputs.tables import (
    write_cluster_csv,
    write_combined_atom_csv,
    write_run_csv,
    write_step_csv,
)

if TYPE_CHECKING:
    from epath.pipeline import EpathPipelineResult


def write_epath_outputs(
    result: EpathPipelineResult,
    options: EpathOutputPipelineOptions,
) -> EpathPipelineOutputFiles:
    """Write the enabled output files for a result and return their paths.

    Every file is named after the output prefix, as `<prefix>_<suffix>`, and
    the writers create its directory when it is missing. A file that is
    switched off is left out of the returned record. Without an output prefix
    nothing is written and the record is empty.

    Nothing is checked here. `epath.pipeline` has already rejected a
    `display_chains` entry that is not in the model.
    """
    if options.output_prefix is None:
        return EpathPipelineOutputFiles()

    prefix = options.output_prefix
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
        if options.write_reports
        else None
    )
    step_csv = (
        write_step_csv(_output_path(prefix, "steps.csv"), path_reports)
        if options.write_reports
        else None
    )
    run_csv = (
        write_run_csv(_output_path(prefix, "run.csv"), (result.run_record,))
        if options.write_reports
        else None
    )

    selected_cif: Path | None = None
    model_cif: Path | None = None
    path_cifs: dict[int, Path] = {}
    if options.write_structure_files:
        bdamage_pipeline_result = result.bdamage_pipeline_result
        selected_cif, model_cif, path_cifs = write_structure_outputs(
            prefix,
            selection_result.selected_atoms,
            path_reports=path_reports,
            cleaned_atoms=bdamage_pipeline_result.prepared_structure.cleaned_atoms,
            bdamage_by_source_index={
                atom_result.source_atom_index: atom_result.bdamage
                for atom_result in bdamage_pipeline_result.bdamage_results
            },
            bdamage_threshold=selection_result.bdamage_threshold,
            display_chains=options.display_chains,
        )

    z_score_falloff = (
        write_z_score_falloff(_output_path(prefix, "zfalloff.png"), selection_result)
        if options.write_z_score_plot
        else None
    )
    # The report lists every file of the run, itself included, so its path is
    # set before the record is built and it is written last.
    summary_report = (
        _output_path(prefix, "summary.txt") if options.write_reports else None
    )
    output_files = EpathPipelineOutputFiles(
        combined_csv=combined_csv,
        cluster_csv=cluster_csv,
        step_csv=step_csv,
        run_csv=run_csv,
        selected_cif=selected_cif,
        model_cif=model_cif,
        path_cifs=path_cifs,
        z_score_falloff=z_score_falloff,
        summary_report=summary_report,
    )

    if summary_report is not None:
        write_summary_txt(
            summary_report,
            result,
            output_files=output_files,
            display_chains=options.display_chains,
        )

    return output_files


def _output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{suffix}")
