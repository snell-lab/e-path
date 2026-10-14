"""Print the terminal messages and run summaries of the E-path command line.

The crystal, BDamage and Bnet sections and the list of files written come from
the builders in `epath.outputs.summary` that also write the summary report, so
the terminal and the file show the same lines for these. The selection and
pathway sections are built here, as a shorter version of the report's, with one
line per pathway and values rounded to fewer digits. Sections are printed
through `epath.cli.style`, which wraps long lines with a hanging indent and
highlights warnings and errors on a colour terminal.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, TextIO

from epath.analysis.endpoints import cofactor_site_summary
from epath.cli.style import print_lines, style_line, supports_color
from epath.outputs.summary import (
    bdamage_summary_lines,
    bnet_summary_lines,
    crystal_summary_lines,
    output_file_entries,
)

if TYPE_CHECKING:
    from epath.cli.run import EpathBatchCliResult, EpathCliResult
    from epath.pipeline import EpathPipelineResult

CITATIONS_MESSAGE = """Please cite:

RABDAM:
  Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
  J Appl Cryst 51, 552-559.

BDamage:
  Gerstel M, Deane CM & Garman EF (2015).
  J Synchrotron Radiat 22, 201-212.

Bnet / Bnet percentile:
  Shelley KL & Garman EF (2022).
  Nat Commun 13, 1314."""

MISSING_STRUCTURE_INPUT_MESSAGE = """epath: error: missing required argument: structure_input

Usage:
  epath STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]

Examples:
  epath 3i9v
  epath example.cif --percentile 2 --output-prefix output/example

Run 'epath --help' for all options."""

INDENT = "  "


def print_error(message: str, *, stream: TextIO) -> None:
    """Print an error message, highlighting the ``epath: error:`` prefix."""
    enabled = supports_color(stream)
    for line in message.splitlines():
        print(style_line(line, enabled=enabled), file=stream)


def print_citations(*, stream: TextIO) -> None:
    """Print citation information for E-path outputs."""
    print(CITATIONS_MESSAGE, file=stream)


def print_epath_summary(
    result: EpathCliResult,
    *,
    total_runtime_seconds: float,
    stream: TextIO,
) -> None:
    """Print the summary of one finished run.

    The crystal, the BDamage calculation and the Bnet results are the same at
    every percentile and are printed once, with crystal warnings in the
    crystal section. The selection, the pathways and the files written follow
    for each percentile, headed by the percentile when there are several.
    """
    runs = result.runs
    first_result = runs[0].pipeline_result
    bdamage_result = first_result.bdamage_pipeline_result
    several = len(runs) > 1

    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output prefix: {result.output_prefix}", file=stream)

    _print_section(
        "Crystal:",
        crystal_summary_lines(bdamage_result.crystal),
        stream=stream,
    )
    _print_section(
        "BDamage calculation:",
        bdamage_summary_lines(bdamage_result),
        stream=stream,
    )
    _print_section(
        "RABDAM Bnet results:",
        bnet_summary_lines(first_result.bnet_pipeline_result),
        stream=stream,
    )
    for run in runs:
        pipeline_result = run.pipeline_result
        heading = (
            f" at {pipeline_result.run_record.cutoff_percent:g}%" if several else ""
        )
        _print_section(
            f"E-path selection{heading}:",
            _selection_lines(pipeline_result),
            stream=stream,
        )
        _print_section(
            f"Pathways{heading}:", _pathway_lines(pipeline_result), stream=stream
        )
        _print_section(
            f"Outputs{heading}:",
            (
                f"{label}: {path}"
                for label, path in output_file_entries(run.output_files)
            ),
            stream=stream,
        )
    if result.percentile_csv is not None:
        _print_section(
            "Percentiles:",
            (f"Run CSV, one row per percentile: {result.percentile_csv}",),
            stream=stream,
        )

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


def print_epath_batch_summary(
    result: EpathBatchCliResult,
    *,
    stream: TextIO,
) -> None:
    """Print the summary of a batch run.

    It gives the number of inputs that succeeded and failed, where the outputs
    went, and the failed inputs by name, in the order they were given.
    """
    lines = [
        f"Successful: {len(result.results)}",
        f"Failed: {len(result.failed_inputs)}",
        f"Output directory: {result.output_dir}",
    ]
    if result.cohort_csv is not None:
        lines.append(f"Cohort CSV: {result.cohort_csv}")
    if result.failed_inputs:
        lines.append("Failed inputs:")
        lines.extend(INDENT + failed_input for failed_input in result.failed_inputs)
    lines.append(f"Total runtime: {result.elapsed_seconds:.1f}s")

    _print_section("Batch summary:", lines, stream=stream)


def _selection_lines(pipeline_result: EpathPipelineResult) -> tuple[str, ...]:
    selection_result = pipeline_result.selection_result
    return (
        "Chains: " + ", ".join(pipeline_result.analyzed_chains),
        f"Method: {selection_result.description}",
        (
            f"Selected atoms: {len(selection_result.selected_atoms)} "
            f"of {len(selection_result.scored_atoms)}"
        ),
        (
            f"BDamage mean: {selection_result.bdamage_mean:.4f}, "
            f"SD: {selection_result.bdamage_standard_deviation:.4f}"
        ),
    )


def _pathway_lines(pipeline_result: EpathPipelineResult) -> tuple[str, ...]:
    reports = pipeline_result.path_reports
    lines = [
        f"Connected components: {pipeline_result.component_statistics.total_count}",
        f"Count: {len(reports)}",
    ]
    if reports:
        anchored = sum(report.is_cofactor_anchored for report in reports)
        lines.append(f"Within range of a cofactor: {anchored}")
    lines.append(
        f"Cofactor sites: {cofactor_site_summary(pipeline_result.cofactor_sites)}"
    )
    lines.extend(
        f"Path {report.path_id} {report.endpoint_summary}: "
        f"{report.residue_count} residues, {report.node_count} nodes, "
        f"{report.total_length:.2f} A, nearest cofactor {report.anchor_summary}"
        for report in reports
    )
    return tuple(lines)


def _print_section(title: str, lines: Iterable[str], *, stream: TextIO) -> None:
    print(file=stream)
    print(title, file=stream)
    print_lines(lines, indent=INDENT, stream=stream)
