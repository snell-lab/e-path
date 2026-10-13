"""Terminal reporting for E-path command-line runs.

The section contents are built by `epath.outputs`, the same builders that
write the persistent summary report, so the terminal and the file never
disagree about a value. This module only decides which sections to show and
how to indent them. Long lines are wrapped with a hanging indent and
warnings and errors are highlighted when the stream is a colour terminal
(see `epath.cli.style`).
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, TextIO

from epath.analysis.endpoints import cofactor_site_summary
from epath.cli.style import print_lines, style_line, supports_color
from epath.outputs.summary import (
    bdamage_summary_lines,
    bnet_summary_lines,
    crystal_summary_lines,
    output_file_entries,
)
from rabdam.bdamage import bdamage_scores_as_tuple

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


def print_missing_structure_input_error(*, stream: TextIO) -> None:
    """Print a compact recovery message for missing structure input."""
    print_error(MISSING_STRUCTURE_INPUT_MESSAGE, stream=stream)


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
    preview_count: int,
    total_runtime_seconds: float,
    stream: TextIO,
) -> None:
    """Print a compact full E-path run summary."""
    pipeline_result = result.pipeline_result
    bdamage_result = pipeline_result.bdamage_pipeline_result
    # The CLI always runs E-path from its own BDamage pipeline.
    assert bdamage_result is not None
    crystal = bdamage_result.crystal

    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output prefix: {result.output_prefix}", file=stream)

    _print_section("Crystal:", crystal_summary_lines(crystal), stream=stream)
    _print_section(
        "BDamage calculation:",
        bdamage_summary_lines(bdamage_result),
        stream=stream,
    )
    _print_section(
        "E-path selection:",
        _selection_lines(pipeline_result),
        stream=stream,
    )
    _print_section(
        "RABDAM Bnet results:",
        bnet_summary_lines(pipeline_result.bnet_pipeline_result),
        stream=stream,
    )
    _print_section("Pathways:", _pathway_lines(pipeline_result), stream=stream)
    _print_section(
        "Outputs:",
        (
            f"{label}: {path}"
            for label, path in output_file_entries(result.output_files)
        ),
        stream=stream,
    )

    if preview_count > 0:
        packing_density_counts = bdamage_result.packing_densities[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(bdamage_result.bdamage_results)[
            :preview_count
        ]
        _print_section(
            "BDamage preview:",
            (
                f"Packing-density counts, first {preview_count}:",
                INDENT + _format_int_values(packing_density_counts),
                f"BDamage scores, first {preview_count}:",
                INDENT + _format_float_values(bdamage_scores),
            ),
            stream=stream,
        )

    if crystal.warnings:
        _print_section(
            "Warnings:",
            (f"WARNING: {warning}" for warning in crystal.warnings),
            stream=stream,
        )

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


def print_epath_batch_summary(
    result: EpathBatchCliResult,
    *,
    stream: TextIO,
) -> None:
    """Print a compact full E-path batch-run summary."""
    lines = [
        f"Successful: {len(result.results)}",
        f"Failed: {len(result.errors)}",
        f"Output directory: {result.output_dir}",
    ]
    if result.cohort_csv is not None:
        lines.append(f"Cohort CSV: {result.cohort_csv}")
    if result.errors:
        lines.append("Failed inputs:")
        lines.extend(
            INDENT + item_error.structure_input for item_error in result.errors
        )
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
        f"Path {report.path_id} "
        f"{report.start_endpoint.label} -> {report.end_endpoint.label}: "
        f"{report.residue_count} residues, {report.node_count} nodes, "
        f"{report.total_length:.2f} A, nearest cofactor {report.anchor_summary}, "
        f"{report.structural_descriptor}"
        for report in reports
    )
    return tuple(lines)


def _print_section(title: str, lines: Iterable[str], *, stream: TextIO) -> None:
    print(file=stream)
    print(title, file=stream)
    print_lines(tuple(lines), indent=INDENT, stream=stream)


def _format_int_values(values: Sequence[int]) -> str:
    return ", ".join(str(value) for value in values)


def _format_float_values(values: Sequence[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)
