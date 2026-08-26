"""Terminal reporting for E-path command-line runs.

The section contents are built by `epath.outputs`, the same builders that
write the persistent summary report, so the terminal and the file never
disagree about a value. This module only decides which sections to show and
how to indent them. Long lines are wrapped with a hanging indent and
warnings and errors are highlighted when the stream is a colour terminal
(see `epath.cli.style`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, TextIO

from epath.analysis.endpoints import cofactor_site_summary
from epath.analysis.pathways import PATH_CLASS_ORDER
from epath.cli.style import print_lines, style_line, supports_color
from epath.outputs.summary import (
    bdamage_parameter_lines,
    bnet_summary_lines,
    crystal_summary_lines,
    output_file_entries,
)
from rabdam.bdamage import bdamage_scores_as_tuple
from rabdam.packing import packing_density_counts_as_tuple

if TYPE_CHECKING:
    from epath.cli.run import EpathBatchCliResult, EpathCliResult
    from epath.workflow import EpathWorkflowResult

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
    workflow_result = result.workflow_result
    bdamage_result = workflow_result.bdamage_workflow_result
    crystal = bdamage_result.crystal if bdamage_result is not None else None

    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output prefix: {result.output_prefix}", file=stream)

    _print_section("Crystal:", crystal_summary_lines(crystal), stream=stream)
    _print_section(
        "BDamage calculation:",
        bdamage_parameter_lines(bdamage_result),
        stream=stream,
    )
    _print_section(
        "E-path selection:",
        _selection_lines(workflow_result),
        stream=stream,
    )
    _print_section(
        "RABDAM Bnet results:",
        bnet_summary_lines(workflow_result.bnet_workflow_result),
        stream=stream,
    )
    _print_section("Pathways:", _pathway_lines(workflow_result), stream=stream)
    _print_section(
        "Outputs:",
        (
            f"{label}: {path}"
            for label, path in output_file_entries(result.output_files)
        ),
        stream=stream,
    )

    if preview_count > 0 and bdamage_result is not None:
        packing_density_counts = packing_density_counts_as_tuple(
            bdamage_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(bdamage_result.bdamage_score_result)[
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

    warnings = crystal.warnings if crystal is not None else ()
    if warnings:
        _print_section(
            "Warnings:",
            (f"WARNING: {warning}" for warning in warnings),
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


def _selection_lines(workflow_result: EpathWorkflowResult) -> tuple[str, ...]:
    selection_result = workflow_result.selection_result
    analyzed_chains = workflow_result.analyzed_chains
    plateau = selection_result.plateau
    plateau_text = (
        "none detected"
        if plateau is None
        else (
            f"{plateau.atom_count} atoms at Z "
            f"{plateau.lowest_z_score:.3f} to {plateau.highest_z_score:.3f}"
        )
    )
    return (
        "Chains: " + (", ".join(analyzed_chains) if analyzed_chains else "none"),
        f"Method: {selection_result.description}",
        (
            f"Selected atoms: {len(selection_result.selected_atoms)} "
            f"of {len(selection_result.scored_atoms)}"
        ),
        (
            f"BDamage mean: {selection_result.bdamage_mean:.4f}, "
            f"SD: {selection_result.bdamage_standard_deviation:.4f}"
        ),
        f"Z-score plateau: {plateau_text}",
    )


def _pathway_lines(workflow_result: EpathWorkflowResult) -> tuple[str, ...]:
    reports = workflow_result.path_reports
    lines = [
        f"Connected components: {workflow_result.component_statistics.total_count}",
        f"Count: {len(reports)}",
    ]
    class_counts = Counter(report.path_class for report in reports)
    if class_counts:
        lines.append(
            "Classes: "
            + ", ".join(
                f"{name} {class_counts[name]}"
                for name in PATH_CLASS_ORDER
                if class_counts[name] > 0
            )
        )
    lines.append(
        f"Cofactor sites: {cofactor_site_summary(workflow_result.cofactor_sites)}"
    )
    lines.extend(
        f"Path {report.path_id} [{report.class_label}] "
        f"{report.start_endpoint.label} -> {report.end_endpoint.label}: "
        f"{report.residue_count} residues, {report.node_count} nodes, "
        f"{report.total_length:.2f} A, {report.structural_descriptor}"
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
