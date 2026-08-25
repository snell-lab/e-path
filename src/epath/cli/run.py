"""Command-line orchestration for E-path.

This module turns parsed arguments into single or batch runs: resolving and
reading inputs, running the RABDAM BDamage stages, and handing the result to
the E-path workflow. Terminal reporting lives in `epath.cli.report`.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO, TypeVar

from epath.cli.args import (
    bdamage_workflow_options_from_args,
    epath_workflow_options_from_args,
    single_output_prefix_from_args,
    structure_inputs_from_args,
)
from epath.cli.report import print_epath_batch_summary, print_epath_summary
from epath.plots import EpathPlotError
from epath.tables import write_run_csv
from epath.workflow import EpathWorkflowResult, run_epath_workflow
from rabdam.bdamage.score import BDamageScoreError
from rabdam.crystal.symmetry import CrystalSymmetryError
from rabdam.crystal.translate import CrystalTranslationError
from rabdam.crystal.trim import CrystalTrimError
from rabdam.packing.density import PackingDensityError
from rabdam.structure.models import StructurePreparationOptions
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowResult,
    WorkflowStageRunner,
    calculate_bdamage_for_structure_data,
)
from xtal.rcsb import ensure_local_structure_file
from xtal.reader import read_structure
from xtal.resolver import ResolvedStructureInput, resolve_structure_input
from xtal.validate import CrystalMetadataError

# One row per structure in a batch, written alongside the per-structure output.
BATCH_COHORT_CSV_NAME = "epath_runs.csv"

RABDAM_CLI_ERRORS = (
    BDamageScoreError,
    BDamageWorkflowError,
    CrystalMetadataError,
    CrystalSymmetryError,
    CrystalTranslationError,
    CrystalTrimError,
    OSError,
    PackingDensityError,
    EpathPlotError,
    ValueError,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EpathCliResult:
    """Result returned by a full E-path CLI run."""

    local_input: ResolvedStructureInput
    output_prefix: Path
    workflow_result: EpathWorkflowResult


@dataclass(frozen=True, slots=True)
class EpathBatchItemError:
    """Failure for one input in a full E-path batch run."""

    structure_input: str
    error: Exception


@dataclass(frozen=True, slots=True)
class EpathBatchCliResult:
    """Result returned by a full E-path batch CLI run."""

    results: tuple[EpathCliResult, ...]
    errors: tuple[EpathBatchItemError, ...]
    output_dir: Path
    elapsed_seconds: float
    cohort_csv: Path | None = None

    @property
    def has_errors(self) -> bool:
        """Return whether any batch item failed."""

        return bool(self.errors)


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    total_runtime_start: float | None = None,
) -> EpathCliResult | EpathBatchCliResult:
    """Run the full E-path workflow from parsed command-line arguments."""

    if total_runtime_start is None:
        total_runtime_start = perf_counter()

    structure_inputs = structure_inputs_from_args(args)
    if not structure_inputs:
        raise ValueError("At least one structure input is required.")

    if len(structure_inputs) == 1:
        return run_single_structure(
            args,
            structure_input=structure_inputs[0],
            output_prefix=single_output_prefix_from_args(args),
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=total_runtime_start,
        )

    return run_batch(
        args,
        structure_inputs=structure_inputs,
        stdout=stdout,
        stderr=stderr,
        total_runtime_start=total_runtime_start,
    )


def run_single_structure(
    args: argparse.Namespace,
    *,
    structure_input: str,
    output_prefix: Path | None,
    stdout: TextIO,
    stderr: TextIO,
    total_runtime_start: float,
    used_batch_output_prefixes: set[Path] | None = None,
) -> EpathCliResult:
    """Run the full E-path workflow for one structure input."""

    # A named generic function rather than a lambda so each workflow stage
    # callback keeps its return type through WorkflowStageRunner.
    def stage_runner(label: str, callback: Callable[[], T]) -> T:
        return run_stage(label, callback, stream=stderr, quiet=args.quiet)

    local_input, bdamage_result = calculate_bdamage_from_args(
        args,
        structure_input=structure_input,
        stage_runner=stage_runner,
    )

    if output_prefix is None:
        output_prefix = batch_output_prefix_path(
            local_input,
            output_dir=Path(args.output_dir).expanduser(),
            used_output_prefixes=used_batch_output_prefixes,
        )

    options = epath_workflow_options_from_args(
        args,
        output_prefix=output_prefix,
    )
    workflow_result = run_epath_workflow(
        bdamage_result,
        structure_path=local_input.local_path,
        options=options,
        stage_runner=stage_runner,
    )
    cli_result = EpathCliResult(
        local_input=local_input,
        output_prefix=output_prefix,
        workflow_result=workflow_result,
    )

    if not args.quiet:
        total_runtime_seconds = perf_counter() - total_runtime_start
        print_epath_summary(
            cli_result,
            preview_count=args.preview_count,
            total_runtime_seconds=total_runtime_seconds,
            stream=stdout,
        )

    return cli_result


def run_batch(
    args: argparse.Namespace,
    *,
    structure_inputs: tuple[str, ...],
    stdout: TextIO,
    stderr: TextIO,
    total_runtime_start: float,
) -> EpathBatchCliResult:
    """Run the full E-path workflow for multiple structure inputs."""

    if args.output_prefix is not None:
        raise ValueError(
            "--output-prefix can only be used with one structure input. "
            "Use --output-dir for batch runs."
        )

    output_dir = Path(args.output_dir).expanduser()
    results: list[EpathCliResult] = []
    errors: list[EpathBatchItemError] = []
    used_output_prefixes: set[Path] = set()

    for index, structure_input in enumerate(structure_inputs, start=1):
        if not args.quiet:
            print(
                f"\nBatch item {index}/{len(structure_inputs)}: {structure_input}",
                file=stderr,
            )

        try:
            result = run_single_structure(
                args,
                structure_input=structure_input,
                output_prefix=None,
                stdout=stdout,
                stderr=stderr,
                total_runtime_start=perf_counter(),
                used_batch_output_prefixes=used_output_prefixes,
            )
        except RABDAM_CLI_ERRORS as error:
            errors.append(
                EpathBatchItemError(
                    structure_input=structure_input,
                    error=error,
                )
            )
            print(f"epath: error: {structure_input}: {error}", file=stderr)
        else:
            results.append(result)

    # One row per structure, so a batch is directly comparable across models.
    cohort_csv = (
        write_run_csv(
            output_dir / BATCH_COHORT_CSV_NAME,
            (result.workflow_result.run_record for result in results),
        )
        if results and not args.no_reports
        else None
    )
    elapsed_seconds = perf_counter() - total_runtime_start
    batch_result = EpathBatchCliResult(
        results=tuple(results),
        errors=tuple(errors),
        output_dir=output_dir,
        elapsed_seconds=elapsed_seconds,
        cohort_csv=cohort_csv,
    )

    if not args.quiet:
        print_epath_batch_summary(batch_result, stream=stdout)

    return batch_result


def calculate_bdamage_from_args(
    args: argparse.Namespace,
    *,
    structure_input: str,
    stage_runner: WorkflowStageRunner,
) -> tuple[ResolvedStructureInput, BDamageWorkflowResult]:
    """Resolve one input, read it, and run the RABDAM BDamage stages."""

    workflow_options = bdamage_workflow_options_from_args(args)
    # Structure preparation is not configurable: only protein atoms are
    # scored, so the RABDAM defaults are the E-path population.
    preparation_options = StructurePreparationOptions()

    resolved = stage_runner(
        "Resolving input",
        lambda: resolve_structure_input(structure_input),
    )
    local_input = stage_runner(
        "Ensuring local structure file",
        lambda: ensure_local_structure_file(
            resolved,
            download_dir=Path(args.download_dir).expanduser(),
            overwrite=args.overwrite_download,
        ),
    )
    structure_data = stage_runner(
        "Reading structure",
        lambda: read_structure(local_input),
    )
    workflow_result = calculate_bdamage_for_structure_data(
        structure_data,
        workflow_options=workflow_options,
        preparation_options=preparation_options,
        stage_runner=stage_runner,
    )

    return local_input, workflow_result


def batch_output_prefix_path(
    local_input: ResolvedStructureInput,
    *,
    output_dir: Path,
    used_output_prefixes: set[Path] | None,
) -> Path:
    """Return a unique output prefix for one resolved input."""

    label = output_label_for_input(local_input)
    candidate = output_dir / label
    if used_output_prefixes is None:
        return candidate

    suffix = 2
    while candidate in used_output_prefixes:
        candidate = output_dir / f"{label}_{suffix}"
        suffix += 1

    used_output_prefixes.add(candidate)
    return candidate


def output_label_for_input(local_input: ResolvedStructureInput) -> str:
    """Return a filesystem-friendly output label for a resolved input."""

    if local_input.structure_id:
        raw_label = local_input.structure_id.upper()
    elif local_input.local_path is not None:
        raw_label = local_input.local_path.stem
    else:
        raw_label = Path(local_input.original_input).stem or "structure"

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_label.strip())
    label = label.strip("._-")
    return label or "structure"


def run_stage(
    label: str,
    callback: Callable[[], T],
    *,
    stream: TextIO,
    quiet: bool,
) -> T:
    """Run one CLI stage and optionally print elapsed time."""

    if quiet:
        return callback()

    print(f"{label}...", file=stream, flush=True)
    start = perf_counter()
    try:
        result = callback()
    except Exception:
        elapsed = perf_counter() - start
        print(f"  failed after {elapsed:.1f}s", file=stream, flush=True)
        raise

    elapsed = perf_counter() - start
    print(f"  done in {elapsed:.1f}s", file=stream, flush=True)
    return result
