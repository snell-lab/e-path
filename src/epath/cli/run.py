"""Command-line orchestration for E-path.

This module turns parsed arguments into single or batch runs: resolving and
reading inputs, running the RABDAM BDamage stages, and handing the result to
the E-path pipeline. Terminal reporting lives in `epath.cli.report`.
"""

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO

from epath.cli.args import (
    bdamage_pipeline_options_from_args,
    epath_pipeline_options_from_args,
    single_output_prefix_from_args,
)
from epath.cli.report import (
    print_epath_batch_summary,
    print_epath_summary,
    print_error,
)
from epath.outputs.files import EpathPipelineOutputFiles
from epath.outputs.tables import write_run_csv
from epath.pipeline import EpathPipelineResult, run_epath_pipeline
from rabdam.pipeline import (
    BDamagePipelineResult,
    PipelineStageRunner,
    run_bdamage_pipeline,
)
from xtal.rcsb import download_rcsb_mmcif
from xtal.reader import read_structure
from xtal.resolver import (
    ResolvedStructureInput,
    is_valid_classic_pdb_id,
    resolve_structure_input,
    split_structure_file_name,
)

# One row per structure in a batch, written alongside the per-structure output.
BATCH_COHORT_CSV_NAME = "epath_runs.csv"

# Every input, crystal and calculation error in xtal and rabdam derives from
# ValueError, so it covers them all.
RABDAM_CLI_ERRORS = (ValueError, OSError)


@dataclass(frozen=True, slots=True)
class EpathCliResult:
    """Result returned by a full E-path CLI run."""

    local_input: ResolvedStructureInput
    output_prefix: Path
    pipeline_result: EpathPipelineResult
    output_files: EpathPipelineOutputFiles


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
    """Run the full E-path pipeline from parsed command-line arguments.

    Raises:
        ValueError: If no structure input was given, or if `--output-prefix`
            is combined with several structure inputs.
    """
    if total_runtime_start is None:
        total_runtime_start = perf_counter()

    structure_inputs = tuple(args.structure_inputs)
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
    """Run the full E-path pipeline for one structure input."""

    # A named generic function rather than a lambda so each pipeline stage
    # callback keeps its return type through PipelineStageRunner.
    def stage_runner[T](label: str, callback: Callable[[], T]) -> T:
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

    options = epath_pipeline_options_from_args(
        args,
        output_prefix=output_prefix,
    )
    pipeline_result, output_files = run_epath_pipeline(
        bdamage_result,
        structure_path=local_input.local_path,
        options=options,
        stage_runner=stage_runner,
    )
    cli_result = EpathCliResult(
        local_input=local_input,
        output_prefix=output_prefix,
        pipeline_result=pipeline_result,
        output_files=output_files,
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
    """Run the full E-path pipeline for multiple structure inputs.

    Raises:
        ValueError: If `--output-prefix` is combined with several structure
            inputs, which are written to separate files.
    """
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
            print_error(f"epath: error: {structure_input}: {error}", stream=stderr)
        else:
            results.append(result)

    # One row per structure, so a batch is directly comparable across models.
    cohort_csv = (
        write_run_csv(
            output_dir / BATCH_COHORT_CSV_NAME,
            (result.pipeline_result.run_record for result in results),
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
    stage_runner: PipelineStageRunner,
) -> tuple[ResolvedStructureInput, BDamagePipelineResult]:
    """Resolve one input, read it, and run the RABDAM BDamage stages."""
    pipeline_options = bdamage_pipeline_options_from_args(args)

    resolved = stage_runner(
        "Resolving input",
        lambda: resolve_structure_input(structure_input),
    )
    local_input = resolved
    if resolved.needs_download:
        local_input = stage_runner(
            "Downloading from RCSB",
            lambda: download_rcsb_mmcif(
                resolved,
                download_dir=Path(args.download_dir).expanduser(),
                overwrite=args.overwrite_download,
            ),
        )
    structure_data = stage_runner(
        "Reading structure",
        lambda: read_structure(local_input),
    )
    pipeline_result = run_bdamage_pipeline(
        structure_data,
        options=pipeline_options,
        stage_runner=stage_runner,
    )

    return local_input, pipeline_result


def batch_output_prefix_path(
    local_input: ResolvedStructureInput,
    *,
    output_dir: Path,
    used_output_prefixes: set[Path] | None,
) -> Path:
    """Return a unique output prefix for one resolved input.

    Each run gets its own directory under `output_dir`, named after the
    structure label. A repeated run of the same structure gets a numbered
    directory (`3I9V_1`, `3I9V_2`, ...) rather than overwriting the last one:
    directories already on disk count as used, as do prefixes handed out
    earlier in the same batch.
    """
    label = output_label_for_input(local_input)

    def prefix_for(run_dir_name: str) -> Path:
        return output_dir / run_dir_name / label

    def is_used(candidate: Path) -> bool:
        if used_output_prefixes is not None and candidate in used_output_prefixes:
            return True
        return candidate.parent.exists()

    candidate = prefix_for(label)
    suffix = 1
    while is_used(candidate):
        candidate = prefix_for(f"{label}_{suffix}")
        suffix += 1

    if used_output_prefixes is not None:
        used_output_prefixes.add(candidate)
    return candidate


def output_label_for_input(local_input: ResolvedStructureInput) -> str:
    """Return a filesystem-friendly output label for a resolved input.

    An RCSB entry is labelled by its upper-case ID. A local file is labelled by
    its name without `.gz` and the format extension, so `1abc.cif.gz` gives
    `1abc`. A wwPDB archive name such as `pdb1abc.ent.gz` gives the upper-case
    ID, `1ABC`, the same label a download of that entry gets. A name with
    nothing usable left falls back to `structure`.
    """
    if local_input.structure_id:
        raw_label = local_input.structure_id.upper()
    else:
        # Without an RCSB ID the input is a local file, which always has a path.
        assert local_input.local_path is not None
        raw_label = _label_from_file_name(local_input.local_path.name)

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_label)
    label = label.strip("._-")
    return label or "structure"


def _label_from_file_name(name: str) -> str:
    stem, suffix = split_structure_file_name(name)
    if (
        suffix == ".ent"
        and len(stem) == 7
        and stem[:3].lower() == "pdb"
        and is_valid_classic_pdb_id(stem[3:])
    ):
        return stem[3:].upper()
    return stem


def run_stage[T](
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
