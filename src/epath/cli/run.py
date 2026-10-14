"""Command-line orchestration for E-path.

This module turns parsed arguments into single or batch runs: resolving and
reading inputs, running the RABDAM BDamage stages, and handing the result to
the E-path pipeline. Terminal reporting lives in `epath.cli.report`.
"""

import argparse
import re
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
    resolve_structure_input,
    split_structure_file_name,
)

# One row per structure in a batch, written alongside the per-structure output.
BATCH_COHORT_CSV_STEM = "epath_runs"

# Every input, crystal and calculation error in xtal, rabdam and epath derives
# from ValueError, so it covers them all. OSError covers files that cannot be
# read or written.
CLI_ERRORS = (ValueError, OSError)


@dataclass(frozen=True, slots=True)
class EpathCliResult:
    """Result of one structure's run.

    `local_input` is the input as it was read, after any download, and
    `output_prefix` is the prefix its output files were written under.
    """

    local_input: ResolvedStructureInput
    output_prefix: Path
    pipeline_result: EpathPipelineResult
    output_files: EpathPipelineOutputFiles


@dataclass(frozen=True, slots=True)
class EpathBatchCliResult:
    """Result of a batch run.

    `results` holds the runs that succeeded and `failed_inputs` the inputs
    whose run failed, as given on the command line, each in input order.
    `elapsed_seconds` covers the whole batch. `cohort_csv` is `None` when no structure succeeded or
    reports are switched off.
    """

    results: tuple[EpathCliResult, ...]
    failed_inputs: tuple[str, ...]
    output_dir: Path
    elapsed_seconds: float
    cohort_csv: Path | None

    @property
    def has_errors(self) -> bool:
        """Return whether any batch item failed."""
        return bool(self.failed_inputs)


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    total_runtime_start: float,
) -> EpathCliResult | EpathBatchCliResult:
    """Run the full E-path pipeline from parsed command-line arguments.

    One structure input is a single run and several are a batch. That there is
    at least one is not checked here: `epath.cli.main` has already rejected an
    empty list. Progress goes to `stderr` and the summaries to `stdout`.
    `total_runtime_start` is the `perf_counter` reading that the reported total
    runtime is measured from.

    A failed single run raises one of `CLI_ERRORS`. A batch raises only
    for `--output-prefix`, and otherwise records each failure and carries on.
    """
    structure_inputs = tuple(args.structure_inputs)
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
) -> EpathCliResult:
    """Run the full E-path pipeline for one structure input and print its summary.

    Without an `output_prefix` the run gets a directory of its own under
    `--output-dir`, chosen by `default_output_prefix`.
    """

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
        output_prefix = default_output_prefix(
            local_input,
            output_dir=Path(args.output_dir).expanduser(),
        )

    options = epath_pipeline_options_from_args(
        args,
        output_prefix=output_prefix,
    )
    pipeline_result, output_files = run_epath_pipeline(
        bdamage_result,
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
        print_epath_summary(
            cli_result,
            total_runtime_seconds=perf_counter() - total_runtime_start,
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
    """Run the full E-path pipeline for several structure inputs in turn.

    Each input is run and summarised as a single structure, in its own
    directory under `--output-dir`. One that fails with any of
    `CLI_ERRORS` is reported on `stderr` and the batch carries on. The
    structures that succeeded then share one cohort CSV, unless `--no-reports`
    is given. Like the run directories, it is numbered rather than written
    over the cohort CSV of an earlier batch.

    `--output-prefix` names the files of one structure, so combining it with
    several inputs raises `ValueError` before anything runs.
    """
    if args.output_prefix is not None:
        raise ValueError(
            "--output-prefix can only be used with one structure input. "
            "Use --output-dir for batch runs."
        )

    output_dir = Path(args.output_dir).expanduser()
    results: list[EpathCliResult] = []
    failed_inputs: list[str] = []

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
            )
        except CLI_ERRORS as error:
            failed_inputs.append(structure_input)
            print_error(f"epath: error: {structure_input}: {error}", stream=stderr)
        else:
            results.append(result)

    # One row per structure, so a batch is directly comparable across models.
    cohort_csv = None
    if results and not args.no_reports:
        cohort_csv = write_run_csv(
            _unused_cohort_csv_path(output_dir),
            (result.pipeline_result.run_record for result in results),
        )

    batch_result = EpathBatchCliResult(
        results=tuple(results),
        failed_inputs=tuple(failed_inputs),
        output_dir=output_dir,
        elapsed_seconds=perf_counter() - total_runtime_start,
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
    """Resolve one input, read it, and run the RABDAM BDamage stages.

    An RCSB entry is downloaded first. The input returned is the local file
    that was read.
    """
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
        options=bdamage_pipeline_options_from_args(args),
        stage_runner=stage_runner,
    )

    return local_input, pipeline_result


def default_output_prefix(
    local_input: ResolvedStructureInput,
    *,
    output_dir: Path,
) -> Path:
    """Return the output prefix of a run that was given none.

    Each run gets its own directory under `output_dir`, named after the
    structure label. A repeated run of the same structure gets a numbered
    directory (`1ABC_1`, `1ABC_2`, ...) rather than overwriting the last one.
    Only directories on disk count as used. That covers the earlier items of
    a batch too, because each one writes its files before the next prefix is
    chosen.
    """
    label = output_label_for_input(local_input)

    candidate = output_dir / label / label
    suffix = 1
    while candidate.parent.exists():
        candidate = output_dir / f"{label}_{suffix}" / label
        suffix += 1

    return candidate


def _unused_cohort_csv_path(output_dir: Path) -> Path:
    """Return the cohort CSV path of a batch, numbered if the name is taken."""
    candidate = output_dir / f"{BATCH_COHORT_CSV_STEM}.csv"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{BATCH_COHORT_CSV_STEM}_{suffix}.csv"
        suffix += 1

    return candidate


def output_label_for_input(local_input: ResolvedStructureInput) -> str:
    """Return a filesystem-friendly output label for a resolved input.

    An RCSB entry is labelled by its upper-case ID. A local file is labelled by
    its name without `.gz` and the format extension, so `1abc.cif.gz` gives
    `1abc`. A name with nothing usable left falls back to `structure`.
    """
    if local_input.structure_id:
        raw_label = local_input.structure_id.upper()
    else:
        # Without an RCSB ID the input is a local file, which always has a path.
        assert local_input.local_path is not None
        raw_label = split_structure_file_name(local_input.local_path.name)[0]

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_label)
    label = label.strip("._-")
    return label or "structure"


def run_stage[T](
    label: str,
    callback: Callable[[], T],
    *,
    stream: TextIO,
    quiet: bool,
) -> T:
    """Run one stage, reporting its label and how long it took on `stream`.

    A stage that raises is reported as failed, with its time, and the exception
    propagates. With `quiet` nothing is printed.
    """
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
