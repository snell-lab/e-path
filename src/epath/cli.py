"""Command-line interface for E-path."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO, TypeVar

from epath import __version__
from epath.atoms import atoms_from_bdamage_workflow
from epath.endpoints import cofactor_site_summary
from epath.outputs import EpathOutputWorkflowOptions
from epath.pathways import PATH_CLASS_ORDER
from epath.plots import EpathPlotError
from epath.tables import write_run_csv
from epath.workflow import (
    EpathClusterWorkflowOptions,
    EpathSelectionWorkflowOptions,
    EpathWorkflowOptions,
    EpathWorkflowResult,
    SelectionMethod,
    analyze_epath_atoms,
)
from rabdam.bdamage.score import (
    BDamageScoreError,
    bdamage_scores_as_tuple,
)
from rabdam.bnet.workflow import BnetWorkflowResult, calculate_bnet_workflow
from rabdam.crystal.symmetry import CrystalSymmetryError
from rabdam.crystal.translate import CrystalTranslationError
from rabdam.crystal.trim import CrystalTrimError
from rabdam.packing.density import (
    PackingDensityError,
    packing_density_counts_as_tuple,
)
from rabdam.structure.models import StructurePreparationOptions
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    BDamageWorkflowResult,
    calculate_bdamage_for_structure_data,
)
from xtal.rcsb import DEFAULT_RCSB_DOWNLOAD_DIR, ensure_local_structure_file
from xtal.reader import read_structure
from xtal.resolver import ResolvedStructureInput, resolve_structure_input
from xtal.validate import CrystalMetadataError, ValidatedCrystal

DEFAULT_OUTPUT_DIR = Path("output")

# One row per structure in a batch, written alongside the per-structure output.
BATCH_COHORT_CSV_NAME = "epath_runs.csv"
DEFAULT_DOWNLOAD_DIR = DEFAULT_RCSB_DOWNLOAD_DIR
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
REMOVED_BDAMAGE_COMMAND_MESSAGE = """\
epath: error: the 'bdamage' subcommand has been removed.

E-path scores protein atoms only, and that population is no longer
configurable. Run 'epath STRUCTURE_INPUT' instead; per-atom BDamage,
packing density, and window-averaged B-factors are written to the
combined '*_epath.csv' output."""

MISSING_STRUCTURE_INPUT_MESSAGE = """epath: error: missing required argument: structure_input

Usage:
  epath STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]
  epath bdamage STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]

Examples:
  epath 3i9v
  epath example.cif --percentile 2 --output-prefix output/example
  epath bdamage 3i9v
  epath bdamage example.cif -o output/example_BDamage.csv

Run 'epath --help' for all options."""

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


def _build_epath_parser() -> argparse.ArgumentParser:
    """Build the full E-path workflow parser."""

    bdamage_defaults = BDamageWorkflowOptions()
    epath_defaults = EpathWorkflowOptions()
    parser = argparse.ArgumentParser(
        prog="epath",
        usage="%(prog)s STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]",
        description="Run the full E-path workflow for structure files or PDB IDs.",
        add_help=False,
    )
    input_group = parser.add_argument_group("input")
    output = parser.add_argument_group("output options")
    epath_selection = parser.add_argument_group("E-path selection options")
    epath_advanced = parser.add_argument_group("advanced E-path parameters")
    bdamage = parser.add_argument_group("advanced BDamage parameters")
    bdamage_debug = parser.add_argument_group("advanced BDamage debugging options")
    download = parser.add_argument_group("download options")
    general = parser.add_argument_group("general options")

    _add_structure_input_argument(input_group)
    output.add_argument(
        "-o",
        "--output-prefix",
        metavar="PREFIX",
        default=None,
        help=(
            "Prefix for E-path output files in a single-structure run. "
            "Default: output/<structure>"
        ),
    )
    output.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for default single-run and batch E-path outputs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    output.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the combined E-path CSV output.",
    )
    output.add_argument(
        "--no-pdb",
        action="store_true",
        help="Do not write the selected-atom PDB output.",
    )
    output.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the BDamage Z-score falloff plot.",
    )
    output.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not write the cluster CSV or summary report.",
    )
    epath_selection.add_argument(
        "--percentile",
        metavar="FLOAT",
        type=percent_float,
        default=epath_defaults.selection.percentile,
        help=(
            "Lowest BDamage percentile to select for E-path analysis. "
            f"Default: {epath_defaults.selection.percentile}"
        ),
    )
    epath_selection.add_argument(
        "--per-chain-percentile",
        action="store_true",
        help="Apply the percentile cutoff separately within each chain.",
    )
    epath_advanced.add_argument(
        "--cluster-distance",
        metavar="ANGSTROM",
        type=positive_float,
        default=epath_defaults.clustering.distance_cutoff,
        help=(
            "Distance cutoff for selected-atom clustering. "
            f"Default: {epath_defaults.clustering.distance_cutoff}"
        ),
    )
    epath_advanced.add_argument(
        "--minimum-cluster-size",
        metavar="INT",
        type=positive_int,
        default=epath_defaults.clustering.minimum_cluster_size,
        help=(
            "Minimum connected-component size reported as an E-path cluster. "
            f"Default: {epath_defaults.clustering.minimum_cluster_size}"
        ),
    )
    _add_bdamage_calculation_arguments(bdamage, defaults=bdamage_defaults)
    _add_bdamage_debug_arguments(bdamage_debug)
    _add_download_arguments(download)
    _add_general_arguments(general)

    return parser


def _add_structure_input_argument(group: Any) -> None:
    group.add_argument(
        "structure_inputs",
        nargs="*",
        metavar="STRUCTURE_INPUT",
        help=(
            "One or more paths to .pdb/.cif/.mmcif files, or 4-character "
            "PDB IDs such as 1LYZ."
        ),
    )


def _add_general_arguments(group: Any) -> None:
    group.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the program version and exit.",
    )
    group.add_argument(
        "--citations",
        action="store_true",
        help="Print citation information and exit.",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress and summary output.",
    )


def _add_download_arguments(group: Any) -> None:
    group.add_argument(
        "--download-dir",
        metavar="DIR",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=(
            "Directory for downloaded RCSB mmCIF files. "
            f"Default: {DEFAULT_DOWNLOAD_DIR}"
        ),
    )
    group.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Download RCSB inputs again even when the mmCIF file already exists.",
    )


def _add_bdamage_calculation_arguments(
    group: Any,
    *,
    defaults: BDamageWorkflowOptions,
) -> None:
    group.add_argument(
        "--packing-density-threshold",
        metavar="FLOAT",
        type=positive_float,
        default=defaults.packing_density_threshold,
        help=(
            "Distance cutoff in Angstroms for neighbour counting. "
            f"Default: {defaults.packing_density_threshold}"
        ),
    )
    group.add_argument(
        "--window-size-fraction",
        metavar="FLOAT",
        type=fraction_float,
        default=defaults.window_size_fraction,
        help=(
            "Fraction of selected atoms used to derive the BDamage window size. "
            f"Default: {defaults.window_size_fraction}"
        ),
    )
    group.add_argument(
        "--minimum-window-size",
        metavar="INT",
        type=positive_int,
        default=defaults.minimum_window_size,
        help=(
            "Minimum BDamage sliding-window atom count. "
            f"Default: {defaults.minimum_window_size}"
        ),
    )
    group.add_argument(
        "--translation-range",
        metavar="INT",
        type=non_negative_int,
        default=defaults.translation_range,
        help=(
            "Number of unit-cell translations to include in each crystal direction. "
            f"Default: {defaults.translation_range}"
        ),
    )


def _add_bdamage_debug_arguments(group: Any) -> None:
    group.add_argument(
        "--materialize-translated-block",
        action="store_true",
        help="Build the full translated crystal block for debugging.",
    )
    group.add_argument(
        "--preview-count",
        metavar="INT",
        type=non_negative_int,
        default=0,
        help="Number of leading packing-density and BDamage values to print. Default: 0",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "bdamage":
        raise SystemExit(REMOVED_BDAMAGE_COMMAND_MESSAGE)

    parser = _build_epath_parser()
    args = parser.parse_args(raw_args)

    args.structure_input = (
        args.structure_inputs[0] if len(args.structure_inputs) == 1 else None
    )
    return args


def positive_float(value: str) -> float:
    """Return a positive finite float for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def fraction_float(value: str) -> float:
    """Return a float in the open interval 0 to 1 for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a number greater than 0 and less than 1, got {value!r}"
        )
    return parsed


def percent_float(value: str) -> float:
    """Return a finite percentage in the closed interval 0 to 100."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError(
            f"expected a percentage from 0 to 100, got {value!r}"
        )
    return parsed


def positive_int(value: str) -> int:
    """Return a positive integer for argparse."""

    parsed = _int_arg(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    """Return a non-negative integer for argparse."""

    parsed = _int_arg(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def workflow_options_from_args(args: argparse.Namespace) -> BDamageWorkflowOptions:
    """Build workflow options from parsed command-line arguments."""

    return BDamageWorkflowOptions(
        packing_density_threshold=args.packing_density_threshold,
        window_size_fraction=args.window_size_fraction,
        minimum_window_size=args.minimum_window_size,
        translation_range=args.translation_range,
        materialize_translated_block=args.materialize_translated_block,
    )


def epath_workflow_options_from_args(
    args: argparse.Namespace,
    *,
    output_prefix: Path | None,
) -> EpathWorkflowOptions:
    """Build full E-path workflow options from parsed command-line arguments."""

    selection_method: SelectionMethod = (
        "percentile_per_chain" if args.per_chain_percentile else "percentile"
    )

    return EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(
            method=selection_method,
            percentile=args.percentile,
        ),
        clustering=EpathClusterWorkflowOptions(
            distance_cutoff=args.cluster_distance,
            minimum_cluster_size=args.minimum_cluster_size,
        ),
        output=EpathOutputWorkflowOptions(
            output_prefix=output_prefix,
            write_combined_csv=not args.no_csv,
            write_structure_files=not args.no_pdb,
            write_z_score_plot=not args.no_plot,
            write_cluster_csv=not args.no_reports,
            write_run_csv=not args.no_reports,
            write_summary_report=not args.no_reports,
        ),
    )


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    total_runtime_start: float | None = None,
) -> EpathCliResult | EpathBatchCliResult:
    """Run the requested workflow from parsed command-line arguments."""

    return run_epath_from_args(
        args,
        stdout=stdout,
        stderr=stderr,
        total_runtime_start=total_runtime_start,
    )


def structure_inputs_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Return one or more structure inputs from parsed CLI arguments."""

    structure_inputs = tuple(getattr(args, "structure_inputs", ()) or ())
    if structure_inputs:
        return structure_inputs

    structure_input = getattr(args, "structure_input", None)
    if structure_input is None:
        return ()

    return (structure_input,)


def calculate_bdamage_workflow_from_args(
    args: argparse.Namespace,
    *,
    structure_input: str,
    stderr: TextIO,
) -> tuple[ResolvedStructureInput, BDamageWorkflowResult]:
    """Run the shared BDamage calculation used by CLI workflows."""

    workflow_options = workflow_options_from_args(args)
    # Structure preparation is not configurable: only protein atoms are
    # scored, so the RABDAM defaults are the E-path population.
    preparation_options = StructurePreparationOptions()

    # A named generic function rather than a lambda so the workflow stage
    # callback keeps its return type through WorkflowStageRunner.
    def stage_runner(label: str, callback: Callable[[], T]) -> T:
        return run_stage(label, callback, stream=stderr, quiet=args.quiet)

    resolved = run_stage(
        "Resolving input",
        lambda: resolve_structure_input(structure_input),
        stream=stderr,
        quiet=args.quiet,
    )
    local_input = run_stage(
        "Ensuring local structure file",
        lambda: ensure_local_structure_file(
            resolved,
            download_dir=Path(args.download_dir).expanduser(),
            overwrite=args.overwrite_download,
        ),
        stream=stderr,
        quiet=args.quiet,
    )
    structure_data = run_stage(
        "Reading structure",
        lambda: read_structure(local_input),
        stream=stderr,
        quiet=args.quiet,
    )
    workflow_result = calculate_bdamage_for_structure_data(
        structure_data,
        workflow_options=workflow_options,
        preparation_options=preparation_options,
        stage_runner=stage_runner,
    )

    return local_input, workflow_result


def run_epath_from_args(
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
        return run_single_epath_structure_from_args(
            args,
            structure_input=structure_inputs[0],
            output_prefix=single_output_prefix_from_args(args),
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=total_runtime_start,
        )

    return run_epath_batch_from_args(
        args,
        structure_inputs=structure_inputs,
        stdout=stdout,
        stderr=stderr,
        total_runtime_start=total_runtime_start,
    )


def run_single_epath_structure_from_args(
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

    local_input, bdamage_result = calculate_bdamage_workflow_from_args(
        args,
        structure_input=structure_input,
        stderr=stderr,
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
    atoms = atoms_from_bdamage_workflow(bdamage_result)
    bnet_workflow_result = run_stage(
        "Calculating Bnet and Bnet percentile",
        lambda: calculate_bnet_workflow(
            prepared_structure=bdamage_result.prepared_structure,
            bdamage_score_result=bdamage_result.bdamage_score_result,
        ),
        stream=stderr,
        quiet=args.quiet,
    )

    workflow_result = run_stage(
        "Running E-path workflow",
        lambda: analyze_epath_atoms(
            atoms,
            structure_path=local_input.local_path,
            options=options,
            bdamage_workflow_result=bdamage_result,
            bnet_workflow_result=bnet_workflow_result,
        ),
        stream=stderr,
        quiet=args.quiet,
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


def run_epath_batch_from_args(
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
            result = run_single_epath_structure_from_args(
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


def single_output_prefix_from_args(args: argparse.Namespace) -> Path | None:
    """Return the E-path output prefix for a single-structure run."""

    if args.output_prefix is not None:
        return Path(args.output_prefix).expanduser()

    return None


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


def print_missing_structure_input_error(*, stream: TextIO) -> None:
    """Print a compact recovery message for missing structure input."""

    print(MISSING_STRUCTURE_INPUT_MESSAGE, file=stream)


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
    selection_result = workflow_result.selection_result
    output_files = workflow_result.output_files
    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output prefix: {result.output_prefix}", file=stream)
    print(file=stream)
    print_crystal_summary(
        workflow_result.bdamage_workflow_result.crystal
        if workflow_result.bdamage_workflow_result is not None
        else None,
        stream=stream,
    )
    print(file=stream)
    print_bdamage_parameter_summary(
        workflow_result.bdamage_workflow_result,
        stream=stream,
    )
    print(file=stream)
    print("E-path selection:", file=stream)
    analyzed_chains = tuple(
        dict.fromkeys(atom.chain_id for atom in workflow_result.atoms)
    )
    print(
        "  Chains: " + (", ".join(analyzed_chains) if analyzed_chains else "none"),
        file=stream,
    )
    print(f"  Method: {selection_result.description}", file=stream)
    print(
        f"  Selected atoms: {len(selection_result.selected_atoms)} "
        f"of {len(selection_result.scored_atoms)}",
        file=stream,
    )
    print(
        f"  BDamage mean: {selection_result.bdamage_mean:.4f}, "
        f"SD: {selection_result.bdamage_standard_deviation:.4f}",
        file=stream,
    )
    plateau = selection_result.plateau
    if plateau is None:
        print("  Z-score plateau: none detected", file=stream)
    else:
        print(
            f"  Z-score plateau: {plateau.atom_count} atoms at Z "
            f"{plateau.lowest_z_score:.3f} to {plateau.highest_z_score:.3f}",
            file=stream,
        )
    print(file=stream)
    print_bnet_summary(workflow_result.bnet_workflow_result, stream=stream)
    print(file=stream)
    print("Pathways:", file=stream)
    print(
        f"  Connected components: {workflow_result.component_statistics.total_count}",
        file=stream,
    )
    print(f"  Count: {len(workflow_result.path_reports)}", file=stream)
    class_counts = Counter(report.path_class for report in workflow_result.path_reports)
    if class_counts:
        print(
            "  Classes: "
            + ", ".join(
                f"{name} {class_counts[name]}"
                for name in PATH_CLASS_ORDER
                if class_counts[name] > 0
            ),
            file=stream,
        )
    print(
        f"  Cofactor sites: {cofactor_site_summary(workflow_result.cofactor_sites)}",
        file=stream,
    )
    for report in workflow_result.path_reports:
        print(
            f"  Path {report.path_id} [{report.class_label}] "
            f"{report.start_endpoint.label} -> {report.end_endpoint.label}: "
            f"{report.residue_count} residues, {report.node_count} nodes, "
            f"{report.total_length:.2f} A, {report.structural_descriptor}",
            file=stream,
        )

    print(file=stream)
    print("Outputs:", file=stream)
    print(f"  Combined CSV: {output_files.combined_csv}", file=stream)
    print(f"  Cluster CSV: {output_files.cluster_csv}", file=stream)
    print(f"  Run CSV: {output_files.run_csv}", file=stream)
    print(f"  Selected PDB: {output_files.selected_pdb}", file=stream)
    print(f"  Model PDB: {output_files.model_pdb}", file=stream)
    print(f"  Z-score falloff: {output_files.z_score_falloff}", file=stream)
    print(f"  Summary report: {output_files.summary_report}", file=stream)

    if preview_count > 0 and workflow_result.bdamage_workflow_result is not None:
        bdamage_result = workflow_result.bdamage_workflow_result
        packing_density_counts = packing_density_counts_as_tuple(
            bdamage_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(bdamage_result.bdamage_score_result)[
            :preview_count
        ]
        print(file=stream)
        print("BDamage preview:", file=stream)
        print(f"  Packing-density counts, first {preview_count}:", file=stream)
        print(f"    {_format_int_values(packing_density_counts)}", file=stream)
        print(f"  BDamage scores, first {preview_count}:", file=stream)
        print(f"    {_format_float_values(bdamage_scores)}", file=stream)

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


def print_bdamage_parameter_summary(
    workflow_result: BDamageWorkflowResult | None,
    *,
    stream: TextIO,
) -> None:
    """Print the BDamage calculation parameters actually used."""

    print("BDamage calculation:", file=stream)
    if workflow_result is None:
        print("  Parameters: not recorded", file=stream)
        return

    options = workflow_result.options
    print(
        f"  Packing density threshold: {options.packing_density_threshold:g} A",
        file=stream,
    )
    print(f"  Window size fraction: {options.window_size_fraction:g}", file=stream)
    print(
        f"  Window size used: {workflow_result.window_size} atoms "
        f"(minimum {options.minimum_window_size})",
        file=stream,
    )
    print(
        f"  Unit-cell translation range: {options.translation_range}",
        file=stream,
    )


def print_crystal_summary(
    crystal: ValidatedCrystal | None,
    *,
    stream: TextIO,
) -> None:
    """Print the validated crystal and asymmetric-unit content."""

    print("Crystal:", file=stream)
    if crystal is None:
        print("  Not validated", file=stream)
        return

    print(f"  Space group: {crystal.space_group_name}", file=stream)
    print(
        f"  Asymmetric units per unit cell: {crystal.operation_count}",
        file=stream,
    )
    content = crystal.content
    if content is None:
        print("  Asymmetric-unit content: not calculated", file=stream)
        return

    print(
        f"  Matthews coefficient: {content.matthews_coefficient:.2f} A^3/Da",
        file=stream,
    )
    print(
        f"  Estimated solvent content: {100.0 * content.solvent_fraction:.1f}%",
        file=stream,
    )
    for warning in crystal.warnings:
        print(f"  WARNING: {warning}", file=stream)


def print_bnet_summary(
    workflow_result: BnetWorkflowResult | None,
    *,
    stream: TextIO,
) -> None:
    """Print the Bnet outputs from the RABDAM workflow."""

    print("RABDAM Bnet results:", file=stream)
    if workflow_result is None:
        print("  Bnet: not calculated", file=stream)
        print("  Bnet percentile: not calculated", file=stream)
        return

    bnet_result = workflow_result.bnet_result
    if bnet_result is None:
        reason = workflow_result.bnet_unavailable_reason or "not calculated"
        print(f"  Bnet: unavailable ({reason})", file=stream)
        print("  Bnet percentile: unavailable (raw Bnet is unavailable)", file=stream)
        return

    print(f"  Protein Bnet sites: {bnet_result.site_count}", file=stream)
    print(f"  Raw protein Bnet: {bnet_result.bnet:.4f}", file=stream)
    percentile = workflow_result.percentile_result
    if percentile is None:
        reason = workflow_result.percentile_unavailable_reason or "not calculated"
        print(f"  Bnet percentile: unavailable ({reason})", file=stream)
        return

    print(f"  Bnet percentile: {percentile.percentile_percent:.2f}%", file=stream)
    print(
        f"  Bnet reference: {percentile.reference_database_id} "
        f"({percentile.reference_entry_count} entries)",
        file=stream,
    )
    print(f"  Resolution used: {percentile.resolution_angstrom:.3g} A", file=stream)
    print(
        "  Local reference set: "
        f"{percentile.local_reference_count} entries, "
        f"{percentile.local_resolution_min:.3g}-"
        f"{percentile.local_resolution_max:.3g} A",
        file=stream,
    )


def print_epath_batch_summary(
    result: EpathBatchCliResult,
    *,
    stream: TextIO,
) -> None:
    """Print a compact full E-path batch-run summary."""

    print(file=stream)
    print("Batch summary:", file=stream)
    print(f"  Successful: {len(result.results)}", file=stream)
    print(f"  Failed: {len(result.errors)}", file=stream)
    print(f"  Output directory: {result.output_dir}", file=stream)
    if result.cohort_csv is not None:
        print(f"  Cohort CSV: {result.cohort_csv}", file=stream)

    if result.errors:
        print("  Failed inputs:", file=stream)
        for item_error in result.errors:
            print(f"    {item_error.structure_input}", file=stream)

    print(f"  Total runtime: {result.elapsed_seconds:.1f}s", file=stream)


def _format_int_values(values: Sequence[int]) -> str:
    """Return integer preview values as a comma-separated string."""

    return ", ".join(str(value) for value in values)


def _format_float_values(values: Sequence[float]) -> str:
    """Return float preview values as a comma-separated string."""

    return ", ".join(f"{value:.3f}" for value in values)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the E-path command-line interface."""

    start = perf_counter()
    args = parse_args(argv)

    if args.citations:
        print_citations(stream=stdout)
        return 0

    if not args.structure_inputs:
        print_missing_structure_input_error(stream=stderr)
        return 2

    try:
        result = run_from_args(
            args,
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=start,
        )
    except RABDAM_CLI_ERRORS as error:
        print(f"epath: error: {error}", file=stderr)
        return 1

    if isinstance(result, EpathBatchCliResult) and result.has_errors:
        return 1

    return 0


def _float_arg(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from error


def _int_arg(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
