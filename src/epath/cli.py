"""Command-line interface for E-path."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, TextIO, TypeVar

from rabdam.bdamage.score import (
    BDamageScoreError,
    bdamage_scores_as_tuple,
)
from epath import __version__
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    BDamageWorkflowResult,
    calculate_bdamage_for_structure_data,
)
from rabdam.crystal.symmetry import CrystalSymmetryError
from rabdam.crystal.translate import CrystalTranslationError
from rabdam.crystal.trim import CrystalTrimError
from rabdam.input.rcsb import DEFAULT_RCSB_DOWNLOAD_DIR, ensure_local_structure_file
from rabdam.input.reader import read_structure
from rabdam.input.resolver import ResolvedStructureInput, resolve_structure_input
from rabdam.packing.density import (
    PackingDensityError,
    packing_density_counts_as_tuple,
)
from rabdam.structure.models import StructurePreparationOptions
from epath.workflow import (
    EpathAnnotationWorkflowOptions,
    EpathClusterWorkflowOptions,
    EpathNullModelWorkflowOptions,
    EpathOutputWorkflowOptions,
    EpathPathWorkflowOptions,
    EpathRateWorkflowOptions,
    EpathSelectionWorkflowOptions,
    EpathWorkflowOptions,
    EpathWorkflowResult,
    SelectionMethod,
    analyze_epath_atoms,
)
from epath.plots import EpathPlotError
from epath.atoms import atoms_from_bdamage_workflow


DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_CSV_PATTERN = "output/<structure>_BDamage.csv"
DEFAULT_DOWNLOAD_DIR = DEFAULT_RCSB_DOWNLOAD_DIR
CITATIONS_MESSAGE = """Please cite:

RABDAM:
  Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
  J Appl Cryst 51, 552-559.

BDamage:
  Gerstel M, Deane CM & Garman EF (2015).
  J Synchrotron Radiat 22, 201-212."""
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

CSV_FIELDNAMES = [
    "REC",
    "ATMNUM",
    "ATMNAME",
    "CONFORMER",
    "RESNAME",
    "CHAIN",
    "RESNUM",
    "INSCODE",
    "XPOS",
    "YPOS",
    "ZPOS",
    "OCC",
    "BFAC",
    "ELEMENT",
    "CHARGE",
    "PD",
    "AVRG_BF",
    "BDAM",
]

RABDAM_CLI_ERRORS = (
    BDamageScoreError,
    BDamageWorkflowError,
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
class RabdamCliResult:
    """Result returned by the CLI runner."""

    local_input: ResolvedStructureInput
    output_csv: Path
    workflow_result: BDamageWorkflowResult


@dataclass(frozen=True, slots=True)
class RabdamBatchItemError:
    """Failure for one input in a batch run."""

    structure_input: str
    error: Exception


@dataclass(frozen=True, slots=True)
class RabdamBatchCliResult:
    """Result returned by a batch CLI run."""

    results: tuple[RabdamCliResult, ...]
    errors: tuple[RabdamBatchItemError, ...]
    output_dir: Path
    elapsed_seconds: float

    @property
    def has_errors(self) -> bool:
        """Return whether any batch item failed."""

        return bool(self.errors)


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

    @property
    def has_errors(self) -> bool:
        """Return whether any batch item failed."""

        return bool(self.errors)


def _build_parser() -> argparse.ArgumentParser:
    """Build the default E-path command-line parser."""

    return _build_epath_parser()


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
    epath_path = parser.add_argument_group("E-path path options")
    epath_optional = parser.add_argument_group(
        "E-path annotation and statistics options"
    )
    epath_advanced = parser.add_argument_group("advanced E-path parameters")
    bdamage = parser.add_argument_group("advanced BDamage parameters")
    bdamage_selection = parser.add_argument_group("advanced BDamage selection options")
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
        help="Do not write selected/path PDB outputs.",
    )
    output.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the BDamage Z-score histogram plot.",
    )
    output.add_argument(
        "--structure-b-factor",
        choices=("bdamage", "b_factor", "z_score"),
        default=epath_defaults.output.structure_b_factor_mode,
        help=(
            "Value written to the B-factor column in E-path PDB outputs. "
            f"Default: {epath_defaults.output.structure_b_factor_mode}"
        ),
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
    epath_selection.add_argument(
        "--z-cutoff",
        metavar="FLOAT",
        type=finite_float,
        default=None,
        help="Select atoms with BDamage Z-score less than or equal to this value.",
    )
    epath_selection.add_argument(
        "--chains",
        metavar="CHAIN[,CHAIN...]",
        type=chain_selection,
        default=None,
        help=(
            "Restrict downstream E-path analysis to these comma-separated chain "
            "IDs (quoted whitespace-separated IDs are also accepted). BDamage is "
            "still calculated from the complete structure."
        ),
    )
    epath_path.add_argument(
        "--donor",
        metavar="CHAIN:RESNUM:ATOM",
        default=None,
        help="Donor endpoint atom for path analysis, for example A:42:CA.",
    )
    epath_path.add_argument(
        "--acceptor",
        metavar="CHAIN:RESNUM:ATOM",
        default=None,
        help="Acceptor endpoint atom for path analysis, for example A:108:CA.",
    )
    epath_optional.add_argument(
        "--annotations",
        action="store_true",
        help="Compute DSSP secondary structure and SASA annotations when available.",
    )
    epath_optional.add_argument(
        "--no-sasa",
        action="store_true",
        help="Skip SASA calculation when annotations are enabled.",
    )
    epath_optional.add_argument(
        "--null-model",
        action="store_true",
        help="Run the Monte Carlo null model for cluster statistics.",
    )
    epath_advanced.add_argument(
        "--histogram-bins",
        metavar="INT",
        type=positive_int,
        default=epath_defaults.selection.histogram_bins,
        help=(
            "Number of bins for BDamage Z-score histogram summaries. "
            f"Default: {epath_defaults.selection.histogram_bins}"
        ),
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
        "--keep-largest-cluster",
        action="store_true",
        help="Keep only the largest selected-atom cluster in E-path outputs.",
    )
    epath_advanced.add_argument(
        "--edge-distance",
        metavar="ANGSTROM",
        type=positive_float,
        default=epath_defaults.path.edge_distance,
        help=(
            "Maximum atom-to-atom graph edge distance for path analysis. "
            f"Default: {epath_defaults.path.edge_distance}"
        ),
    )
    epath_advanced.add_argument(
        "--snap-radius",
        metavar="ANGSTROM",
        type=non_negative_float,
        default=epath_defaults.path.snap_radius,
        help=(
            "Maximum distance for snapping donor/acceptor endpoints to selected atoms. "
            f"Default: {epath_defaults.path.snap_radius}"
        ),
    )
    epath_advanced.add_argument(
        "--lambda-ev",
        metavar="EV",
        type=positive_float,
        default=epath_defaults.rates.reorganization_energy_ev,
        help=(
            "Reorganization energy used for path rate calculation. "
            f"Default: {epath_defaults.rates.reorganization_energy_ev}"
        ),
    )
    epath_advanced.add_argument(
        "--delta-g-ev",
        metavar="EV",
        type=finite_float,
        default=epath_defaults.rates.delta_g_ev,
        help=(
            "Reaction free energy used for path rate calculation. "
            f"Default: {epath_defaults.rates.delta_g_ev}"
        ),
    )
    epath_advanced.add_argument(
        "--temperature-k",
        metavar="K",
        type=positive_float,
        default=epath_defaults.rates.temperature_k,
        help=(
            "Temperature used for path rate calculation. "
            f"Default: {epath_defaults.rates.temperature_k}"
        ),
    )
    epath_advanced.add_argument(
        "--beta-ainv",
        "--beta-Ainv",
        dest="beta_per_angstrom",
        metavar="FLOAT",
        type=positive_float,
        default=epath_defaults.rates.beta_per_angstrom,
        help=(
            "Distance-decay beta value in inverse Angstroms. "
            f"Default: {epath_defaults.rates.beta_per_angstrom}"
        ),
    )
    epath_advanced.add_argument(
        "--null-iterations",
        metavar="INT",
        type=positive_int,
        default=epath_defaults.null_model.iterations,
        help=(
            "Monte Carlo iterations when --null-model is enabled. "
            f"Default: {epath_defaults.null_model.iterations}"
        ),
    )
    epath_advanced.add_argument(
        "--null-seed",
        metavar="INT",
        type=_int_arg,
        default=epath_defaults.null_model.seed,
        help=(
            "Random seed for the Monte Carlo null model. "
            f"Default: {epath_defaults.null_model.seed}"
        ),
    )
    _add_bdamage_calculation_arguments(bdamage, defaults=bdamage_defaults)
    _add_bdamage_selection_arguments(bdamage_selection)
    _add_bdamage_debug_arguments(bdamage_debug)
    _add_download_arguments(download)
    _add_general_arguments(general)

    return parser


def _build_bdamage_parser() -> argparse.ArgumentParser:
    """Build the BDamage-only command-line parser."""

    defaults = BDamageWorkflowOptions()
    parser = argparse.ArgumentParser(
        prog="epath bdamage",
        usage="%(prog)s STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]",
        description="Run only the BDamage workflow for structure files or PDB IDs.",
        add_help=False,
    )
    input_group = parser.add_argument_group("input")
    output_download = parser.add_argument_group("output and download options")
    bdamage = parser.add_argument_group("BDamage calculation parameters")
    selection = parser.add_argument_group("BDamage selection options")
    advanced = parser.add_argument_group("advanced/debugging options")
    general = parser.add_argument_group("general options")

    _add_structure_input_argument(input_group)
    output_download.add_argument(
        "-o",
        "--output-csv",
        metavar="PATH",
        default=None,
        help=(
            "Path for the per-atom BDamage CSV in a single-structure run. "
            f"Default: {DEFAULT_OUTPUT_CSV_PATTERN}"
        ),
    )
    output_download.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for default single-run output and batch output CSVs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    _add_download_arguments(output_download)
    _add_bdamage_calculation_arguments(bdamage, defaults=defaults)
    _add_bdamage_selection_arguments(selection)
    _add_bdamage_debug_arguments(advanced)
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


def _add_bdamage_selection_arguments(group: Any) -> None:
    group.add_argument(
        "--keep-hydrogens",
        action="store_true",
        help="Keep hydrogen atoms during structure preparation.",
    )
    group.add_argument(
        "--include-hetatm",
        action="store_true",
        help="Include HETATM records in the BDamage atom selection.",
    )
    group.add_argument(
        "--include-nucleic-acid",
        action="store_true",
        help="Include nucleic-acid atoms in the BDamage atom selection.",
    )
    group.add_argument(
        "--remove-atom-serial",
        metavar="INT",
        action="append",
        default=[],
        type=positive_int,
        help="Remove an atom serial from the BDamage selection. May be used repeatedly.",
    )
    group.add_argument(
        "--add-atom-serial",
        metavar="INT",
        action="append",
        default=[],
        type=positive_int,
        help="Add an atom serial to the BDamage selection. May be used repeatedly.",
    )
    group.add_argument(
        "--remove-component",
        metavar="NAME",
        action="append",
        default=[],
        type=component_name,
        help="Remove a residue/component name from the BDamage selection.",
    )
    group.add_argument(
        "--add-component",
        metavar="NAME",
        action="append",
        default=[],
        type=component_name,
        help="Add a residue/component name to the BDamage selection.",
    )


def _add_bdamage_debug_arguments(group: Any) -> None:
    group.add_argument(
        "--materialize-translated-block",
        action="store_true",
        help="Build the full translated crystal block for debugging.",
    )
    group.add_argument(
        "--allow-non-protein-selection",
        action="store_true",
        help="Allow BDamage scoring when the selected atoms contain no protein atoms.",
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
        parser = _build_bdamage_parser()
        args = parser.parse_args(raw_args[1:])
        args.command = "bdamage"
    else:
        parser = _build_epath_parser()
        args = parser.parse_args(raw_args)
        args.command = "epath"

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


def finite_float(value: str) -> float:
    """Return a finite float for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {value!r}")
    return parsed


def non_negative_float(value: str) -> float:
    """Return a non-negative finite float for argparse."""

    parsed = _float_arg(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
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


def component_name(value: str) -> str:
    """Return a normalized residue/component name for argparse."""

    cleaned = value.strip().upper()
    if not cleaned:
        raise argparse.ArgumentTypeError("component name cannot be empty")
    return cleaned


def chain_selection(value: str) -> tuple[str, ...]:
    """Return unique chain IDs from a comma- or space-separated CLI value."""

    chain_ids = tuple(dict.fromkeys(value.replace(",", " ").split()))
    if not chain_ids:
        raise argparse.ArgumentTypeError("at least one chain ID is required")
    return chain_ids


def workflow_options_from_args(args: argparse.Namespace) -> BDamageWorkflowOptions:
    """Build workflow options from parsed command-line arguments."""

    return BDamageWorkflowOptions(
        packing_density_threshold=args.packing_density_threshold,
        window_size_fraction=args.window_size_fraction,
        minimum_window_size=args.minimum_window_size,
        translation_range=args.translation_range,
        materialize_translated_block=args.materialize_translated_block,
    )


def preparation_options_from_args(
    args: argparse.Namespace,
) -> StructurePreparationOptions:
    """Build structure-preparation options from parsed command-line arguments."""

    return StructurePreparationOptions(
        remove_hydrogens=not args.keep_hydrogens,
        include_hetatm_in_selection=args.include_hetatm,
        include_nucleic_acid_in_selection=args.include_nucleic_acid,
        require_protein_selection=not args.allow_non_protein_selection,
        remove_atom_serials=frozenset(args.remove_atom_serial),
        add_atom_serials=frozenset(args.add_atom_serial),
        remove_component_names=frozenset(args.remove_component),
        add_component_names=frozenset(args.add_component),
    )


def epath_workflow_options_from_args(
    args: argparse.Namespace,
    *,
    output_prefix: Path | None,
) -> EpathWorkflowOptions:
    """Build full E-path workflow options from parsed command-line arguments."""

    if args.z_cutoff is not None and args.per_chain_percentile:
        raise ValueError("--z-cutoff cannot be combined with --per-chain-percentile.")

    selection_method: SelectionMethod = "percentile"
    if args.z_cutoff is not None:
        selection_method = "z_cutoff"
    elif args.per_chain_percentile:
        selection_method = "percentile_per_chain"

    return EpathWorkflowOptions(
        allowed_chains=args.chains,
        selection=EpathSelectionWorkflowOptions(
            method=selection_method,
            z_cutoff=0.0 if args.z_cutoff is None else args.z_cutoff,
            percentile=args.percentile,
            histogram_bins=args.histogram_bins,
        ),
        clustering=EpathClusterWorkflowOptions(
            distance_cutoff=args.cluster_distance,
            keep_largest_only=args.keep_largest_cluster,
        ),
        path=EpathPathWorkflowOptions(
            donor_token=args.donor,
            acceptor_token=args.acceptor,
            edge_distance=args.edge_distance,
            snap_radius=args.snap_radius,
        ),
        rates=EpathRateWorkflowOptions(
            reorganization_energy_ev=args.lambda_ev,
            delta_g_ev=args.delta_g_ev,
            temperature_k=args.temperature_k,
            beta_per_angstrom=args.beta_per_angstrom,
        ),
        annotations=EpathAnnotationWorkflowOptions(
            enabled=args.annotations,
            disable_sasa=args.no_sasa,
        ),
        null_model=EpathNullModelWorkflowOptions(
            enabled=args.null_model,
            iterations=args.null_iterations,
            seed=args.null_seed,
        ),
        output=EpathOutputWorkflowOptions(
            output_prefix=output_prefix,
            write_combined_csv=not args.no_csv,
            write_structure_files=not args.no_pdb,
            write_z_score_plot=not args.no_plot,
            structure_b_factor_mode=args.structure_b_factor,
        ),
    )


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    total_runtime_start: float | None = None,
) -> (
    RabdamCliResult
    | RabdamBatchCliResult
    | EpathCliResult
    | EpathBatchCliResult
):
    """Run the requested workflow from parsed command-line arguments."""

    if args.command == "bdamage":
        return run_bdamage_from_args(
            args,
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=total_runtime_start,
        )

    return run_epath_from_args(
        args,
        stdout=stdout,
        stderr=stderr,
        total_runtime_start=total_runtime_start,
    )


def run_bdamage_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    total_runtime_start: float | None = None,
) -> RabdamCliResult | RabdamBatchCliResult:
    """Run the BDamage workflow from parsed command-line arguments."""

    if total_runtime_start is None:
        total_runtime_start = perf_counter()

    structure_inputs = structure_inputs_from_args(args)
    if not structure_inputs:
        raise ValueError("At least one structure input is required.")

    if len(structure_inputs) == 1:
        return run_single_structure_from_args(
            args,
            structure_input=structure_inputs[0],
            output_csv=single_output_csv_from_args(args),
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=total_runtime_start,
        )

    return run_batch_from_args(
        args,
        structure_inputs=structure_inputs,
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
    preparation_options = preparation_options_from_args(args)

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
        stage_runner=lambda label, callback: run_stage(
            label,
            callback,
            stream=stderr,
            quiet=args.quiet,
        ),
    )

    return local_input, workflow_result


def run_single_structure_from_args(
    args: argparse.Namespace,
    *,
    structure_input: str,
    output_csv: Path | None,
    stdout: TextIO,
    stderr: TextIO,
    total_runtime_start: float,
    used_batch_output_paths: set[Path] | None = None,
) -> RabdamCliResult:
    """Run the BDamage workflow for one structure input."""

    local_input, workflow_result = calculate_bdamage_workflow_from_args(
        args,
        structure_input=structure_input,
        stderr=stderr,
    )

    if output_csv is None:
        output_csv = batch_output_csv_path(
            local_input,
            output_dir=Path(args.output_dir).expanduser(),
            used_output_paths=used_batch_output_paths,
        )

    run_stage(
        "Writing BDamage CSV",
        lambda: write_bdamage_csv(
            output_path=output_csv,
            prepared_structure=workflow_result.prepared_structure,
            bdamage_score_result=workflow_result.bdamage_score_result,
        ),
        stream=stderr,
        quiet=args.quiet,
    )

    cli_result = RabdamCliResult(
        local_input=local_input,
        output_csv=output_csv,
        workflow_result=workflow_result,
    )

    if not args.quiet:
        total_runtime_seconds = perf_counter() - total_runtime_start
        print_summary(
            cli_result,
            preview_count=args.preview_count,
            total_runtime_seconds=total_runtime_seconds,
            stream=stdout,
        )

    return cli_result


def run_batch_from_args(
    args: argparse.Namespace,
    *,
    structure_inputs: tuple[str, ...],
    stdout: TextIO,
    stderr: TextIO,
    total_runtime_start: float,
) -> RabdamBatchCliResult:
    """Run the BDamage workflow for multiple structure inputs."""

    if args.output_csv is not None:
        raise ValueError(
            "--output-csv can only be used with one structure input. "
            "Use --output-dir for batch runs."
        )

    output_dir = Path(args.output_dir).expanduser()
    results: list[RabdamCliResult] = []
    errors: list[RabdamBatchItemError] = []
    used_output_paths: set[Path] = set()

    for index, structure_input in enumerate(structure_inputs, start=1):
        if not args.quiet:
            print(
                f"\nBatch item {index}/{len(structure_inputs)}: {structure_input}",
                file=stderr,
            )

        try:
            result = run_single_structure_from_args(
                args,
                structure_input=structure_input,
                output_csv=None,
                stdout=stdout,
                stderr=stderr,
                total_runtime_start=perf_counter(),
                used_batch_output_paths=used_output_paths,
            )
        except RABDAM_CLI_ERRORS as error:
            errors.append(
                RabdamBatchItemError(
                    structure_input=structure_input,
                    error=error,
                )
            )
            print(f"epath: error: {structure_input}: {error}", file=stderr)
        else:
            results.append(result)

    elapsed_seconds = perf_counter() - total_runtime_start
    batch_result = RabdamBatchCliResult(
        results=tuple(results),
        errors=tuple(errors),
        output_dir=output_dir,
        elapsed_seconds=elapsed_seconds,
    )

    if not args.quiet:
        print_batch_summary(batch_result, stream=stdout)

    return batch_result


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
    warning_handler: Callable[[str], None] | None = None
    if not args.quiet:
        warning_handler = lambda message: print(
            f"epath: warning: {message}",
            file=stderr,
        )

    workflow_result = run_stage(
        "Running E-path workflow",
        lambda: analyze_epath_atoms(
            atoms,
            structure_path=local_input.local_path,
            options=options,
            warn=warning_handler,
            bdamage_workflow_result=bdamage_result,
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

    elapsed_seconds = perf_counter() - total_runtime_start
    batch_result = EpathBatchCliResult(
        results=tuple(results),
        errors=tuple(errors),
        output_dir=output_dir,
        elapsed_seconds=elapsed_seconds,
    )

    if not args.quiet:
        print_epath_batch_summary(batch_result, stream=stdout)

    return batch_result


def single_output_csv_from_args(args: argparse.Namespace) -> Path | None:
    """Return the output CSV path for a single-structure run."""

    if args.output_csv is not None:
        return Path(args.output_csv).expanduser()

    return None


def single_output_prefix_from_args(args: argparse.Namespace) -> Path | None:
    """Return the E-path output prefix for a single-structure run."""

    if args.output_prefix is not None:
        return Path(args.output_prefix).expanduser()

    return None


def batch_output_csv_path(
    local_input: ResolvedStructureInput,
    *,
    output_dir: Path,
    used_output_paths: set[Path] | None,
) -> Path:
    """Return a unique batch output CSV path for one resolved input."""

    label = output_label_for_input(local_input)
    candidate = output_dir / f"{label}_BDamage.csv"
    if used_output_paths is None:
        return candidate

    suffix = 2
    while candidate in used_output_paths:
        candidate = output_dir / f"{label}_{suffix}_BDamage.csv"
        suffix += 1

    used_output_paths.add(candidate)
    return candidate


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


def write_bdamage_csv(
    *,
    output_path: Path,
    prepared_structure,
    bdamage_score_result,
) -> None:
    """Write per-atom BDamage results as a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for atom, score in zip(
            prepared_structure.selected_atoms,
            bdamage_score_result.atom_results,
            strict=True,
        ):
            record = atom.record
            writer.writerow(
                {
                    "REC": record.record_type,
                    "ATMNUM": record.atom_serial,
                    "ATMNAME": record.atom_name,
                    "CONFORMER": record.altloc or ".",
                    "RESNAME": record.residue_name,
                    "CHAIN": record.chain_id or ".",
                    "RESNUM": record.residue_number,
                    "INSCODE": record.insertion_code or ".",
                    "XPOS": record.x,
                    "YPOS": record.y,
                    "ZPOS": record.z,
                    "OCC": record.occupancy,
                    "BFAC": record.b_factor,
                    "ELEMENT": record.element,
                    "CHARGE": "?",
                    "PD": score.packing_density,
                    "AVRG_BF": score.average_b_factor,
                    "BDAM": score.bdamage,
                }
            )


def print_summary(
    result: RabdamCliResult,
    *,
    preview_count: int,
    total_runtime_seconds: float,
    stream: TextIO,
) -> None:
    """Print a compact successful-run summary."""

    workflow_result = result.workflow_result
    print(file=stream)
    print("Done.", file=stream)
    print(file=stream)
    print(f"Input: {result.local_input.local_path}", file=stream)
    print(f"Output CSV: {result.output_csv}", file=stream)
    print(file=stream)
    print("Crystallographic expansion:", file=stream)
    print(
        "  Symmetry-expanded atoms: "
        f"{len(workflow_result.symmetry_expanded_structure.atoms)}",
        file=stream,
    )
    print(
        "  Translated atoms before trimming: "
        f"{workflow_result.trimmed_block.original_atom_count}",
        file=stream,
    )
    print(
        "  Neighbour-block atoms after trimming: "
        f"{workflow_result.trimmed_block.atom_count}",
        file=stream,
    )
    print(file=stream)
    print("BDamage summary:", file=stream)
    print(
        f"  Selected atoms: {workflow_result.prepared_structure.report.selected_atom_count}",
        file=stream,
    )
    print(f"  Window size: {workflow_result.window_size}", file=stream)

    if preview_count > 0:
        packing_density_counts = packing_density_counts_as_tuple(
            workflow_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(
            workflow_result.bdamage_score_result
        )[:preview_count]
        print(file=stream)
        print("Preview:", file=stream)
        print(f"  Packing-density counts, first {preview_count}:", file=stream)
        print(f"    {_format_int_values(packing_density_counts)}", file=stream)
        print(f"  BDamage scores, first {preview_count}:", file=stream)
        print(f"    {_format_float_values(bdamage_scores)}", file=stream)

    if workflow_result.translated_block is not None:
        print(file=stream)
        print("Debug:", file=stream)
        print("  Translated block materialized: True", file=stream)

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


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
    print(f"  Z-score mode: {selection_result.z_mode:.3f}", file=stream)
    print(file=stream)
    print("Clusters:", file=stream)
    print(f"  Count: {len(workflow_result.clusters)}", file=stream)
    largest_cluster = max(
        workflow_result.clusters,
        key=lambda cluster: cluster.atom_count,
        default=None,
    )
    if largest_cluster is not None:
        print(f"  Largest atom count: {largest_cluster.atom_count}", file=stream)
        print(
            f"  Largest MST length: {largest_cluster.metrics.mst_total_length:.3f} A",
            file=stream,
        )

    if workflow_result.path is not None:
        print(file=stream)
        print("Path:", file=stream)
        print(f"  Atoms: {workflow_result.path.atom_count}", file=stream)
        print(f"  Steps: {workflow_result.path.step_count}", file=stream)
        print(f"  Total length: {workflow_result.path.total_length:.3f} A", file=stream)
        if workflow_result.marcus_rate is not None:
            print(f"  Rate proxy: {workflow_result.marcus_rate.rate:.6g}", file=stream)

    print(file=stream)
    print("Outputs:", file=stream)
    print(f"  Combined CSV: {output_files.combined_csv}", file=stream)
    print(f"  Selected PDB: {output_files.selected_pdb}", file=stream)
    print(f"  Path PDB: {output_files.path_pdb}", file=stream)
    print(f"  Z-score plot: {output_files.z_score_histogram}", file=stream)

    if workflow_result.warnings:
        print(file=stream)
        print("Warnings:", file=stream)
        for warning in workflow_result.warnings:
            print(f"  {warning}", file=stream)

    if preview_count > 0 and workflow_result.bdamage_workflow_result is not None:
        bdamage_result = workflow_result.bdamage_workflow_result
        packing_density_counts = packing_density_counts_as_tuple(
            bdamage_result.packing_density_result
        )[:preview_count]
        bdamage_scores = bdamage_scores_as_tuple(
            bdamage_result.bdamage_score_result
        )[:preview_count]
        print(file=stream)
        print("BDamage preview:", file=stream)
        print(f"  Packing-density counts, first {preview_count}:", file=stream)
        print(f"    {_format_int_values(packing_density_counts)}", file=stream)
        print(f"  BDamage scores, first {preview_count}:", file=stream)
        print(f"    {_format_float_values(bdamage_scores)}", file=stream)

    print(file=stream)
    print(f"Total runtime: {total_runtime_seconds:.1f}s", file=stream)


def print_batch_summary(
    result: RabdamBatchCliResult,
    *,
    stream: TextIO,
) -> None:
    """Print a compact batch-run summary."""

    print(file=stream)
    print("Batch summary:", file=stream)
    print(f"  Successful: {len(result.results)}", file=stream)
    print(f"  Failed: {len(result.errors)}", file=stream)
    print(f"  Output directory: {result.output_dir}", file=stream)

    if result.errors:
        print("  Failed inputs:", file=stream)
        for item_error in result.errors:
            print(f"    {item_error.structure_input}", file=stream)

    print(f"  Total runtime: {result.elapsed_seconds:.1f}s", file=stream)


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

    if (
        isinstance(result, (RabdamBatchCliResult, EpathBatchCliResult))
        and result.has_errors
    ):
        return 1

    return 0


def _float_arg(value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a number, got {value!r}"
        ) from error


def _int_arg(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
