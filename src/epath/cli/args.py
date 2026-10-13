"""Command-line argument parsing for E-path.

This module owns the argparse parser, the argument validators, and the
translation from a parsed namespace into workflow option objects. It does not
run anything.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from epath import __version__
from epath.outputs.files import EpathOutputWorkflowOptions
from epath.workflow import (
    EpathClusterWorkflowOptions,
    EpathSelectionWorkflowOptions,
    EpathWorkflowOptions,
)
from rabdam.workflow import BDamageWorkflowOptions
from xtal.rcsb import DEFAULT_RCSB_DOWNLOAD_DIR
from xtal.resolver import COMPRESSED_SUFFIX, SUPPORTED_STRUCTURE_SUFFIXES

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_DOWNLOAD_DIR = DEFAULT_RCSB_DOWNLOAD_DIR

REMOVED_BDAMAGE_COMMAND_MESSAGE = """\
epath: error: the 'bdamage' subcommand has been removed.

E-path scores protein atoms only, and that population is no longer
configurable. Run 'epath STRUCTURE_INPUT' instead; per-atom BDamage,
packing density, and window-averaged B-factors are written to the
combined '*_epath.csv' output."""


def build_parser() -> argparse.ArgumentParser:
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
    _add_output_arguments(output)
    _add_epath_selection_arguments(epath_selection, defaults=epath_defaults)
    _add_epath_advanced_arguments(epath_advanced, defaults=epath_defaults)
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
            f"One or more paths to {'/'.join(SUPPORTED_STRUCTURE_SUFFIXES)} "
            f"files, optionally gzipped ({COMPRESSED_SUFFIX}), or 4-character "
            "PDB IDs such as 1LYZ."
        ),
    )


def _add_output_arguments(group: Any) -> None:
    group.add_argument(
        "-o",
        "--output-prefix",
        metavar="PREFIX",
        default=None,
        help=(
            "Prefix for E-path output files in a single-structure run. "
            "Default: output/<structure>"
        ),
    )
    group.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for default single-run and batch E-path outputs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    group.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the combined E-path CSV output.",
    )
    group.add_argument(
        "--no-cif",
        action="store_true",
        help=(
            "Do not write the coordinate outputs: the selected-atom, cleaned "
            "model, and per-pathway mmCIF files."
        ),
    )
    group.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the BDamage Z-score falloff plot.",
    )
    group.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not write the cluster, step, and run CSVs or the summary report.",
    )
    group.add_argument(
        "--chains",
        metavar="IDS",
        type=chain_list,
        default=None,
        help=(
            "Comma-separated chain identifiers to keep in the mmCIF outputs, for "
            "example A,B. The analysis itself always covers every chain; this "
            "only trims the coordinate files. Default: all chains"
        ),
    )


def _add_epath_selection_arguments(
    group: Any,
    *,
    defaults: EpathWorkflowOptions,
) -> None:
    group.add_argument(
        "--percentile",
        metavar="FLOAT",
        type=percent_float,
        default=defaults.selection.percentile,
        help=(
            "Lowest BDamage percentile to select for E-path analysis. "
            f"Default: {defaults.selection.percentile}"
        ),
    )


def _add_epath_advanced_arguments(
    group: Any,
    *,
    defaults: EpathWorkflowOptions,
) -> None:
    group.add_argument(
        "--cluster-distance",
        metavar="ANGSTROM",
        type=positive_float,
        default=defaults.clustering.distance_cutoff,
        help=(
            "Distance cutoff for selected-atom clustering. "
            f"Default: {defaults.clustering.distance_cutoff}"
        ),
    )
    group.add_argument(
        "--minimum-cluster-size",
        metavar="INT",
        type=positive_int,
        default=defaults.clustering.minimum_cluster_size,
        help=(
            "Minimum connected-component size reported as an E-path cluster. "
            f"Default: {defaults.clustering.minimum_cluster_size}"
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
    """Parse command-line arguments.

    `SystemExit` is raised for the removed `bdamage` subcommand, as well as by
    argparse itself for invalid arguments, `--help` and `--version`.
    """
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "bdamage":
        raise SystemExit(REMOVED_BDAMAGE_COMMAND_MESSAGE)

    return build_parser().parse_args(raw_args)


def chain_list(value: str) -> tuple[str, ...]:
    """Return a tuple of unique chain identifiers from a comma-separated list."""
    chains = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    if any(not chain for chain in chains):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated chain identifiers, got {value!r}"
        )
    return chains


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


def single_output_prefix_from_args(args: argparse.Namespace) -> Path | None:
    """Return the E-path output prefix for a single-structure run."""
    if args.output_prefix is not None:
        return Path(args.output_prefix).expanduser()

    return None


def bdamage_workflow_options_from_args(
    args: argparse.Namespace,
) -> BDamageWorkflowOptions:
    """Build BDamage workflow options from parsed command-line arguments."""
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
    return EpathWorkflowOptions(
        selection=EpathSelectionWorkflowOptions(
            percentile=args.percentile,
        ),
        clustering=EpathClusterWorkflowOptions(
            distance_cutoff=args.cluster_distance,
            minimum_cluster_size=args.minimum_cluster_size,
        ),
        output=EpathOutputWorkflowOptions(
            output_prefix=output_prefix,
            write_combined_csv=not args.no_csv,
            write_structure_files=not args.no_cif,
            write_z_score_plot=not args.no_plot,
            write_cluster_csv=not args.no_reports,
            write_step_csv=not args.no_reports,
            write_run_csv=not args.no_reports,
            write_summary_report=not args.no_reports,
            display_chains=args.chains,
        ),
    )


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
