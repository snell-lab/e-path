"""Command-line argument parsing for E-path.

This module owns the argparse parser, the argument validators, and the
translation from a parsed namespace into pipeline option objects. It does not
run anything. `epath.cli.run` takes the namespace and the options from here.

The validators reject an out-of-range value while the arguments are parsed, so
argparse reports it as a usage error. This is the only place the ranges are
checked: the pipelines take the option objects as they are.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from epath import __version__
from epath.outputs.files import EpathOutputPipelineOptions
from epath.pipeline import (
    EpathClusterPipelineOptions,
    EpathPipelineOptions,
    EpathSelectionPipelineOptions,
)
from rabdam.pipeline import BDamagePipelineOptions
from xtal.rcsb import DEFAULT_RCSB_DOWNLOAD_DIR
from xtal.resolver import COMPRESSED_SUFFIX, SUPPORTED_STRUCTURE_SUFFIXES

DEFAULT_OUTPUT_DIR = "output"


def build_parser() -> argparse.ArgumentParser:
    """Build the full E-path pipeline parser.

    The defaults of the calculation options are read from the pipeline option
    classes, so the command line and the library cannot drift apart.
    """
    bdamage_defaults = BDamagePipelineOptions()
    epath_defaults = EpathPipelineOptions()
    parser = argparse.ArgumentParser(
        prog="epath",
        usage="%(prog)s STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]",
        description="Run the full E-path pipeline for structure files or PDB IDs.",
        # `-h` is added by hand so that it is listed under the general options.
        add_help=False,
    )
    # The help lists the groups in the order they are created here.
    input_group = parser.add_argument_group("input")
    output = parser.add_argument_group("output options")
    epath_selection = parser.add_argument_group("E-path selection options")
    epath_advanced = parser.add_argument_group("advanced E-path parameters")
    bdamage = parser.add_argument_group("advanced BDamage parameters")
    download = parser.add_argument_group("download options")
    general = parser.add_argument_group("general options")

    _add_structure_input_argument(input_group)
    _add_output_arguments(output)
    _add_epath_selection_arguments(epath_selection, defaults=epath_defaults)
    _add_epath_advanced_arguments(epath_advanced, defaults=epath_defaults)
    _add_bdamage_calculation_arguments(bdamage, defaults=bdamage_defaults)
    _add_download_arguments(download)
    _add_general_arguments(general)

    return parser


# The groups below are typed `Any` because argparse does not export the class
# of an argument group.
def _add_structure_input_argument(group: Any) -> None:
    group.add_argument(
        "structure_inputs",
        # Not required here, so that `--citations` works without an input.
        # `epath.cli.main` reports a missing input itself.
        nargs="*",
        metavar="STRUCTURE_INPUT",
        help=(
            f"One or more paths to {'/'.join(SUPPORTED_STRUCTURE_SUFFIXES)} "
            f"files, optionally gzipped ({COMPRESSED_SUFFIX}), or 4-character "
            "PDB IDs such as 1ABC."
        ),
    )


def _add_output_arguments(group: Any) -> None:
    group.add_argument(
        "-o",
        "--output-prefix",
        metavar="PREFIX",
        help=(
            "Prefix for E-path output files in a single-structure run. "
            "Default: <structure>/<structure> under --output-dir, in a "
            "numbered directory if that one exists."
        ),
    )
    group.add_argument(
        "--output-dir",
        metavar="DIR",
        default=DEFAULT_OUTPUT_DIR,
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
        help=(
            "Comma-separated chain identifiers to keep in the mmCIF outputs, for "
            "example A,B. The analysis itself always covers every chain; this "
            "only trims the coordinate files. Default: all chains"
        ),
    )


def _add_epath_selection_arguments(
    group: Any,
    *,
    defaults: EpathPipelineOptions,
) -> None:
    group.add_argument(
        "--percentile",
        metavar="FLOAT[,FLOAT...]",
        type=percentile_list,
        default=(defaults.selection.percentile,),
        help=(
            "Lowest BDamage percentile to select for E-path analysis. Several "
            "comma-separated values, for example 1,2,3, run the analysis at "
            "each from one BDamage calculation, with the outputs of each in a "
            "directory of their own. "
            f"Default: {defaults.selection.percentile}"
        ),
    )


def _add_epath_advanced_arguments(
    group: Any,
    *,
    defaults: EpathPipelineOptions,
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
        default=str(DEFAULT_RCSB_DOWNLOAD_DIR),
        help=(
            "Directory for downloaded RCSB mmCIF files. "
            f"Default: {DEFAULT_RCSB_DOWNLOAD_DIR}"
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
    defaults: BDamagePipelineOptions,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, taken from `sys.argv` when `argv` is None.

    `SystemExit` is raised by argparse for invalid arguments, `--help` and
    `--version`.
    """
    return build_parser().parse_args(argv)


def chain_list(value: str) -> tuple[str, ...]:
    """Return the chain identifiers of a comma-separated list.

    Spaces around an identifier are dropped, and a repeated identifier is kept
    once, at its first position. An empty identifier, as in `A,,B`, is
    rejected.
    """
    chains = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    if any(not chain for chain in chains):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated chain identifiers, got {value!r}"
        )
    return chains


def percentile_list(value: str) -> tuple[float, ...]:
    """Return the percentages of a comma-separated list, each from 0 to 100.

    Spaces around a value are dropped, and a repeated value is kept once, at
    its first position. An empty value, as in `1,,2`, is rejected.
    """
    items = [item.strip() for item in value.split(",")]
    if any(not item for item in items):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated percentages, got {value!r}"
        )
    return tuple(dict.fromkeys(percent_float(item) for item in items))


def positive_float(value: str) -> float:
    """Return a positive finite float for argparse."""
    parsed = _float_arg(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def fraction_float(value: str) -> float:
    """Return a float in the open interval 0 to 1 for argparse."""
    parsed = _float_arg(value)
    # Written as a negation so that NaN is rejected too.
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a number greater than 0 and less than 1, got {value!r}"
        )
    return parsed


def percent_float(value: str) -> float:
    """Return a percentage in the closed interval 0 to 100 for argparse."""
    parsed = _float_arg(value)
    # Written as a negation so that NaN is rejected too.
    if not 0.0 <= parsed <= 100.0:
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
    """Return the `--output-prefix` of a single-structure run, with `~` expanded.

    None means the option was not given, and the caller then derives a prefix
    under `--output-dir`.
    """
    if args.output_prefix is not None:
        return Path(args.output_prefix).expanduser()

    return None


def bdamage_pipeline_options_from_args(
    args: argparse.Namespace,
) -> BDamagePipelineOptions:
    """Build BDamage pipeline options from parsed command-line arguments."""
    return BDamagePipelineOptions(
        packing_density_threshold=args.packing_density_threshold,
        window_size_fraction=args.window_size_fraction,
        minimum_window_size=args.minimum_window_size,
        translation_range=args.translation_range,
    )


def epath_pipeline_options_from_args(
    args: argparse.Namespace,
    *,
    percentile: float,
    output_prefix: Path | None,
) -> EpathPipelineOptions:
    """Build the E-path pipeline options of one percentile of `--percentile`.

    `--no-reports` switches off the cluster, step and run CSVs and the summary
    report together.
    """
    return EpathPipelineOptions(
        selection=EpathSelectionPipelineOptions(
            percentile=percentile,
        ),
        clustering=EpathClusterPipelineOptions(
            distance_cutoff=args.cluster_distance,
            minimum_cluster_size=args.minimum_cluster_size,
        ),
        output=EpathOutputPipelineOptions(
            output_prefix=output_prefix,
            write_combined_csv=not args.no_csv,
            write_structure_files=not args.no_cif,
            write_z_score_plot=not args.no_plot,
            write_reports=not args.no_reports,
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
