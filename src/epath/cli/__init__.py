"""Command-line interface for E-path, with `main` as its entry point.

- `args` parses arguments into workflow options.
- `run` resolves inputs and runs the RABDAM and E-path stages.
- `report` prints the terminal summaries.
- `style` colours and wraps terminal output.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from time import perf_counter
from typing import TextIO

from epath.cli.args import parse_args
from epath.cli.report import (
    print_citations,
    print_error,
    print_missing_structure_input_error,
)
from epath.cli.run import (
    RABDAM_CLI_ERRORS,
    EpathBatchCliResult,
    EpathBatchItemError,
    EpathCliResult,
    run_from_args,
)

__all__ = [
    "RABDAM_CLI_ERRORS",
    "EpathBatchCliResult",
    "EpathBatchItemError",
    "EpathCliResult",
    "main",
    "parse_args",
    "run_from_args",
]


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
        print_error(f"epath: error: {error}", stream=stderr)
        return 1

    if isinstance(result, EpathBatchCliResult) and result.has_errors:
        return 1

    return 0
