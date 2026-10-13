"""Command-line interface for E-path, with `main` as its entry point.

- `args` parses arguments into pipeline options.
- `run` resolves inputs and runs the RABDAM and E-path stages.
- `report` prints the terminal summaries.
- `style` colours and wraps terminal output.
"""

import sys
from collections.abc import Sequence
from time import perf_counter
from typing import TextIO

from epath.cli.args import parse_args
from epath.cli.report import (
    MISSING_STRUCTURE_INPUT_MESSAGE,
    print_citations,
    print_error,
)
from epath.cli.run import CLI_ERRORS, EpathBatchCliResult, run_from_args


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the E-path command-line interface and return its exit status.

    The status is 0 on success, 1 when a run fails or any structure of a batch
    does, and 2 when no structure input is given.
    """
    start = perf_counter()
    args = parse_args(argv)

    if args.citations:
        print_citations(stream=stdout)
        return 0

    if not args.structure_inputs:
        print_error(MISSING_STRUCTURE_INPUT_MESSAGE, stream=stderr)
        return 2

    try:
        result = run_from_args(
            args,
            stdout=stdout,
            stderr=stderr,
            total_runtime_start=start,
        )
    except CLI_ERRORS as error:
        print_error(f"epath: error: {error}", stream=stderr)
        return 1

    if isinstance(result, EpathBatchCliResult) and result.has_errors:
        return 1

    return 0
