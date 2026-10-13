"""Colour and wrap the terminal output of `epath.cli.report`.

Colour is emitted only on an interactive terminal, so redirected or piped
output stays plain text. The ``NO_COLOR`` environment variable turns it off
everywhere and ``FORCE_COLOR`` turns it on everywhere, with ``NO_COLOR``
winning when both are set.
"""

import os
import shutil
import sys
import textwrap
from collections.abc import Iterable
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"

# The line prefixes that are highlighted, each with its colour.
_HIGHLIGHTS = (
    ("WARNING:", BOLD + YELLOW),
    ("epath: error:", BOLD + RED),
)

_DEFAULT_WIDTH = 100
_MAX_WIDTH = 120


def supports_color(stream: TextIO) -> bool:
    """Return whether ANSI colour should be emitted on `stream`.

    An empty ``NO_COLOR`` or ``FORCE_COLOR`` counts as unset.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not (os.environ.get("FORCE_COLOR") or stream.isatty()):
        return False
    if sys.platform == "win32":
        # Windows Terminal and modern consoles understand ANSI sequences once
        # virtual-terminal processing is enabled; ``os.system("")`` does that.
        os.system("")
    return True


def style_line(line: str, *, enabled: bool) -> str:
    """Highlight a leading ``WARNING:`` or ``epath: error:`` in `line`.

    Only the prefix is coloured, and indentation before it is kept. Any other
    line is returned unchanged, as is every line when `enabled` is false.
    """
    if not enabled:
        return line
    stripped = line.lstrip()
    leading = line[: len(line) - len(stripped)]
    for prefix, colour in _HIGHLIGHTS:
        if stripped.startswith(prefix):
            return leading + colour + prefix + RESET + stripped[len(prefix) :]
    return line


def terminal_width(stream: TextIO) -> int:
    """Return the width, in columns, to wrap output on `stream` to.

    This is 100 when `stream` is not a terminal. Otherwise it is the width of
    the terminal, or 120 if the terminal is wider than that.
    """
    if not stream.isatty():
        return _DEFAULT_WIDTH
    columns = shutil.get_terminal_size(fallback=(_DEFAULT_WIDTH, 24)).columns
    return min(columns, _MAX_WIDTH)


def wrap_indented(line: str, *, indent: str, width: int) -> list[str]:
    """Wrap `line` under `indent`, with continuation lines indented one step more.

    Any indentation already present in `line` is preserved and continuation
    lines are indented by that plus two extra spaces so the wrapped text reads
    as a single item rather than as new entries.

    `width` counts the indentation. Lines break only at spaces, so a word
    longer than the width is kept whole and overruns it. A blank line gives
    one line with no trailing spaces.
    """
    stripped = line.lstrip(" ")
    initial = indent + line[: len(line) - len(stripped)]
    wrapped = textwrap.wrap(
        stripped,
        width=width,
        initial_indent=initial,
        subsequent_indent=initial + "  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    # textwrap returns no lines at all for a blank line.
    return wrapped or [initial.rstrip()]


def print_lines(lines: Iterable[str], *, indent: str, stream: TextIO) -> None:
    """Print `lines` indented, wrapped to the terminal width, and styled."""
    enabled = supports_color(stream)
    width = terminal_width(stream)
    for line in lines:
        for wrapped in wrap_indented(line, indent=indent, width=width):
            print(style_line(wrapped, enabled=enabled), file=stream)
