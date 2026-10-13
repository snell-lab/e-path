"""Terminal styling helpers: ANSI colour and indented line wrapping.

Colour is only emitted when the stream is an interactive terminal and the
``NO_COLOR`` environment variable is unset, so redirected or piped output
stays plain text.
"""

import os
import shutil
import sys
import textwrap
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"

WARNING_PREFIX = "WARNING:"
ERROR_PREFIX = "error:"

_DEFAULT_WIDTH = 100
_MAX_WIDTH = 120


def supports_color(stream: TextIO) -> bool:
    """Return whether ANSI colour should be emitted on `stream`."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if sys.platform == "win32":
        # Windows Terminal and modern consoles understand ANSI sequences once
        # virtual-terminal processing is enabled; ``os.system("")`` does that.
        os.system("")
    return stream.isatty()


def colorize(text: str, *codes: str, enabled: bool) -> str:
    """Wrap `text` in the given ANSI codes when colour is enabled."""
    if not enabled:
        return text
    return "".join(codes) + text + RESET


def style_line(line: str, *, enabled: bool) -> str:
    """Highlight a well-known prefix (warning or error) in `line`."""
    stripped = line.lstrip()
    leading = line[: len(line) - len(stripped)]
    if stripped.startswith(WARNING_PREFIX):
        prefix = colorize(WARNING_PREFIX, BOLD, YELLOW, enabled=enabled)
        return leading + prefix + stripped[len(WARNING_PREFIX) :]
    if stripped.startswith("epath: " + ERROR_PREFIX):
        prefix = colorize("epath: " + ERROR_PREFIX, BOLD, RED, enabled=enabled)
        return leading + prefix + stripped[len("epath: " + ERROR_PREFIX) :]
    return line


def terminal_width(stream: TextIO) -> int:
    """Return a sensible wrapping width for `stream`."""
    if not stream.isatty():
        return _DEFAULT_WIDTH
    return min(shutil.get_terminal_size((_DEFAULT_WIDTH, 24)).columns, _MAX_WIDTH)


def wrap_indented(line: str, *, indent: str, width: int) -> list[str]:
    """Wrap `line` under `indent`, with continuation lines indented one step more.

    Any indentation already present in `line` is preserved and continuation
    lines are indented by that plus two extra spaces so the wrapped text reads
    as a single item rather than as new entries.
    """
    stripped = line.lstrip(" ")
    existing = line[: len(line) - len(stripped)]
    initial = indent + existing
    subsequent = initial + "  "
    if not stripped:
        return [initial.rstrip()]
    wrapped = textwrap.wrap(
        stripped,
        width=width,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [initial + stripped]


def print_lines(
    lines: list[str] | tuple[str, ...],
    *,
    indent: str,
    stream: TextIO,
) -> None:
    """Print `lines` indented, wrapped to the terminal width, and styled."""
    enabled = supports_color(stream)
    width = terminal_width(stream)
    for line in lines:
        for wrapped in wrap_indented(line, indent=indent, width=width):
            print(style_line(wrapped, enabled=enabled), file=stream)
