"""Tests for terminal styling and wrapping helpers."""

import io

import pytest

from epath.cli.style import (
    BOLD,
    RESET,
    YELLOW,
    print_lines,
    style_line,
    supports_color,
    wrap_indented,
)


def test_color_disabled_for_non_tty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert supports_color(io.StringIO()) is False


def test_no_color_env_wins_over_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert supports_color(io.StringIO()) is False


def test_style_line_highlights_warning_prefix_only() -> None:
    line = "  WARNING: something is off"
    assert style_line(line, enabled=False) == line
    assert style_line(line, enabled=True) == (
        f"  {BOLD}{YELLOW}WARNING:{RESET} something is off"
    )


def test_wrap_indented_uses_hanging_indent() -> None:
    words = " ".join(["word"] * 12)
    wrapped = wrap_indented(words, indent="  ", width=30)
    assert len(wrapped) > 1
    assert wrapped[0].startswith("  word")
    assert all(line.startswith("    word") for line in wrapped[1:])
    assert " ".join(line.strip() for line in wrapped) == words


def test_print_lines_is_plain_when_piped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    stream = io.StringIO()
    print_lines(("WARNING: x", "y"), indent="  ", stream=stream)
    assert stream.getvalue() == "  WARNING: x\n  y\n"
