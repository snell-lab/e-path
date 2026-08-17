"""Plot outputs for E-path analysis."""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from epath.selection import (
    DEFAULT_HISTOGRAM_BINS,
    EpathScoredAtom,
    EpathSelectionResult,
)

DEFAULT_Z_HISTOGRAM_DPI = 300
DEFAULT_Z_HISTOGRAM_FIGSIZE = (6.0, 4.0)


class EpathPlotError(RuntimeError):
    """Raised when an E-path plot cannot be generated."""


def plot_z_hist(
    prefix: str | Path,
    z_all: Sequence[float] | np.ndarray,
    z_mode: float,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    dpi: int = DEFAULT_Z_HISTOGRAM_DPI,
) -> Path | None:
    """Write the E-path Z-score histogram beside an output prefix."""

    output_prefix = Path(prefix)
    return write_z_score_histogram(
        output_prefix.with_name(f"{output_prefix.name}_zscores.png"),
        z_all,
        z_mode=z_mode,
        bins=bins,
        dpi=dpi,
    )


def write_selection_z_score_histogram(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    dpi: int = DEFAULT_Z_HISTOGRAM_DPI,
) -> Path | None:
    """Write the Z-score histogram for a selection result."""

    return write_z_score_histogram(
        output_path,
        [scored_atom.bdamage_z_score for scored_atom in selection_result.scored_atoms],
        z_mode=selection_result.z_mode,
        bins=bins,
        dpi=dpi,
    )


def write_scored_atom_z_score_histogram(
    output_path: str | Path,
    scored_atoms: Iterable[EpathScoredAtom],
    *,
    z_mode: float,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    dpi: int = DEFAULT_Z_HISTOGRAM_DPI,
) -> Path | None:
    """Write the Z-score histogram for scored atoms."""

    return write_z_score_histogram(
        output_path,
        [scored_atom.bdamage_z_score for scored_atom in scored_atoms],
        z_mode=z_mode,
        bins=bins,
        dpi=dpi,
    )


def write_z_score_histogram(
    output_path: str | Path,
    z_scores: Sequence[float] | np.ndarray,
    *,
    z_mode: float,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    dpi: int = DEFAULT_Z_HISTOGRAM_DPI,
) -> Path | None:
    """Write a Z-score histogram with the histogram mode marked."""

    z_score_array = _z_score_array(z_scores)
    if z_score_array.size == 0:
        return None

    bins = _positive_int(bins, name="bins")
    dpi = _positive_int(dpi, name="dpi")
    z_mode = _float_value(z_mode, name="z_mode")
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    pyplot = _pyplot()
    figure = pyplot.figure(figsize=DEFAULT_Z_HISTOGRAM_FIGSIZE)
    try:
        pyplot.hist(
            z_score_array,
            bins=bins,
            color="lightgray",
            edgecolor="black",
        )
        if not math.isnan(z_mode):
            pyplot.axvline(
                z_mode,
                color="red",
                linestyle="--",
                linewidth=1.5,
            )
            pyplot.axvspan(
                float(np.min(z_score_array)),
                z_mode,
                color="steelblue",
                alpha=0.20,
            )

        pyplot.xlabel("Z-score (Bdam)")
        pyplot.ylabel("Count")
        pyplot.title("Z-score distribution (mode shown)")
        pyplot.tight_layout()
        figure.savefig(output_file, dpi=dpi)
    finally:
        pyplot.close(figure)

    return output_file


def _pyplot() -> Any:
    try:
        _ensure_matplotlib_config_dir()
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        return importlib.import_module("matplotlib.pyplot")
    except Exception as error:
        raise EpathPlotError(
            "matplotlib is required to generate E-path plots."
        ) from error


def _ensure_matplotlib_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return

    config_dir = Path(tempfile.gettempdir()) / "epath-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _z_score_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise EpathPlotError("z_scores must be one-dimensional.")

    if array.size == 0:
        return array

    if not np.all(np.isfinite(array)):
        raise EpathPlotError("z_scores must contain only finite numbers.")

    return array


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise EpathPlotError(f"{name} must be a positive integer, got {value!r}.")

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise EpathPlotError(
            f"{name} must be a positive integer, got {value!r}."
        ) from error

    if number <= 0:
        raise EpathPlotError(f"{name} must be a positive integer, got {number!r}.")

    return number


def _float_value(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathPlotError(f"{name} must be numeric, got {value!r}.")

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise EpathPlotError(f"{name} must be numeric, got {value!r}.") from error
