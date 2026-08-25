"""Plot outputs for E-path analysis.

The falloff plot ranks atoms by ascending BDamage Z-score and shows the
negative tail, which is the region E-path selects from. A plateau in that
falloff, where many consecutive atoms share a Z-score, indicates a commonly
protected region rather than a smooth decay.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from epath.analysis.selection import EpathSelectionResult, ZScorePlateau

DEFAULT_PLOT_DPI = 300
DEFAULT_PLOT_FIGSIZE = (6.0, 4.0)


class EpathPlotError(RuntimeError):
    """Raised when an E-path plot cannot be generated."""


def plot_z_falloff(
    prefix: str | Path,
    selection_result: EpathSelectionResult,
    *,
    dpi: int = DEFAULT_PLOT_DPI,
) -> Path | None:
    """Write the Z-score falloff plot beside an output prefix."""

    output_prefix = Path(prefix)
    return write_z_score_falloff(
        output_prefix.with_name(f"{output_prefix.name}_zfalloff.png"),
        selection_result,
        dpi=dpi,
    )


def write_z_score_falloff(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
    *,
    dpi: int = DEFAULT_PLOT_DPI,
) -> Path | None:
    """Write rank-ordered BDamage Z-scores for the negative tail.

    The selection threshold and any detected plateau are marked, so the plot
    shows both what was selected and whether the falloff flattens within it.
    """

    z_scores = _z_score_array(
        [scored_atom.bdamage_z_score for scored_atom in selection_result.scored_atoms]
    )
    if z_scores.size == 0:
        return None

    dpi = _positive_int(dpi, name="dpi")
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    ranked = np.sort(z_scores)
    # The method only ever selects from the negative tail, so plotting the full
    # range would compress the region of interest into the left edge.
    tail = ranked[ranked <= 0.0]
    if tail.size == 0:
        tail = ranked[: max(1, ranked.size // 50)]

    ranks = np.arange(1, tail.size + 1)
    threshold = selection_result.z_threshold

    pyplot = _pyplot()
    figure = pyplot.figure(figsize=DEFAULT_PLOT_FIGSIZE)
    try:
        pyplot.plot(ranks, tail, color="black", linewidth=1.0)
        if math.isfinite(threshold):
            pyplot.axhline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"selection threshold Z = {threshold:.3f}",
            )
        _mark_plateau(pyplot, selection_result.plateau)
        pyplot.xlabel("Atom rank by ascending BDamage Z-score")
        pyplot.ylabel("BDamage Z-score")
        pyplot.title("BDamage Z-score falloff")
        pyplot.legend(loc="lower right", fontsize="small")
        figure.tight_layout()
        figure.savefig(output_file, dpi=dpi)
    finally:
        pyplot.close(figure)

    return output_file


def _mark_plateau(pyplot: Any, plateau: ZScorePlateau | None) -> None:
    if plateau is None:
        return

    pyplot.axvspan(
        plateau.first_rank,
        plateau.first_rank + plateau.atom_count - 1,
        color="tab:blue",
        alpha=0.2,
        label=f"plateau, {plateau.atom_count} atoms",
    )


def _pyplot() -> Any:
    # Imported here rather than at module level so the config directory and
    # the non-interactive backend are set before matplotlib initialises, and
    # so importing the E-path package does not pay matplotlib's import cost.
    _ensure_matplotlib_config_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def _ensure_matplotlib_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return

    config_dir = Path(tempfile.gettempdir()) / "epath-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _z_score_array(
    values: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
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
