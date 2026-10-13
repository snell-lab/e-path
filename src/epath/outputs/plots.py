"""Plot outputs for E-path analysis.

The falloff plot ranks atoms by ascending BDamage Z-score and shows the
negative tail, which is the region E-path selects from.
"""

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from epath.analysis.selection import EpathSelectionResult

DEFAULT_PLOT_DPI = 300
DEFAULT_PLOT_FIGSIZE = (6.0, 4.0)


def plot_z_falloff(
    prefix: str | Path,
    selection_result: EpathSelectionResult,
) -> Path:
    """Write the Z-score falloff plot beside an output prefix."""
    output_prefix = Path(prefix)
    return write_z_score_falloff(
        output_prefix.with_name(f"{output_prefix.name}_zfalloff.png"),
        selection_result,
    )


def write_z_score_falloff(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
    *,
    dpi: int = DEFAULT_PLOT_DPI,
) -> Path:
    """Write rank-ordered BDamage Z-scores for the negative tail.

    The selection threshold is marked, so the plot shows where the cutoff
    falls on the falloff.
    """
    z_scores = np.asarray(
        [scored_atom.bdamage_z_score for scored_atom in selection_result.scored_atoms],
        dtype=float,
    )
    output_file = Path(output_path)
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
        pyplot.xlabel("Atom rank by ascending BDamage Z-score")
        pyplot.ylabel("BDamage Z-score")
        pyplot.title("BDamage Z-score falloff")
        # A legend with no labelled artists only produces a warning.
        if math.isfinite(threshold):
            pyplot.legend(loc="lower right", fontsize="small")
        figure.tight_layout()
        figure.savefig(output_file, dpi=dpi)
    finally:
        pyplot.close(figure)

    return output_file


def _pyplot() -> Any:
    # Imported here rather than at module level so the config directory and
    # the non-interactive backend are set before matplotlib initialises, and
    # so importing the E-path package does not pay matplotlib's import cost.
    # Matplotlib ships no type information, so it is opaque to the type
    # checkers and reaches the plotting helpers above as `Any`.
    _ensure_matplotlib_config_dir()
    import matplotlib  # pyright: ignore[reportMissingTypeStubs]

    matplotlib.use("Agg", force=True)  # pyright: ignore[reportUnknownMemberType]
    from matplotlib import pyplot  # pyright: ignore[reportMissingTypeStubs]

    return pyplot


def _ensure_matplotlib_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return

    config_dir = Path(tempfile.gettempdir()) / "epath-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
