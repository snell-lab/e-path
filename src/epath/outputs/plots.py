"""Draw the Z-score falloff plot of an E-path run.

The falloff plot ranks atoms by ascending BDamage Z-score and shows the
negative tail, which is the region E-path selects from. `epath.outputs.write`
calls `write_z_score_falloff` when the run asks for the plot.
"""

from pathlib import Path
from typing import Any

from epath.analysis.selection import EpathSelectionResult

PLOT_DPI = 300
# Width and height, in inches.
PLOT_FIGSIZE = (6.0, 4.0)


def write_z_score_falloff(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
) -> Path:
    """Write rank-ordered BDamage Z-scores for the negative tail.

    Only the atoms with a Z-score of zero or below are drawn, lowest first.
    The Z-scores are centred on their mean, so at least one atom qualifies.

    The selected atoms are shaded and the last of them is marked with a
    dashed line, so the plot shows where the cutoff falls on the falloff.
    Ties at the cutoff share a Z-score, so the selection is exactly the first
    ranks of the curve.

    Missing parent directories are created and an existing file is replaced.
    The path written is returned.
    """
    # The method only ever selects from the negative tail, so plotting the full
    # range would compress the region of interest into the left edge.
    tail = sorted(
        scored_atom.bdamage_z_score
        for scored_atom in selection_result.scored_atoms
        if scored_atom.bdamage_z_score <= 0.0
    )
    threshold = selection_result.z_threshold
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pyplot = _pyplot()
    figure = pyplot.figure(figsize=PLOT_FIGSIZE)
    try:
        pyplot.plot(range(1, len(tail) + 1), tail, color="black", linewidth=1.0)
        selected_count = len(selection_result.selected_atoms)
        if selected_count:
            pyplot.axvspan(0, selected_count, color="tab:cyan", alpha=0.15, linewidth=0)
            pyplot.axvline(
                selected_count,
                color="tab:cyan",
                linestyle="--",
                linewidth=1.0,
                label=f"{selected_count} atoms selected, Z = {threshold:.2f}",
            )
            pyplot.legend(loc="lower right", fontsize="small")
        pyplot.xlim(left=0)
        pyplot.xlabel("Atom rank by ascending BDamage Z-score")
        pyplot.ylabel("BDamage Z-score")
        pyplot.title("BDamage Z-score falloff")
        figure.tight_layout()
        figure.savefig(output_file, dpi=PLOT_DPI)
    finally:
        pyplot.close(figure)

    return output_file


def _pyplot() -> Any:
    # Imported here rather than at module level so the non-interactive backend
    # is set before pyplot initialises, and so importing the E-path package
    # does not pay matplotlib's import cost. Older matplotlib releases ship no
    # type information, so pyplot is returned as `Any` and the type checks
    # pass with any release.
    import matplotlib  # pyright: ignore[reportMissingTypeStubs]

    matplotlib.use("Agg")  # pyright: ignore[reportUnknownMemberType]
    from matplotlib import pyplot  # pyright: ignore[reportMissingTypeStubs]

    return pyplot
