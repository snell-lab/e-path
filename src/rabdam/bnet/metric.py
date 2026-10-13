"""Protein Bnet, a whole-structure damage metric built from BDamage scores.

Bnet says whether the atoms most prone to radiation damage, the Asp and Glu
carboxyl oxygens, score higher than the protein as a whole. A value near 1
means they look like every other atom, and a value well above 1 means they
stand out.

Two groups of atoms go into it. The median BDamage of every scored atom gives
the typical level for the structure. A smooth density curve, a Gaussian kernel
density estimate (KDE), is fit to the BDamage of the carboxyl oxygens alone.
Bnet is the area of that curve at or above the median divided by the area
below it.

`calculate_protein_bnet` goes from a scored structure to the result. Beneath
it, `protein_bnet_site_bdamage_values` picks out the carboxyl oxygens and
`bnet_from_bdamage_values` does the arithmetic on plain numbers.
`rabdam.bnet.pipeline` calls this module, then passes the result to
`rabdam.bnet.percentile` for ranking.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

from rabdam.bdamage import BDamageAtomResult, bdamage_scores_as_tuple
from rabdam.structure.models import PreparedStructure

# Asp/Glu carboxyl oxygens are used because decarboxylation makes them
# established protein radiation-damage markers. DAS and DGL are the D
# enantiomers, counted alongside the L forms.
_ASPARTATE_CARBOXYL_OXYGENS = frozenset({"OD1", "OD2"})
_GLUTAMATE_CARBOXYL_OXYGENS = frozenset({"OE1", "OE2"})
_CARBOXYL_OXYGENS_BY_RESIDUE = {
    "ASP": _ASPARTATE_CARBOXYL_OXYGENS,
    "DAS": _ASPARTATE_CARBOXYL_OXYGENS,
    "GLU": _GLUTAMATE_CARBOXYL_OXYGENS,
    "DGL": _GLUTAMATE_CARBOXYL_OXYGENS,
}

# The area under the density curve is measured by sampling it at evenly spaced
# points and summing the trapezia between them. Changing either value below
# moves the sample points, and with them every Bnet value.
#
# The number of trapezia. They lie between 100 sample points.
TRAPEZIUM_COUNT = 99
# How far the sampling extends beyond the lowest and highest site values, in
# bandwidths. The curve has fallen to almost nothing by then.
KDE_TAIL_BANDWIDTH_COUNT = 3.0


class BnetMetricError(ValueError):
    """Report a structure or BDamage values from which no Bnet can be calculated.

    `rabdam.bnet.pipeline` catches it and records Bnet as unavailable, so a
    structure without usable sites does not fail the run.
    """


@dataclass(frozen=True, slots=True)
class ProteinBnetResult:
    """Store protein Bnet and the number of sites it was calculated from.

    `site_count` is the number of Asp/Glu carboxyl oxygens among the scored
    atoms.
    """

    bnet: float
    site_count: int


class _GaussianKDE(Protocol):
    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def scotts_factor(self) -> float: ...


def calculate_protein_bnet(
    *,
    prepared_structure: PreparedStructure,
    bdamage_results: Sequence[BDamageAtomResult],
) -> ProteinBnetResult:
    """Return protein Bnet for a prepared structure and its BDamage scores.

    `bdamage_results` must hold one result per selected atom of
    `prepared_structure`, in the same order. A different length raises a plain
    `ValueError`.

    A `BnetMetricError` is raised when the selected atoms include no Asp/Glu
    carboxyl oxygen, or when Bnet cannot be calculated from the ones found:
    there is only one, their BDamage values have exactly zero variance, or no
    KDE area falls below the median.
    """
    site_values = protein_bnet_site_bdamage_values(
        prepared_structure=prepared_structure,
        bdamage_results=bdamage_results,
    )

    if not site_values:
        raise BnetMetricError(
            "Cannot calculate protein Bnet because no Asp/Glu carboxyl oxygen "
            "Bnet sites were selected."
        )

    bnet = bnet_from_bdamage_values(
        all_bdamage_values=bdamage_scores_as_tuple(bdamage_results),
        bnet_site_bdamage_values=site_values,
    )
    return ProteinBnetResult(bnet=bnet, site_count=len(site_values))


def protein_bnet_site_bdamage_values(
    *,
    prepared_structure: PreparedStructure,
    bdamage_results: Sequence[BDamageAtomResult],
) -> tuple[float, ...]:
    """Return the BDamage of each Asp/Glu carboxyl oxygen among the scored atoms.

    The scores are paired with the selected atoms by position, as in
    `calculate_protein_bnet`. The values keep selected-atom order, and a
    structure without any such atom gives none.
    """
    return tuple(
        bdamage_result.bdamage
        for atom, bdamage_result in zip(
            prepared_structure.selected_atoms,
            bdamage_results,
            strict=True,
        )
        if is_protein_bnet_site(
            residue_name=atom.record.residue_name,
            atom_name=atom.record.atom_name,
        )
    )


def is_protein_bnet_site(*, residue_name: str, atom_name: str) -> bool:
    """Return whether the names identify an Asp/Glu carboxyl oxygen.

    Case and surrounding whitespace are ignored, and each residue matches only
    its own oxygens, so OE1 of an Asp is not a site. Only the names are
    examined. That the atom belongs to the protein is left to the caller.
    """
    carboxyl_oxygens = _CARBOXYL_OXYGENS_BY_RESIDUE.get(residue_name.strip().upper())
    return (
        carboxyl_oxygens is not None and atom_name.strip().upper() in carboxyl_oxygens
    )


def bnet_from_bdamage_values(
    *,
    all_bdamage_values: Sequence[float],
    bnet_site_bdamage_values: Sequence[float],
) -> float:
    """Return Bnet from plain BDamage values.

    `all_bdamage_values` holds the BDamage of every scored atom and supplies
    only the median that splits the density. `bnet_site_bdamage_values` holds
    the BDamage of the Asp/Glu carboxyl oxygens. A Gaussian kernel density
    estimate with Scott's bandwidth is fit to them and integrated by the
    trapezium rule over `TRAPEZIUM_COUNT` equal intervals. Bnet is the area of
    the intervals that start at or above the median divided by the area of
    those that start below it.

    Both sequences must hold only finite values, and `all_bdamage_values` at
    least one.

    A `BnetMetricError` is raised when there are fewer than two sites, when
    their variance is exactly zero, or when the median lies at or below the
    start of the grid, which leaves no area below it to divide by. Sites that
    differ only by rounding noise, as a model with uniform B-factors gives,
    still have a variance and get a Bnet.
    """
    all_values = np.asarray(all_bdamage_values, dtype=np.float64)
    site_values = np.asarray(bnet_site_bdamage_values, dtype=np.float64)

    if site_values.size < 2:
        raise BnetMetricError(
            "Cannot calculate Bnet from fewer than two Bnet-site BDamage values."
        )

    # The all-atom median supplies the baseline while the KDE contains only
    # damage-sensitive Asp/Glu sites.
    median_bdamage = float(np.median(all_values))

    # SciPy rejects only a covariance of exactly zero
    try:
        kde = cast(_GaussianKDE, gaussian_kde(site_values, bw_method="scott"))
    except np.linalg.LinAlgError as error:
        raise BnetMetricError(
            "Cannot calculate Bnet because the Bnet-site BDamage values have "
            "zero variance, so no density can be fit to them."
        ) from error

    x_values = _bnet_kde_x_values(site_values, kde)
    y_values = kde(x_values)

    below_median_area, above_median_area = _split_trapezium_areas(
        x_values=x_values,
        y_values=y_values,
        split_value=median_bdamage,
    )

    if below_median_area <= 0.0:
        raise BnetMetricError(
            "Cannot calculate Bnet because the all-atom median BDamage "
            f"({median_bdamage!r}) lies below the whole Bnet-site density, so "
            "the below-median KDE area is zero."
        )

    return above_median_area / below_median_area


def _bnet_kde_x_values(
    site_values: NDArray[np.float64],
    kde: _GaussianKDE,
) -> NDArray[np.float64]:
    """Return the evenly spaced points at which the density is integrated."""
    # This bandwidth uses the population standard deviation, whereas the
    # kernel SciPy fits uses the sample one, so the tails are slightly shorter
    # than three kernel widths. Unifying the two would move every grid point
    # and with it every Bnet value.
    bandwidth = kde.scotts_factor() * float(np.std(site_values, ddof=0))
    tail_length = KDE_TAIL_BANDWIDTH_COUNT * bandwidth

    return np.linspace(
        float(site_values.min() - tail_length),
        float(site_values.max() + tail_length),
        TRAPEZIUM_COUNT + 1,
    )


def _split_trapezium_areas(
    *,
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    split_value: float,
) -> tuple[float, float]:
    """Split trapezoidal KDE area according to each interval's left edge."""
    below_split_area = 0.0
    above_split_area = 0.0

    for left_x, right_x, left_y, right_y in zip(
        x_values[:-1],
        x_values[1:],
        y_values[:-1],
        y_values[1:],
        strict=True,
    ):
        trapezium_area = float((right_x - left_x) * (left_y + right_y) / 2.0)

        # The trapezium spanning the median is assigned by its left edge.
        if left_x < split_value:
            below_split_area += trapezium_area
        else:
            above_split_area += trapezium_area

    return below_split_area, above_split_area
