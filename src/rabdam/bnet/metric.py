"""Calculate raw Bnet from damage-sensitive BDamage values.

Bnet is the ratio of KDE area above versus below the all-atom median BDamage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]

# These sampling choices match RABDAM 2 so raw Bnet values remain comparable.
TRAPEZIUM_COUNT = 99
KDE_TAIL_BANDWIDTH_COUNT = 3.0


class _GaussianKDE(Protocol):
    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def scotts_factor(self) -> float: ...


@dataclass(frozen=True, slots=True)
class BnetResult:
    """Store raw Bnet and its KDE integration components."""

    bnet: float
    median_bdamage: float
    below_median_area: float
    above_median_area: float
    site_count: int


def calculate_bnet(
    *,
    all_bdamage_values: Sequence[float],
    bnet_site_bdamage_values: Sequence[float],
) -> BnetResult:
    """Calculate raw Bnet from all-atom and damage-sensitive BDamage values.

    A Gaussian KDE with Scott bandwidth is fit to the Bnet-site values. Bnet is
    the integrated area at or above the all-atom median divided by the area
    below it.

    Args:
        all_bdamage_values: Finite BDamage values for every selected atom.
        bnet_site_bdamage_values: Finite BDamage values for damage-sensitive
            Asp/Glu carboxyl oxygens.

    Returns:
        The Bnet ratio, split point, component areas, and Bnet-site count.

    Raises:
        ValueError: If there are fewer than two sites, the sites have zero
            variance, or no KDE area falls below the median.
    """
    all_values = np.asarray(all_bdamage_values, dtype=float)
    site_values = np.asarray(bnet_site_bdamage_values, dtype=float)

    if site_values.size < 2:
        raise ValueError(
            "Cannot calculate Bnet from fewer than two Bnet-site BDamage values."
        )

    # The all-atom median supplies the baseline while the KDE contains only
    # damage-sensitive Asp/Glu sites.
    median_bdamage = float(np.median(all_values))

    kde = _build_gaussian_kde_scott(site_values)
    x_values = _bnet_kde_x_values(site_values, kde)
    y_values = kde(x_values)

    below_median_area, above_median_area = _split_trapezium_areas(
        x_values=x_values,
        y_values=y_values,
        split_value=median_bdamage,
    )

    if below_median_area <= 0.0:
        raise ValueError(
            "Cannot calculate Bnet because the below-median KDE area is zero."
        )

    return BnetResult(
        bnet=above_median_area / below_median_area,
        median_bdamage=median_bdamage,
        below_median_area=below_median_area,
        above_median_area=above_median_area,
        site_count=site_values.size,
    )


def _build_gaussian_kde_scott(
    sample_values: NDArray[np.float64],
) -> _GaussianKDE:
    """Fit a Gaussian KDE with Scott bandwidth to one-dimensional samples."""
    try:
        return cast(
            _GaussianKDE,
            gaussian_kde(dataset=sample_values, bw_method="scott"),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Cannot calculate KDE because Bnet-site values have zero variance."
        ) from error


def _bnet_kde_x_values(
    sample_values: NDArray[np.float64],
    kde: _GaussianKDE,
) -> NDArray[np.float64]:
    """Build evenly spaced integration points spanning three tail bandwidths."""
    sample_standard_deviation = float(np.std(sample_values, ddof=0))
    tail_width = kde.scotts_factor() * sample_standard_deviation

    x_min = float(np.min(sample_values) - (KDE_TAIL_BANDWIDTH_COUNT * tail_width))
    x_max = float(np.max(sample_values) + (KDE_TAIL_BANDWIDTH_COUNT * tail_width))

    return np.linspace(x_min, x_max, TRAPEZIUM_COUNT + 1)


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

        # RABDAM assigns the trapezium spanning the median by its left edge.
        if left_x < split_value:
            below_split_area += trapezium_area
        else:
            above_split_area += trapezium_area

    return below_split_area, above_split_area
