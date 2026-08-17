"""Calculate resolution-matched Bnet percentiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import SupportsFloat, SupportsIndex

import numpy as np

from rabdam.bnet.reference import BnetReferenceDatabase

# A 1,000-structure neighbourhood follows the published RABDAM definition and
# balances resolution matching against reference-sample size.
DEFAULT_NEAREST_RESOLUTION_COUNT = 1000
BNET_PERCENTILE_RANK_METHOD = "nearest_reference"


class BnetPercentileError(ValueError):
    """Report invalid input to a Bnet percentile calculation."""


@dataclass(frozen=True, slots=True)
class ResolutionMatchedReferenceSet:
    """Store reference rows spanning a local resolution interval."""

    indices: tuple[int, ...]
    resolution_min: float
    resolution_max: float

    @property
    def count(self) -> int:
        """Return the number of matched reference rows."""
        return len(self.indices)


@dataclass(frozen=True, slots=True)
class BnetPercentileResult:
    """Store a resolution-matched Bnet rank and its reference context."""

    bnet: float
    resolution_angstrom: float
    percentile_percent: float
    percentile_fraction: float
    rank: int
    rank_method: str
    reference_database_id: str
    reference_entry_count: int
    nearest_resolution_count: int
    local_reference_count: int
    local_resolution_min: float
    local_resolution_max: float
    local_bnet_min: float
    local_bnet_max: float
    nearest_reference_bnet: float


def calculate_bnet_percentile(
    *,
    bnet: float,
    resolution_angstrom: float,
    reference_database: BnetReferenceDatabase,
    nearest_resolution_count: int = DEFAULT_NEAREST_RESOLUTION_COUNT,
) -> BnetPercentileResult:
    """Rank Bnet against a resolution-matched reference neighbourhood.

    Args:
        bnet: Raw Bnet value to rank.
        resolution_angstrom: Positive experimental resolution in angstroms.
        reference_database: Validated reference entries and metadata.
        nearest_resolution_count: Minimum number of nearest-resolution rows used
            to establish the local interval.

    Returns:
        A discrete nearest-reference percentile and the local range and database
        metadata needed to interpret it.

    Raises:
        BnetPercentileError: If numeric inputs are invalid or the requested
            neighbourhood cannot be selected.
    """
    bnet_value = _finite_float(bnet, name="bnet")
    resolution_value = _finite_float(
        resolution_angstrom,
        name="resolution_angstrom",
    )
    if resolution_value <= 0.0:
        raise BnetPercentileError(
            f"resolution_angstrom must be greater than zero, got {resolution_value!r}."
        )

    nearest_resolution_count = _positive_integer(
        nearest_resolution_count,
        name="nearest_resolution_count",
    )

    matched_set = find_resolution_matched_reference_set(
        resolution_angstrom=resolution_value,
        reference_database=reference_database,
        nearest_resolution_count=nearest_resolution_count,
    )

    local_bnet_values = np.sort(
        reference_database.bnet_values[list(matched_set.indices)],
        kind="stable",
    )
    local_count = int(local_bnet_values.size)

    rank, nearest_reference_bnet = _rank_bnet_by_nearest_reference_value(
        bnet_value,
        local_bnet_values=local_bnet_values,
    )

    percentile_percent = (100.0 * rank) / local_count

    return BnetPercentileResult(
        bnet=bnet_value,
        resolution_angstrom=resolution_value,
        percentile_percent=percentile_percent,
        percentile_fraction=percentile_percent / 100.0,
        rank=rank,
        rank_method=BNET_PERCENTILE_RANK_METHOD,
        reference_database_id=reference_database.metadata.database_id,
        reference_entry_count=reference_database.entry_count,
        nearest_resolution_count=nearest_resolution_count,
        local_reference_count=local_count,
        local_resolution_min=matched_set.resolution_min,
        local_resolution_max=matched_set.resolution_max,
        local_bnet_min=float(local_bnet_values[0]),
        local_bnet_max=float(local_bnet_values[-1]),
        nearest_reference_bnet=nearest_reference_bnet,
    )


def find_resolution_matched_reference_set(
    *,
    resolution_angstrom: float,
    reference_database: BnetReferenceDatabase,
    nearest_resolution_count: int = DEFAULT_NEAREST_RESOLUTION_COUNT,
) -> ResolutionMatchedReferenceSet:
    """Select the nearest-resolution interval, including boundary ties.

    The initial nearest-neighbour selection establishes inclusive minimum and
    maximum resolutions. All database rows inside that interval are returned,
    so the final count can exceed `nearest_resolution_count`.

    Raises:
        BnetPercentileError: If numeric inputs are invalid, the requested count
            exceeds the database, or no local rows can be selected.
    """
    resolution_value = _finite_float(
        resolution_angstrom,
        name="resolution_angstrom",
    )
    if resolution_value <= 0.0:
        raise BnetPercentileError(
            f"resolution_angstrom must be greater than zero, got {resolution_value!r}."
        )

    nearest_resolution_count = _positive_integer(
        nearest_resolution_count,
        name="nearest_resolution_count",
    )
    if nearest_resolution_count > reference_database.entry_count:
        raise BnetPercentileError(
            "nearest_resolution_count is larger than the reference database: "
            f"{nearest_resolution_count!r} > {reference_database.entry_count!r}."
        )

    resolutions = reference_database.resolution_values
    distances = np.abs(resolutions - resolution_value)

    # Stable sort preserves CSV order for exact distance ties while avoiding
    # repeated nearest-neighbour searches.
    nearest_indices = np.argsort(distances, kind="mergesort")[:nearest_resolution_count]
    nearest_resolutions = resolutions[nearest_indices]

    resolution_min = float(np.min(nearest_resolutions))
    resolution_max = float(np.max(nearest_resolutions))

    # Expanding back to the full range keeps every boundary tie instead of an
    # arbitrary subset chosen by CSV order.
    local_mask = (resolutions >= resolution_min) & (resolutions <= resolution_max)
    local_indices = tuple(int(index) for index in np.flatnonzero(local_mask))

    if not local_indices:
        raise BnetPercentileError(
            "No reference entries were found in the calculated resolution range."
        )

    return ResolutionMatchedReferenceSet(
        indices=local_indices,
        resolution_min=resolution_min,
        resolution_max=resolution_max,
    )


def _rank_bnet_by_nearest_reference_value(
    bnet: float,
    *,
    local_bnet_values: np.ndarray,
) -> tuple[int, float]:
    """Return the one-based rank and value of the nearest local reference."""
    if local_bnet_values.ndim != 1 or local_bnet_values.size == 0:
        raise BnetPercentileError("local_bnet_values must be a non-empty 1D array.")

    # RABDAM defines a discrete reference rank rather than an interpolated ECDF.
    nearest_index = int(np.abs(local_bnet_values - bnet).argmin())
    return nearest_index + 1, float(local_bnet_values[nearest_index])


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise BnetPercentileError(f"{name} must be numeric, got {value!r}.")

    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        raise BnetPercentileError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BnetPercentileError(f"{name} must be numeric, got {value!r}.") from error

    if not math.isfinite(number):
        raise BnetPercentileError(f"{name} must be finite, got {number!r}.")

    return number


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise BnetPercentileError(f"{name} must be a positive integer, got {value!r}.")

    return int(value)


__all__ = [
    "BNET_PERCENTILE_RANK_METHOD",
    "DEFAULT_NEAREST_RESOLUTION_COUNT",
    "BnetPercentileError",
    "BnetPercentileResult",
    "ResolutionMatchedReferenceSet",
    "calculate_bnet_percentile",
    "find_resolution_matched_reference_set",
]
