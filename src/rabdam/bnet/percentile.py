"""Rank a Bnet value against reference structures of similar resolution.

`rabdam.bnet.pipeline` calls this with the Bnet from `rabdam.bnet.metric` and
the database loaded by `rabdam.bnet.reference`.
"""

from dataclasses import dataclass

import numpy as np

from rabdam.bnet.reference import BnetReferenceDatabase

DEFAULT_NEAREST_RESOLUTION_COUNT = 1000


@dataclass(frozen=True, slots=True)
class BnetPercentileResult:
    """Store a Bnet percentile and the reference set it was ranked against.

    `calculate_bnet_percentile` says how `percentile_percent` is derived.
    `local_reference_count` is the number of rows ranked against, those
    between `local_resolution_min` and `local_resolution_max` angstroms, and
    `reference_entry_count` is the size of the whole database.
    """

    resolution_angstrom: float
    percentile_percent: float
    reference_database_id: str
    reference_entry_count: int
    local_reference_count: int
    local_resolution_min: float
    local_resolution_max: float


@dataclass(frozen=True, slots=True)
class ResolutionMatchedReferenceSet:
    """Store the reference rows inside one inclusive resolution interval.

    `indices` are positions in the database's entry order, ascending. The
    interval bounds are in angstroms.
    """

    indices: tuple[int, ...]
    resolution_min: float
    resolution_max: float


def calculate_bnet_percentile(
    *,
    bnet: float,
    resolution_angstrom: float,
    reference_database: BnetReferenceDatabase,
    nearest_resolution_count: int = DEFAULT_NEAREST_RESOLUTION_COUNT,
) -> BnetPercentileResult:
    """Return the percentile of `bnet` among the resolution-matched references.

    The comparison set comes from `find_resolution_matched_reference_set`. Its
    Bnet values are put in ascending order, and the rank is the one-based
    position of the value closest to `bnet`, the lowest such position when
    several are equally close. The percentile is that rank as a percentage of
    the set size. It therefore runs from just above 0 to 100, and a Bnet
    outside the local range takes the first or the last rank.

    Nothing is checked here. `bnet` must be finite and `resolution_angstrom`
    finite and positive, as the values `rabdam.bnet.pipeline` passes are.
    """
    matched_set = find_resolution_matched_reference_set(
        resolution_angstrom=resolution_angstrom,
        reference_database=reference_database,
        nearest_resolution_count=nearest_resolution_count,
    )
    local_bnet_values = np.sort(
        reference_database.bnet_values[list(matched_set.indices)]
    )
    local_count = local_bnet_values.size

    rank = int(np.abs(local_bnet_values - bnet).argmin()) + 1
    percentile_percent = (100.0 * rank) / local_count

    return BnetPercentileResult(
        resolution_angstrom=resolution_angstrom,
        percentile_percent=percentile_percent,
        reference_database_id=reference_database.database_id,
        reference_entry_count=reference_database.entry_count,
        local_reference_count=local_count,
        local_resolution_min=matched_set.resolution_min,
        local_resolution_max=matched_set.resolution_max,
    )


def find_resolution_matched_reference_set(
    *,
    resolution_angstrom: float,
    reference_database: BnetReferenceDatabase,
    nearest_resolution_count: int = DEFAULT_NEAREST_RESOLUTION_COUNT,
) -> ResolutionMatchedReferenceSet:
    """Return the reference rows spanning the nearest resolutions.

    The `nearest_resolution_count` rows closest in resolution to
    `resolution_angstrom` set an inclusive resolution interval, and every row
    inside that interval is returned. Rows sharing a boundary resolution are
    therefore all kept, and the set can be larger than the count asked for.

    Only resolutions equal to a boundary are added back. Two resolutions the
    same distance either side of the target are ordered by floating-point
    rounding, or by database order when the distances are exactly equal, so
    when the count runs out between them the interval reaches further on one
    side of the target than on the other.
    """
    resolutions = reference_database.resolution_values
    distances = np.abs(resolutions - resolution_angstrom)

    # A stable sort takes the earliest of several rows at the same distance.
    nearest_indices = np.argsort(distances, kind="stable")[:nearest_resolution_count]
    nearest_resolutions = resolutions[nearest_indices]

    resolution_min = float(np.min(nearest_resolutions))
    resolution_max = float(np.max(nearest_resolutions))

    # Expanding back to the full range keeps every boundary tie instead of an
    # arbitrary subset chosen by CSV order.
    local_mask = (resolutions >= resolution_min) & (resolutions <= resolution_max)
    local_indices = tuple(int(index) for index in np.flatnonzero(local_mask))

    return ResolutionMatchedReferenceSet(
        indices=local_indices,
        resolution_min=resolution_min,
        resolution_max=resolution_max,
    )
