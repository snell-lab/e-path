"""Run the Bnet steps for one structure and record why any were unavailable.

`metric` gives protein Bnet, and `reference` and `percentile` rank it
against the bundled database. This module chains them for `epath.pipeline`. A
structure that Bnet cannot describe, such as one with no Asp/Glu sites or no
recorded resolution, comes back with a reason for the run summary to print
rather than as an exception.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rabdam.bdamage import BDamageAtomResult
from rabdam.bnet.metric import (
    BnetMetricError,
    ProteinBnetResult,
    calculate_protein_bnet,
)
from rabdam.bnet.percentile import (
    BnetPercentileError,
    BnetPercentileResult,
    calculate_bnet_percentile,
)
from rabdam.bnet.reference import (
    BnetReferenceError,
    load_default_bnet_reference_database,
)
from rabdam.structure.models import PreparedStructure


@dataclass(frozen=True, slots=True)
class BnetPipelineResult:
    """Store protein Bnet, its percentile, and why either is missing.

    `run_bnet_pipeline` sets each reason exactly when the matching result
    is `None`. The percentile needs Bnet, so without `bnet_result` both
    reasons are set. The reasons are text for the run summaries.
    """

    bnet_result: ProteinBnetResult | None = None
    percentile_result: BnetPercentileResult | None = None
    bnet_unavailable_reason: str | None = None
    percentile_unavailable_reason: str | None = None


def run_bnet_pipeline(
    *,
    prepared_structure: PreparedStructure,
    bdamage_results: Sequence[BDamageAtomResult],
) -> BnetPipelineResult:
    """Calculate protein Bnet and its resolution-matched percentile.

    `bdamage_results` must hold the scores calculated for
    `prepared_structure`. A different number of scores than selected atoms
    raises `ValueError`.

    Nothing is raised for a structure that Bnet cannot describe. Bnet is
    unavailable when `calculate_protein_bnet` rejects the structure, usually
    for having too few scored Asp/Glu carboxyl oxygens. The percentile is then
    unavailable as well, as it is when the structure records no resolution,
    when the bundled reference database cannot be loaded, or when
    `calculate_bnet_percentile` rejects its input. The matching reason field
    says which.

    Bnet values are reported by E-path but do not participate in its atom
    selection or clustering calculations.
    """
    try:
        bnet_result = calculate_protein_bnet(
            prepared_structure=prepared_structure,
            bdamage_results=bdamage_results,
        )
    except BnetMetricError as error:
        reason = str(error)
        return BnetPipelineResult(
            bnet_unavailable_reason=reason,
            percentile_unavailable_reason=f"Bnet is unavailable: {reason}",
        )

    resolution_angstrom = prepared_structure.metadata.resolution_angstrom
    if resolution_angstrom is None:
        return BnetPipelineResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason="structure resolution is unavailable",
        )

    # The bundled database is package data, so failing to load it is a broken
    # installation rather than bad input. It costs only the percentile, and
    # the rest of the run is still worth finishing.
    try:
        reference_database = load_default_bnet_reference_database()
    except (BnetReferenceError, OSError) as error:
        return BnetPipelineResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason=(
                f"default reference database could not be loaded: {error}"
            ),
        )

    try:
        percentile_result = calculate_bnet_percentile(
            bnet=bnet_result.bnet,
            resolution_angstrom=resolution_angstrom,
            reference_database=reference_database,
        )
    except BnetPercentileError as error:
        return BnetPipelineResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason=str(error),
        )

    return BnetPipelineResult(
        bnet_result=bnet_result,
        percentile_result=percentile_result,
    )
