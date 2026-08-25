"""Calculate the structure-level Bnet outputs used by RABDAM."""

from __future__ import annotations

from dataclasses import dataclass

from rabdam.bdamage import BDamageScoreResult
from rabdam.bnet.calculate import (
    ProteinBnetCalculationError,
    ProteinBnetResult,
    calculate_protein_bnet,
)
from rabdam.bnet.percentile import (
    BnetPercentileError,
    BnetPercentileResult,
    calculate_bnet_percentile,
)
from rabdam.bnet.reference import (
    DEFAULT_REFERENCE_DATABASE_CSV_PATH,
    BnetReferenceError,
    load_default_bnet_reference_database,
)
from rabdam.structure.models import PreparedStructure


@dataclass(frozen=True, slots=True)
class BnetWorkflowResult:
    """Store raw protein Bnet and its resolution-matched percentile."""

    bnet_result: ProteinBnetResult | None = None
    percentile_result: BnetPercentileResult | None = None
    bnet_unavailable_reason: str | None = None
    percentile_unavailable_reason: str | None = None


def calculate_bnet_workflow(
    *,
    prepared_structure: PreparedStructure,
    bdamage_score_result: BDamageScoreResult,
) -> BnetWorkflowResult:
    """Calculate the Bnet portion of the complete RABDAM workflow.

    Structures without the required protein sites or resolution retain an
    explicit unavailable result. Bnet values are reported by ePath but do not
    participate in its atom selection or clustering calculations.
    """
    try:
        bnet_result = calculate_protein_bnet(
            prepared_structure=prepared_structure,
            bdamage_score_result=bdamage_score_result,
        )
    except ProteinBnetCalculationError as error:
        reason = str(error)
        return BnetWorkflowResult(
            bnet_unavailable_reason=reason,
            percentile_unavailable_reason=f"raw Bnet is unavailable: {reason}",
        )

    resolution_angstrom = prepared_structure.metadata.resolution_angstrom
    if resolution_angstrom is None:
        return BnetWorkflowResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason="structure resolution is unavailable",
        )

    try:
        reference_database = load_default_bnet_reference_database()
    except (BnetReferenceError, OSError) as error:
        return BnetWorkflowResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason=(
                f"default reference database could not be loaded: {error}"
            ),
        )

    if reference_database is None:
        return BnetWorkflowResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason=(
                "default reference database not found: "
                f"{DEFAULT_REFERENCE_DATABASE_CSV_PATH}"
            ),
        )

    try:
        percentile_result = calculate_bnet_percentile(
            bnet=bnet_result.bnet,
            resolution_angstrom=resolution_angstrom,
            reference_database=reference_database,
        )
    except BnetPercentileError as error:
        return BnetWorkflowResult(
            bnet_result=bnet_result,
            percentile_unavailable_reason=str(error),
        )

    return BnetWorkflowResult(
        bnet_result=bnet_result,
        percentile_result=percentile_result,
    )


__all__ = [
    "BnetWorkflowResult",
    "calculate_bnet_workflow",
]
