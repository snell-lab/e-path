"""High-level protein Bnet calculation.

This module connects RABDAM's prepared structure and per-atom BDamage results
to the raw Bnet metric calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bdamage.score import BDamageScoreResult
import bnet.metric as bnet_metric
from bnet.metric import BnetResult
from bnet.sites import (
    BnetSite,
    BnetSiteSelectionError,
    ProteinBnetSiteSelection,
    all_selected_bdamage_values,
    select_protein_bnet_sites,
)
from structure.models import PreparedStructure


class ProteinBnetCalculationError(ValueError):
    """Raised when protein Bnet cannot be calculated."""


@dataclass(frozen=True, slots=True)
class ProteinBnetResult:
    """Protein Bnet result and the sites used to calculate it."""

    metric: BnetResult
    site_selection: ProteinBnetSiteSelection

    @property
    def bnet(self) -> float:
        """Return the raw protein Bnet value."""

        return self.metric.bnet

    @property
    def median_bdamage(self) -> float:
        """Return the all-selected-atom median BDamage value."""

        return self.metric.median_bdamage

    @property
    def left_area(self) -> float:
        """Return the below-median KDE area."""

        return self.metric.left_area

    @property
    def right_area(self) -> float:
        """Return the above-median KDE area."""

        return self.metric.right_area

    @property
    def site_count(self) -> int:
        """Return the number of selected protein Bnet sites."""

        return len(self.sites)

    @property
    def sites(self) -> tuple[BnetSite, ...]:
        """Return the selected protein Bnet sites."""

        return self.site_selection.sites


def calculate_protein_bnet(
    *,
    prepared_structure: PreparedStructure,
    bdamage_score_result: BDamageScoreResult,
) -> ProteinBnetResult:
    """Calculate raw protein Bnet from a prepared structure and BDamage result.

    This function:
      1. collects all selected-atom BDamage values,
      2. selects Asp/Glu carboxyl oxygen Bnet sites,
      3. calculates the raw Bnet metric.

    Bnet-percentile is intentionally not calculated here. It requires a
    reference database and resolution-dependent comparison step.
    """

    try:
        all_bdamage_values = all_selected_bdamage_values(bdamage_score_result)
        site_selection = select_protein_bnet_sites(
            prepared_structure=prepared_structure,
            bdamage_score_result=bdamage_score_result,
        )
    except BnetSiteSelectionError as error:
        raise ProteinBnetCalculationError(
            f"Cannot calculate protein Bnet: {error}"
        ) from error

    if not site_selection.sites:
        raise ProteinBnetCalculationError(
            "Cannot calculate protein Bnet because no Asp/Glu carboxyl oxygen "
            "Bnet sites were selected."
        )

    try:
        metric = bnet_metric.calculate_bnet(
            all_bdamage_values=all_bdamage_values,
            bnet_site_bdamage_values=site_selection.bdamage_values,
        )
    except ValueError as error:
        raise ProteinBnetCalculationError(
            f"Cannot calculate protein Bnet: {error}"
        ) from error

    if metric.site_count != len(site_selection.sites):
        raise ProteinBnetCalculationError(
            "Protein Bnet site count does not match metric site count: "
            f"{len(site_selection.sites)} selected sites, "
            f"{metric.site_count} metric sites."
        )

    return ProteinBnetResult(
        metric=metric,
        site_selection=site_selection,
    )
