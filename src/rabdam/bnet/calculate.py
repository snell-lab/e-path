"""Calculate raw protein Bnet from prepared structures and BDamage scores."""

from __future__ import annotations

from dataclasses import dataclass

import rabdam.bnet.metric as bnet_metric
from rabdam.bdamage import BDamageScoreResult, bdamage_scores_as_tuple
from rabdam.bnet.metric import BnetResult
from rabdam.bnet.sites import (
    BnetSite,
    ProteinBnetSiteSelection,
    select_protein_bnet_sites,
)
from rabdam.structure.models import PreparedStructure


class ProteinBnetCalculationError(ValueError):
    """Report an invalid or unsuccessful protein Bnet calculation."""


@dataclass(frozen=True, slots=True)
class ProteinBnetResult:
    """Store a protein Bnet metric and its selected damage-sensitive sites."""

    metric: BnetResult
    site_selection: ProteinBnetSiteSelection

    @property
    def bnet(self) -> float:
        """Return the raw Bnet ratio."""
        return self.metric.bnet

    @property
    def median_bdamage(self) -> float:
        """Return the all-selected-atom median BDamage."""
        return self.metric.median_bdamage

    @property
    def below_median_area(self) -> float:
        """Return KDE area assigned below the median BDamage."""
        return self.metric.below_median_area

    @property
    def above_median_area(self) -> float:
        """Return KDE area assigned at or above the median BDamage."""
        return self.metric.above_median_area

    @property
    def site_count(self) -> int:
        """Return the number of damage-sensitive sites."""
        return len(self.sites)

    @property
    def sites(self) -> tuple[BnetSite, ...]:
        """Return selected sites in prepared-atom order."""
        return self.site_selection.sites


def calculate_protein_bnet(
    *,
    prepared_structure: PreparedStructure,
    bdamage_score_result: BDamageScoreResult,
) -> ProteinBnetResult:
    """Calculate protein Bnet from aligned prepared atoms and BDamage scores.

    The KDE is fit to Asp/Glu carboxyl-oxygen BDamage values, while its split
    point is the median BDamage across all selected atoms.

    Returns:
        The raw Bnet metric and the sites used to calculate it.

    Raises:
        ProteinBnetCalculationError: If no usable sites exist or the Bnet
            metric cannot be calculated.
    """
    site_selection = select_protein_bnet_sites(
        prepared_structure=prepared_structure,
        bdamage_score_result=bdamage_score_result,
    )

    if not site_selection.sites:
        raise ProteinBnetCalculationError(
            "Cannot calculate protein Bnet because no Asp/Glu carboxyl oxygen "
            "Bnet sites were selected."
        )

    try:
        metric = bnet_metric.calculate_bnet(
            all_bdamage_values=bdamage_scores_as_tuple(bdamage_score_result),
            bnet_site_bdamage_values=site_selection.bdamage_values,
        )
    except ValueError as error:
        raise ProteinBnetCalculationError(
            f"Cannot calculate protein Bnet: {error}"
        ) from error

    return ProteinBnetResult(
        metric=metric,
        site_selection=site_selection,
    )
