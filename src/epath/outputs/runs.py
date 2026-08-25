"""One machine-readable row per E-path run, for comparing models.

A run summary is written for people to read, but comparing many models means
reading the same values back out again. This module records them once per run
in the form the E-path cohort tables use: one row per structure, holding the
selection depth, the damage context, and what the connectivity analysis found.

Concatenating these rows across runs gives the cohort table directly. Grouping
those rows into cohorts is left to the caller, because which group a structure
belongs to is a prior judgement about its biology, not something the analysis
can determine.

The rows are written by `epath.outputs.tables`, with the other CSV outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from epath.analysis.clustering import ComponentStatistics, EpathCluster
from epath.analysis.selection import EpathSelectionResult
from rabdam.bnet.workflow import BnetWorkflowResult


@dataclass(frozen=True, slots=True)
class EpathRunRecord:
    """Run-level results for one structure."""

    structure: str
    input_path: str
    selection_method: str
    cutoff_percent: float
    total_atoms: int
    atoms_selected: int
    selected_percent: float
    equivalent_z_score: float
    bdamage_threshold: float | None
    bnet: float | None
    bnet_percentile: float | None
    total_components: int
    minimum_cluster_size: int
    clusters_reported: int
    largest_component_atoms: int
    largest_component_reported: bool
    largest_component_length: float | None
    largest_component_mean_step: float | None
    largest_component_largest_edge: float | None
    largest_component_span: float | None


def build_run_record(
    *,
    structure_path: str | Path | None,
    selection_result: EpathSelectionResult,
    clusters: Sequence[EpathCluster],
    statistics: ComponentStatistics,
    minimum_cluster_size: int,
    cutoff_percent: float,
    bnet_workflow_result: BnetWorkflowResult | None = None,
) -> EpathRunRecord:
    """Collect one structure's run-level results into a single record.

    The largest component is reported whether or not it reached the reporting
    threshold, so a model that produced no pathways still records what its
    connectivity analysis found.
    """
    path = Path(structure_path) if structure_path is not None else None
    metrics = statistics.largest_metrics
    bnet, bnet_percentile = _bnet_values(bnet_workflow_result)

    return EpathRunRecord(
        structure=path.stem if path is not None else "",
        input_path=str(path) if path is not None else "",
        selection_method=selection_result.method,
        cutoff_percent=cutoff_percent,
        total_atoms=len(selection_result.scored_atoms),
        atoms_selected=len(selection_result.selected_atoms),
        selected_percent=selection_result.selected_percent,
        equivalent_z_score=selection_result.z_threshold,
        bdamage_threshold=selection_result.bdamage_threshold,
        bnet=bnet,
        bnet_percentile=bnet_percentile,
        total_components=statistics.total_count,
        minimum_cluster_size=minimum_cluster_size,
        clusters_reported=len(clusters),
        largest_component_atoms=statistics.largest_atom_count,
        largest_component_reported=(
            statistics.largest_atom_count >= minimum_cluster_size
        ),
        largest_component_length=metrics.mst_total_length if metrics else None,
        largest_component_mean_step=metrics.mean_step if metrics else None,
        largest_component_largest_edge=metrics.mst_largest_edge if metrics else None,
        largest_component_span=metrics.span if metrics else None,
    )


def _bnet_values(
    workflow_result: BnetWorkflowResult | None,
) -> tuple[float | None, float | None]:
    if workflow_result is None or workflow_result.bnet_result is None:
        return None, None

    percentile = workflow_result.percentile_result

    return (
        workflow_result.bnet_result.bnet,
        percentile.percentile_percent if percentile is not None else None,
    )
