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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from epath import __version__
from epath.analysis.clustering import ComponentStatistics, EpathCluster
from epath.analysis.selection import EpathSelectionResult
from rabdam.bnet.pipeline import BnetPipelineResult
from xtal.resolver import split_structure_file_name


@dataclass(frozen=True, slots=True)
class EpathRunRecord:
    """Run-level results for one structure, one field per column of the run CSV.

    `structure` is the input file's name without its format extension and
    `input_path` the file's full path. `epath_version` is the version of
    E-path that made the record. `cutoff_percent` is the percentile
    asked for and `selected_percent` the share of the scored atoms that was
    selected. `equivalent_z_score` is `bdamage_threshold` as a
    Z-score among the scored atoms.

    `clusters_reported` counts the components of at least
    `minimum_cluster_size` atoms, out of `total_components`. The
    `largest_component_*` fields describe the largest component whether or not
    it was reported. Its lengths are in angstroms, and are `None` when no atom
    was selected and so there is no component. `bnet` and `bnet_percentile`
    are `None` when they could not be calculated.
    """

    structure: str
    input_path: str
    epath_version: str
    cutoff_percent: float
    total_atoms: int
    atoms_selected: int
    selected_percent: float
    equivalent_z_score: float
    bdamage_threshold: float
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


def build_run_record(
    *,
    structure_path: Path,
    selection_result: EpathSelectionResult,
    clusters: Sequence[EpathCluster],
    statistics: ComponentStatistics,
    minimum_cluster_size: int,
    cutoff_percent: float,
    bnet_pipeline_result: BnetPipelineResult,
) -> EpathRunRecord:
    """Collect one structure's run-level results into a single record.

    `clusters` holds the components that reached `minimum_cluster_size` atoms,
    and `statistics` describes every component. The largest component is
    therefore recorded whether or not it reached the reporting threshold, so a
    model that produced no pathways still records what its connectivity
    analysis found.

    `structure_path` is the file the structure was read from. A Bnet or a
    percentile that could not be calculated is recorded as `None`.
    """
    metrics = statistics.largest_metrics
    bnet = bnet_pipeline_result.bnet_result
    percentile = bnet_pipeline_result.percentile_result

    return EpathRunRecord(
        structure=split_structure_file_name(structure_path.name)[0],
        input_path=str(structure_path),
        epath_version=__version__,
        cutoff_percent=cutoff_percent,
        total_atoms=len(selection_result.scored_atoms),
        atoms_selected=len(selection_result.selected_atoms),
        selected_percent=selection_result.selected_percent,
        equivalent_z_score=selection_result.z_threshold,
        bdamage_threshold=selection_result.bdamage_threshold,
        bnet=bnet.bnet if bnet else None,
        bnet_percentile=percentile.percentile_percent if percentile else None,
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
    )
