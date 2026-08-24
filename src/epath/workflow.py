"""End-to-end E-path analysis workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from epath.atoms import EpathAtom, atoms_from_bdamage_workflow
from epath.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    ComponentStatistics,
    EpathCluster,
    cluster_selected_atoms,
    component_statistics,
)
from epath.endpoints import CofactorSite
from epath.outputs import (
    EpathOutputWorkflowOptions,
    EpathWorkflowOutputFiles,
    write_epath_outputs,
)
from epath.pathways import PathAnalysisOptions, PathReport, analyze_paths
from epath.runs import EpathRunRecord, build_run_record
from epath.selection import (
    EpathSelectionResult,
    select_atoms_by_percentile,
    select_atoms_by_percentile_per_chain,
)
from rabdam.bnet.workflow import BnetWorkflowResult, calculate_bnet_workflow
from rabdam.workflow import BDamageWorkflowResult

SelectionMethod = Literal["percentile", "percentile_per_chain"]


class EpathWorkflowError(ValueError):
    """Raised when E-path cannot run the analysis workflow."""


@dataclass(frozen=True, slots=True)
class EpathSelectionWorkflowOptions:
    """Options controlling E-path atom selection."""

    method: SelectionMethod = "percentile"
    percentile: float = 2.0


@dataclass(frozen=True, slots=True)
class EpathClusterWorkflowOptions:
    """Options controlling E-path cluster construction."""

    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM
    minimum_cluster_size: int = 8


@dataclass(frozen=True, slots=True)
class EpathWorkflowOptions:
    """Options for the complete E-path analysis workflow."""

    selection: EpathSelectionWorkflowOptions = field(
        default_factory=EpathSelectionWorkflowOptions
    )
    clustering: EpathClusterWorkflowOptions = field(
        default_factory=EpathClusterWorkflowOptions
    )
    paths: PathAnalysisOptions = field(default_factory=PathAnalysisOptions)
    output: EpathOutputWorkflowOptions = field(
        default_factory=EpathOutputWorkflowOptions
    )


@dataclass(frozen=True, slots=True)
class EpathWorkflowResult:
    """Complete result of an E-path analysis workflow."""

    atoms: tuple[EpathAtom, ...]
    selection_result: EpathSelectionResult
    clusters: tuple[EpathCluster, ...]
    component_statistics: ComponentStatistics
    path_reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]
    run_record: EpathRunRecord
    output_files: EpathWorkflowOutputFiles
    bdamage_workflow_result: BDamageWorkflowResult | None = None
    bnet_workflow_result: BnetWorkflowResult | None = None


def run_epath_workflow(
    bdamage_workflow_result: BDamageWorkflowResult,
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
) -> EpathWorkflowResult:
    """Run E-path analysis from a completed RABDAM BDamage workflow.

    This is the library entry point for callers that already hold a BDamage
    workflow result. The command-line interface calls `analyze_epath_atoms`
    directly.
    """

    atoms = atoms_from_bdamage_workflow(bdamage_workflow_result)
    if structure_path is None:
        structure_path = bdamage_workflow_result.prepared_structure.metadata.source_path

    return analyze_epath_atoms(
        atoms,
        structure_path=structure_path,
        options=options,
        bdamage_workflow_result=bdamage_workflow_result,
    )


def analyze_epath_atoms(
    atoms: Iterable[EpathAtom],
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
    bdamage_workflow_result: BDamageWorkflowResult | None = None,
    bnet_workflow_result: BnetWorkflowResult | None = None,
) -> EpathWorkflowResult:
    """Run E-path selection, clustering, MST analysis, and outputs."""

    if options is None:
        options = EpathWorkflowOptions()

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise EpathWorkflowError("E-path analysis requires at least one atom.")

    selection_result = _select_atoms(atom_tuple, options.selection)
    components = cluster_selected_atoms(
        selection_result.selected_atoms,
        distance_cutoff=options.clustering.distance_cutoff,
    )
    minimum_cluster_size = _positive_int(
        options.clustering.minimum_cluster_size,
        name="minimum_cluster_size",
    )
    # Components below the reporting threshold are not pathways, but the
    # connectivity analysis still found them, so their count and the geometry
    # of the largest one are recorded before they are dropped.
    statistics = component_statistics(components)
    clusters = tuple(
        cluster for cluster in components if cluster.atom_count >= minimum_cluster_size
    )
    cleaned_atoms = (
        bdamage_workflow_result.prepared_structure.cleaned_atoms
        if bdamage_workflow_result is not None
        else None
    )
    path_analysis = analyze_paths(
        clusters,
        scored_atoms=atom_tuple,
        cleaned_atoms=cleaned_atoms,
        distance_cutoff=options.clustering.distance_cutoff,
        options=options.paths,
    )
    path_reports = path_analysis.reports
    analyzed_chains = tuple(dict.fromkeys(atom.chain_id for atom in atom_tuple))
    if bnet_workflow_result is None and bdamage_workflow_result is not None:
        bnet_workflow_result = calculate_bnet_workflow(
            prepared_structure=bdamage_workflow_result.prepared_structure,
            bdamage_score_result=bdamage_workflow_result.bdamage_score_result,
        )
    run_record = build_run_record(
        structure_path=structure_path,
        selection_result=selection_result,
        clusters=clusters,
        statistics=statistics,
        minimum_cluster_size=minimum_cluster_size,
        cutoff_percent=options.selection.percentile,
        bnet_workflow_result=bnet_workflow_result,
    )
    output_files = write_epath_outputs(
        selection_result=selection_result,
        clusters=clusters,
        path_reports=path_reports,
        cofactor_sites=path_analysis.cofactor_sites,
        component_statistics=statistics,
        run_record=run_record,
        analyzed_chains=analyzed_chains,
        structure_path=structure_path,
        minimum_cluster_size=minimum_cluster_size,
        options=options.output,
        bnet_workflow_result=bnet_workflow_result,
        crystal=(
            bdamage_workflow_result.crystal
            if bdamage_workflow_result is not None
            else None
        ),
        cleaned_atoms=cleaned_atoms,
        bdamage_workflow_result=bdamage_workflow_result,
    )

    return EpathWorkflowResult(
        atoms=atom_tuple,
        selection_result=selection_result,
        clusters=clusters,
        component_statistics=statistics,
        path_reports=path_reports,
        cofactor_sites=path_analysis.cofactor_sites,
        run_record=run_record,
        output_files=output_files,
        bdamage_workflow_result=bdamage_workflow_result,
        bnet_workflow_result=bnet_workflow_result,
    )


def _select_atoms(
    atoms: tuple[EpathAtom, ...],
    options: EpathSelectionWorkflowOptions,
) -> EpathSelectionResult:
    if options.method == "percentile":
        return select_atoms_by_percentile(atoms, percentile=options.percentile)

    if options.method == "percentile_per_chain":
        return select_atoms_by_percentile_per_chain(
            atoms,
            percentile=options.percentile,
        )

    raise EpathWorkflowError(f"Unknown selection method: {options.method!r}.")


def _positive_int(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise EpathWorkflowError(f"{name} must be a positive integer, got {value!r}.")
    return value
