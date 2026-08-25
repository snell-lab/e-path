"""End-to-end E-path analysis workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

from epath.analysis.atoms import EpathAtom, atoms_from_bdamage_workflow
from epath.analysis.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    ComponentStatistics,
    EpathCluster,
    cluster_selected_atoms,
    component_statistics,
)
from epath.analysis.endpoints import CofactorSite
from epath.analysis.pathways import PathAnalysisOptions, PathReport, analyze_paths
from epath.analysis.selection import (
    EpathSelectionResult,
    select_atoms_by_percentile,
    select_atoms_by_percentile_per_chain,
)
from epath.outputs.files import (
    EpathOutputWorkflowOptions,
    EpathWorkflowOutputFiles,
)
from epath.outputs.runs import EpathRunRecord, build_run_record
from epath.outputs.write import write_epath_outputs
from rabdam.bnet.workflow import BnetWorkflowResult, calculate_bnet_workflow
from rabdam.structure.models import PreparedAtom
from rabdam.workflow import BDamageWorkflowResult, WorkflowStageRunner
from xtal.validate import ValidatedCrystal

SelectionMethod = Literal["percentile", "percentile_per_chain"]

T = TypeVar("T")


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
    bdamage_workflow_result: BDamageWorkflowResult | None = None
    bnet_workflow_result: BnetWorkflowResult | None = None

    @property
    def analyzed_chains(self) -> tuple[str, ...]:
        """Chain identifiers of the scored atoms, in first-seen order."""
        return tuple(dict.fromkeys(atom.chain_id for atom in self.atoms))

    @property
    def structure_path(self) -> Path | None:
        """Path of the analysed structure, when one was recorded."""
        return Path(self.run_record.input_path) if self.run_record.input_path else None

    @property
    def crystal(self) -> ValidatedCrystal | None:
        """Validated crystal from the BDamage workflow, when available."""
        if self.bdamage_workflow_result is None:
            return None
        return self.bdamage_workflow_result.crystal

    @property
    def cleaned_atoms(self) -> tuple[PreparedAtom, ...] | None:
        """Cleaned model atoms from the BDamage workflow, when available."""
        if self.bdamage_workflow_result is None:
            return None
        return tuple(self.bdamage_workflow_result.prepared_structure.cleaned_atoms)


def run_epath_workflow(
    bdamage_workflow_result: BDamageWorkflowResult,
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
    stage_runner: WorkflowStageRunner | None = None,
) -> tuple[EpathWorkflowResult, EpathWorkflowOutputFiles]:
    """Run E-path analysis from a completed RABDAM BDamage workflow.

    This is the single entry point for callers that hold a BDamage workflow
    result; the command-line interface uses it too, passing a `stage_runner`
    that reports each named stage as it runs.
    """
    atoms = atoms_from_bdamage_workflow(bdamage_workflow_result)
    if structure_path is None:
        structure_path = bdamage_workflow_result.prepared_structure.metadata.source_path

    return analyze_epath_atoms(
        atoms,
        structure_path=structure_path,
        options=options,
        bdamage_workflow_result=bdamage_workflow_result,
        stage_runner=stage_runner,
    )


def analyze_epath_atoms(
    atoms: Iterable[EpathAtom],
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
    bdamage_workflow_result: BDamageWorkflowResult | None = None,
    stage_runner: WorkflowStageRunner | None = None,
) -> tuple[EpathWorkflowResult, EpathWorkflowOutputFiles]:
    """Run E-path selection, clustering, MST analysis, and outputs.

    Returns the analysis result and, separately, the files written from it.

    Selection, clustering, and pathway geometry need only the scored `atoms`.
    Everything that must see the rest of the model depends on
    `bdamage_workflow_result`, whose cleaned `PreparedAtom`s carry the
    cofactor, solvent, and chain information that E-path atoms do not:

    - endpoint classification (cofactor sites, chain interfaces, surface)
    - the cleaned model PDB with BDamage in the B-factor column
    - Bnet and the Bnet percentile

    Without it those degrade quietly: paths are still reported, but with no
    cofactor or interface detection, and no model PDB or Bnet.

    Bnet is calculated here whenever a BDamage workflow result is available,
    so every caller sees the same sequence of stages. `stage_runner` wraps
    each named stage for timing or progress reporting; without it the stages
    run silently.

    Raises:
        EpathWorkflowError: If `atoms` is empty.
    """
    if options is None:
        options = EpathWorkflowOptions()

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise EpathWorkflowError("E-path analysis requires at least one atom.")

    selection_result = _run_stage(
        "Selecting E-path atoms",
        lambda: _select_atoms(atom_tuple, options.selection),
        stage_runner=stage_runner,
    )
    components = _run_stage(
        "Clustering selected atoms",
        lambda: cluster_selected_atoms(
            selection_result.selected_atoms,
            distance_cutoff=options.clustering.distance_cutoff,
        ),
        stage_runner=stage_runner,
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
    path_analysis = _run_stage(
        "Analyzing pathways",
        lambda: analyze_paths(
            clusters,
            scored_atoms=atom_tuple,
            cleaned_atoms=cleaned_atoms,
            distance_cutoff=options.clustering.distance_cutoff,
            options=options.paths,
        ),
        stage_runner=stage_runner,
    )
    bnet_workflow_result = (
        _run_stage(
            "Calculating Bnet and Bnet percentile",
            lambda: _calculate_bnet(bdamage_workflow_result),
            stage_runner=stage_runner,
        )
        if bdamage_workflow_result is not None
        else None
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
    result = EpathWorkflowResult(
        atoms=atom_tuple,
        selection_result=selection_result,
        clusters=clusters,
        component_statistics=statistics,
        path_reports=path_analysis.reports,
        cofactor_sites=path_analysis.cofactor_sites,
        run_record=run_record,
        bdamage_workflow_result=bdamage_workflow_result,
        bnet_workflow_result=bnet_workflow_result,
    )
    output_files = _run_stage(
        "Writing E-path outputs",
        lambda: write_epath_outputs(result, options.output),
        stage_runner=stage_runner,
    )

    return result, output_files


def _calculate_bnet(
    bdamage_workflow_result: BDamageWorkflowResult,
) -> BnetWorkflowResult:
    return calculate_bnet_workflow(
        prepared_structure=bdamage_workflow_result.prepared_structure,
        bdamage_score_result=bdamage_workflow_result.bdamage_score_result,
    )


def _run_stage(
    label: str,
    callback: Callable[[], T],
    *,
    stage_runner: WorkflowStageRunner | None,
) -> T:
    if stage_runner is None:
        return callback()

    return stage_runner(label, callback)


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
