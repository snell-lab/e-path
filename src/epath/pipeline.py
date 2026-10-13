"""End-to-end E-path analysis pipeline."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from epath.analysis.atoms import EpathAtom, atoms_from_bdamage_pipeline
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
)
from epath.outputs.files import (
    EpathOutputPipelineOptions,
    EpathPipelineOutputFiles,
)
from epath.outputs.runs import EpathRunRecord, build_run_record
from epath.outputs.write import write_epath_outputs
from rabdam.bnet.pipeline import BnetPipelineResult, run_bnet_pipeline
from rabdam.pipeline import BDamagePipelineResult, PipelineStageRunner
from rabdam.structure.models import PreparedAtom
from xtal.validate import ValidatedCrystal


class EpathPipelineError(ValueError):
    """Raised when E-path cannot run the analysis pipeline."""


@dataclass(frozen=True, slots=True)
class EpathSelectionPipelineOptions:
    """Options controlling E-path atom selection."""

    percentile: float = 2.0


@dataclass(frozen=True, slots=True)
class EpathClusterPipelineOptions:
    """Options controlling E-path cluster construction."""

    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM
    minimum_cluster_size: int = 8


@dataclass(frozen=True, slots=True)
class EpathPipelineOptions:
    """Options for the complete E-path analysis pipeline."""

    selection: EpathSelectionPipelineOptions = field(
        default_factory=EpathSelectionPipelineOptions
    )
    clustering: EpathClusterPipelineOptions = field(
        default_factory=EpathClusterPipelineOptions
    )
    paths: PathAnalysisOptions = field(default_factory=PathAnalysisOptions)
    output: EpathOutputPipelineOptions = field(
        default_factory=EpathOutputPipelineOptions
    )


@dataclass(frozen=True, slots=True)
class EpathPipelineResult:
    """Complete result of an E-path analysis pipeline."""

    atoms: tuple[EpathAtom, ...]
    selection_result: EpathSelectionResult
    clusters: tuple[EpathCluster, ...]
    component_statistics: ComponentStatistics
    path_reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]
    run_record: EpathRunRecord
    bdamage_pipeline_result: BDamagePipelineResult | None = None
    bnet_pipeline_result: BnetPipelineResult | None = None

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
        """Validated crystal from the BDamage pipeline, when available."""
        if self.bdamage_pipeline_result is None:
            return None
        return self.bdamage_pipeline_result.crystal

    @property
    def cleaned_atoms(self) -> tuple[PreparedAtom, ...] | None:
        """Cleaned model atoms from the BDamage pipeline, when available."""
        if self.bdamage_pipeline_result is None:
            return None
        return self.bdamage_pipeline_result.prepared_structure.cleaned_atoms


def run_epath_pipeline(
    bdamage_pipeline_result: BDamagePipelineResult,
    *,
    structure_path: str | Path | None = None,
    options: EpathPipelineOptions | None = None,
    stage_runner: PipelineStageRunner | None = None,
) -> tuple[EpathPipelineResult, EpathPipelineOutputFiles]:
    """Run E-path analysis from a completed RABDAM BDamage pipeline.

    This is the single entry point for callers that hold a BDamage pipeline
    result; the command-line interface uses it too, passing a `stage_runner`
    that reports each named stage as it runs.
    """
    atoms = atoms_from_bdamage_pipeline(bdamage_pipeline_result)
    if structure_path is None:
        structure_path = bdamage_pipeline_result.prepared_structure.metadata.source_path

    return analyze_epath_atoms(
        atoms,
        structure_path=structure_path,
        options=options,
        bdamage_pipeline_result=bdamage_pipeline_result,
        stage_runner=stage_runner,
    )


def analyze_epath_atoms(
    atoms: Iterable[EpathAtom],
    *,
    structure_path: str | Path | None = None,
    options: EpathPipelineOptions | None = None,
    bdamage_pipeline_result: BDamagePipelineResult | None = None,
    stage_runner: PipelineStageRunner | None = None,
) -> tuple[EpathPipelineResult, EpathPipelineOutputFiles]:
    """Run E-path selection, clustering, MST analysis, and outputs.

    Returns the analysis result and, separately, the files written from it.

    Selection, clustering, and pathway geometry need only the scored `atoms`.
    Everything that must see the rest of the model depends on
    `bdamage_pipeline_result`, whose cleaned `PreparedAtom`s carry the
    cofactor, solvent, and chain information that E-path atoms do not:

    - endpoint classification (cofactor sites, chain interfaces, surface)
    - the cleaned model PDB with BDamage in the B-factor column
    - Bnet and the Bnet percentile

    Without it those degrade quietly: paths are still reported, but with no
    cofactor or interface detection, and no model PDB or Bnet.

    Bnet is calculated here whenever a BDamage pipeline result is available,
    so every caller sees the same sequence of stages. `stage_runner` wraps
    each named stage for timing or progress reporting; without it the stages
    run silently.

    Raises:
        EpathPipelineError: If `atoms` is empty.
    """
    if options is None:
        options = EpathPipelineOptions()

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise EpathPipelineError("E-path analysis requires at least one atom.")

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
        bdamage_pipeline_result.prepared_structure.cleaned_atoms
        if bdamage_pipeline_result is not None
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
    bnet_pipeline_result = (
        _run_stage(
            "Calculating Bnet and Bnet percentile",
            lambda: _calculate_bnet(bdamage_pipeline_result),
            stage_runner=stage_runner,
        )
        if bdamage_pipeline_result is not None
        else None
    )
    run_record = build_run_record(
        structure_path=structure_path,
        selection_result=selection_result,
        clusters=clusters,
        statistics=statistics,
        minimum_cluster_size=minimum_cluster_size,
        cutoff_percent=options.selection.percentile,
        bnet_pipeline_result=bnet_pipeline_result,
    )
    result = EpathPipelineResult(
        atoms=atom_tuple,
        selection_result=selection_result,
        clusters=clusters,
        component_statistics=statistics,
        path_reports=path_analysis.reports,
        cofactor_sites=path_analysis.cofactor_sites,
        run_record=run_record,
        bdamage_pipeline_result=bdamage_pipeline_result,
        bnet_pipeline_result=bnet_pipeline_result,
    )
    output_files = _run_stage(
        "Writing E-path outputs",
        lambda: write_epath_outputs(result, options.output),
        stage_runner=stage_runner,
    )

    return result, output_files


def _calculate_bnet(
    bdamage_pipeline_result: BDamagePipelineResult,
) -> BnetPipelineResult:
    return run_bnet_pipeline(
        prepared_structure=bdamage_pipeline_result.prepared_structure,
        bdamage_results=bdamage_pipeline_result.bdamage_results,
    )


def _run_stage[T](
    label: str,
    callback: Callable[[], T],
    *,
    stage_runner: PipelineStageRunner | None,
) -> T:
    if stage_runner is None:
        return callback()

    return stage_runner(label, callback)


def _select_atoms(
    atoms: tuple[EpathAtom, ...],
    options: EpathSelectionPipelineOptions,
) -> EpathSelectionResult:
    return select_atoms_by_percentile(atoms, percentile=options.percentile)


def _positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise EpathPipelineError(f"{name} must be a positive integer, got {value!r}.")
    return value
