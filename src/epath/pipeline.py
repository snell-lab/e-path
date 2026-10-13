"""Run the E-path analysis from per-atom BDamage scores to written outputs.

`rabdam.pipeline` supplies the BDamage result this starts from. The stages
then run in order, each from its own module. `epath.analysis` selects the
lowest-BDamage atoms, clusters them and describes each cluster as a pathway,
`rabdam.bnet` adds Bnet and its percentile, and `epath.outputs` writes the
files. `epath.cli` calls `run_epath_pipeline` once per structure.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

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


class EpathPipelineError(ValueError):
    """Report invalid input to the E-path analysis pipeline."""


@dataclass(frozen=True, slots=True)
class EpathSelectionPipelineOptions:
    """Set how many atoms are selected as pathway candidates.

    `percentile` is the percentage of the scored atoms to select, taken from
    the low end of BDamage. It must lie between 0 and 100, which the selection
    stage checks.
    """

    percentile: float = 2.0


@dataclass(frozen=True, slots=True)
class EpathClusterPipelineOptions:
    """Set how selected atoms are joined into clusters and which are reported.

    Two selected atoms are connected when they lie within `distance_cutoff`
    angstroms of each other. A connected component is reported as a cluster
    only when it holds at least `minimum_cluster_size` atoms.
    """

    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM
    minimum_cluster_size: int = 8


@dataclass(frozen=True, slots=True)
class EpathPipelineOptions:
    """Group the options of every stage of the E-path pipeline.

    Constructing one checks nothing; `run_epath_pipeline` and its stages do.
    """

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
    """Store the results of an E-path analysis.

    `selection_result` holds every scored atom and the ones selected from
    them. `clusters` holds the connected components that reached the
    minimum cluster size, and `path_reports` describes each of them as a
    pathway, in the same order. `component_statistics` covers every component,
    reported or not. `cofactor_sites` holds the redox cofactors of the model,
    which the pathways were labelled against. `run_record` is the row written
    to `*_run.csv`.

    `bdamage_pipeline_result` is the BDamage result the analysis started from,
    with the cleaned model and the validated crystal, and
    `bnet_pipeline_result` the Bnet calculated from it.
    """

    selection_result: EpathSelectionResult
    clusters: tuple[EpathCluster, ...]
    component_statistics: ComponentStatistics
    path_reports: tuple[PathReport, ...]
    cofactor_sites: tuple[CofactorSite, ...]
    run_record: EpathRunRecord
    bdamage_pipeline_result: BDamagePipelineResult
    bnet_pipeline_result: BnetPipelineResult

    @property
    def analyzed_chains(self) -> tuple[str, ...]:
        """Chain identifiers of the scored atoms, in first-seen order."""
        return tuple(
            dict.fromkeys(
                atom.record.chain_id for atom in self.selection_result.scored_atoms
            )
        )


def run_epath_pipeline(
    bdamage_pipeline_result: BDamagePipelineResult,
    *,
    options: EpathPipelineOptions | None = None,
    stage_runner: PipelineStageRunner | None = None,
) -> tuple[EpathPipelineResult, EpathPipelineOutputFiles]:
    """Run the E-path analysis on a BDamage result and write its outputs.

    The atoms that received a BDamage score are selected, clustered and
    described as pathways. The rest of the cleaned model supplies what the
    scored atoms cannot. It holds the cofactor sites and the chains that the
    pathway endpoints are labelled against, and it is the model written with
    BDamage in its B-factor column. Bnet and its percentile are calculated
    from the same result. The analysis result is returned with, separately,
    the files written from it.

    Without `options` the defaults of `EpathPipelineOptions` apply. A
    `stage_runner` is handed each stage as a label and a callback, so a caller
    can time or report the stages, as the command line does.

    An `EpathPipelineError` is raised before any stage runs when
    `minimum_cluster_size` is below 1 or a display chain is not in the cleaned
    model. The percentile and the distance cutoff are checked by the selection
    and clustering stages, which raise their own errors.
    """
    if options is None:
        options = EpathPipelineOptions()

    prepared_structure = bdamage_pipeline_result.prepared_structure
    cleaned_atoms = prepared_structure.cleaned_atoms
    minimum_cluster_size = options.clustering.minimum_cluster_size
    if minimum_cluster_size <= 0:
        raise EpathPipelineError(
            "minimum_cluster_size must be a positive integer, "
            f"got {minimum_cluster_size!r}."
        )

    # A chain that holds only solvent or ligands is still a chain of the
    # model, so the display chains are checked against the cleaned model, not
    # the scored atoms.
    model_chains = dict.fromkeys(atom.record.chain_id for atom in cleaned_atoms)
    unknown_chains = [
        chain
        for chain in options.output.display_chains or ()
        if chain not in model_chains
    ]
    if unknown_chains:
        raise EpathPipelineError(
            "The coordinate files cannot be limited to chains that are not in "
            f"the model: {', '.join(unknown_chains)}. "
            f"The model has chains {', '.join(model_chains)}."
        )

    selection_result = _run_stage(
        "Selecting E-path atoms",
        lambda: select_atoms_by_percentile(
            prepared_structure.selected_atoms,
            bdamage_pipeline_result.bdamage_results,
            percentile=options.selection.percentile,
        ),
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
    # Components below the reporting threshold are not pathways, but the
    # connectivity analysis still found them, so their count and the geometry
    # of the largest one are recorded before they are dropped.
    statistics = component_statistics(components)
    clusters = tuple(
        cluster for cluster in components if cluster.atom_count >= minimum_cluster_size
    )
    path_analysis = _run_stage(
        "Analyzing pathways",
        lambda: analyze_paths(
            clusters,
            scored_atoms=selection_result.scored_atoms,
            cleaned_atoms=cleaned_atoms,
            distance_cutoff=options.clustering.distance_cutoff,
            options=options.paths,
        ),
        stage_runner=stage_runner,
    )
    bnet_pipeline_result = _run_stage(
        "Calculating Bnet and Bnet percentile",
        lambda: run_bnet_pipeline(
            prepared_structure=prepared_structure,
            bdamage_results=bdamage_pipeline_result.bdamage_results,
        ),
        stage_runner=stage_runner,
    )
    run_record = build_run_record(
        structure_path=prepared_structure.metadata.source_path,
        selection_result=selection_result,
        clusters=clusters,
        statistics=statistics,
        minimum_cluster_size=minimum_cluster_size,
        cutoff_percent=options.selection.percentile,
        bnet_pipeline_result=bnet_pipeline_result,
    )
    result = EpathPipelineResult(
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


def _run_stage[T](
    label: str,
    callback: Callable[[], T],
    *,
    stage_runner: PipelineStageRunner | None,
) -> T:
    if stage_runner is None:
        return callback()

    return stage_runner(label, callback)
