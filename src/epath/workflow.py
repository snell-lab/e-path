"""End-to-end E-path analysis workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from epath.annotations import EpathAtomAnnotation, annotate_atoms, compute_annotation_maps
from epath.atoms import EpathAtom, atoms_from_bdamage_workflow
from epath.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    EpathCluster,
    cluster_selected_atoms,
    keep_largest_cluster,
)
from epath.endpoints import (
    DEFAULT_SNAP_RADIUS_ANGSTROM,
    EpathResolvedEndpoint,
    resolve_endpoint,
)
from epath.null import (
    DEFAULT_NULL_ITERATIONS,
    DEFAULT_NULL_SEED,
    EpathNullModelComparison,
    EpathNullModelResult,
    compare_cluster_to_null,
    run_null_model,
)
from epath.outputs import (
    EpathOutputWorkflowOptions,
    EpathWorkflowOutputFiles,
    write_epath_outputs,
)
from epath.paths import (
    DEFAULT_PATH_EDGE_DISTANCE_ANGSTROM,
    EpathPath,
    find_shortest_path,
)
from epath.rates import (
    DEFAULT_BETA_PER_ANGSTROM,
    DEFAULT_DELTA_G_EV,
    DEFAULT_REORGANIZATION_ENERGY_EV,
    DEFAULT_TEMPERATURE_K,
    EpathClusterRate,
    EpathMarcusRate,
    cluster_geometric_rates,
    marcus_rate_for_path,
)
from epath.selection import (
    DEFAULT_HISTOGRAM_BINS,
    EpathSelectionResult,
    select_atoms_by_percentile,
    select_atoms_by_percentile_per_chain,
    select_atoms_by_z,
)
from rabdam.workflow import BDamageWorkflowResult


SelectionMethod = Literal["z_cutoff", "percentile", "percentile_per_chain"]
WarningHandler = Callable[[str], None]


class EpathWorkflowError(ValueError):
    """Raised when E-path cannot run the analysis workflow."""


@dataclass(frozen=True, slots=True)
class EpathSelectionWorkflowOptions:
    """Options controlling E-path atom selection."""

    method: SelectionMethod = "percentile"
    z_cutoff: float = 0.0
    percentile: float = 2.0
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS


@dataclass(frozen=True, slots=True)
class EpathClusterWorkflowOptions:
    """Options controlling E-path cluster construction."""

    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM
    keep_largest_only: bool = False


@dataclass(frozen=True, slots=True)
class EpathPathWorkflowOptions:
    """Options controlling donor-to-acceptor path calculation."""

    donor_token: str | None = None
    acceptor_token: str | None = None
    edge_distance: float = DEFAULT_PATH_EDGE_DISTANCE_ANGSTROM
    snap_radius: float = DEFAULT_SNAP_RADIUS_ANGSTROM
    allowed_chains: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EpathRateWorkflowOptions:
    """Options controlling path rate-proxy calculations."""

    enabled: bool = True
    reorganization_energy_ev: float = DEFAULT_REORGANIZATION_ENERGY_EV
    delta_g_ev: float = DEFAULT_DELTA_G_EV
    temperature_k: float = DEFAULT_TEMPERATURE_K
    beta_per_angstrom: float = DEFAULT_BETA_PER_ANGSTROM


@dataclass(frozen=True, slots=True)
class EpathAnnotationWorkflowOptions:
    """Options controlling optional structural annotations."""

    enabled: bool = False
    disable_sasa: bool = False


@dataclass(frozen=True, slots=True)
class EpathNullModelWorkflowOptions:
    """Options controlling the Monte Carlo null model."""

    enabled: bool = False
    iterations: int = DEFAULT_NULL_ITERATIONS
    seed: int = DEFAULT_NULL_SEED


@dataclass(frozen=True, slots=True)
class EpathWorkflowOptions:
    """Options for the complete E-path analysis workflow."""

    selection: EpathSelectionWorkflowOptions = field(
        default_factory=EpathSelectionWorkflowOptions
    )
    clustering: EpathClusterWorkflowOptions = field(
        default_factory=EpathClusterWorkflowOptions
    )
    path: EpathPathWorkflowOptions = field(default_factory=EpathPathWorkflowOptions)
    rates: EpathRateWorkflowOptions = field(default_factory=EpathRateWorkflowOptions)
    annotations: EpathAnnotationWorkflowOptions = field(
        default_factory=EpathAnnotationWorkflowOptions
    )
    null_model: EpathNullModelWorkflowOptions = field(
        default_factory=EpathNullModelWorkflowOptions
    )
    output: EpathOutputWorkflowOptions = field(default_factory=EpathOutputWorkflowOptions)
    allowed_chains: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EpathWorkflowResult:
    """Complete result of an E-path analysis workflow."""

    atoms: tuple[EpathAtom, ...]
    selection_result: EpathSelectionResult
    clusters: tuple[EpathCluster, ...]
    cluster_rates: tuple[EpathClusterRate, ...]
    annotations: tuple[EpathAtomAnnotation, ...]
    null_model_result: EpathNullModelResult | None
    null_comparisons: Mapping[int, EpathNullModelComparison]
    endpoints: Mapping[str, EpathResolvedEndpoint]
    path: EpathPath | None
    marcus_rate: EpathMarcusRate | None
    output_files: EpathWorkflowOutputFiles
    warnings: tuple[str, ...]
    bdamage_workflow_result: BDamageWorkflowResult | None = None


def run_epath_workflow(
    bdamage_workflow_result: BDamageWorkflowResult,
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
    warn: WarningHandler | None = None,
) -> EpathWorkflowResult:
    """Run E-path analysis from a completed RABDAM BDamage workflow."""

    atoms = atoms_from_bdamage_workflow(bdamage_workflow_result)
    if structure_path is None:
        structure_path = bdamage_workflow_result.prepared_structure.metadata.source_path

    return analyze_epath_atoms(
        atoms,
        structure_path=structure_path,
        options=options,
        warn=warn,
        bdamage_workflow_result=bdamage_workflow_result,
    )


def analyze_epath_atoms(
    atoms: Iterable[EpathAtom],
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
    warn: WarningHandler | None = None,
    bdamage_workflow_result: BDamageWorkflowResult | None = None,
) -> EpathWorkflowResult:
    """Run E-path selection, clustering, optional path analysis, and outputs."""

    if options is None:
        options = EpathWorkflowOptions()

    warnings: list[str] = []
    warning_handler = _warning_handler(warnings, warn)
    all_atoms = tuple(atoms)
    if not all_atoms:
        raise EpathWorkflowError("E-path analysis requires at least one atom.")
    atom_tuple = _filter_atoms_by_chain(
        all_atoms,
        allowed_chains=options.allowed_chains,
    )

    selection_result = _select_atoms(atom_tuple, options.selection)
    clusters = cluster_selected_atoms(
        selection_result.selected_atoms,
        distance_cutoff=options.clustering.distance_cutoff,
    )
    if options.clustering.keep_largest_only:
        clusters = keep_largest_cluster(clusters)

    cluster_rates = cluster_geometric_rates(
        clusters,
        beta_per_angstrom=options.rates.beta_per_angstrom,
    )

    null_model_result, null_comparisons = _run_null_model(
        selection_result=selection_result,
        clusters=clusters,
        options=options,
    )

    endpoints, path = _resolve_path(
        atoms=atom_tuple,
        selection_result=selection_result,
        options=options.path,
        warn=warning_handler,
    )

    marcus_rate = (
        marcus_rate_for_path(
            path,
            reorganization_energy_ev=options.rates.reorganization_energy_ev,
            delta_g_ev=options.rates.delta_g_ev,
            temperature_k=options.rates.temperature_k,
            beta_per_angstrom=options.rates.beta_per_angstrom,
        )
        if path is not None and options.rates.enabled
        else None
    )

    annotations = _annotate_atoms(
        atoms=atom_tuple,
        structure_path=structure_path,
        options=options.annotations,
        warn=warning_handler,
    )

    analyzed_chains = tuple(dict.fromkeys(atom.chain_id for atom in atom_tuple))
    output_files = write_epath_outputs(
        selection_result=selection_result,
        clusters=clusters,
        annotations=annotations,
        path=path,
        marcus_rate=marcus_rate,
        cluster_rates=cluster_rates,
        endpoints=endpoints,
        null_model_result=null_model_result,
        null_comparisons=null_comparisons,
        analyzed_chains=analyzed_chains,
        warnings=tuple(warnings),
        structure_path=structure_path,
        requested_donor=options.path.donor_token,
        requested_acceptor=options.path.acceptor_token,
        edge_distance=options.path.edge_distance,
        snap_radius=options.path.snap_radius,
        reorganization_energy_ev=options.rates.reorganization_energy_ev,
        delta_g_ev=options.rates.delta_g_ev,
        temperature_k=options.rates.temperature_k,
        beta_per_angstrom=options.rates.beta_per_angstrom,
        options=options.output,
        histogram_bins=options.selection.histogram_bins,
    )

    return EpathWorkflowResult(
        atoms=atom_tuple,
        selection_result=selection_result,
        clusters=clusters,
        cluster_rates=cluster_rates,
        annotations=annotations,
        null_model_result=null_model_result,
        null_comparisons=null_comparisons,
        endpoints=endpoints,
        path=path,
        marcus_rate=marcus_rate,
        output_files=output_files,
        warnings=tuple(warnings),
        bdamage_workflow_result=bdamage_workflow_result,
    )


def _filter_atoms_by_chain(
    atoms: tuple[EpathAtom, ...],
    *,
    allowed_chains: tuple[str, ...] | None,
) -> tuple[EpathAtom, ...]:
    """Restrict downstream E-path analysis to requested chain identifiers."""

    if allowed_chains is None:
        return atoms

    normalized_chains = tuple(
        dict.fromkeys(
            chain_id.strip()
            for chain_id in allowed_chains
            if chain_id.strip()
        )
    )
    if not normalized_chains:
        raise EpathWorkflowError("At least one non-empty chain ID is required.")

    available_chains = {atom.chain_id for atom in atoms}
    missing_chains = tuple(
        chain_id
        for chain_id in normalized_chains
        if chain_id not in available_chains
    )
    if missing_chains:
        missing_text = ", ".join(missing_chains)
        available_text = ", ".join(sorted(available_chains)) or "none"
        raise EpathWorkflowError(
            f"Requested chain(s) have no BDamage-scored atoms: {missing_text}. "
            f"Chains available for E-path analysis: {available_text}."
        )

    allowed_chain_set = set(normalized_chains)
    return tuple(
        atom
        for atom in atoms
        if atom.chain_id in allowed_chain_set
    )


def _select_atoms(
    atoms: tuple[EpathAtom, ...],
    options: EpathSelectionWorkflowOptions,
) -> EpathSelectionResult:
    if options.method == "z_cutoff":
        return select_atoms_by_z(
            atoms,
            z_cutoff=options.z_cutoff,
            histogram_bins=options.histogram_bins,
        )

    if options.method == "percentile":
        return select_atoms_by_percentile(
            atoms,
            percentile=options.percentile,
            histogram_bins=options.histogram_bins,
        )

    if options.method == "percentile_per_chain":
        return select_atoms_by_percentile_per_chain(
            atoms,
            percentile=options.percentile,
            histogram_bins=options.histogram_bins,
        )

    raise EpathWorkflowError(f"Unknown selection method: {options.method!r}.")


def _run_null_model(
    *,
    selection_result: EpathSelectionResult,
    clusters: tuple[EpathCluster, ...],
    options: EpathWorkflowOptions,
) -> tuple[EpathNullModelResult | None, dict[int, EpathNullModelComparison]]:
    if not options.null_model.enabled:
        return None, {}

    null_model_result = run_null_model(
        selection_result.scored_atoms,
        sample_size=len(selection_result.selected_atoms),
        distance_cutoff=options.clustering.distance_cutoff,
        iterations=options.null_model.iterations,
        seed=options.null_model.seed,
    )
    if not clusters:
        return null_model_result, {}

    largest_cluster = max(clusters, key=lambda cluster: cluster.atom_count)
    return null_model_result, {
        largest_cluster.cluster_id: compare_cluster_to_null(
            largest_cluster,
            null_model_result,
        )
    }


def _resolve_path(
    *,
    atoms: tuple[EpathAtom, ...],
    selection_result: EpathSelectionResult,
    options: EpathPathWorkflowOptions,
    warn: WarningHandler,
) -> tuple[dict[str, EpathResolvedEndpoint], EpathPath | None]:
    if options.donor_token is None and options.acceptor_token is None:
        return {}, None

    if options.donor_token is None or options.acceptor_token is None:
        warn("Donor and acceptor tokens are both required for path analysis.")
        return {}, None

    if len(selection_result.selected_atoms) <= 1:
        warn("Path analysis skipped because fewer than two atoms were selected.")
        return {}, None

    endpoints: dict[str, EpathResolvedEndpoint] = {}
    donor = resolve_endpoint(
        options.donor_token,
        all_atoms=atoms,
        selected_atoms=selection_result.selected_atoms,
        snap_radius=options.snap_radius,
        allowed_chains=options.allowed_chains,
    )
    if donor is None:
        warn(f"Donor endpoint could not be resolved: {options.donor_token!r}.")
    else:
        endpoints["donor"] = donor

    acceptor = resolve_endpoint(
        options.acceptor_token,
        all_atoms=atoms,
        selected_atoms=selection_result.selected_atoms,
        snap_radius=options.snap_radius,
        allowed_chains=options.allowed_chains,
    )
    if acceptor is None:
        warn(f"Acceptor endpoint could not be resolved: {options.acceptor_token!r}.")
    else:
        endpoints["acceptor"] = acceptor

    if donor is None or acceptor is None:
        return endpoints, None

    path = find_shortest_path(
        selection_result.selected_atoms,
        source_index=donor.selected_index,
        target_index=acceptor.selected_index,
        edge_distance=options.edge_distance,
    )
    if path is None:
        warn("No connected donor-to-acceptor path was found in selected atoms.")

    return endpoints, path


def _annotate_atoms(
    *,
    atoms: tuple[EpathAtom, ...],
    structure_path: str | Path | None,
    options: EpathAnnotationWorkflowOptions,
    warn: WarningHandler,
) -> tuple[EpathAtomAnnotation, ...]:
    if not options.enabled:
        return annotate_atoms(atoms)

    if structure_path is None:
        warn("Structural annotations skipped because no structure path was provided.")
        return annotate_atoms(atoms)

    annotation_maps = compute_annotation_maps(
        structure_path,
        atoms,
        disable_sasa=options.disable_sasa,
        warn=warn,
    )
    return annotate_atoms(
        atoms,
        secondary_structure_map=annotation_maps.secondary_structure_by_residue,
        sasa_map=annotation_maps.sasa_by_atom,
    )


def _warning_handler(
    warnings: list[str],
    external_warn: WarningHandler | None,
) -> WarningHandler:
    def handle_warning(message: str) -> None:
        warnings.append(message)
        if external_warn is not None:
            external_warn(message)

    return handle_warning
