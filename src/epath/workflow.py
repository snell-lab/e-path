"""End-to-end E-path analysis workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from epath.atoms import EpathAtom, atoms_from_bdamage_workflow
from epath.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    EpathCluster,
    cluster_selected_atoms,
    keep_largest_cluster,
)
from epath.outputs import (
    EpathOutputWorkflowOptions,
    EpathWorkflowOutputFiles,
    write_epath_outputs,
)
from epath.selection import (
    DEFAULT_HISTOGRAM_BINS,
    EpathSelectionResult,
    select_atoms_by_percentile,
    select_atoms_by_percentile_per_chain,
    select_atoms_by_z,
)
from rabdam.bnet.workflow import BnetWorkflowResult, calculate_bnet_workflow
from rabdam.workflow import BDamageWorkflowResult

SelectionMethod = Literal["z_cutoff", "percentile", "percentile_per_chain"]


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
    minimum_cluster_size: int = 8
    keep_largest_only: bool = False


@dataclass(frozen=True, slots=True)
class EpathWorkflowOptions:
    """Options for the complete E-path analysis workflow."""

    selection: EpathSelectionWorkflowOptions = field(
        default_factory=EpathSelectionWorkflowOptions
    )
    clustering: EpathClusterWorkflowOptions = field(
        default_factory=EpathClusterWorkflowOptions
    )
    output: EpathOutputWorkflowOptions = field(
        default_factory=EpathOutputWorkflowOptions
    )
    allowed_chains: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EpathWorkflowResult:
    """Complete result of an E-path analysis workflow."""

    atoms: tuple[EpathAtom, ...]
    selection_result: EpathSelectionResult
    clusters: tuple[EpathCluster, ...]
    output_files: EpathWorkflowOutputFiles
    bdamage_workflow_result: BDamageWorkflowResult | None = None
    bnet_workflow_result: BnetWorkflowResult | None = None


def run_epath_workflow(
    bdamage_workflow_result: BDamageWorkflowResult,
    *,
    structure_path: str | Path | None = None,
    options: EpathWorkflowOptions | None = None,
) -> EpathWorkflowResult:
    """Run E-path analysis from a completed RABDAM BDamage workflow."""

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
    minimum_cluster_size = _positive_int(
        options.clustering.minimum_cluster_size,
        name="minimum_cluster_size",
    )
    clusters = tuple(
        cluster for cluster in clusters if cluster.atom_count >= minimum_cluster_size
    )
    if options.clustering.keep_largest_only:
        clusters = keep_largest_cluster(clusters)

    analyzed_chains = tuple(dict.fromkeys(atom.chain_id for atom in atom_tuple))
    if bnet_workflow_result is None and bdamage_workflow_result is not None:
        bnet_workflow_result = calculate_bnet_workflow(
            prepared_structure=bdamage_workflow_result.prepared_structure,
            bdamage_score_result=bdamage_workflow_result.bdamage_score_result,
        )
    output_files = write_epath_outputs(
        selection_result=selection_result,
        clusters=clusters,
        analyzed_chains=analyzed_chains,
        structure_path=structure_path,
        minimum_cluster_size=minimum_cluster_size,
        options=options.output,
        histogram_bins=options.selection.histogram_bins,
        bnet_workflow_result=bnet_workflow_result,
    )

    return EpathWorkflowResult(
        atoms=atom_tuple,
        selection_result=selection_result,
        clusters=clusters,
        output_files=output_files,
        bdamage_workflow_result=bdamage_workflow_result,
        bnet_workflow_result=bnet_workflow_result,
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
            chain_id.strip() for chain_id in allowed_chains if chain_id.strip()
        )
    )
    if not normalized_chains:
        raise EpathWorkflowError("At least one non-empty chain ID is required.")

    available_chains = {atom.chain_id for atom in atoms}
    missing_chains = tuple(
        chain_id for chain_id in normalized_chains if chain_id not in available_chains
    )
    if missing_chains:
        missing_text = ", ".join(missing_chains)
        available_text = ", ".join(sorted(available_chains)) or "none"
        raise EpathWorkflowError(
            f"Requested chain(s) have no BDamage-scored atoms: {missing_text}. "
            f"Chains available for E-path analysis: {available_text}."
        )

    allowed_chain_set = set(normalized_chains)
    return tuple(atom for atom in atoms if atom.chain_id in allowed_chain_set)


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


def _positive_int(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise EpathWorkflowError(f"{name} must be a positive integer, got {value!r}.")
    return value
