"""Coordinate E-path output files and human-readable reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from epath.annotations import EpathAtomAnnotation
from epath.clustering import EpathCluster
from epath.endpoints import EpathResolvedEndpoint, atom_token, atom_token_help_text
from epath.null import EpathNullModelComparison, EpathNullModelResult
from epath.paths import EpathPath
from epath.plots import plot_z_hist
from epath.rates import EpathClusterRate, EpathMarcusRate
from epath.selection import EpathSelectionResult
from epath.structures import PdbBFactorMode, write_structure_outputs
from epath.tables import write_cluster_csv, write_combined_atom_csv


@dataclass(frozen=True, slots=True)
class EpathOutputWorkflowOptions:
    """Options controlling E-path output files."""

    output_prefix: str | Path | None = None
    write_combined_csv: bool = True
    write_structure_files: bool = True
    write_z_score_plot: bool = True
    structure_b_factor_mode: PdbBFactorMode = "bdamage"
    write_cluster_csv: bool = True
    write_summary_report: bool = True
    write_marcus_report: bool = True


@dataclass(frozen=True, slots=True)
class EpathWorkflowOutputFiles:
    """Files written by an E-path workflow run."""

    combined_csv: Path | None = None
    selected_pdb: Path | None = None
    path_pdb: Path | None = None
    z_score_histogram: Path | None = None
    cluster_csv: Path | None = None
    summary_report: Path | None = None
    marcus_report: Path | None = None


def write_epath_outputs(
    *,
    selection_result: EpathSelectionResult,
    clusters: tuple[EpathCluster, ...],
    annotations: tuple[EpathAtomAnnotation, ...],
    path: EpathPath | None,
    marcus_rate: EpathMarcusRate | None,
    cluster_rates: tuple[EpathClusterRate, ...],
    endpoints: Mapping[str, EpathResolvedEndpoint],
    null_model_result: EpathNullModelResult | None,
    null_comparisons: Mapping[int, EpathNullModelComparison],
    analyzed_chains: Sequence[str],
    warnings: Sequence[str],
    structure_path: str | Path | None,
    requested_donor: str | None,
    requested_acceptor: str | None,
    minimum_cluster_size: int,
    edge_distance: float,
    snap_radius: float,
    reorganization_energy_ev: float,
    delta_g_ev: float,
    temperature_k: float,
    beta_per_angstrom: float,
    options: EpathOutputWorkflowOptions,
    histogram_bins: int,
) -> EpathWorkflowOutputFiles:
    """Write enabled E-path artifacts and return their paths."""

    if options.output_prefix is None:
        return EpathWorkflowOutputFiles()

    prefix = Path(options.output_prefix)
    combined_csv: Path | None = None
    if options.write_combined_csv:
        combined_csv = write_combined_atom_csv(
            _output_path(prefix, "epath.csv"),
            selection_result,
            clusters=clusters,
            annotations=annotations,
            path=path,
            marcus_rate=marcus_rate,
            cluster_rates=cluster_rates,
            endpoint_roles=endpoints,
            null_comparisons=null_comparisons,
        )

    cluster_csv: Path | None = None
    if options.write_cluster_csv:
        cluster_csv = write_cluster_csv(
            _output_path(prefix, "clusters.csv"),
            clusters,
            cluster_rates=cluster_rates,
            null_comparisons=null_comparisons,
        )

    selected_pdb: Path | None = None
    path_pdb: Path | None = None
    if options.write_structure_files:
        structure_files = write_structure_outputs(
            prefix,
            selection_result.selected_atoms,
            path=path,
            b_factor_mode=options.structure_b_factor_mode,
        )
        selected_pdb = structure_files.selected_pdb
        path_pdb = structure_files.path_pdb

    z_score_histogram: Path | None = None
    if options.write_z_score_plot:
        z_score_histogram = plot_z_hist(
            prefix,
            [atom.bdamage_z_score for atom in selection_result.scored_atoms],
            selection_result.z_mode,
            bins=histogram_bins,
        )

    marcus_report: Path | None = None
    endpoint_analysis_requested = (
        requested_donor is not None or requested_acceptor is not None
    )
    if options.write_marcus_report and endpoint_analysis_requested:
        marcus_report = write_marcus_txt(
            _output_path(prefix, "marcus.txt"),
            requested_donor=requested_donor,
            requested_acceptor=requested_acceptor,
            endpoints=endpoints,
            path=path,
            marcus_rate=marcus_rate,
            edge_distance=edge_distance,
            snap_radius=snap_radius,
            reorganization_energy_ev=reorganization_energy_ev,
            delta_g_ev=delta_g_ev,
            temperature_k=temperature_k,
            beta_per_angstrom=beta_per_angstrom,
            analyzed_chains=analyzed_chains,
        )

    summary_report = (
        _output_path(prefix, "summary.txt")
        if options.write_summary_report
        else None
    )
    output_files = EpathWorkflowOutputFiles(
        combined_csv=combined_csv,
        cluster_csv=cluster_csv,
        selected_pdb=selected_pdb,
        path_pdb=path_pdb,
        z_score_histogram=z_score_histogram,
        summary_report=summary_report,
        marcus_report=marcus_report,
    )

    if summary_report is not None:
        write_summary_txt(
            summary_report,
            structure_path=structure_path,
            analyzed_chains=analyzed_chains,
            selection_result=selection_result,
            clusters=clusters,
            cluster_rates=cluster_rates,
            null_model_result=null_model_result,
            null_comparisons=null_comparisons,
            requested_donor=requested_donor,
            requested_acceptor=requested_acceptor,
            minimum_cluster_size=minimum_cluster_size,
            endpoints=endpoints,
            path=path,
            marcus_rate=marcus_rate,
            warnings=warnings,
            output_files=output_files,
        )

    return output_files


def write_summary_txt(
    output_path: str | Path,
    *,
    structure_path: str | Path | None,
    analyzed_chains: Sequence[str],
    selection_result: EpathSelectionResult,
    clusters: Sequence[EpathCluster],
    cluster_rates: Sequence[EpathClusterRate],
    null_model_result: EpathNullModelResult | None,
    null_comparisons: Mapping[int, EpathNullModelComparison],
    requested_donor: str | None,
    requested_acceptor: str | None,
    minimum_cluster_size: int,
    endpoints: Mapping[str, EpathResolvedEndpoint],
    path: EpathPath | None,
    marcus_rate: EpathMarcusRate | None,
    warnings: Sequence[str],
    output_files: EpathWorkflowOutputFiles,
) -> Path:
    """Write a persistent human-readable E-path analysis summary."""

    lines = [
        "E-path Analysis Summary",
        "=======================",
        "",
        (
            "Input structure: "
            f"{structure_path if structure_path is not None else 'not provided'}"
        ),
        f"Chains analyzed: {_chain_text(analyzed_chains)}",
        "",
        "Selection",
        "---------",
        f"Method: {selection_result.description}",
        (
            f"Selected atoms: {len(selection_result.selected_atoms)} / "
            f"{len(selection_result.scored_atoms)} "
            f"({_number(selection_result.selected_percent)}%)"
        ),
        f"Z-score mode: {_number(selection_result.z_mode)}",
        f"Z threshold: {_number(selection_result.z_threshold)}",
    ]
    if selection_result.bdamage_threshold is not None:
        lines.append(
            "BDamage threshold: "
            f"{_number(selection_result.bdamage_threshold)}"
        )
    if selection_result.boundary_z_score is not None:
        lines.append(
            "Boundary Z-score among selected atoms: "
            f"{_number(selection_result.boundary_z_score)}"
        )

    selected_chain_counts = Counter(
        atom.atom.chain_id for atom in selection_result.selected_atoms
    )
    lines.extend(("", "Selected atoms by chain", "-----------------------"))
    if selected_chain_counts:
        lines.extend(
            f"Chain {chain_id}: {count}"
            for chain_id, count in selected_chain_counts.items()
        )
    else:
        lines.append("No atoms selected.")

    if selection_result.chain_thresholds:
        lines.extend(
            ("", "Per-chain percentile thresholds", "-------------------------------")
        )
        for threshold in selection_result.chain_thresholds:
            lines.append(
                f"Chain {threshold.chain_id}: atoms={threshold.atom_count}, "
                f"selected={threshold.selected_count}, "
                f"BDamage threshold={_number(threshold.bdamage_threshold)}, "
                f"Z threshold={_number(threshold.z_threshold)}"
            )

    rates_by_cluster = {rate.cluster_id: rate for rate in cluster_rates}
    lines.extend(
        (
            "",
            "Clusters",
            "--------",
            f"Minimum reported cluster size: {minimum_cluster_size} atoms",
            f"Cluster count: {len(clusters)}",
        )
    )
    if not clusters:
        lines.append("No selected-atom clusters.")
    for cluster in clusters:
        metrics = cluster.metrics
        rate = rates_by_cluster.get(cluster.cluster_id)
        lines.extend(
            (
                "",
                f"Cluster {cluster.cluster_id}",
                f"  Atoms: {cluster.atom_count}",
                f"  MST total length: {_number(metrics.mst_total_length)} A",
                f"  MST mean step: {_number(metrics.mean_step)} A",
                f"  MST largest edge: {_number(metrics.mst_largest_edge)} A",
                f"  Span: {_number(metrics.span)} A",
            )
        )
        if rate is not None:
            lines.extend(
                (
                    f"  Geometric rate: {_number(rate.rate)}",
                    f"  Normalized rate: {_number(rate.normalized_rate)}",
                )
            )
        comparison = null_comparisons.get(cluster.cluster_id)
        if comparison is not None:
            lines.extend(_null_comparison_lines(comparison, indent="  "))

    lines.extend(("", "Null model", "----------"))
    if null_model_result is None:
        lines.append("Not run.")
    else:
        lines.extend(
            (
                f"Iterations: {null_model_result.iterations}",
                f"Sample size: {null_model_result.sample_size}",
                f"Population size: {null_model_result.population_size}",
                f"Distance cutoff: {_number(null_model_result.distance_cutoff)} A",
                f"Random seed: {null_model_result.seed}",
                (
                    "Largest cluster size: "
                    f"mean={_number(null_model_result.largest_mean)}, "
                    f"std={_number(null_model_result.largest_std)}"
                ),
                (
                    "Component count: "
                    f"mean={_number(null_model_result.ncomp_mean)}, "
                    f"std={_number(null_model_result.ncomp_std)}"
                ),
                (
                    "Largest-cluster MST length: "
                    f"mean={_number(null_model_result.mst_mean)} A, "
                    f"std={_number(null_model_result.mst_std)} A"
                ),
            )
        )

    lines.extend(("", "Donor-to-acceptor path", "----------------------"))
    lines.extend(
        _endpoint_summary_lines(
            requested_donor=requested_donor,
            requested_acceptor=requested_acceptor,
            endpoints=endpoints,
        )
    )
    if path is None:
        lines.append(
            "Connected path: "
            + _missing_path_explanation(
                requested_donor=requested_donor,
                requested_acceptor=requested_acceptor,
                endpoints=endpoints,
            )
        )
    else:
        lines.extend(
            (
                "Connected path: found",
                f"Path atoms: {path.atom_count}",
                f"Path steps: {path.step_count}",
                f"Path length: {_number(path.total_length)} A",
                f"Largest path step: {_number(path.largest_step)} A",
                "Path nodes:",
            )
        )
        lines.extend(f"  {atom_token(atom.atom)}" for atom in path.atoms)
    if marcus_rate is not None:
        lines.append(f"Marcus-style rate: {_number(marcus_rate.rate)}")

    lines.extend(("", "Warnings", "--------"))
    lines.extend(warnings if warnings else ("None.",))

    lines.extend(("", "Files written", "-------------"))
    file_entries = _output_file_entries(output_files)
    if file_entries:
        lines.extend(f"{label}: {path}" for label, path in file_entries)
    else:
        lines.append("None.")

    return _write_text(output_path, lines)


def write_marcus_txt(
    output_path: str | Path,
    *,
    requested_donor: str | None,
    requested_acceptor: str | None,
    endpoints: Mapping[str, EpathResolvedEndpoint],
    path: EpathPath | None,
    marcus_rate: EpathMarcusRate | None,
    edge_distance: float,
    snap_radius: float,
    reorganization_energy_ev: float,
    delta_g_ev: float,
    temperature_k: float,
    beta_per_angstrom: float,
    analyzed_chains: Sequence[str],
) -> Path:
    """Write endpoint syntax, resolution, path, and Marcus-rate details."""

    lines = [
        "Marcus-style Pathway Analysis",
        "=============================",
        "",
        "Atom token input format",
        "-----------------------",
        atom_token_help_text(),
        "",
        "Run settings",
        "------------",
        f"Chains analyzed: {_chain_text(analyzed_chains)}",
        f"Donor requested: {requested_donor or 'None'}",
        f"Acceptor requested: {requested_acceptor or 'None'}",
        f"Edge distance cutoff: {_number(edge_distance)} A",
        f"Snap radius: {_number(snap_radius)} A",
        f"Reorganization energy: {_number(reorganization_energy_ev)} eV",
        f"Delta G: {_number(delta_g_ev)} eV",
        f"Temperature: {_number(temperature_k)} K",
        f"Beta: {_number(beta_per_angstrom)} A^-1",
        "",
        "Endpoint resolution",
        "-------------------",
    ]
    lines.extend(
        _endpoint_summary_lines(
            requested_donor=requested_donor,
            requested_acceptor=requested_acceptor,
            endpoints=endpoints,
        )
    )
    lines.extend(("", "Path result", "-----------"))
    if path is None:
        lines.append(
            _missing_path_explanation(
                requested_donor=requested_donor,
                requested_acceptor=requested_acceptor,
                endpoints=endpoints,
            )
        )
    else:
        lines.extend(
            (
                f"Path length: {_number(path.total_length)} A",
                f"Path atoms: {path.atom_count}",
                f"Path steps: {path.step_count}",
                f"Largest step: {_number(path.largest_step)} A",
                "",
                "Path nodes (in order)",
                "---------------------",
            )
        )
        lines.extend(atom_token(atom.atom) for atom in path.atoms)

    lines.extend(("", "Marcus-style rate", "-----------------"))
    if marcus_rate is None:
        lines.append("Not calculated.")
    else:
        lines.extend(
            (
                f"Rate proxy: {_number(marcus_rate.rate)}",
                (
                    "Reorganization exponent: "
                    f"{_number(marcus_rate.reorganization_exponent)}"
                ),
                f"Distance exponent: {_number(marcus_rate.distance_exponent)}",
            )
        )

    return _write_text(output_path, lines)


def _endpoint_summary_lines(
    *,
    requested_donor: str | None,
    requested_acceptor: str | None,
    endpoints: Mapping[str, EpathResolvedEndpoint],
) -> tuple[str, ...]:
    lines: list[str] = []
    for role, requested_token in (
        ("donor", requested_donor),
        ("acceptor", requested_acceptor),
    ):
        label = role.capitalize()
        endpoint = endpoints.get(role)
        if requested_token is None:
            lines.append(f"{label}: not requested")
            continue
        if endpoint is None:
            lines.append(f"{label}: {requested_token} (unresolved)")
            continue
        lines.extend(
            (
                f"{label} requested: {requested_token}",
                f"{label} target: {atom_token(endpoint.target_atom)}",
                f"{label} selected node: {atom_token(endpoint.selected_atom.atom)}",
                f"{label} exact selected match: {endpoint.exact_match}",
                f"{label} snap distance: {_number(endpoint.snap_distance)} A",
            )
        )
    return tuple(lines)


def _null_comparison_lines(
    comparison: EpathNullModelComparison,
    *,
    indent: str,
) -> tuple[str, ...]:
    return (
        f"{indent}Null largest-cluster Z: {_number(comparison.largest_z_score)}",
        (
            f"{indent}Null largest-cluster p(X >= observed): "
            f"{_number(comparison.largest_p_value_greater)}"
        ),
        f"{indent}Null MST Z: {_number(comparison.mst_z_score)}",
        (
            f"{indent}Null MST p(X <= observed): "
            f"{_number(comparison.mst_p_value_less)}"
        ),
    )


def _missing_path_explanation(
    *,
    requested_donor: str | None,
    requested_acceptor: str | None,
    endpoints: Mapping[str, EpathResolvedEndpoint],
) -> str:
    if requested_donor is None and requested_acceptor is None:
        return "not requested"
    if requested_donor is None or requested_acceptor is None:
        return "not calculated because both donor and acceptor are required"
    if "donor" not in endpoints or "acceptor" not in endpoints:
        return "not calculated because one or both endpoints could not be resolved"
    return "not found in the selected-atom graph"


def _output_file_entries(
    output_files: EpathWorkflowOutputFiles,
) -> tuple[tuple[str, Path], ...]:
    entries = (
        ("Combined atom CSV", output_files.combined_csv),
        ("Cluster CSV", output_files.cluster_csv),
        ("Selected-atoms PDB", output_files.selected_pdb),
        ("Path PDB", output_files.path_pdb),
        ("Z-score histogram", output_files.z_score_histogram),
        ("Marcus report", output_files.marcus_report),
        ("Summary report", output_files.summary_report),
    )
    return tuple((label, path) for label, path in entries if path is not None)


def _output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{suffix}")


def _chain_text(chains: Sequence[str]) -> str:
    return ", ".join(chains) if chains else "none"


def _number(value: float) -> str:
    return f"{float(value):.6g}"


def _write_text(output_path: str | Path, lines: Sequence[str]) -> Path:
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_file
