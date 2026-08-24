"""Coordinate E-path output files and human-readable reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from epath.clustering import EpathCluster
from epath.endpoints import CofactorSite, cofactor_site_summary
from epath.pathways import PATH_CLASS_ORDER, PathReport
from epath.plots import plot_z_falloff
from epath.residues import RESIDUE_CLASS_ORDER
from epath.selection import EpathSelectionResult
from epath.structures import write_structure_outputs
from epath.tables import write_cluster_csv, write_combined_atom_csv
from rabdam.bnet.workflow import BnetWorkflowResult
from rabdam.structure.models import PreparedAtom
from rabdam.workflow import BDamageWorkflowResult
from xtal.validate import ValidatedCrystal


@dataclass(frozen=True, slots=True)
class EpathOutputWorkflowOptions:
    """Options controlling E-path output files."""

    output_prefix: str | Path | None = None
    write_combined_csv: bool = True
    write_structure_files: bool = True
    write_z_score_plot: bool = True
    write_cluster_csv: bool = True
    write_summary_report: bool = True


@dataclass(frozen=True, slots=True)
class EpathWorkflowOutputFiles:
    """Files written by an E-path workflow run."""

    combined_csv: Path | None = None
    selected_pdb: Path | None = None
    model_pdb: Path | None = None
    z_score_falloff: Path | None = None
    cluster_csv: Path | None = None
    summary_report: Path | None = None


def write_epath_outputs(
    *,
    selection_result: EpathSelectionResult,
    clusters: tuple[EpathCluster, ...],
    path_reports: Sequence[PathReport],
    cofactor_sites: Sequence[CofactorSite],
    analyzed_chains: Sequence[str],
    structure_path: str | Path | None,
    minimum_cluster_size: int,
    options: EpathOutputWorkflowOptions,
    bnet_workflow_result: BnetWorkflowResult | None = None,
    crystal: ValidatedCrystal | None = None,
    cleaned_atoms: Sequence[PreparedAtom] | None = None,
    bdamage_workflow_result: BDamageWorkflowResult | None = None,
) -> EpathWorkflowOutputFiles:
    """Write enabled E-path artifacts and return their paths."""

    if options.output_prefix is None:
        return EpathWorkflowOutputFiles()

    prefix = Path(options.output_prefix)
    combined_csv = (
        write_combined_atom_csv(
            _output_path(prefix, "epath.csv"),
            selection_result,
            clusters=clusters,
        )
        if options.write_combined_csv
        else None
    )
    cluster_csv = (
        write_cluster_csv(_output_path(prefix, "clusters.csv"), path_reports)
        if options.write_cluster_csv
        else None
    )

    selected_pdb: Path | None = None
    model_pdb: Path | None = None
    if options.write_structure_files:
        structure_files = write_structure_outputs(
            prefix,
            selection_result.selected_atoms,
            cleaned_atoms=cleaned_atoms,
            bdamage_by_source_index={
                scored_atom.atom.source_atom_index: scored_atom.atom.bdamage
                for scored_atom in selection_result.scored_atoms
            },
            bdamage_threshold=selection_result.bdamage_threshold,
        )
        selected_pdb = structure_files.selected_pdb
        model_pdb = structure_files.model_pdb

    z_score_falloff = (
        plot_z_falloff(prefix, selection_result) if options.write_z_score_plot else None
    )
    summary_report = (
        _output_path(prefix, "summary.txt") if options.write_summary_report else None
    )
    output_files = EpathWorkflowOutputFiles(
        combined_csv=combined_csv,
        cluster_csv=cluster_csv,
        selected_pdb=selected_pdb,
        model_pdb=model_pdb,
        z_score_falloff=z_score_falloff,
        summary_report=summary_report,
    )

    if summary_report is not None:
        write_summary_txt(
            summary_report,
            structure_path=structure_path,
            analyzed_chains=analyzed_chains,
            selection_result=selection_result,
            path_reports=path_reports,
            cofactor_sites=cofactor_sites,
            minimum_cluster_size=minimum_cluster_size,
            output_files=output_files,
            bnet_workflow_result=bnet_workflow_result,
            crystal=crystal,
            bdamage_workflow_result=bdamage_workflow_result,
        )

    return output_files


def write_summary_txt(
    output_path: str | Path,
    *,
    structure_path: str | Path | None,
    analyzed_chains: Sequence[str],
    selection_result: EpathSelectionResult,
    path_reports: Sequence[PathReport],
    cofactor_sites: Sequence[CofactorSite],
    minimum_cluster_size: int,
    output_files: EpathWorkflowOutputFiles,
    bnet_workflow_result: BnetWorkflowResult | None = None,
    crystal: ValidatedCrystal | None = None,
    bdamage_workflow_result: BDamageWorkflowResult | None = None,
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
        f"BDamage mean: {_number(selection_result.bdamage_mean)}",
        (
            "BDamage standard deviation: "
            f"{_number(selection_result.bdamage_standard_deviation)}"
        ),
        f"Z threshold: {_number(selection_result.z_threshold)}",
    ]
    if selection_result.bdamage_threshold is not None:
        lines.append(
            f"BDamage threshold: {_number(selection_result.bdamage_threshold)}"
        )
    if selection_result.boundary_z_score is not None:
        lines.append(
            "Boundary Z-score among selected atoms: "
            f"{_number(selection_result.boundary_z_score)}"
        )

    lines.extend(
        (
            (
                "Selected BDamage mean: "
                f"{_number(selection_result.selected_bdamage_mean)}"
            ),
            (
                "Selected BDamage standard deviation: "
                f"{_number(selection_result.selected_bdamage_standard_deviation)}"
            ),
        )
    )
    plateau = selection_result.plateau
    if plateau is None:
        lines.append("Z-score plateau: none detected in the selected atoms")
    else:
        lines.append(
            f"Z-score plateau: {plateau.atom_count} atoms from rank "
            f"{plateau.first_rank}, Z {_number(plateau.lowest_z_score)} to "
            f"{_number(plateau.highest_z_score)}"
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

    lines.extend(("", "BDamage calculation", "-------------------"))
    lines.extend(_bdamage_parameter_lines(bdamage_workflow_result))

    lines.extend(("", "Crystal", "-------"))
    lines.extend(_crystal_summary_lines(crystal))

    lines.extend(("", "RABDAM Bnet results", "--------------------"))
    lines.extend(_bnet_summary_lines(bnet_workflow_result))

    lines.extend(
        _pathway_summary_lines(
            path_reports,
            cofactor_sites=cofactor_sites,
            minimum_cluster_size=minimum_cluster_size,
        )
    )

    lines.extend(("", "Files written", "-------------"))
    file_entries = _output_file_entries(output_files)
    if file_entries:
        lines.extend(f"{label}: {path}" for label, path in file_entries)
    else:
        lines.append("None.")

    return _write_text(output_path, lines)


def _pathway_summary_lines(
    path_reports: Sequence[PathReport],
    *,
    cofactor_sites: Sequence[CofactorSite],
    minimum_cluster_size: int,
) -> tuple[str, ...]:
    """Render the reported pathways, one block per path."""

    lines = [
        "",
        "Pathways",
        "--------",
        f"Minimum reported cluster size: {minimum_cluster_size} atoms",
        f"Cofactor sites: {cofactor_site_summary(cofactor_sites)}",
        f"Path count: {len(path_reports)}{_path_class_counts(path_reports)}",
    ]
    if not path_reports:
        lines.append("No selected-atom clusters reached the reporting threshold.")
        return tuple(lines)

    for report in path_reports:
        metrics = report.cluster.metrics
        topology = report.topology
        lines.extend(
            (
                "",
                (
                    f"Path {report.path_id} [{report.class_label}] "
                    f"{report.start_endpoint.label} -> {report.end_endpoint.label}"
                ),
                (
                    f"  Endpoint atoms: {report.start_endpoint.atom_token} "
                    f"({report.start_endpoint.residue}) to "
                    f"{report.end_endpoint.atom_token} "
                    f"({report.end_endpoint.residue})"
                ),
                f"  Chains: {_chain_text(report.chains)}",
                (
                    f"  Residues: {report.residue_count}   "
                    f"Nodes: {report.node_count}   "
                    f"Length: {_number(metrics.mst_total_length)} A"
                ),
                (
                    f"  MST mean step: {_number(metrics.mean_step)} A   "
                    f"largest edge: {_number(metrics.mst_largest_edge)} A   "
                    f"span: {_number(metrics.span)} A"
                ),
                (
                    f"  Topology: {topology.topology} "
                    f"(span/length {_number(topology.linearity)}, "
                    f"junction nodes {_number(topology.branch_fraction)})"
                ),
                f"  Composition: {_composition_text(report)}",
                f"  Descriptor: {report.structural_descriptor}",
                f"  Path: {report.residue_path}",
            )
        )

    return tuple(lines)


def _path_class_counts(path_reports: Sequence[PathReport]) -> str:
    if not path_reports:
        return ""

    counts = Counter(report.path_class for report in path_reports)
    reported = [
        f"{name} {counts[name]}" for name in PATH_CLASS_ORDER if counts[name] > 0
    ]
    return f" ({', '.join(reported)})"


def _composition_text(report: PathReport) -> str:
    fractions = report.composition.fractions
    counts = report.composition.counts
    reported = [
        f"{name} {counts[name]}/{report.residue_count} ({_number(fractions[name])})"
        for name in RESIDUE_CLASS_ORDER
        if counts[name] > 0
    ]
    return ", ".join(reported) if reported else "no residues"


def _bdamage_parameter_lines(
    workflow_result: BDamageWorkflowResult | None,
) -> tuple[str, ...]:
    if workflow_result is None:
        return ("BDamage parameters: not recorded",)

    options = workflow_result.options
    return (
        f"Packing density threshold: {_number(options.packing_density_threshold)} A",
        f"Window size fraction: {_number(options.window_size_fraction)}",
        f"Minimum window size: {options.minimum_window_size} atoms",
        f"Window size used: {workflow_result.window_size} atoms",
        f"Unit-cell translation range: {options.translation_range}",
    )


def _crystal_summary_lines(
    crystal: ValidatedCrystal | None,
) -> tuple[str, ...]:
    if crystal is None:
        return ("Crystal: not validated",)

    lines = [
        f"Space group: {crystal.space_group_name}",
        f"Asymmetric units per unit cell: {crystal.operation_count}",
        f"Unit-cell volume: {_number(crystal.unit_cell_volume_angstrom3)} A^3",
    ]
    content = crystal.content
    if content is None:
        lines.append("Asymmetric-unit content: not calculated")
        return tuple(lines)

    lines.extend(
        (
            (
                "Modelled macromolecule mass: "
                f"{_number(content.macromolecule_mass_dalton)} Da "
                f"({content.macromolecule_atom_count} atoms"
                f"{', hydrogens excluded' if content.hydrogens_excluded else ''})"
            ),
            (f"Matthews coefficient: {_number(content.matthews_coefficient)} A^3/Da"),
            (
                "Estimated solvent content: "
                f"{_number(100.0 * content.solvent_fraction)}%"
            ),
        )
    )
    lines.extend(f"WARNING: {warning}" for warning in crystal.warnings)
    return tuple(lines)


def _bnet_summary_lines(
    workflow_result: BnetWorkflowResult | None,
) -> tuple[str, ...]:
    if workflow_result is None:
        return ("Bnet: not calculated", "Bnet percentile: not calculated")

    bnet_result = workflow_result.bnet_result
    if bnet_result is None:
        reason = workflow_result.bnet_unavailable_reason or "not calculated"
        return (
            f"Bnet: unavailable ({reason})",
            "Bnet percentile: unavailable (raw Bnet is unavailable)",
        )

    lines = [
        f"Protein Bnet sites: {bnet_result.site_count}",
        f"Raw protein Bnet: {_number(bnet_result.bnet)}",
    ]
    percentile = workflow_result.percentile_result
    if percentile is None:
        reason = workflow_result.percentile_unavailable_reason or "not calculated"
        lines.append(f"Bnet percentile: unavailable ({reason})")
        return tuple(lines)

    lines.extend(
        (
            f"Bnet percentile: {_number(percentile.percentile_percent)}%",
            (
                f"Bnet reference: {percentile.reference_database_id} "
                f"({percentile.reference_entry_count} entries)"
            ),
            f"Resolution used: {_number(percentile.resolution_angstrom)} A",
            (
                f"Local reference set: {percentile.local_reference_count} entries, "
                f"{_number(percentile.local_resolution_min)}-"
                f"{_number(percentile.local_resolution_max)} A"
            ),
        )
    )
    return tuple(lines)


def _output_file_entries(
    output_files: EpathWorkflowOutputFiles,
) -> tuple[tuple[str, Path], ...]:
    entries = (
        ("Combined atom CSV", output_files.combined_csv),
        ("Cluster CSV", output_files.cluster_csv),
        ("Selected-atoms PDB", output_files.selected_pdb),
        ("Cleaned model PDB", output_files.model_pdb),
        ("Z-score falloff plot", output_files.z_score_falloff),
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
