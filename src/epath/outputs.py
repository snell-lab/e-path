"""Coordinate E-path output files and human-readable reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from epath.clustering import EpathCluster
from epath.plots import plot_z_hist
from epath.selection import EpathSelectionResult
from epath.structures import PdbBFactorMode, write_structure_outputs
from epath.tables import write_cluster_csv, write_combined_atom_csv
from rabdam.bnet.workflow import BnetWorkflowResult
from xtal.validate import ValidatedCrystal


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


@dataclass(frozen=True, slots=True)
class EpathWorkflowOutputFiles:
    """Files written by an E-path workflow run."""

    combined_csv: Path | None = None
    selected_pdb: Path | None = None
    z_score_histogram: Path | None = None
    cluster_csv: Path | None = None
    summary_report: Path | None = None


def write_epath_outputs(
    *,
    selection_result: EpathSelectionResult,
    clusters: tuple[EpathCluster, ...],
    analyzed_chains: Sequence[str],
    structure_path: str | Path | None,
    minimum_cluster_size: int,
    options: EpathOutputWorkflowOptions,
    bnet_workflow_result: BnetWorkflowResult | None = None,
    crystal: ValidatedCrystal | None = None,
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
        write_cluster_csv(_output_path(prefix, "clusters.csv"), clusters)
        if options.write_cluster_csv
        else None
    )

    selected_pdb: Path | None = None
    if options.write_structure_files:
        selected_pdb = write_structure_outputs(
            prefix,
            selection_result.selected_atoms,
            b_factor_mode=options.structure_b_factor_mode,
        ).selected_pdb

    z_score_histogram = (
        plot_z_hist(
            prefix,
            [atom.bdamage_z_score for atom in selection_result.scored_atoms],
            selection_result.z_mode,
        )
        if options.write_z_score_plot
        else None
    )
    summary_report = (
        _output_path(prefix, "summary.txt") if options.write_summary_report else None
    )
    output_files = EpathWorkflowOutputFiles(
        combined_csv=combined_csv,
        cluster_csv=cluster_csv,
        selected_pdb=selected_pdb,
        z_score_histogram=z_score_histogram,
        summary_report=summary_report,
    )

    if summary_report is not None:
        write_summary_txt(
            summary_report,
            structure_path=structure_path,
            analyzed_chains=analyzed_chains,
            selection_result=selection_result,
            clusters=clusters,
            minimum_cluster_size=minimum_cluster_size,
            output_files=output_files,
            bnet_workflow_result=bnet_workflow_result,
            crystal=crystal,
        )

    return output_files


def write_summary_txt(
    output_path: str | Path,
    *,
    structure_path: str | Path | None,
    analyzed_chains: Sequence[str],
    selection_result: EpathSelectionResult,
    clusters: Sequence[EpathCluster],
    minimum_cluster_size: int,
    output_files: EpathWorkflowOutputFiles,
    bnet_workflow_result: BnetWorkflowResult | None = None,
    crystal: ValidatedCrystal | None = None,
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
            f"BDamage threshold: {_number(selection_result.bdamage_threshold)}"
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

    lines.extend(("", "Crystal", "-------"))
    lines.extend(_crystal_summary_lines(crystal))

    lines.extend(("", "RABDAM Bnet results", "--------------------"))
    lines.extend(_bnet_summary_lines(bnet_workflow_result))

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

    lines.extend(("", "Files written", "-------------"))
    file_entries = _output_file_entries(output_files)
    if file_entries:
        lines.extend(f"{label}: {path}" for label, path in file_entries)
    else:
        lines.append("None.")

    return _write_text(output_path, lines)


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
        ("Z-score histogram", output_files.z_score_histogram),
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
