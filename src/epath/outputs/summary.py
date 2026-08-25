"""Render the human-readable E-path analysis summary.

`write_summary_txt` builds the persistent summary report. The
`*_summary_lines` helpers are also used by the CLI to print the same values
to the terminal, so both reports stay in step.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from epath.analysis.clustering import ComponentStatistics
from epath.analysis.endpoints import CofactorSite, cofactor_site_summary
from epath.analysis.pathways import PATH_CLASS_ORDER, PathReport
from epath.analysis.residues import RESIDUE_CLASS_ORDER
from epath.outputs.files import EpathWorkflowOutputFiles
from rabdam.bnet.workflow import BnetWorkflowResult
from rabdam.workflow import BDamageWorkflowResult
from xtal.validate import ValidatedCrystal

if TYPE_CHECKING:
    from epath.workflow import EpathWorkflowResult


def write_summary_txt(
    output_path: str | Path,
    result: EpathWorkflowResult,
    *,
    output_files: EpathWorkflowOutputFiles,
) -> Path:
    """Write a persistent human-readable E-path analysis summary."""

    selection_result = result.selection_result
    structure_path = result.structure_path
    lines = [
        "E-path Analysis Summary",
        "=======================",
        "",
        (
            "Input structure: "
            f"{structure_path if structure_path is not None else 'not provided'}"
        ),
        f"Chains analyzed: {_chain_text(result.analyzed_chains)}",
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
    lines.extend(bdamage_parameter_lines(result.bdamage_workflow_result))

    lines.extend(("", "Crystal", "-------"))
    lines.extend(crystal_summary_lines(result.crystal))

    lines.extend(("", "RABDAM Bnet results", "--------------------"))
    lines.extend(bnet_summary_lines(result.bnet_workflow_result))

    lines.extend(
        _pathway_summary_lines(
            result.path_reports,
            cofactor_sites=result.cofactor_sites,
            statistics=result.component_statistics,
            minimum_cluster_size=result.run_record.minimum_cluster_size,
        )
    )

    lines.extend(("", "Files written", "-------------"))
    file_entries = output_file_entries(output_files)
    if file_entries:
        lines.extend(f"{label}: {path}" for label, path in file_entries)
    else:
        lines.append("None.")

    return _write_text(output_path, lines)


def _pathway_summary_lines(
    path_reports: Sequence[PathReport],
    *,
    cofactor_sites: Sequence[CofactorSite],
    statistics: ComponentStatistics,
    minimum_cluster_size: int,
) -> tuple[str, ...]:
    """Render the reported pathways, one block per path."""

    lines = [
        "",
        "Pathways",
        "--------",
        f"Minimum reported cluster size: {minimum_cluster_size} atoms",
        f"Connected components found: {statistics.total_count}",
        f"Cofactor sites: {cofactor_site_summary(cofactor_sites)}",
        f"Path count: {len(path_reports)}{_path_class_counts(path_reports)}",
    ]
    if not path_reports:
        lines.extend(_unreported_component_lines(statistics))
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
                (
                    f"  Steps: {report.edges.intra_residue} within residues, "
                    f"{report.edges.peptide} peptide, "
                    f"{report.edges.tertiary} tertiary, "
                    f"{report.edges.interchain} interchain "
                    f"(peptide share of the "
                    f"{report.edges.between_residues} between-residue steps: "
                    f"{_number(report.edges.peptide_fraction)})"
                ),
                f"  Composition: {_composition_text(report)}",
                f"  Descriptor: {report.structural_descriptor}",
                f"  Path: {report.residue_path}",
            )
        )

    return tuple(lines)


def _unreported_component_lines(
    statistics: ComponentStatistics,
) -> tuple[str, ...]:
    """Describe the largest component when none reached the threshold."""

    if statistics.largest_metrics is None:
        return ("No selected atoms were connected into a component.",)

    metrics = statistics.largest_metrics
    return (
        "No connected component reached the reporting threshold.",
        (
            f"Largest component: {statistics.largest_atom_count} atoms, "
            f"length {_number(metrics.mst_total_length)} A, "
            f"mean step {_number(metrics.mean_step)} A, "
            f"largest edge {_number(metrics.mst_largest_edge)} A, "
            f"span {_number(metrics.span)} A"
        ),
    )


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


def bdamage_parameter_lines(
    workflow_result: BDamageWorkflowResult | None,
) -> tuple[str, ...]:
    """Return the BDamage calculation parameters as report lines."""

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


def crystal_summary_lines(
    crystal: ValidatedCrystal | None,
) -> tuple[str, ...]:
    """Return the validated crystal and asymmetric-unit content as report lines."""

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


def bnet_summary_lines(
    workflow_result: BnetWorkflowResult | None,
) -> tuple[str, ...]:
    """Return the RABDAM Bnet results as report lines."""

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


def output_file_entries(
    output_files: EpathWorkflowOutputFiles,
) -> tuple[tuple[str, Path], ...]:
    """Return (label, path) pairs for every output file that was written."""

    entries: tuple[tuple[str, Path | None], ...] = (
        ("Combined atom CSV", output_files.combined_csv),
        ("Cluster CSV", output_files.cluster_csv),
        ("Step CSV", output_files.step_csv),
        ("Run CSV", output_files.run_csv),
        ("Selected-atoms PDB", output_files.selected_pdb),
        ("Cleaned model PDB", output_files.model_pdb),
        ("Z-score falloff plot", output_files.z_score_falloff),
        ("Summary report", output_files.summary_report),
        *(
            (f"Pathway PDB {index}", path)
            for index, path in enumerate(output_files.path_pdbs, start=1)
        ),
    )
    return tuple((label, path) for label, path in entries if path is not None)


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
