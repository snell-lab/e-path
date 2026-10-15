"""Render the human-readable E-path analysis summary.

`write_summary_txt` writes the summary report, the last file that
`epath.outputs.write` produces for a run. `epath.cli.report` prints the lines
of the `*_summary_lines` functions and the entries of `output_file_entries`
to the terminal, so the terminal and the file show the same values.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from epath import __version__
from epath.analysis.clustering import ComponentStatistics
from epath.analysis.cofactors import cofactor_site_summary
from epath.analysis.pathways import PathReport
from epath.analysis.residues import RESIDUE_CLASS_ORDER
from epath.outputs.files import EpathPipelineOutputFiles
from rabdam.bnet.pipeline import BnetPipelineResult
from rabdam.pipeline import BDamagePipelineResult
from rabdam.structure.models import RemovedAtomCounts
from xtal.validate import ValidatedCrystal

if TYPE_CHECKING:
    # `epath.pipeline` imports `epath.outputs.write`, which imports this module.
    from epath.pipeline import EpathPipelineResult


def write_summary_txt(
    output_path: str | Path,
    result: EpathPipelineResult,
    *,
    output_files: EpathPipelineOutputFiles,
    display_chains: Sequence[str] | None = None,
) -> Path:
    """Write the summary report of an E-path run and return its path.

    The report gives the selection statistics, the selected atoms in each
    chain, the BDamage parameters, the crystal, Bnet, every reported pathway,
    and the files in `output_files`.

    `display_chains` and `output_files` are only reported. Nothing here is
    limited to the display chains, which are named only when `output_files`
    holds coordinate files for them to apply to. The files are listed as
    given, whether or not they exist yet. Missing parent directories of
    `output_path` are created, and an existing file is replaced.
    """
    selection_result = result.selection_result
    bdamage_result = result.bdamage_pipeline_result
    lines = [
        "E-path Analysis Summary",
        "=======================",
        "",
        f"E-path version: {__version__}",
        f"Input structure: {result.run_record.input_path}",
        f"Chains analyzed: {_chain_text(result.analyzed_chains)}",
    ]
    if output_files.selected_cif is not None:
        lines.append(
            "Chains in coordinate files: "
            + (
                "all"
                if display_chains is None
                else f"{_chain_text(display_chains)} (display filter only)"
            )
        )
    lines.extend(
        (
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
                "BDamage standard deviation (population): "
                f"{_number(selection_result.bdamage_standard_deviation)}"
            ),
            f"Z threshold: {_number(selection_result.z_threshold)}",
            f"BDamage threshold: {_number(selection_result.bdamage_threshold)}",
        )
    )
    # These three describe the selected atoms, so there is nothing to report
    # when the selection is empty.
    if selection_result.selected_atoms:
        lines.extend(
            (
                (
                    "Boundary Z-score among selected atoms: "
                    f"{_number(selection_result.boundary_z_score)}"
                ),
                (
                    "Selected BDamage mean: "
                    f"{_number(selection_result.selected_bdamage_mean)}"
                ),
                (
                    "Selected BDamage standard deviation (population): "
                    f"{_number(selection_result.selected_bdamage_standard_deviation)}"
                ),
            )
        )
    selected_chain_counts = Counter(
        atom.record.chain_id for atom in selection_result.selected_atoms
    )
    lines.extend(("", "Selected atoms by chain", "-----------------------"))
    if selected_chain_counts:
        lines.extend(
            f"Chain {chain_id}: {count}"
            for chain_id, count in selected_chain_counts.items()
        )
    else:
        lines.append("No atoms selected.")

    lines.extend(("", "BDamage calculation", "-------------------"))
    lines.extend(bdamage_summary_lines(bdamage_result))

    lines.extend(("", "Crystal", "-------"))
    lines.extend(crystal_summary_lines(bdamage_result.crystal))

    lines.extend(("", "RABDAM Bnet results", "-------------------"))
    lines.extend(bnet_summary_lines(result.bnet_pipeline_result))

    lines.extend(("", "Pathways", "--------"))
    lines.extend(_pathway_summary_lines(result))

    lines.extend(("", "Files written", "-------------"))
    file_entries = output_file_entries(output_files)
    if file_entries:
        lines.extend(f"{label}: {path}" for label, path in file_entries)
    else:
        lines.append("None.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def _pathway_summary_lines(result: EpathPipelineResult) -> tuple[str, ...]:
    """Return the pathway totals, then one block of lines per reported path.

    The paths come in the order of `result.path_reports`. When there are none,
    the largest component that was found is described instead.
    """
    path_reports = result.path_reports
    statistics = result.component_statistics
    lines = [
        f"Cluster distance: {_number(result.run_record.cluster_distance)} A",
        "Minimum reported cluster size: "
        f"{result.run_record.minimum_cluster_size} atoms",
        f"Cofactor range: {_number(result.run_record.cofactor_range)} A",
        f"Connected components found: {statistics.total_count}",
        f"Cofactor sites: {cofactor_site_summary(result.cofactor_sites)}",
    ]
    if not path_reports:
        lines.append("Path count: 0")
        lines.extend(_unreported_component_lines(statistics))
        return tuple(lines)

    near_cofactor = sum(report.is_near_cofactor for report in path_reports)
    lines.append(
        f"Path count: {len(path_reports)} ({near_cofactor} within range of a cofactor)"
    )
    for report in path_reports:
        metrics = report.cluster.metrics
        lines.extend(
            (
                "",
                f"Path {report.path_id}",
                f"  Cofactors in range: {_nearby_cofactor_text(report)}",
                f"  Chains: {_chain_text(report.chains)}",
                (
                    f"  Residues: {report.residue_count}   "
                    f"Nodes: {report.node_count}   "
                    f"Length: {_number(metrics.mst_total_length)} A"
                ),
                (
                    f"  MST mean step: {_number(metrics.mean_step)} A   "
                    f"largest edge: {_number(metrics.mst_largest_edge)} A"
                ),
                (
                    f"  Steps: {report.edges.intra_residue} within residues, "
                    f"{report.edges.peptide} peptide, "
                    f"{report.edges.tertiary} tertiary, "
                    f"{report.edges.interchain} interchain "
                    f"({_peptide_share_text(report)})"
                ),
                f"  Composition: {_composition_text(report)}",
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
            f"largest edge {_number(metrics.mst_largest_edge)} A"
        ),
    )


def _nearby_cofactor_text(report: PathReport) -> str:
    if not report.nearby_cofactors:
        return "none"

    return ", ".join(
        f"{cofactor.label} at {_number(cofactor.distance)} A from "
        f"{cofactor.atom_token} ({cofactor.residue})"
        for cofactor in report.nearby_cofactors
    )


def _peptide_share_text(report: PathReport) -> str:
    edges = report.edges
    if edges.peptide_fraction is None:
        return "no between-residue steps, so no peptide share"

    return (
        f"peptide share of the {edges.between_residues} between-residue steps: "
        f"{_number(edges.peptide_fraction)}"
    )


def _composition_text(report: PathReport) -> str:
    fractions = report.composition.fractions
    counts = report.composition.counts
    reported = [
        f"{name} {counts[name]}/{report.residue_count} ({_number(fractions[name])})"
        for name in RESIDUE_CLASS_ORDER
        if counts[name] > 0
    ]
    return ", ".join(reported)


def bdamage_summary_lines(
    pipeline_result: BDamagePipelineResult,
) -> tuple[str, ...]:
    """Return the atoms removed before scoring and the BDamage parameters."""
    options = pipeline_result.options
    return (
        removed_atoms_line(pipeline_result.prepared_structure.removed),
        f"Packing density threshold: {_number(options.packing_density_threshold)} A",
        f"Window size fraction: {_number(options.window_size_fraction)}",
        f"Minimum window size: {options.minimum_window_size} atoms",
        f"Window size used: {pipeline_result.window_size} atoms",
        f"Unit-cell translation range: {options.translation_range}",
    )


def removed_atoms_line(removed: RemovedAtomCounts) -> str:
    """Return the non-zero removal counts from structure preparation as one line."""
    # The labels that start with "with" describe the atoms and take no plural.
    counts = (
        (removed.hydrogens, "hydrogen"),
        (removed.non_finite_coordinates, "with non-finite coordinates"),
        (removed.invalid_occupancy, "with invalid occupancy"),
        (removed.non_positive_b_factor, "without a usable B-factor"),
        (removed.alternate_conformers, "alternate conformer"),
    )
    reported = ", ".join(
        f"{count} {label}{'s' if count > 1 and not label.startswith('with') else ''}"
        for count, label in counts
        if count > 0
    )
    return f"Atoms removed before scoring: {reported or 'none'}"


def crystal_summary_lines(crystal: ValidatedCrystal) -> tuple[str, ...]:
    """Return the validated crystal and asymmetric-unit content as report lines.

    The crystal's warnings come last, each prefixed with "WARNING: ".
    """
    lines = [
        f"Space group: {crystal.space_group_name}",
        f"Asymmetric units per unit cell: {crystal.operation_count}",
        f"Unit-cell volume: {_number(crystal.unit_cell_volume_angstrom3)} A^3",
    ]
    lines.extend(_content_lines(crystal))
    lines.extend(f"WARNING: {warning}" for warning in crystal.warnings)
    return tuple(lines)


def _content_lines(crystal: ValidatedCrystal) -> tuple[str, ...]:
    """Return the macromolecule mass, Matthews coefficient, and solvent content."""
    content = crystal.content
    if content is None:
        return ("Asymmetric-unit content: not calculated",)

    implicit_hydrogens = (
        f", including {_number(content.implicit_hydrogen_mass_dalton)} Da of "
        "implicit hydrogens"
        if content.implicit_hydrogen_mass_dalton > 0.0
        else ""
    )
    lines = [
        "Macromolecule mass: "
        f"{_number(content.macromolecule_mass_dalton)} Da "
        f"({content.macromolecule_atom_count} modelled atoms{implicit_hydrogens})"
    ]
    # The split only matters to the solvent estimate when both are present, and
    # a protein-only line would just repeat the total.
    if content.nucleic_acid_mass_dalton > 0.0:
        lines.append(
            "Protein and nucleic-acid mass: "
            f"{_number(content.protein_mass_dalton)} Da and "
            f"{_number(content.nucleic_acid_mass_dalton)} Da"
        )
    lines.extend(
        (
            f"Matthews coefficient: {_number(content.matthews_coefficient)} A^3/Da",
            (
                "Estimated solvent content: "
                f"{_number(100.0 * content.solvent_fraction)}%"
            ),
        )
    )
    return tuple(lines)


def bnet_summary_lines(pipeline_result: BnetPipelineResult) -> tuple[str, ...]:
    """Return the RABDAM Bnet results as report lines.

    A result that lacks Bnet or its percentile is reported with the reason
    that `run_bnet_pipeline` always gives for it.
    """
    bnet_result = pipeline_result.bnet_result
    if bnet_result is None:
        return (
            f"Bnet: unavailable ({pipeline_result.bnet_unavailable_reason})",
            # The percentile's own reason would only repeat the one above.
            "Bnet percentile: unavailable (Bnet is unavailable)",
        )

    lines = [
        f"Protein Bnet sites: {bnet_result.site_count}",
        f"Protein Bnet: {_number(bnet_result.bnet)}",
    ]
    percentile = pipeline_result.percentile_result
    if percentile is None:
        lines.append(
            "Bnet percentile: unavailable "
            f"({pipeline_result.percentile_unavailable_reason})"
        )
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
    output_files: EpathPipelineOutputFiles,
) -> tuple[tuple[str, Path], ...]:
    """Return a (label, path) pair for every file recorded in `output_files`.

    The order is fixed, with the pathway files last, each labelled with its
    path number.
    """
    entries: tuple[tuple[str, Path | None], ...] = (
        ("Combined atom CSV", output_files.combined_csv),
        ("Cluster CSV", output_files.cluster_csv),
        ("Step CSV", output_files.step_csv),
        ("Run CSV", output_files.run_csv),
        ("Selected-atoms mmCIF", output_files.selected_cif),
        ("Cleaned model mmCIF", output_files.model_cif),
        ("Z-score falloff plot", output_files.z_score_falloff),
        ("Summary report", output_files.summary_report),
        *(
            (f"Pathway mmCIF {path_id}", path)
            for path_id, path in output_files.path_cifs.items()
        ),
    )
    return tuple((label, path) for label, path in entries if path is not None)


def _chain_text(chains: Sequence[str]) -> str:
    return ", ".join(chains) if chains else "none"


def _number(value: float) -> str:
    """Format a value to six significant figures, without trailing zeros."""
    return f"{value:.6g}"
