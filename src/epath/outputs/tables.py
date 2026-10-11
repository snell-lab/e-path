"""CSV table outputs for E-path analysis results."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from epath.analysis.atoms import EpathAtom, atom_token
from epath.analysis.clustering import EpathCluster
from epath.analysis.pathways import PathReport
from epath.analysis.selection import EpathScoredAtom, EpathSelectionResult
from epath.outputs.runs import EpathRunRecord

COMBINED_ATOM_FIELDNAMES = [
    "source_atom_index",
    "bdamage_atom_index",
    "atom_serial",
    "record_type",
    "atom_token",
    "chain_id",
    "residue_name",
    "residue_number",
    "insertion_code",
    "atom_name",
    "element",
    "x",
    "y",
    "z",
    "occupancy",
    "b_factor",
    "packing_density",
    "average_b_factor",
    "bdamage",
    "bdamage_z_score",
    "selected",
    "cluster_id",
    "cluster_member_index",
]

# The leading columns are the E-path pathway table: the path, where it
# starts and ends, the cofactor it comes nearest to, and what it is made of.
# The columns after them are the geometry and the measurements behind the
# descriptor.
RUN_FIELDNAMES = [
    "structure",
    "input_path",
    "selection_method",
    "cutoff_percent",
    "total_atoms",
    "atoms_selected",
    "selected_percent",
    "equivalent_z_score",
    "bdamage_threshold",
    "bnet",
    "bnet_percentile",
    "total_components",
    "minimum_cluster_size",
    "clusters_reported",
    "largest_component_atoms",
    "largest_component_reported",
    "largest_component_length",
    "largest_component_mean_step",
    "largest_component_largest_edge",
    "largest_component_span",
]

STEP_FIELDNAMES = [
    "cluster_id",
    "step_index",
    "length",
    "kind",
    "first_atom",
    "first_residue",
    "second_atom",
    "second_residue",
]

CLUSTER_FIELDNAMES = [
    "cluster_id",
    "endpoint_start",
    "endpoint_end",
    "residue_path",
    "residue_count",
    "atom_count",
    "mst_total_length",
    "anchor_cofactor",
    "anchor_distance",
    "anchor_atom",
    "structural_descriptor",
    "chains",
    "mean_step",
    "largest_edge",
    "span",
    "topology",
    "linearity",
    "branch_fraction",
    "intra_residue_edges",
    "peptide_edges",
    "tertiary_edges",
    "interchain_edges",
    "peptide_edge_fraction",
    "aromatic_fraction",
    "sulfur_fraction",
    "charged_fraction",
    "polar_fraction",
    "hydrophobic_fraction",
    "endpoint_start_kind",
    "endpoint_start_atom",
    "endpoint_end_kind",
    "endpoint_end_atom",
]


def write_combined_atom_csv(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
    *,
    clusters: Iterable[EpathCluster] = (),
) -> Path:
    """Write a combined per-atom E-path CSV table."""
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = combined_atom_rows(selection_result, clusters=clusters)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMBINED_ATOM_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_format_row(row) for row in rows)

    return output_file


def write_run_csv(
    output_path: str | Path,
    records: Iterable[EpathRunRecord],
) -> Path:
    """Write one CSV row per run.

    A single run writes one row. Passing the records of a batch writes the
    cohort table for those structures.
    """
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            _format_row_for_fields(_run_row(record), RUN_FIELDNAMES)
            for record in records
        )

    return output_file


def _run_row(record: EpathRunRecord) -> dict[str, Any]:
    return {
        "structure": record.structure,
        "input_path": record.input_path,
        "selection_method": record.selection_method,
        "cutoff_percent": record.cutoff_percent,
        "total_atoms": record.total_atoms,
        "atoms_selected": record.atoms_selected,
        "selected_percent": record.selected_percent,
        "equivalent_z_score": record.equivalent_z_score,
        "bdamage_threshold": record.bdamage_threshold,
        "bnet": record.bnet,
        "bnet_percentile": record.bnet_percentile,
        "total_components": record.total_components,
        "minimum_cluster_size": record.minimum_cluster_size,
        "clusters_reported": record.clusters_reported,
        "largest_component_atoms": record.largest_component_atoms,
        "largest_component_reported": record.largest_component_reported,
        "largest_component_length": record.largest_component_length,
        "largest_component_mean_step": record.largest_component_mean_step,
        "largest_component_largest_edge": record.largest_component_largest_edge,
        "largest_component_span": record.largest_component_span,
    }


def write_step_csv(
    output_path: str | Path,
    path_reports: Iterable[PathReport],
) -> Path:
    """Write one CSV row per step of every reported pathway.

    The summary reports the total, mean, and largest step of a path. This
    records each step individually, because electron tunnelling falls away
    sharply with distance and a single long step can limit a whole path.
    """
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STEP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            _format_row_for_fields(
                {
                    "cluster_id": report.path_id,
                    "step_index": step.step_index,
                    "length": step.length,
                    "kind": step.kind,
                    "first_atom": step.first_atom,
                    "first_residue": step.first_residue,
                    "second_atom": step.second_atom,
                    "second_residue": step.second_residue,
                },
                STEP_FIELDNAMES,
            )
            for report in path_reports
            for step in report.steps
        )

    return output_file


def write_cluster_csv(
    output_path: str | Path,
    path_reports: Iterable[PathReport],
) -> Path:
    """Write one CSV row per reported E-path pathway."""
    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = cluster_rows(path_reports)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLUSTER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            _format_row_for_fields(row, CLUSTER_FIELDNAMES) for row in rows
        )

    return output_file


def cluster_rows(
    path_reports: Iterable[PathReport],
) -> tuple[dict[str, Any], ...]:
    """Return one standalone CSV row for each reported E-path pathway."""
    return tuple(_cluster_row(report) for report in path_reports)


def _cluster_row(report: PathReport) -> dict[str, Any]:
    metrics = report.cluster.metrics
    fractions = report.composition.fractions
    anchor = report.anchor

    return {
        "cluster_id": report.path_id,
        "endpoint_start": report.start_endpoint.label,
        "endpoint_end": report.end_endpoint.label,
        "residue_path": report.residue_path,
        "residue_count": report.residue_count,
        "atom_count": report.node_count,
        "mst_total_length": metrics.mst_total_length,
        "anchor_cofactor": anchor.label if anchor is not None else "",
        "anchor_distance": anchor.distance if anchor is not None else "",
        "anchor_atom": anchor.atom_token if anchor is not None else "",
        "structural_descriptor": report.structural_descriptor,
        "chains": "/".join(report.chains),
        "mean_step": metrics.mean_step,
        "largest_edge": metrics.mst_largest_edge,
        "span": metrics.span,
        "topology": report.topology.topology,
        "linearity": report.topology.linearity,
        "branch_fraction": report.topology.branch_fraction,
        "intra_residue_edges": report.edges.intra_residue,
        "peptide_edges": report.edges.peptide,
        "tertiary_edges": report.edges.tertiary,
        "interchain_edges": report.edges.interchain,
        "peptide_edge_fraction": report.edges.peptide_fraction,
        "aromatic_fraction": fractions["aromatic"],
        "sulfur_fraction": fractions["sulfur"],
        "charged_fraction": fractions["charged"],
        "polar_fraction": fractions["polar"],
        "hydrophobic_fraction": fractions["hydrophobic"],
        "endpoint_start_kind": report.start_endpoint.kind,
        "endpoint_start_atom": report.start_endpoint.atom_token,
        "endpoint_end_kind": report.end_endpoint.kind,
        "endpoint_end_atom": report.end_endpoint.atom_token,
    }


def combined_atom_rows(
    selection_result: EpathSelectionResult,
    *,
    clusters: Iterable[EpathCluster] = (),
) -> tuple[dict[str, Any], ...]:
    """Return combined per-atom rows for E-path CSV output."""
    selected_keys = {
        _scored_atom_identity(scored_atom)
        for scored_atom in selection_result.selected_atoms
    }
    cluster_by_atom = _cluster_membership_by_atom(clusters)

    rows: list[dict[str, Any]] = []
    for scored_atom in selection_result.scored_atoms:
        atom = scored_atom.atom
        atom_id = _atom_identity(atom)
        row: dict[str, Any] = {
            "source_atom_index": atom.source_atom_index,
            "bdamage_atom_index": atom.bdamage_atom_index,
            "atom_serial": atom.atom_serial,
            "record_type": atom.record_type,
            "atom_token": atom_token(atom),
            "chain_id": atom.chain_id,
            "residue_name": atom.residue_name,
            "residue_number": atom.residue_number,
            "insertion_code": atom.insertion_code,
            "atom_name": atom.atom_name,
            "element": atom.element,
            "x": atom.x,
            "y": atom.y,
            "z": atom.z,
            "occupancy": atom.occupancy,
            "b_factor": atom.b_factor,
            "packing_density": atom.packing_density,
            "average_b_factor": atom.average_b_factor,
            "bdamage": scored_atom.bdamage,
            "bdamage_z_score": scored_atom.bdamage_z_score,
            "selected": atom_id in selected_keys,
        }
        row.update(_cluster_columns(cluster_by_atom.get(atom_id)))
        rows.append(row)

    return tuple(rows)


def _cluster_membership_by_atom(
    clusters: Iterable[EpathCluster],
) -> dict[tuple[int, int | None], tuple[EpathCluster, int]]:
    memberships: dict[tuple[int, int | None], tuple[EpathCluster, int]] = {}
    for cluster in clusters:
        for member_index, scored_atom in enumerate(cluster.atoms, start=1):
            memberships[_scored_atom_identity(scored_atom)] = (
                cluster,
                member_index,
            )

    return memberships


def _cluster_columns(
    cluster_membership: tuple[EpathCluster, int] | None,
) -> dict[str, Any]:
    if cluster_membership is None:
        return {"cluster_id": None, "cluster_member_index": None}

    cluster, member_index = cluster_membership
    return {
        "cluster_id": cluster.cluster_id,
        "cluster_member_index": member_index,
    }


def _format_row(row: Mapping[str, Any]) -> dict[str, str]:
    return _format_row_for_fields(row, COMBINED_ATOM_FIELDNAMES)


def _format_row_for_fields(
    row: Mapping[str, Any],
    fieldnames: Iterable[str],
) -> dict[str, str]:
    return {fieldname: _format_cell(row.get(fieldname)) for fieldname in fieldnames}


def _format_cell(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"

    return str(value)


def _scored_atom_identity(scored_atom: EpathScoredAtom) -> tuple[int, int | None]:
    return _atom_identity(scored_atom.atom)


def _atom_identity(atom: EpathAtom) -> tuple[int, int | None]:
    return atom.source_atom_index, atom.atom_serial
