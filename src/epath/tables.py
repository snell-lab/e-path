"""CSV table outputs for E-path analysis results."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from epath.atoms import EpathAtom
from epath.clustering import EpathCluster
from epath.selection import (
    ChainPercentileThreshold,
    EpathScoredAtom,
    EpathSelectionResult,
)

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
    "selection_method",
    "selection_description",
    "selection_selected_percent",
    "selection_z_mode",
    "selection_z_threshold",
    "selection_bdamage_threshold",
    "selection_boundary_z_score",
    "chain_bdamage_threshold",
    "chain_z_threshold",
    "cluster_id",
    "cluster_member_index",
    "cluster_atom_count",
    "cluster_mst_total_length",
    "cluster_mean_step",
    "cluster_largest_edge",
    "cluster_span",
]

CLUSTER_FIELDNAMES = [
    "cluster_id",
    "atom_count",
    "mst_total_length",
    "mean_step",
    "largest_edge",
    "span",
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


def write_cluster_csv(
    output_path: str | Path,
    clusters: Iterable[EpathCluster],
) -> Path:
    """Write one CSV row per reported E-path cluster."""

    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = cluster_rows(clusters)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CLUSTER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            _format_row_for_fields(row, CLUSTER_FIELDNAMES) for row in rows
        )

    return output_file


def cluster_rows(
    clusters: Iterable[EpathCluster],
) -> tuple[dict[str, Any], ...]:
    """Return one standalone CSV row for each E-path cluster."""

    return tuple(
        {
            "cluster_id": cluster.cluster_id,
            "atom_count": cluster.atom_count,
            "mst_total_length": cluster.metrics.mst_total_length,
            "mean_step": cluster.metrics.mean_step,
            "largest_edge": cluster.metrics.mst_largest_edge,
            "span": cluster.metrics.span,
        }
        for cluster in clusters
    )


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
    chain_thresholds = _chain_thresholds_by_chain(selection_result.chain_thresholds)
    cluster_by_atom = _cluster_membership_by_atom(clusters)

    rows: list[dict[str, Any]] = []
    for scored_atom in selection_result.scored_atoms:
        atom = scored_atom.atom
        atom_id = _atom_identity(atom)
        chain_threshold = chain_thresholds.get(atom.chain_id)
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
            "selection_method": selection_result.method,
            "selection_description": selection_result.description,
            "selection_selected_percent": selection_result.selected_percent,
            "selection_z_mode": selection_result.z_mode,
            "selection_z_threshold": selection_result.z_threshold,
            "selection_bdamage_threshold": selection_result.bdamage_threshold,
            "selection_boundary_z_score": selection_result.boundary_z_score,
            "chain_bdamage_threshold": (
                chain_threshold.bdamage_threshold if chain_threshold else None
            ),
            "chain_z_threshold": (
                chain_threshold.z_threshold if chain_threshold else None
            ),
        }
        row.update(_cluster_columns(cluster_by_atom.get(atom_id)))
        rows.append(row)

    return tuple(rows)


def atom_token(atom: EpathAtom) -> str:
    """Return an atom token in Chain:Resnum[icode]:Atom form."""

    residue_number = "" if atom.residue_number is None else str(atom.residue_number)
    insertion_code = atom.insertion_code.strip()
    return f"{atom.chain_id}:{residue_number}{insertion_code}:{atom.atom_name.strip()}"


def _chain_thresholds_by_chain(
    chain_thresholds: Iterable[ChainPercentileThreshold],
) -> dict[str, ChainPercentileThreshold]:
    return {threshold.chain_id: threshold for threshold in chain_thresholds}


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
        return {
            "cluster_id": None,
            "cluster_member_index": None,
            "cluster_atom_count": None,
            "cluster_mst_total_length": None,
            "cluster_mean_step": None,
            "cluster_largest_edge": None,
            "cluster_span": None,
        }

    cluster, member_index = cluster_membership
    return {
        "cluster_id": cluster.cluster_id,
        "cluster_member_index": member_index,
        "cluster_atom_count": cluster.atom_count,
        "cluster_mst_total_length": cluster.metrics.mst_total_length,
        "cluster_mean_step": cluster.metrics.mean_step,
        "cluster_largest_edge": cluster.metrics.mst_largest_edge,
        "cluster_span": cluster.metrics.span,
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
