"""CSV table outputs for E-path analysis results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import math
from pathlib import Path
from typing import Any

from epath.annotations import EpathAtomAnnotation
from epath.atoms import EpathAtom
from epath.clustering import EpathCluster
from epath.endpoints import EpathResolvedEndpoint, atom_token
from epath.null import EpathNullModelComparison
from epath.paths import EpathPath
from epath.rates import EpathClusterRate, EpathMarcusRate
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
    "secondary_structure",
    "sasa_angstrom_squared",
    "cluster_id",
    "cluster_member_index",
    "cluster_atom_count",
    "cluster_mst_total_length",
    "cluster_mean_step",
    "cluster_largest_edge",
    "cluster_span",
    "cluster_rate",
    "cluster_rate_normalized",
    "on_path",
    "path_order",
    "path_total_length",
    "path_step_count",
    "path_largest_step",
    "path_edge_from_previous",
    "path_edge_to_next",
    "endpoint_roles",
    "endpoint_tokens",
    "endpoint_exact_match",
    "endpoint_snap_distance",
    "marcus_rate",
    "marcus_path_length",
    "marcus_reorganization_energy_ev",
    "marcus_delta_g_ev",
    "marcus_temperature_k",
    "marcus_beta_per_angstrom",
    "marcus_reorganization_exponent",
    "marcus_distance_exponent",
    "null_observed_largest_cluster_size",
    "null_observed_mst_total_length",
    "null_largest_z_score",
    "null_largest_p_value_greater",
    "null_mst_z_score",
    "null_mst_p_value_less",
]


class EpathTableError(ValueError):
    """Raised when E-path table output cannot be built."""


def write_combined_atom_csv(
    output_path: str | Path,
    selection_result: EpathSelectionResult,
    *,
    clusters: Iterable[EpathCluster] = (),
    annotations: Iterable[EpathAtomAnnotation] = (),
    path: EpathPath | None = None,
    marcus_rate: EpathMarcusRate | None = None,
    cluster_rates: Iterable[EpathClusterRate] = (),
    endpoint_roles: Mapping[str, EpathResolvedEndpoint] | None = None,
    null_comparisons: Mapping[int, EpathNullModelComparison] | None = None,
) -> Path:
    """Write a combined per-atom E-path CSV table."""

    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = combined_atom_rows(
        selection_result,
        clusters=clusters,
        annotations=annotations,
        path=path,
        marcus_rate=marcus_rate,
        cluster_rates=cluster_rates,
        endpoint_roles=endpoint_roles,
        null_comparisons=null_comparisons,
    )
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMBINED_ATOM_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_format_row(row) for row in rows)

    return output_file


def combined_atom_rows(
    selection_result: EpathSelectionResult,
    *,
    clusters: Iterable[EpathCluster] = (),
    annotations: Iterable[EpathAtomAnnotation] = (),
    path: EpathPath | None = None,
    marcus_rate: EpathMarcusRate | None = None,
    cluster_rates: Iterable[EpathClusterRate] = (),
    endpoint_roles: Mapping[str, EpathResolvedEndpoint] | None = None,
    null_comparisons: Mapping[int, EpathNullModelComparison] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return combined per-atom rows for E-path CSV output."""

    scored_atoms = tuple(selection_result.scored_atoms)
    selected_keys = {
        _scored_atom_identity(scored_atom)
        for scored_atom in selection_result.selected_atoms
    }
    annotation_by_atom = {
        _atom_identity(annotation.atom): annotation
        for annotation in annotations
    }
    chain_thresholds = _chain_thresholds_by_chain(
        selection_result.chain_thresholds
    )
    cluster_by_atom = _cluster_membership_by_atom(clusters)
    cluster_rates_by_id = {
        cluster_rate.cluster_id: cluster_rate
        for cluster_rate in cluster_rates
    }
    path_by_atom = _path_membership_by_atom(path)
    endpoints_by_atom = _endpoint_roles_by_atom(endpoint_roles or {})

    rows: list[dict[str, Any]] = []
    for scored_atom in scored_atoms:
        atom = scored_atom.atom
        atom_id = _atom_identity(atom)
        annotation = annotation_by_atom.get(atom_id)
        chain_threshold = chain_thresholds.get(atom.chain_id)
        cluster_membership = cluster_by_atom.get(atom_id)
        path_membership = path_by_atom.get(atom_id)
        endpoint_membership = endpoints_by_atom.get(atom_id, ())

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
            "secondary_structure": (
                annotation.secondary_structure if annotation else None
            ),
            "sasa_angstrom_squared": (
                annotation.solvent_accessible_surface_area
                if annotation
                else None
            ),
        }
        row.update(
            _cluster_columns(
                cluster_membership,
                cluster_rates_by_id=cluster_rates_by_id,
                null_comparisons=null_comparisons or {},
            )
        )
        row.update(_path_columns(path_membership, path=path))
        row.update(_endpoint_columns(endpoint_membership))
        row.update(_marcus_columns(marcus_rate, on_path=path_membership is not None))
        rows.append(row)

    return tuple(rows)


def _chain_thresholds_by_chain(
    chain_thresholds: Iterable[ChainPercentileThreshold],
) -> dict[str, ChainPercentileThreshold]:
    return {
        threshold.chain_id: threshold
        for threshold in chain_thresholds
    }


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


def _path_membership_by_atom(
    path: EpathPath | None,
) -> dict[tuple[int, int | None], tuple[int, float | None, float | None]]:
    if path is None:
        return {}

    memberships: dict[tuple[int, int | None], tuple[int, float | None, float | None]] = {}
    for path_index, scored_atom in enumerate(path.atoms):
        previous_edge = (
            path.edge_lengths[path_index - 1]
            if path_index > 0
            else None
        )
        next_edge = (
            path.edge_lengths[path_index]
            if path_index < len(path.edge_lengths)
            else None
        )
        memberships[_scored_atom_identity(scored_atom)] = (
            path_index + 1,
            previous_edge,
            next_edge,
        )

    return memberships


def _endpoint_roles_by_atom(
    endpoint_roles: Mapping[str, EpathResolvedEndpoint],
) -> dict[tuple[int, int | None], tuple[tuple[str, EpathResolvedEndpoint], ...]]:
    grouped: dict[tuple[int, int | None], list[tuple[str, EpathResolvedEndpoint]]] = {}
    for role, endpoint in endpoint_roles.items():
        grouped.setdefault(
            _scored_atom_identity(endpoint.selected_atom),
            [],
        ).append((role, endpoint))

    return {
        atom_id: tuple(role_entries)
        for atom_id, role_entries in grouped.items()
    }


def _cluster_columns(
    cluster_membership: tuple[EpathCluster, int] | None,
    *,
    cluster_rates_by_id: Mapping[int, EpathClusterRate],
    null_comparisons: Mapping[int, EpathNullModelComparison],
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
            "cluster_rate": None,
            "cluster_rate_normalized": None,
            "null_observed_largest_cluster_size": None,
            "null_observed_mst_total_length": None,
            "null_largest_z_score": None,
            "null_largest_p_value_greater": None,
            "null_mst_z_score": None,
            "null_mst_p_value_less": None,
        }

    cluster, member_index = cluster_membership
    cluster_rate = cluster_rates_by_id.get(cluster.cluster_id)
    null_comparison = null_comparisons.get(cluster.cluster_id)
    return {
        "cluster_id": cluster.cluster_id,
        "cluster_member_index": member_index,
        "cluster_atom_count": cluster.atom_count,
        "cluster_mst_total_length": cluster.metrics.mst_total_length,
        "cluster_mean_step": cluster.metrics.mean_step,
        "cluster_largest_edge": cluster.metrics.mst_largest_edge,
        "cluster_span": cluster.metrics.span,
        "cluster_rate": cluster_rate.rate if cluster_rate else None,
        "cluster_rate_normalized": (
            cluster_rate.normalized_rate if cluster_rate else None
        ),
        "null_observed_largest_cluster_size": (
            null_comparison.observed_largest_cluster_size
            if null_comparison
            else None
        ),
        "null_observed_mst_total_length": (
            null_comparison.observed_mst_total_length
            if null_comparison
            else None
        ),
        "null_largest_z_score": (
            null_comparison.largest_z_score if null_comparison else None
        ),
        "null_largest_p_value_greater": (
            null_comparison.largest_p_value_greater if null_comparison else None
        ),
        "null_mst_z_score": (
            null_comparison.mst_z_score if null_comparison else None
        ),
        "null_mst_p_value_less": (
            null_comparison.mst_p_value_less if null_comparison else None
        ),
    }


def _path_columns(
    path_membership: tuple[int, float | None, float | None] | None,
    *,
    path: EpathPath | None,
) -> dict[str, Any]:
    if path_membership is None or path is None:
        return {
            "on_path": False,
            "path_order": None,
            "path_total_length": None,
            "path_step_count": None,
            "path_largest_step": None,
            "path_edge_from_previous": None,
            "path_edge_to_next": None,
        }

    path_order, previous_edge, next_edge = path_membership
    return {
        "on_path": True,
        "path_order": path_order,
        "path_total_length": path.total_length,
        "path_step_count": path.step_count,
        "path_largest_step": path.largest_step,
        "path_edge_from_previous": previous_edge,
        "path_edge_to_next": next_edge,
    }


def _endpoint_columns(
    endpoints: tuple[tuple[str, EpathResolvedEndpoint], ...],
) -> dict[str, Any]:
    return {
        "endpoint_roles": ";".join(role for role, _ in endpoints) or None,
        "endpoint_tokens": ";".join(endpoint.token for _, endpoint in endpoints)
        or None,
        "endpoint_exact_match": ";".join(
            _bool_text(endpoint.exact_match) for _, endpoint in endpoints
        )
        or None,
        "endpoint_snap_distance": ";".join(
            _format_cell(endpoint.snap_distance) for _, endpoint in endpoints
        )
        or None,
    }


def _marcus_columns(
    marcus_rate: EpathMarcusRate | None,
    *,
    on_path: bool,
) -> dict[str, Any]:
    if marcus_rate is None or not on_path:
        return {
            "marcus_rate": None,
            "marcus_path_length": None,
            "marcus_reorganization_energy_ev": None,
            "marcus_delta_g_ev": None,
            "marcus_temperature_k": None,
            "marcus_beta_per_angstrom": None,
            "marcus_reorganization_exponent": None,
            "marcus_distance_exponent": None,
        }

    return {
        "marcus_rate": marcus_rate.rate,
        "marcus_path_length": marcus_rate.path_length_angstrom,
        "marcus_reorganization_energy_ev": marcus_rate.reorganization_energy_ev,
        "marcus_delta_g_ev": marcus_rate.delta_g_ev,
        "marcus_temperature_k": marcus_rate.temperature_k,
        "marcus_beta_per_angstrom": marcus_rate.beta_per_angstrom,
        "marcus_reorganization_exponent": marcus_rate.reorganization_exponent,
        "marcus_distance_exponent": marcus_rate.distance_exponent,
    }


def _format_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        fieldname: _format_cell(row.get(fieldname))
        for fieldname in COMBINED_ATOM_FIELDNAMES
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return _bool_text(value)

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"

    return str(value)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _scored_atom_identity(scored_atom: EpathScoredAtom) -> tuple[int, int | None]:
    return _atom_identity(scored_atom.atom)


def _atom_identity(atom: EpathAtom) -> tuple[int, int | None]:
    return atom.source_atom_index, atom.atom_serial
