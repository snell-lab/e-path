"""Write the CSV tables of an E-path run.

There are four: one row per scored atom, one per reported pathway, one per
step of each pathway, and one per run. `epath.outputs.write` calls the writers
for a single run, and the batch command calls `write_run_csv` with the record
of every structure to make the cohort table. The run records are built in
`epath.outputs.runs`.

Each `*_FIELDNAMES` list holds the columns of one table, in the order they are
written. Every writer creates the directory of its file when it is missing,
replaces a file that is already there, and returns the path it wrote.
"""

import csv
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path

from epath.analysis.clustering import EpathCluster
from epath.analysis.labels import atom_token
from epath.analysis.pathways import PathReport
from epath.analysis.selection import EpathSelectionResult
from epath.outputs.runs import EpathRunRecord

# Twelve significant figures is more than any input carries, and it hides
# binary rounding noise such as 0.30000000000000004.
_FLOAT_FORMAT = ".12g"

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

# These are the fields of `EpathRunRecord`, which `write_run_csv` reads by name.
RUN_FIELDNAMES = [
    "structure",
    "input_path",
    "epath_version",
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

# The leading columns, up to `nearby_cofactor_atoms`, are the E-path pathway
# table: the path, its size, and the cofactors within reach of it. The columns
# after them are its geometry, steps, and composition.
CLUSTER_FIELDNAMES = [
    "cluster_id",
    "residue_path",
    "residue_count",
    "atom_count",
    "mst_total_length",
    "nearby_cofactors",
    "nearby_cofactor_distances",
    "nearby_cofactor_atoms",
    "chains",
    "mean_step",
    "largest_edge",
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
]


def write_combined_atom_csv(
    output_path: Path,
    selection_result: EpathSelectionResult,
    *,
    clusters: Iterable[EpathCluster],
) -> Path:
    """Write one CSV row per scored atom, in scored-atom order.

    `selected` is `true` for the atoms of the selection. `cluster_id` and
    `cluster_member_index` are filled for the atoms of `clusters` and left
    blank for every other atom. The member index counts from 1 in the order
    the cluster holds its atoms.
    """
    return _write_csv(
        output_path,
        COMBINED_ATOM_FIELDNAMES,
        _atom_rows(selection_result, clusters),
    )


def write_run_csv(
    output_path: Path,
    records: Iterable[EpathRunRecord],
) -> Path:
    """Write one CSV row per run, in the order of `records`.

    A single run writes one row. Passing the records of a batch writes the
    cohort table for those structures. The columns are the fields of the
    record, and a value that is None is left blank.
    """
    return _write_csv(
        output_path,
        RUN_FIELDNAMES,
        (asdict(record) for record in records),
    )


def write_step_csv(
    output_path: Path,
    path_reports: Iterable[PathReport],
) -> Path:
    """Write one CSV row per step of every reported pathway.

    The summary reports the total, mean, and largest step of a path. This
    records each step individually, because electron tunnelling falls away
    sharply with distance and a single long step can limit a whole path.

    The pathways come in the order of `path_reports` and the steps of each in
    the order the report holds them. `cluster_id` is the pathway's identifier,
    the one the pathway table uses.
    """
    return _write_csv(
        output_path,
        STEP_FIELDNAMES,
        (
            {
                "cluster_id": report.path_id,
                "step_index": step.step_index,
                "length": step.length,
                "kind": step.kind,
                "first_atom": step.first_atom,
                "first_residue": step.first_residue,
                "second_atom": step.second_atom,
                "second_residue": step.second_residue,
            }
            for report in path_reports
            for step in report.steps
        ),
    )


def write_cluster_csv(
    output_path: Path,
    path_reports: Iterable[PathReport],
) -> Path:
    """Write one CSV row per reported pathway, in the order of `path_reports`.

    The three nearby-cofactor columns list the cofactors in range, the nearest
    first, separated by "/" and in the same order in each column. They are
    blank for a pathway with no cofactor in range.
    """
    return _write_csv(
        output_path,
        CLUSTER_FIELDNAMES,
        (_cluster_row(report) for report in path_reports),
    )


def _cluster_row(report: PathReport) -> dict[str, object]:
    metrics = report.cluster.metrics
    fractions = report.composition.fractions
    nearby = report.nearby_cofactors

    return {
        "cluster_id": report.path_id,
        "residue_path": report.residue_path,
        "residue_count": report.residue_count,
        "atom_count": report.node_count,
        "mst_total_length": metrics.mst_total_length,
        "nearby_cofactors": "/".join(cofactor.label for cofactor in nearby),
        "nearby_cofactor_distances": "/".join(
            _format_cell(cofactor.distance) for cofactor in nearby
        ),
        "nearby_cofactor_atoms": "/".join(cofactor.atom_token for cofactor in nearby),
        "chains": "/".join(report.chains),
        "mean_step": metrics.mean_step,
        "largest_edge": metrics.mst_largest_edge,
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
    }


def _atom_rows(
    selection_result: EpathSelectionResult,
    clusters: Iterable[EpathCluster],
) -> Iterator[dict[str, object]]:
    # `source_atom_index` is an atom's position in the input file, so it
    # identifies the atom on its own.
    selected_indices = {
        scored_atom.record.source_atom_index
        for scored_atom in selection_result.selected_atoms
    }
    membership_by_index = {
        scored_atom.record.source_atom_index: (cluster.cluster_id, member_index)
        for cluster in clusters
        for member_index, scored_atom in enumerate(cluster.atoms, start=1)
    }

    for scored_atom in selection_result.scored_atoms:
        record = scored_atom.record
        bdamage_result = scored_atom.bdamage_result
        cluster_id, cluster_member_index = membership_by_index.get(
            record.source_atom_index,
            (None, None),
        )
        yield {
            "source_atom_index": record.source_atom_index,
            "bdamage_atom_index": bdamage_result.bdamage_atom_index,
            "atom_serial": record.atom_serial,
            "record_type": "HETATM" if record.is_hetatm else "ATOM",
            "atom_token": atom_token(record),
            "chain_id": record.chain_id,
            "residue_name": record.residue_name,
            "residue_number": record.residue_number,
            "insertion_code": record.insertion_code,
            "atom_name": record.atom_name,
            "element": record.element,
            "x": record.x,
            "y": record.y,
            "z": record.z,
            "occupancy": record.occupancy,
            "b_factor": record.b_factor,
            "packing_density": bdamage_result.packing_density,
            "average_b_factor": bdamage_result.average_b_factor,
            "bdamage": bdamage_result.bdamage,
            "bdamage_z_score": scored_atom.bdamage_z_score,
            "selected": record.source_atom_index in selected_indices,
            "cluster_id": cluster_id,
            "cluster_member_index": cluster_member_index,
        }


def _write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write a header and the named value of every row, and return the path.

    A row must hold a value for every field name. Values it holds under other
    names are not written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(
            [_format_cell(row[fieldname]) for fieldname in fieldnames] for row in rows
        )

    return output_path


def _format_cell(value: object) -> str:
    """Return a value as the text of a CSV cell.

    None gives an empty cell and a boolean gives `true` or `false`. A float is
    written to twelve significant figures, with NaN as `nan`. Anything else is
    written as `str` gives it.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        return format(value, _FLOAT_FORMAT)

    return str(value)
