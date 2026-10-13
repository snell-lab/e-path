"""Bnet: site selection, the KDE metric, the reference database, percentiles."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import approx, synthetic_atoms, write_synthetic_pdb

import rabdam.bnet.reference as bnet_reference
from rabdam.bnet.metric import (
    BnetMetricError,
    bnet_from_bdamage_values,
    calculate_protein_bnet,
    is_protein_bnet_site,
    protein_bnet_site_bdamage_values,
)
from rabdam.bnet.percentile import (
    BnetPercentileError,
    calculate_bnet_percentile,
    find_resolution_matched_reference_set,
)
from rabdam.bnet.pipeline import run_bnet_pipeline
from rabdam.bnet.reference import (
    DEFAULT_REFERENCE_DATABASE_CSV_PATH,
    BnetReferenceDatabase,
    BnetReferenceEntry,
    BnetReferenceError,
    load_bnet_reference_database,
    load_default_bnet_reference_database,
)
from rabdam.pipeline import BDamagePipelineResult, run_bdamage_pipeline
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input


@pytest.fixture(scope="module")
def bdamage_result(synthetic_pdb: Path) -> BDamagePipelineResult:
    """Return the BDamage pipeline result for the synthetic crystal."""
    return run_bdamage_pipeline(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )


@pytest.mark.parametrize(
    ("residue", "atom", "expected"),
    [
        ("ASP", "OD1", True),
        ("asp", "od2", True),
        ("GLU", "OE2", True),
        ("DGL", "OE1", True),
        ("DAS", "OD2", True),
        ("ASP", "OE1", False),
        ("GLU", "OD1", False),
        ("ASN", "OD1", False),
        ("ASP", "CG", False),
    ],
)
def test_bnet_site_recognition(residue: str, atom: str, expected: bool) -> None:
    assert is_protein_bnet_site(residue_name=residue, atom_name=atom) is expected


def test_site_values_are_those_of_the_carboxyl_oxygens_in_prepared_order(
    bdamage_result: BDamagePipelineResult,
) -> None:
    prepared = bdamage_result.prepared_structure
    scores = bdamage_result.bdamage_results

    values = protein_bnet_site_bdamage_values(
        prepared_structure=prepared, bdamage_results=scores
    )

    # The synthetic model has exactly these four carboxyl oxygens.
    sites = [
        ("GLU", 4, "OE1"),
        ("GLU", 4, "OE2"),
        ("ASP", 10, "OD1"),
        ("ASP", 10, "OD2"),
    ]
    bdamage_by_site = {
        (
            atom.record.residue_name,
            atom.record.residue_number,
            atom.record.atom_name,
        ): score.bdamage
        for atom, score in zip(prepared.selected_atoms, scores, strict=True)
    }
    assert values == tuple(bdamage_by_site[site] for site in sites)


def test_symmetric_sites_around_the_median_give_bnet_near_one() -> None:
    sites = np.linspace(-1.0, 1.0, 21)
    result = bnet_from_bdamage_values(
        all_bdamage_values=[-5.0, 0.0, 5.0], bnet_site_bdamage_values=sites.tolist()
    )

    assert result == approx(1.0, rel=0.1)


def test_bnet_grows_as_sites_move_above_the_median() -> None:
    all_values = [1.0] * 50
    low = bnet_from_bdamage_values(
        all_bdamage_values=all_values, bnet_site_bdamage_values=[0.5, 0.6, 0.7, 0.8]
    )
    high = bnet_from_bdamage_values(
        all_bdamage_values=all_values, bnet_site_bdamage_values=[1.2, 1.3, 1.4, 1.5]
    )

    assert low < 0.1
    assert high > 10.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"all_bdamage_values": [1.0], "bnet_site_bdamage_values": [1.0]},
            "fewer than two",
        ),
        (
            {"all_bdamage_values": [1.0], "bnet_site_bdamage_values": [2.0, 2.0]},
            "zero variance",
        ),
    ],
)
def test_bnet_input_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(BnetMetricError, match=message):
        bnet_from_bdamage_values(**kwargs)


def test_protein_bnet_uses_the_selected_sites(
    bdamage_result: BDamagePipelineResult,
) -> None:
    result = calculate_protein_bnet(
        prepared_structure=bdamage_result.prepared_structure,
        bdamage_results=bdamage_result.bdamage_results,
    )
    assert result.site_count == 4 and result.bnet > 0.0


def _database(
    resolutions: Sequence[float], bnets: Sequence[float]
) -> BnetReferenceDatabase:
    """Return a reference database with the given resolutions and Bnet values."""
    return BnetReferenceDatabase(
        entries=[
            BnetReferenceEntry(f"{i:04d}", r, b)
            for i, (r, b) in enumerate(zip(resolutions, bnets, strict=True))
        ],
        database_id="test-db",
    )


def test_nearest_resolution_set_includes_boundary_ties() -> None:
    database = _database([1.0, 2.0, 2.0, 2.0, 3.0], [10.0, 30.0, 10.0, 20.0, 40.0])

    matched = find_resolution_matched_reference_set(
        resolution_angstrom=2.05,
        reference_database=database,
        nearest_resolution_count=2,
    )

    assert matched.indices == (1, 2, 3)
    assert (matched.resolution_min, matched.resolution_max) == (2.0, 2.0)


def test_percentile_is_the_rank_of_the_nearest_local_reference_value() -> None:
    database = _database([1.0, 2.0, 2.0, 2.0, 3.0], [10.0, 30.0, 10.0, 20.0, 40.0])

    result = calculate_bnet_percentile(
        bnet=21.0,
        resolution_angstrom=2.0,
        reference_database=database,
        nearest_resolution_count=2,
    )

    # The local values are 10, 20 and 30, and 20 is the nearest to 21: rank 2
    # of 3.
    assert result.local_reference_count == 3
    assert result.percentile_percent == approx(200 / 3)
    assert (
        result.reference_database_id == "test-db" and result.reference_entry_count == 5
    )

    top = calculate_bnet_percentile(
        bnet=1000.0,
        resolution_angstrom=2.0,
        reference_database=database,
        nearest_resolution_count=2,
    )
    assert top.percentile_percent == 100.0

    overrides: tuple[dict[str, Any], ...] = (
        {"bnet": float("nan")},
        {"resolution_angstrom": 0.0},
        {"nearest_resolution_count": 0},
        {"nearest_resolution_count": 6},
    )
    for bad in overrides:
        with pytest.raises(BnetPercentileError):
            arguments: dict[str, Any] = {
                "bnet": 1.0,
                "resolution_angstrom": 2.0,
                "reference_database": database,
                "nearest_resolution_count": 2,
                **bad,
            }
            calculate_bnet_percentile(**arguments)


def test_reference_csv_loading_reads_the_columns_and_a_manifest(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "ref.csv"
    csv_path.write_text(
        ",PDB code,Resolution (A),Bnet\n0,6Q5R,1.61,173.36\n\n1,7udy,2.4,143.8\n",
        encoding="utf-8",
    )
    (tmp_path / "ref.manifest.json").write_text(
        json.dumps({"database_id": "demo", "entry_count": 2})
    )

    database = load_bnet_reference_database(csv_path)

    assert database.entry_count == 2
    assert [entry.pdb_id for entry in database.entries] == ["6Q5R", "7udy"]
    assert database.resolution_values.tolist() == [1.61, 2.4]
    assert database.bnet_values.tolist() == [173.36, 143.8]
    assert not database.bnet_values.flags.writeable
    assert database.database_id == "demo"

    unnamed = tmp_path / "other.csv"
    unnamed.write_text("PDB code,Resolution (A),Bnet\nX,1.0,2.0\n")
    assert load_bnet_reference_database(unnamed).database_id == "other"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("PDB code,Bnet\nX,1.0\n", r"missing the column 'Resolution \(A\)'"),
        ("pdb_id,resolution,bnet\nX,1.0,2.0\n", "missing the column 'PDB code'"),
        ("PDB code,Resolution (A),Bnet\nX,1.0,2.0\nx,1.5,3.0\n", "Duplicate pdb_id"),
        ("PDB code,Resolution (A),Bnet\nX,abc,2.0\n", "non-numeric"),
        ("PDB code,Resolution (A),Bnet\nX,0.0,2.0\n", "greater than zero"),
        ("PDB code,Resolution (A),Bnet\nX,1.0,-2.0\n", "greater than or equal to zero"),
        ("PDB code,Resolution (A),Bnet\n", "must contain entries"),
    ],
)
def test_reference_csv_validation(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(BnetReferenceError, match=message):
        load_bnet_reference_database(path)
    with pytest.raises(FileNotFoundError):
        load_bnet_reference_database(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"entry_count": 3}, "holds 2 entries"),
        ({"sha256": "0" * 64}, "SHA-256"),
    ],
)
def test_a_reference_csv_that_disagrees_with_its_manifest_is_rejected(
    tmp_path: Path, manifest: dict[str, Any], message: str
) -> None:
    csv_path = tmp_path / "ref.csv"
    csv_path.write_text("PDB code,Resolution (A),Bnet\n1ABC,1.5,2.0\n2ABC,1.6,3.0\n")
    (tmp_path / "ref.manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(BnetReferenceError, match=message):
        load_bnet_reference_database(csv_path)


def test_a_reference_csv_matching_its_manifest_digest_loads(tmp_path: Path) -> None:
    csv_path = tmp_path / "ref.csv"
    csv_path.write_bytes(b"PDB code,Resolution (A),Bnet\r\n1ABC,1.5,2.0\r\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (tmp_path / "ref.manifest.json").write_text(
        json.dumps({"entry_count": 1, "sha256": digest})
    )

    assert load_bnet_reference_database(csv_path).entry_count == 1


def test_the_bundled_reference_database_is_read_once() -> None:
    assert (
        load_default_bnet_reference_database() is load_default_bnet_reference_database()
    )


def test_bundled_reference_database_is_consistent_with_its_manifest() -> None:
    manifest = json.loads(
        DEFAULT_REFERENCE_DATABASE_CSV_PATH.with_suffix(".manifest.json").read_text()
    )
    database = load_default_bnet_reference_database()

    assert database is not None
    assert database.entry_count == manifest["entry_count"]
    assert database.database_id == manifest["database_id"]
    assert (
        database.resolution_values.min() > 0.0
        and database.resolution_values.max()
        <= manifest["build"]["settings"]["max_resolution_angstrom"]
    )
    assert database.bnet_values.min() >= 0.0


def test_bnet_pipeline_reports_full_results(
    bdamage_result: BDamagePipelineResult,
) -> None:
    result = run_bnet_pipeline(
        prepared_structure=bdamage_result.prepared_structure,
        bdamage_results=bdamage_result.bdamage_results,
    )

    assert result.bnet_result is not None and result.percentile_result is not None
    assert (
        result.bnet_unavailable_reason is None
        and result.percentile_unavailable_reason is None
    )
    assert result.percentile_result.resolution_angstrom == approx(1.8)
    assert result.percentile_result.local_reference_count >= 1000


def test_bnet_pipeline_explains_a_missing_reference_database(
    bdamage_result: BDamagePipelineResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bnet_reference, "DEFAULT_REFERENCE_DATABASE_CSV_PATH", tmp_path / "none.csv"
    )

    result = run_bnet_pipeline(
        prepared_structure=bdamage_result.prepared_structure,
        bdamage_results=bdamage_result.bdamage_results,
    )

    assert result.bnet_result is not None and result.percentile_result is None
    assert result.percentile_unavailable_reason is not None
    assert result.percentile_unavailable_reason.startswith(
        "default reference database could not be loaded"
    )


def test_bnet_pipeline_explains_missing_resolution(tmp_path: Path) -> None:
    path = write_synthetic_pdb(tmp_path / "nores.pdb", resolution=None)
    bdamage = run_bdamage_pipeline(read_structure(resolve_structure_input(str(path))))

    result = run_bnet_pipeline(
        prepared_structure=bdamage.prepared_structure,
        bdamage_results=bdamage.bdamage_results,
    )

    assert result.bnet_result is not None
    assert result.percentile_result is None
    assert result.percentile_unavailable_reason == "structure resolution is unavailable"


def test_bnet_pipeline_explains_missing_sites(tmp_path: Path) -> None:
    atoms = [
        atom for atom in synthetic_atoms() if atom.residue_name not in {"ASP", "GLU"}
    ]
    path = write_synthetic_pdb(tmp_path / "nosites.pdb", atoms=atoms)
    bdamage = run_bdamage_pipeline(read_structure(resolve_structure_input(str(path))))

    result = run_bnet_pipeline(
        prepared_structure=bdamage.prepared_structure,
        bdamage_results=bdamage.bdamage_results,
    )

    assert result.bnet_result is None and result.percentile_result is None
    assert (
        result.bnet_unavailable_reason is not None
        and "Asp/Glu" in result.bnet_unavailable_reason
    )
    assert (
        result.percentile_unavailable_reason is not None
        and result.percentile_unavailable_reason.startswith("Bnet is unavailable")
    )
