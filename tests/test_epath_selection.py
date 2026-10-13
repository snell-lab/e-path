"""E-path atom selection: Z-scores and percentiles."""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from support import approx, make_scored_atom

from epath.analysis.selection import (
    EpathSelectionError,
    score_bdamage_atoms,
    select_atoms_by_percentile,
)
from rabdam.bdamage import BDamageAtomResult
from rabdam.structure.models import PreparedAtom
from rabdam.structure.prepare import classify_atom


def _atoms(
    bdamage_values: Sequence[float],
) -> tuple[list[PreparedAtom], list[BDamageAtomResult]]:
    """Return atoms with the given BDamage values, and their BDamage results."""
    scored = [
        make_scored_atom(
            source_atom_index=i, bdamage_atom_index=i + 1, atom_serial=i + 1, bdamage=b
        )
        for i, b in enumerate(bdamage_values)
    ]
    return (
        [classify_atom(atom.record) for atom in scored],
        [atom.bdamage_result for atom in scored],
    )


def test_z_scores_use_the_population_standard_deviation() -> None:
    values = [0.5, 1.0, 1.5, 2.0]
    scored = score_bdamage_atoms(*_atoms(values))

    z = np.asarray([atom.bdamage_z_score for atom in scored])
    assert z.mean() == approx(0.0, abs=1e-12)
    assert z.std() == approx(1.0)
    assert z.tolist() == approx(((np.asarray(values) - 1.25) / np.std(values)).tolist())
    assert [
        atom.bdamage_z_score for atom in score_bdamage_atoms(*_atoms([2.0, 2.0, 2.0]))
    ] == [0.0, 0.0, 0.0]
    with pytest.raises(EpathSelectionError, match="finite"):
        score_bdamage_atoms(*_atoms([1.0, float("inf")]))


def test_lowest_percentile_selects_the_smallest_bdamage_values() -> None:
    values = list(range(100, 0, -1))  # 100 down to 1, so input order is not rank order
    result = select_atoms_by_percentile(*_atoms(values), percentile=2.0)

    assert sorted(atom.bdamage_result.bdamage for atom in result.selected_atoms) == [
        1.0,
        2.0,
    ]
    assert result.selected_percent == approx(2.0)
    assert result.bdamage_threshold == approx(np.percentile(values, 2.0))
    assert result.boundary_z_score == approx(
        max(a.bdamage_z_score for a in result.selected_atoms)
    )
    assert result.z_threshold == approx(
        (result.bdamage_threshold - np.mean(values)) / np.std(values)
    )
    assert result.bdamage_mean == approx(50.5)
    assert result.selected_bdamage_mean == approx(1.5)
    assert result.selected_bdamage_standard_deviation == approx(0.5)
    assert (
        "lowest 2.000% by BDamage" in result.description
        and "selected 2 of 100" in result.description
    )


def test_selected_atoms_stay_in_input_order_when_the_count_is_cut() -> None:
    # The 25th percentile of these ten values is 3.25, which admits the three
    # atoms up to 3.0, but 25% of ten atoms rounds to two.
    values = [9.0, 3.0, 8.0, 1.0, 7.0, 6.0, 2.0, 5.0, 4.0, 10.0]
    result = select_atoms_by_percentile(*_atoms(values), percentile=25.0)

    assert result.bdamage_threshold == approx(3.25)
    assert [atom.bdamage_result.bdamage for atom in result.selected_atoms] == [1.0, 2.0]
    assert [atom.record.source_atom_index for atom in result.selected_atoms] == [3, 6]

    # Of two atoms with the same BDamage, the cut drops the later one.
    tied = select_atoms_by_percentile(*_atoms([2.0, 1.0, 1.0, 3.0]), percentile=25.0)
    assert [atom.record.source_atom_index for atom in tied.selected_atoms] == [1]


def test_a_count_exactly_halfway_rounds_to_the_even_number() -> None:
    values = [float(value) for value in range(1, 11)]

    # 5% of ten atoms is half an atom, which rounds down to none.
    assert (
        select_atoms_by_percentile(*_atoms(values), percentile=5.0).selected_atoms == ()
    )
    # 15% is one and a half, which rounds up to two.
    assert (
        len(select_atoms_by_percentile(*_atoms(values), percentile=15.0).selected_atoms)
        == 2
    )


def test_percentile_extremes() -> None:
    atoms = _atoms([3.0, 1.0, 2.0])

    none = select_atoms_by_percentile(*atoms, percentile=0.0)
    assert none.selected_atoms == () and none.bdamage_threshold == 1.0
    assert math.isnan(none.boundary_z_score) and math.isnan(none.selected_bdamage_mean)

    everything = select_atoms_by_percentile(*atoms, percentile=100.0)
    assert everything.selected_atoms == everything.scored_atoms
    assert everything.bdamage_threshold == 3.0 and everything.selected_percent == 100.0
