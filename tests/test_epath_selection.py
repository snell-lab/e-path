"""E-path atom selection: Z-scores, percentiles, per-chain cutoffs, plateaus."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from support import approx, make_epath_atom, make_scored_atom

from epath.analysis.atoms import EpathAtom
from epath.analysis.selection import (
    EpathSelectionError,
    find_z_score_plateau,
    score_bdamage_atoms,
    select_atoms_by_percentile,
    select_atoms_by_percentile_per_chain,
)


def _atoms(
    bdamage_values: Sequence[float], chain_ids: Sequence[str] | None = None
) -> tuple[EpathAtom, ...]:
    """Return E-path atoms with the given BDamage values and chain IDs."""
    chain_ids = chain_ids or ["A"] * len(bdamage_values)
    return tuple(
        make_epath_atom(
            source_atom_index=i,
            bdamage_atom_index=i + 1,
            atom_serial=i + 1,
            bdamage=float(b),
            chain_id=c,
        )
        for i, (b, c) in enumerate(zip(bdamage_values, chain_ids, strict=True))
    )


def test_z_scores_use_the_population_standard_deviation() -> None:
    values = [0.5, 1.0, 1.5, 2.0]
    scored = score_bdamage_atoms(_atoms(values))

    z = np.asarray([atom.bdamage_z_score for atom in scored])
    assert z.mean() == approx(0.0, abs=1e-12)
    assert z.std() == approx(1.0)
    assert z.tolist() == approx(((np.asarray(values) - 1.25) / np.std(values)).tolist())
    assert [
        atom.bdamage_z_score for atom in score_bdamage_atoms(_atoms([2.0, 2.0, 2.0]))
    ] == [0.0, 0.0, 0.0]
    assert score_bdamage_atoms(()) == ()
    with pytest.raises(EpathSelectionError, match="finite"):
        score_bdamage_atoms(_atoms([1.0, float("inf")]))


def test_lowest_percentile_selects_the_smallest_bdamage_values() -> None:
    values = list(range(100, 0, -1))  # 100 down to 1, so input order is not rank order
    result = select_atoms_by_percentile(_atoms(values), percentile=2.0)

    assert sorted(atom.bdamage for atom in result.selected_atoms) == [1.0, 2.0]
    assert result.method == "percentile"
    assert result.selected_percent == approx(2.0)
    assert result.bdamage_threshold == approx(np.percentile(values, 2.0))
    assert result.boundary_z_score == approx(
        max(a.bdamage_z_score for a in result.selected_atoms)
    )
    assert result.bdamage_threshold is not None
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
    assert result.chain_thresholds == () and result.plateau is None


def test_percentile_extremes() -> None:
    atoms = _atoms([3.0, 1.0, 2.0])

    none = select_atoms_by_percentile(atoms, percentile=0.0)
    assert (
        none.selected_atoms == ()
        and math.isnan(none.z_threshold)
        and math.isnan(none.selected_bdamage_mean)
    )

    everything = select_atoms_by_percentile(atoms, percentile=100.0)
    assert len(everything.selected_atoms) == 3 and everything.selected_percent == 100.0


@pytest.mark.parametrize("percentile", [-1.0, 100.5, float("nan"), True, "two"])
def test_invalid_percentiles_are_rejected(percentile: Any) -> None:
    with pytest.raises(EpathSelectionError):
        select_atoms_by_percentile(_atoms([1.0, 2.0]), percentile=percentile)


def test_per_chain_selection_stratifies_the_cutoff() -> None:
    # Chain A holds the 50 lowest values, so a global cutoff would select
    # nothing from chain B.
    values = list(range(1, 101))
    chains = ["A"] * 50 + ["B"] * 50

    result = select_atoms_by_percentile_per_chain(
        _atoms(values, chains), percentile=10.0
    )

    by_chain = {
        c: sorted(a.bdamage for a in result.selected_atoms if a.chain_id == c)
        for c in "AB"
    }
    assert by_chain == {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [51.0, 52.0, 53.0, 54.0, 55.0],
    }
    assert result.method == "percentile_per_chain"
    assert result.bdamage_threshold is None and result.boundary_z_score is None
    assert [t.chain_id for t in result.chain_thresholds] == ["A", "B"]
    assert [t.selected_count for t in result.chain_thresholds] == [5, 5]
    assert [t.atom_count for t in result.chain_thresholds] == [50, 50]
    assert result.z_threshold == approx(
        float(np.median([t.z_threshold for t in result.chain_thresholds]))
    )
    assert result.selected_percent == approx(10.0)

    full = select_atoms_by_percentile_per_chain(
        _atoms(values, chains), percentile=100.0
    )
    assert len(full.selected_atoms) == 100


def test_plateau_is_a_local_shelf_in_the_rank_ordered_z_scores() -> None:
    before = np.linspace(-3.0, -2.0, 5)  # 5 atoms per Z unit
    shelf = np.linspace(-1.9, -1.8, 5)  # 50 atoms per Z unit
    after = np.linspace(-1.7, -0.7, 5)  # 5 atoms per Z unit
    scored = [make_scored_atom(z) for z in np.concatenate([after, shelf, before])]

    plateau = find_z_score_plateau(scored)

    assert plateau is not None
    assert plateau.atom_count == 5 and plateau.first_rank == 6
    assert plateau.lowest_z_score == approx(-1.9) and plateau.highest_z_score == approx(
        -1.8
    )


def test_smooth_falloff_has_no_plateau() -> None:
    ranks = np.arange(1, 61)
    smooth = [make_scored_atom(float(z)) for z in -3.0 + 0.0008 * ranks**2]
    assert find_z_score_plateau(smooth) is None
    assert (
        find_z_score_plateau(
            [make_scored_atom(float(z)) for z in np.linspace(-3, -1, 14)]
        )
        is None
    )
    assert find_z_score_plateau(()) is None
