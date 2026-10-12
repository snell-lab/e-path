"""E-path atom selection: Z-scores and percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from support import approx, make_epath_atom

from epath.analysis.atoms import EpathAtom
from epath.analysis.selection import (
    EpathSelectionError,
    score_bdamage_atoms,
    select_atoms_by_percentile,
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
