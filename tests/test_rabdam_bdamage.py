"""Packing-density counting and the BDamage sliding-window normalization."""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from support import approx, make_metadata, make_prepared_atom

from rabdam.bdamage import (
    BDamageAtomInput,
    BDamageScoreError,
    bdamage_scores_as_tuple,
    calculate_bdamage_scores,
    calculate_bdamage_scores_for_structure,
    centered_window_average_b_factors,
    sort_atom_inputs_for_bdamage,
)
from rabdam.packing import calculate_packing_density
from rabdam.pipeline import (
    bdamage_window_size_from_fraction,
)
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
)

type Position = tuple[float, float, float]


def _on_x_axis(x: float) -> Position:
    """Return the position of an atom image on the x axis."""
    return (x, 0.0, 0.0)


def _packing_density(
    selected: Sequence[PreparedAtom],
    neighbours: Sequence[Position],
    threshold: float = 7.0,
) -> tuple[int, ...]:
    """Run the production search on a hand-built set of surrounding atoms."""
    return calculate_packing_density(
        selected_atoms=selected,
        neighbour_coordinates=np.asarray(neighbours, dtype=np.float64).reshape(-1, 3),
        packing_density_threshold=threshold,
    )


def test_neighbours_are_counted_strictly_inside_the_threshold() -> None:
    selected = (make_prepared_atom(source_atom_index=0, x=0.0),)
    neighbours = (
        # The atom itself is returned by the search but never counted.
        _on_x_axis(0.0),
        _on_x_axis(1.0),  # inside
        _on_x_axis(6.999),  # inside
        _on_x_axis(7.0),  # exactly on the threshold: excluded
        _on_x_axis(7.001),  # outside
        _on_x_axis(-3.0),  # inside, on the other side
        _on_x_axis(5.0),  # inside
    )

    assert _packing_density(selected, neighbours) == (4,)


def _inputs(
    b_factors: Sequence[float],
    packing_densities: Sequence[int],
    serials: Sequence[int] | None = None,
) -> tuple[BDamageAtomInput, ...]:
    """Return one-based BDamage inputs from aligned B-factors and densities."""
    serials = serials or list(range(1, len(b_factors) + 1))
    return tuple(
        BDamageAtomInput(
            bdamage_atom_index=i + 1,
            source_atom_index=i,
            atom_serial=serials[i],
            b_factor=b,
            packing_density=pd,
        )
        for i, (b, pd) in enumerate(zip(b_factors, packing_densities, strict=True))
    )


def test_hand_computed_window_averages_reuse_the_edge_windows() -> None:
    inputs = _inputs([1.0, 2.0, 3.0, 4.0, 5.0], [10, 20, 30, 40, 50])

    averages = centered_window_average_b_factors(inputs, window_size=3)
    assert averages == approx((2.0, 2.0, 3.0, 4.0, 4.0))

    result = calculate_bdamage_scores(atom_inputs=inputs, window_size=3)
    assert bdamage_scores_as_tuple(result) == approx((0.5, 1.0, 1.0, 1.0, 1.25))
    assert len(result) == 5
    assert [r.average_b_factor for r in result] == approx([2, 2, 3, 4, 4])
    assert centered_window_average_b_factors(inputs, window_size=5) == approx(
        (3.0,) * 5
    )


def test_results_are_returned_in_input_order_after_sorting_by_density() -> None:
    # Input order is the reverse of packing-density order.
    inputs = _inputs([5.0, 4.0, 3.0, 2.0, 1.0], [50, 40, 30, 20, 10])

    result = calculate_bdamage_scores(atom_inputs=inputs, window_size=3)

    assert [r.bdamage_atom_index for r in result] == [1, 2, 3, 4, 5]
    assert [r.packing_density for r in result] == [50, 40, 30, 20, 10]
    assert bdamage_scores_as_tuple(result) == approx((1.25, 1.0, 1.0, 1.0, 0.5))


def test_density_ties_are_broken_by_serial_then_source_index() -> None:
    inputs = _inputs([1.0, 2.0, 3.0, 4.0], [7, 7, 7, 7], serials=[40, 10, 30, 20])
    ordered = sort_atom_inputs_for_bdamage(inputs)
    assert [a.atom_serial for a in ordered] == [10, 20, 30, 40]

    missing_serial = (
        BDamageAtomInput(
            bdamage_atom_index=1,
            source_atom_index=9,
            atom_serial=None,
            b_factor=1.0,
            packing_density=7,
        ),
        BDamageAtomInput(
            bdamage_atom_index=2,
            source_atom_index=3,
            atom_serial=None,
            b_factor=1.0,
            packing_density=7,
        ),
    )
    assert [
        a.source_atom_index for a in sort_atom_inputs_for_bdamage(missing_serial)
    ] == [3, 9]


def _prepared(*atoms: PreparedAtom) -> PreparedStructure:
    return PreparedStructure(
        source_atoms=tuple(atom.record for atom in atoms),
        cleaned_atoms=atoms,
        selected_atoms=atoms,
        metadata=make_metadata(),
        removed=RemovedAtomCounts(0, 0, 0, 0, 0),
    )


def test_structure_scores_pair_atoms_with_packing_densities() -> None:
    prepared = _prepared(
        make_prepared_atom(source_atom_index=0),
        make_prepared_atom(source_atom_index=1, atom_serial=2),
    )

    result = calculate_bdamage_scores_for_structure(
        prepared_structure=prepared, packing_densities=(5, 9), window_size=1
    )
    assert bdamage_scores_as_tuple(result) == approx((1.0, 1.0))
    assert [r.packing_density for r in result] == [5, 9]

    # A B-factor that is not finite and positive is rejected where it enters
    # the score calculation, naming the source atom.
    for b_factor in (0.0, -1.0, math.inf, math.nan):
        bad = _prepared(
            make_prepared_atom(source_atom_index=0),
            make_prepared_atom(source_atom_index=1, atom_serial=2, b_factor=b_factor),
        )
        with pytest.raises(BDamageScoreError, match="for source atom 1"):
            calculate_bdamage_scores_for_structure(
                prepared_structure=bad, packing_densities=(5, 9), window_size=1
            )


@pytest.mark.parametrize(
    ("atom_count", "fraction", "minimum", "expected"),
    [
        (100, 0.02, 11, 11),  # round(2) -> 3, raised to the minimum
        (1000, 0.02, 11, 21),  # 20 is even, so 21
        (1050, 0.02, 11, 21),  # 21 already odd
        (100, 0.02, 10, 11),  # an even minimum is raised to the next odd size
        (2000, 0.05, 1, 101),
    ],
)
def test_window_size_from_fraction(
    atom_count: int, fraction: float, minimum: int, expected: int
) -> None:
    window = bdamage_window_size_from_fraction(
        atom_count=atom_count,
        window_size_fraction=fraction,
        minimum_window_size=minimum,
    )
    assert window == expected and window % 2 == 1
