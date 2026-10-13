"""Packing-density counting and the BDamage sliding-window normalization."""

from __future__ import annotations

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
from rabdam.crystal.translate import TranslatedAtom
from rabdam.packing import (
    PackingDensityAtomResult,
    PackingDensityError,
    PackingDensityResult,
    calculate_packing_density_from_arrays,
    packing_density_counts_as_tuple,
)
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
)
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    bdamage_window_size_from_fraction,
    validate_workflow_options,
)


def _neighbour(
    index: int,
    source: int,
    x: float,
    *,
    identity: bool = True,
    offset: tuple[int, int, int] = (0, 0, 0),
) -> TranslatedAtom:
    """Return a neighbour image on the x axis at distance `x` from the origin."""
    return TranslatedAtom(
        translated_atom_index=index,
        unit_cell_atom_index=index,
        source_atom_index=source,
        symmetry_operation_index=1 if identity else 2,
        is_identity_symmetry_operation=identity,
        translation_a=offset[0],
        translation_b=offset[1],
        translation_c=offset[2],
        x=x,
        y=0.0,
        z=0.0,
    )


def _packing_density(
    selected: Sequence[PreparedAtom],
    neighbours: Sequence[TranslatedAtom],
    threshold: float = 7.0,
) -> PackingDensityResult:
    """Run the production array path on neighbour images given as objects."""
    return calculate_packing_density_from_arrays(
        selected_atoms=selected,
        neighbour_coordinates=np.asarray(
            [(n.x, n.y, n.z) for n in neighbours], dtype=np.float64
        ).reshape(-1, 3),
        source_atom_indices=np.asarray(
            [n.source_atom_index for n in neighbours], dtype=np.int64
        ),
        is_identity_symmetry_operation=np.asarray(
            [n.is_identity_symmetry_operation for n in neighbours], dtype=np.bool_
        ),
        translation_offsets=np.asarray(
            [(n.translation_a, n.translation_b, n.translation_c) for n in neighbours],
            dtype=np.int64,
        ).reshape(-1, 3),
        packing_density_threshold=threshold,
    )


def _brute_force_counts(
    selected: Sequence[PreparedAtom],
    neighbours: Sequence[TranslatedAtom],
    threshold: float,
) -> tuple[int, ...]:
    """Count neighbours strictly inside the threshold pair by pair.

    This is an independent reference for the KD-tree search. One is subtracted
    for each atom's own central-cell copy, which every test cloud contains.
    """
    return tuple(
        sum(
            math.dist((a.record.x, a.record.y, a.record.z), (n.x, n.y, n.z)) < threshold
            for n in neighbours
        )
        - 1
        for a in selected
    )


def test_neighbours_are_counted_strictly_inside_the_threshold() -> None:
    selected = (make_prepared_atom(source_atom_index=0, x=0.0),)
    neighbours = (
        # The atom's own central-cell copy is returned by the search but never counted.
        _neighbour(1, 0, 0.0),
        _neighbour(2, 1, 1.0),  # inside
        _neighbour(3, 2, 6.999),  # inside
        _neighbour(4, 3, 7.0),  # exactly on the threshold: excluded
        _neighbour(5, 4, 7.001),  # outside
        # Symmetry and lattice images of the atom itself are ordinary neighbours.
        _neighbour(6, 0, -3.0, identity=False),
        _neighbour(7, 0, 5.0, offset=(1, 0, 0)),
    )

    result = _packing_density(selected, neighbours)

    assert packing_density_counts_as_tuple(result) == (4,)
    assert result.atom_results[0].source_atom_index == 0
    assert result.atom_results[0].packing_density_atom_index == 1
    assert result.selected_atom_count == 1 and result.neighbour_atom_count == 7


def test_kd_tree_counts_match_a_brute_force_count() -> None:
    selected = (
        make_prepared_atom(source_atom_index=0, x=0.0),
        make_prepared_atom(source_atom_index=5, x=10.0, atom_serial=6),
    )
    neighbours = (
        _neighbour(1, 0, 0.0),
        _neighbour(2, 5, 10.0),
        _neighbour(3, 9, 4.0),
        _neighbour(4, 9, 15.0),
    )

    result = _packing_density(selected, neighbours)

    # Atom 0 sees only the image at 4.0; atom 5 sees both images of atom 9.
    assert (
        packing_density_counts_as_tuple(result)
        == _brute_force_counts(selected, neighbours, 7.0)
        == (1, 2)
    )


def test_missing_self_copy_and_misaligned_arrays_are_rejected() -> None:
    selected = (make_prepared_atom(source_atom_index=0, x=0.0),)
    with pytest.raises(PackingDensityError, match="central-cell copy"):
        _packing_density(selected, (_neighbour(1, 7, 1.0),))
    with pytest.raises(PackingDensityError, match="central-cell copy"):
        _packing_density(selected, (_neighbour(1, 0, 9.0),))
    with pytest.raises(PackingDensityError, match="shape"):
        calculate_packing_density_from_arrays(
            selected_atoms=selected,
            neighbour_coordinates=np.zeros((2, 3)),
            source_atom_indices=np.zeros(3, dtype=np.int64),
            is_identity_symmetry_operation=np.ones(2, dtype=bool),
            translation_offsets=np.zeros((2, 3), dtype=np.int64),
            packing_density_threshold=7.0,
        )


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
    assert result.window_size == 3 and result.selected_atom_count == 5
    assert [r.sorted_packing_density_index for r in result.atom_results] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [r.average_b_factor for r in result.atom_results] == approx([2, 2, 3, 4, 4])
    assert centered_window_average_b_factors(inputs, window_size=5) == approx(
        (3.0,) * 5
    )


def test_results_are_returned_in_input_order_after_sorting_by_density() -> None:
    # Input order is the reverse of packing-density order.
    inputs = _inputs([5.0, 4.0, 3.0, 2.0, 1.0], [50, 40, 30, 20, 10])

    result = calculate_bdamage_scores(atom_inputs=inputs, window_size=3)

    assert [r.bdamage_atom_index for r in result.atom_results] == [1, 2, 3, 4, 5]
    assert [r.sorted_packing_density_index for r in result.atom_results] == [
        5,
        4,
        3,
        2,
        1,
    ]
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
        cleaned_atoms=atoms,
        selected_atoms=atoms,
        metadata=make_metadata(),
        removed=RemovedAtomCounts(0, 0, 0, 0, 0),
    )


def test_structure_scores_pair_atoms_with_packing_rows() -> None:
    prepared = _prepared(
        make_prepared_atom(source_atom_index=0),
        make_prepared_atom(source_atom_index=1, atom_serial=2),
    )

    def packing(*rows: PackingDensityAtomResult) -> PackingDensityResult:
        return PackingDensityResult(
            atom_results=rows,
            packing_density_threshold=7.0,
            selected_atom_count=len(rows),
            neighbour_atom_count=1,
        )

    good = packing(
        PackingDensityAtomResult(1, 0, 1, 5), PackingDensityAtomResult(2, 1, 2, 9)
    )
    result = calculate_bdamage_scores_for_structure(
        prepared_structure=prepared, packing_density_result=good, window_size=1
    )
    assert bdamage_scores_as_tuple(result) == approx((1.0, 1.0))
    assert [r.packing_density for r in result.atom_results] == [5, 9]

    # A B-factor that is not finite and positive is rejected where it enters
    # the score calculation, naming the source atom.
    for b_factor in (0.0, -1.0, math.inf, math.nan):
        bad = _prepared(
            make_prepared_atom(source_atom_index=0),
            make_prepared_atom(source_atom_index=1, atom_serial=2, b_factor=b_factor),
        )
        with pytest.raises(BDamageScoreError, match="for source atom 1"):
            calculate_bdamage_scores_for_structure(
                prepared_structure=bad, packing_density_result=good, window_size=1
            )


@pytest.mark.parametrize(
    ("atom_count", "fraction", "minimum", "expected"),
    [
        (100, 0.02, 11, 11),  # round(2) -> 3, raised to the minimum
        (1000, 0.02, 11, 21),  # 20 is even, so 21
        (1050, 0.02, 11, 21),  # 21 already odd
        (5000, 0.02, 11, 101),
        (100, 0.02, 10, 11),  # an even minimum is raised to the next odd size
        (100, 0.02, 12, 13),
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


@pytest.mark.parametrize(
    "options",
    [
        BDamageWorkflowOptions(packing_density_threshold=0.0),
        BDamageWorkflowOptions(window_size_fraction=0.0),
        BDamageWorkflowOptions(window_size_fraction=1.0),
        BDamageWorkflowOptions(window_size_fraction=1.5),
        BDamageWorkflowOptions(minimum_window_size=0),
        BDamageWorkflowOptions(translation_range=-1),
    ],
)
def test_workflow_option_validation(options: BDamageWorkflowOptions) -> None:
    with pytest.raises(BDamageWorkflowError):
        validate_workflow_options(options)
    validate_workflow_options(BDamageWorkflowOptions())
