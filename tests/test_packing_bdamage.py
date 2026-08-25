"""Packing-density counting and the BDamage sliding-window normalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    validate_window_size,
)
from rabdam.crystal.translate import TranslatedAtom
from rabdam.packing import (
    PackingDensityAtomResult,
    PackingDensityError,
    PackingDensityResult,
    calculate_packing_density,
    calculate_packing_density_from_arrays,
    packing_density_counts_as_tuple,
)
from rabdam.structure.models import PreparedStructure, StructurePreparationReport
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

    result = calculate_packing_density(
        selected_atoms=selected,
        neighbour_atoms=neighbours,
        packing_density_threshold=7.0,
    )

    assert packing_density_counts_as_tuple(result) == (4,)
    assert result.atom_results[0].source_atom_index == 0
    assert result.atom_results[0].packing_density_atom_index == 1
    assert result.selected_atom_count == 1 and result.neighbour_atom_count == 7


def test_array_path_matches_object_path() -> None:
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

    from_objects = calculate_packing_density(
        selected_atoms=selected,
        neighbour_atoms=neighbours,
        packing_density_threshold=7.0,
    )
    from_arrays = calculate_packing_density_from_arrays(
        selected_atoms=selected,
        neighbour_coordinates=np.asarray([(n.x, n.y, n.z) for n in neighbours]),
        source_atom_indices=np.asarray(
            [n.source_atom_index for n in neighbours], dtype=np.int64
        ),
        is_identity_symmetry_operation=np.asarray(
            [n.is_identity_symmetry_operation for n in neighbours], dtype=np.bool_
        ),
        translation_offsets=np.zeros((4, 3), dtype=np.int64),
        packing_density_threshold=7.0,
    )

    # Atom 0 sees only the image at 4.0; atom 5 sees both images of atom 9.
    assert (
        packing_density_counts_as_tuple(from_objects)
        == packing_density_counts_as_tuple(from_arrays)
        == (1, 2)
    )


def test_missing_self_copy_and_invalid_input_are_rejected() -> None:
    selected = (make_prepared_atom(source_atom_index=0, x=0.0),)
    with pytest.raises(PackingDensityError, match="central-cell copy"):
        calculate_packing_density(
            selected_atoms=selected,
            neighbour_atoms=(_neighbour(1, 7, 1.0),),
            packing_density_threshold=7.0,
        )
    with pytest.raises(PackingDensityError, match="central-cell copy"):
        calculate_packing_density(
            selected_atoms=selected,
            neighbour_atoms=(_neighbour(1, 0, 9.0),),
            packing_density_threshold=7.0,
        )
    with pytest.raises(PackingDensityError, match="threshold"):
        calculate_packing_density(
            selected_atoms=selected,
            neighbour_atoms=(_neighbour(1, 0, 0.0),),
            packing_density_threshold=0.0,
        )
    with pytest.raises(PackingDensityError, match="empty selected"):
        calculate_packing_density(
            selected_atoms=(),
            neighbour_atoms=(_neighbour(1, 0, 0.0),),
            packing_density_threshold=7.0,
        )
    with pytest.raises(PackingDensityError, match="empty neighbour"):
        calculate_packing_density(
            selected_atoms=selected, neighbour_atoms=(), packing_density_threshold=7.0
        )
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


@pytest.mark.parametrize(
    ("window", "message"),
    [(0, "positive"), (4, "odd"), (7, "larger"), (3.0, "integer")],
)
def test_window_size_validation(window: Any, message: str) -> None:
    with pytest.raises(BDamageScoreError, match=message):
        validate_window_size(window_size=window, atom_count=5)


def test_atom_input_validation() -> None:
    with pytest.raises(BDamageScoreError, match="empty"):
        calculate_bdamage_scores(atom_inputs=(), window_size=1)
    with pytest.raises(BDamageScoreError, match="positive B-factors"):
        calculate_bdamage_scores(
            atom_inputs=_inputs([1.0, 0.0, 1.0], [1, 2, 3]), window_size=1
        )
    with pytest.raises(BDamageScoreError, match="non-negative"):
        calculate_bdamage_scores(
            atom_inputs=_inputs([1.0, 1.0], [1, -2]), window_size=1
        )
    duplicate = _inputs([1.0, 1.0], [1, 2])
    duplicate = (duplicate[0], BDamageAtomInput(1, 1, 2, 1.0, 2))
    with pytest.raises(BDamageScoreError, match="Duplicate"):
        calculate_bdamage_scores(atom_inputs=duplicate, window_size=1)
    gap = (BDamageAtomInput(1, 0, 1, 1.0, 1), BDamageAtomInput(3, 1, 2, 1.0, 1))
    with pytest.raises(BDamageScoreError, match="contiguous"):
        calculate_bdamage_scores(atom_inputs=gap, window_size=1)


def test_structure_scores_require_aligned_packing_results() -> None:
    atoms = (
        make_prepared_atom(source_atom_index=0),
        make_prepared_atom(source_atom_index=1, atom_serial=2),
    )
    prepared = PreparedStructure(
        cleaned_atoms=atoms,
        selected_atoms=atoms,
        metadata=make_metadata(),
        report=StructurePreparationReport(2, 2, 2, 0, 0, 0, 0, 0),
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

    with pytest.raises(BDamageScoreError, match="count does not match"):
        calculate_bdamage_scores_for_structure(
            prepared_structure=prepared,
            packing_density_result=packing(PackingDensityAtomResult(1, 0, 1, 5)),
            window_size=1,
        )
    with pytest.raises(BDamageScoreError, match="different source atoms"):
        calculate_bdamage_scores_for_structure(
            prepared_structure=prepared,
            packing_density_result=packing(
                PackingDensityAtomResult(1, 0, 1, 5),
                PackingDensityAtomResult(2, 7, 2, 9),
            ),
            window_size=1,
        )
    with pytest.raises(BDamageScoreError, match="different atom serials"):
        calculate_bdamage_scores_for_structure(
            prepared_structure=prepared,
            packing_density_result=packing(
                PackingDensityAtomResult(1, 0, 1, 5),
                PackingDensityAtomResult(2, 1, 99, 9),
            ),
            window_size=1,
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
    "kwargs",
    [
        {"atom_count": 0},
        {"atom_count": 10.0},
        {"window_size_fraction": 0.0},
        {"window_size_fraction": 1.0},
        {"minimum_window_size": 0},
    ],
)
def test_window_size_from_fraction_rejects_bad_input(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "atom_count": 100,
        "window_size_fraction": 0.02,
        "minimum_window_size": 11,
        **kwargs,
    }
    with pytest.raises(BDamageWorkflowError):
        bdamage_window_size_from_fraction(**arguments)


@pytest.mark.parametrize(
    "options",
    [
        BDamageWorkflowOptions(packing_density_threshold=0.0),
        BDamageWorkflowOptions(window_size_fraction=1.5),
        BDamageWorkflowOptions(minimum_window_size=0),
        BDamageWorkflowOptions(translation_range=-1),
        BDamageWorkflowOptions(materialize_translated_block=1),  # type: ignore[arg-type]
    ],
)
def test_workflow_option_validation(options: BDamageWorkflowOptions) -> None:
    with pytest.raises(BDamageWorkflowError):
        validate_workflow_options(options)
    validate_workflow_options(BDamageWorkflowOptions())
