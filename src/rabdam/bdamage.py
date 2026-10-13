"""Calculate per-atom BDamage scores.

BDamage is an atom's B-factor divided by the average B-factor of atoms with
similar packing density.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rabdam.packing import PackingDensityAtomResult, PackingDensityResult
from rabdam.structure.models import PreparedAtom, PreparedStructure


class BDamageScoreError(ValueError):
    """Report invalid input to a BDamage score calculation."""


@dataclass(frozen=True, slots=True)
class BDamageAtomInput:
    """Store one selected atom's B-factor and packing density."""

    bdamage_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    b_factor: float
    packing_density: int


@dataclass(frozen=True, slots=True)
class BDamageAtomResult:
    """Store one atom's BDamage score and normalization details."""

    bdamage_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    b_factor: float
    packing_density: int
    average_b_factor: float
    bdamage: float
    sorted_packing_density_index: int


@dataclass(frozen=True, slots=True)
class BDamageScoreResult:
    """Store per-atom BDamage results and calculation summary values."""

    atom_results: tuple[BDamageAtomResult, ...]
    window_size: int
    selected_atom_count: int


def calculate_bdamage_scores_for_structure(
    *,
    prepared_structure: PreparedStructure,
    packing_density_result: PackingDensityResult,
    window_size: int,
) -> BDamageScoreResult:
    """Calculate BDamage scores for a prepared structure.

    Packing-density rows are built one per selected atom in selected-atom
    order, so they are paired with the selected atoms by position. Scores are
    returned in selected-atom order.
    """
    atom_inputs = tuple(
        bdamage_input_from_prepared_atom(
            bdamage_atom_index=index,
            atom=atom,
            packing_density_atom_result=packing_result,
        )
        for index, (atom, packing_result) in enumerate(
            zip(
                prepared_structure.selected_atoms,
                packing_density_result.atom_results,
                strict=True,
            ),
            start=1,
        )
    )

    return calculate_bdamage_scores(
        atom_inputs=atom_inputs,
        window_size=window_size,
    )


def bdamage_input_from_prepared_atom(
    *,
    bdamage_atom_index: int,
    atom: PreparedAtom,
    packing_density_atom_result: PackingDensityAtomResult,
) -> BDamageAtomInput:
    """Combine an aligned prepared atom and packing-density result.

    This is where B-factors enter the score calculation, so a B-factor that is
    not finite and positive raises `BDamageScoreError` here.
    """
    b_factor = atom.record.b_factor
    if not math.isfinite(b_factor) or b_factor <= 0:
        raise BDamageScoreError(
            "BDamage requires finite positive B-factors, "
            f"got {b_factor!r} for source atom "
            f"{atom.record.source_atom_index!r}."
        )

    return BDamageAtomInput(
        bdamage_atom_index=bdamage_atom_index,
        source_atom_index=atom.record.source_atom_index,
        atom_serial=atom.record.atom_serial,
        b_factor=b_factor,
        packing_density=packing_density_atom_result.neighbour_count,
    )


def calculate_bdamage_scores(
    *,
    atom_inputs: Iterable[BDamageAtomInput],
    window_size: int,
) -> BDamageScoreResult:
    """Normalize atom B-factors within packing-density windows.

    Atoms are sorted by packing density before centered-window averages are
    computed. Results are restored to their original one-based BDamage order.
    The workflow entry point has already checked that there are atoms and that
    the window fits them.

    Args:
        atom_inputs: Selected atoms with B-factors and neighbour counts.
        window_size: Positive odd number of atoms in each normalization window.

    Returns:
        Per-atom BDamage values and intermediate normalization values in input
        order.
    """
    atom_input_tuple = tuple(atom_inputs)
    sorted_inputs = sort_atom_inputs_for_bdamage(atom_input_tuple)
    sorted_average_b_factors = centered_window_average_b_factors(
        sorted_inputs,
        window_size=window_size,
    )

    sorted_results = tuple(
        BDamageAtomResult(
            bdamage_atom_index=atom_input.bdamage_atom_index,
            source_atom_index=atom_input.source_atom_index,
            atom_serial=atom_input.atom_serial,
            b_factor=atom_input.b_factor,
            packing_density=atom_input.packing_density,
            average_b_factor=average_b_factor,
            bdamage=atom_input.b_factor / average_b_factor,
            sorted_packing_density_index=sorted_index,
        )
        for sorted_index, (atom_input, average_b_factor) in enumerate(
            zip(sorted_inputs, sorted_average_b_factors, strict=True), start=1
        )
    )

    results_by_original_index = {
        result.bdamage_atom_index: result for result in sorted_results
    }

    return BDamageScoreResult(
        atom_results=tuple(
            results_by_original_index[index]
            for index in range(1, len(atom_input_tuple) + 1)
        ),
        window_size=window_size,
        selected_atom_count=len(atom_input_tuple),
    )


def sort_atom_inputs_for_bdamage(
    atom_inputs: Iterable[BDamageAtomInput],
) -> tuple[BDamageAtomInput, ...]:
    """Sort atom inputs deterministically by increasing packing density."""
    # Deterministic tie-breakers preserve RABDAM's window membership when many
    # atoms share the same packing density.
    return tuple(
        sorted(
            atom_inputs,
            key=lambda atom_input: (
                atom_input.packing_density,
                _atom_serial_sort_value(atom_input),
                atom_input.source_atom_index,
            ),
        )
    )


def _atom_serial_sort_value(atom_input: BDamageAtomInput) -> int:
    if atom_input.atom_serial is not None:
        return atom_input.atom_serial

    # Serials are almost always present; falling back to the reader index keeps
    # the sort deterministic when one is missing.
    return atom_input.source_atom_index


def centered_window_average_b_factors(
    sorted_atom_inputs: Iterable[BDamageAtomInput],
    *,
    window_size: int,
) -> tuple[float, ...]:
    """Compute centered-window B-factor averages for sorted atoms.

    The first and last complete windows are reused at the distribution edges,
    so every atom is normalized by exactly `window_size` B-factors.

    The workflow entry point has already checked `window_size` against the
    atom count, and `bdamage_input_from_prepared_atom` each B-factor.

    Returns:
        Average B-factors aligned with the sorted input atoms.
    """
    sorted_input_tuple = tuple(sorted_atom_inputs)
    b_factors = np.asarray(
        [atom_input.b_factor for atom_input in sorted_input_tuple],
        dtype=np.float64,
    )
    half_window = math.floor(window_size / 2)

    # Cumulative sums make each window average an O(1) subtraction instead of
    # re-summing window_size B-factors per atom.
    prefix_sums = np.empty(len(b_factors) + 1, dtype=np.float64)
    prefix_sums[0] = 0.0
    np.cumsum(b_factors, out=prefix_sums[1:])

    first_window_average = _window_average_from_prefix_sums(
        prefix_sums,
        start=0,
        window_size=window_size,
    )
    last_window_average = _window_average_from_prefix_sums(
        prefix_sums,
        start=len(b_factors) - window_size,
        window_size=window_size,
    )

    averages: list[float] = []
    for index in range(len(b_factors)):
        # Reusing a complete edge window avoids changing the normalization
        # merely because an atom is near an end of the sorted distribution.
        if index <= half_window - 1:
            averages.append(first_window_average)
            continue

        if index >= len(b_factors) - half_window:
            averages.append(last_window_average)
            continue

        start = index - half_window
        averages.append(
            _window_average_from_prefix_sums(
                prefix_sums,
                start=start,
                window_size=window_size,
            )
        )

    return tuple(averages)


def _window_average_from_prefix_sums(
    prefix_sums: NDArray[np.float64],
    *,
    start: int,
    window_size: int,
) -> float:
    return float((prefix_sums[start + window_size] - prefix_sums[start]) / window_size)


def bdamage_scores_as_tuple(
    result: BDamageScoreResult,
) -> tuple[float, ...]:
    """Return BDamage values in selected-atom order."""
    return tuple(atom_result.bdamage for atom_result in result.atom_results)
