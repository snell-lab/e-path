"""Calculate per-atom BDamage scores.

BDamage is an atom's B-factor divided by the average B-factor of atoms with
similar packing density.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from rabdam.packing.density import PackingDensityAtomResult, PackingDensityResult
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

    Packing-density rows must align one-to-one with selected atoms by both
    source index and atom serial.

    Returns:
        Per-atom scores in selected-atom order and the effective window size.

    Raises:
        BDamageScoreError: If the selected atoms and packing-density results do
            not align or any scoring input is invalid.
    """
    selected_atoms = tuple(prepared_structure.selected_atoms)
    packing_results = tuple(packing_density_result.atom_results)

    if len(selected_atoms) != len(packing_results):
        raise BDamageScoreError(
            "Selected atom count does not match packing-density result count: "
            f"{len(selected_atoms)} selected atoms, "
            f"{len(packing_results)} packing-density results."
        )

    atom_inputs = tuple(
        bdamage_input_from_prepared_atom(
            bdamage_atom_index=index,
            atom=atom,
            packing_density_atom_result=packing_result,
        )
        for index, (atom, packing_result) in enumerate(
            zip(selected_atoms, packing_results), start=1
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

    Raises:
        BDamageScoreError: If atom identities differ or the B-factor is not
            finite and positive.
    """
    if atom.record.source_atom_index != packing_density_atom_result.source_atom_index:
        raise BDamageScoreError(
            "Selected atom and packing-density result refer to different source "
            "atoms: "
            f"{atom.record.source_atom_index!r} != "
            f"{packing_density_atom_result.source_atom_index!r}."
        )

    if atom.record.atom_serial != packing_density_atom_result.atom_serial:
        raise BDamageScoreError(
            "Selected atom and packing-density result refer to different atom "
            "serials: "
            f"{atom.record.atom_serial!r} != "
            f"{packing_density_atom_result.atom_serial!r}."
        )

    b_factor = float(atom.record.b_factor)
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

    Args:
        atom_inputs: Selected atoms with B-factors and neighbour counts.
        window_size: Positive odd number of atoms in each normalization window.

    Returns:
        Per-atom BDamage values and intermediate normalization values in input
        order.

    Raises:
        BDamageScoreError: If inputs are empty, invalid, duplicated, noncontiguous,
            or incompatible with the window size.
    """
    atom_input_tuple = tuple(atom_inputs)
    if not atom_input_tuple:
        raise BDamageScoreError("Cannot calculate BDamage for an empty atom list.")

    validate_window_size(window_size=window_size, atom_count=len(atom_input_tuple))
    validate_atom_inputs(atom_input_tuple)

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
            zip(sorted_inputs, sorted_average_b_factors), start=1
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


def validate_window_size(*, window_size: int, atom_count: int) -> None:
    """Validate a centered BDamage normalization-window size.

    Raises:
        BDamageScoreError: If the size is not a positive odd integer no larger
            than the atom count.
    """
    if type(window_size) is not int:
        raise BDamageScoreError(f"window_size must be an integer, got {window_size!r}.")

    if window_size <= 0:
        raise BDamageScoreError(f"window_size must be positive, got {window_size!r}.")

    if window_size % 2 == 0:
        raise BDamageScoreError(f"window_size must be odd, got {window_size!r}.")

    if window_size > atom_count:
        raise BDamageScoreError(
            "window_size cannot be larger than the number of selected atoms: "
            f"window_size={window_size!r}, atom_count={atom_count!r}."
        )


def validate_atom_inputs(atom_inputs: tuple[BDamageAtomInput, ...]) -> None:
    """Validate atom identities, B-factors, and packing-density values.

    Raises:
        BDamageScoreError: If one-based indices are invalid, B-factors are not
            finite and positive, or neighbour counts are negative.
    """
    seen_indices: set[int] = set()

    for atom_input in atom_inputs:
        if atom_input.bdamage_atom_index <= 0:
            raise BDamageScoreError(
                "bdamage_atom_index must be one-based and positive, "
                f"got {atom_input.bdamage_atom_index!r}."
            )

        if atom_input.bdamage_atom_index in seen_indices:
            raise BDamageScoreError(
                "Duplicate bdamage_atom_index found: "
                f"{atom_input.bdamage_atom_index!r}."
            )
        seen_indices.add(atom_input.bdamage_atom_index)

        if not math.isfinite(atom_input.b_factor) or atom_input.b_factor <= 0:
            raise BDamageScoreError(
                "BDamage requires finite positive B-factors, "
                f"got {atom_input.b_factor!r}."
            )

        if atom_input.packing_density < 0:
            raise BDamageScoreError(
                "packing_density must be non-negative, "
                f"got {atom_input.packing_density!r}."
            )

    expected_indices = set(range(1, len(atom_inputs) + 1))
    if seen_indices != expected_indices:
        raise BDamageScoreError(
            "bdamage_atom_index values must be contiguous one-based positions "
            f"from 1 to {len(atom_inputs)}."
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

    Returns:
        Average B-factors aligned with the sorted input atoms.
    """
    sorted_input_tuple = tuple(sorted_atom_inputs)
    validate_window_size(window_size=window_size, atom_count=len(sorted_input_tuple))

    b_factors = np.asarray(
        [atom_input.b_factor for atom_input in sorted_input_tuple],
        dtype=np.float64,
    )
    half_window = math.floor(window_size / 2)

    if not np.all(np.isfinite(b_factors)):
        raise BDamageScoreError("Cannot calculate the mean of non-finite values.")

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
    prefix_sums: np.ndarray,
    *,
    start: int,
    window_size: int,
) -> float:
    average = float(
        (prefix_sums[start + window_size] - prefix_sums[start]) / window_size
    )
    if average <= 0:
        raise BDamageScoreError(f"Average B-factor must be positive, got {average!r}.")

    return average


def bdamage_scores_as_tuple(
    result: BDamageScoreResult,
) -> tuple[float, ...]:
    """Return BDamage values in selected-atom order."""
    return tuple(atom_result.bdamage for atom_result in result.atom_results)
