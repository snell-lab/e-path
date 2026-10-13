"""Calculate per-atom BDamage scores.

BDamage is an atom's B-factor divided by the average B-factor of atoms with
similar packing density. This is the last stage of the BDamage calculation:
`rabdam.packing` supplies the packing densities, and `rabdam.pipeline` chooses
the window size and checks it against the selection before calling in here.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from rabdam.structure.models import PreparedAtom, PreparedStructure


class BDamageScoreError(ValueError):
    """Report invalid input to a BDamage score calculation."""


@dataclass(frozen=True, slots=True)
class BDamageAtomInput:
    """Store one selected atom's B-factor and packing density.

    `bdamage_atom_index` is the atom's position among the selected atoms,
    counted from 1. `source_atom_index` and `atom_serial` are copied from the
    atom's `xtal.reader.AtomRecord`. `packing_density` is the atom's neighbour
    count.
    """

    bdamage_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    b_factor: float
    packing_density: int


@dataclass(frozen=True, slots=True)
class BDamageAtomResult:
    """Store one atom's BDamage score and normalization details.

    `average_b_factor` is the mean B-factor of the atom's packing-density
    window, and `bdamage` is the atom's B-factor divided by it. The other
    fields are those of the atom's `BDamageAtomInput`.
    """

    bdamage_atom_index: int
    source_atom_index: int
    packing_density: int
    average_b_factor: float
    bdamage: float


def calculate_bdamage_scores_for_structure(
    *,
    prepared_structure: PreparedStructure,
    packing_densities: Sequence[int],
    window_size: int,
) -> tuple[BDamageAtomResult, ...]:
    """Calculate BDamage scores for a prepared structure.

    `packing_densities` holds one neighbour count per selected atom, in
    selected-atom order, so the two are paired by position. Scores are
    returned in selected-atom order. A selected atom whose B-factor is not
    finite and positive raises `BDamageScoreError`.
    """
    atom_inputs = tuple(
        _bdamage_input_from_prepared_atom(
            bdamage_atom_index=index,
            atom=atom,
            packing_density=packing_density,
        )
        for index, (atom, packing_density) in enumerate(
            zip(prepared_structure.selected_atoms, packing_densities, strict=True),
            start=1,
        )
    )

    return calculate_bdamage_scores(
        atom_inputs=atom_inputs,
        window_size=window_size,
    )


def _bdamage_input_from_prepared_atom(
    *,
    bdamage_atom_index: int,
    atom: PreparedAtom,
    packing_density: int,
) -> BDamageAtomInput:
    """Combine a prepared atom with its packing density.

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
        packing_density=packing_density,
    )


def calculate_bdamage_scores(
    *,
    atom_inputs: Iterable[BDamageAtomInput],
    window_size: int,
) -> tuple[BDamageAtomResult, ...]:
    """Normalize atom B-factors within packing-density windows.

    The atoms are sorted by packing density, and each B-factor is divided by
    the mean B-factor of the `window_size` atoms centered on it in that order.
    Results are returned in order of `bdamage_atom_index`, which is the input
    order when the inputs are numbered from 1 as
    `calculate_bdamage_scores_for_structure` numbers them.

    `window_size` is the positive odd number of atoms in each window. The
    pipeline entry point has already checked that the window fits the atoms.
    """
    sorted_inputs = sort_atom_inputs_for_bdamage(atom_inputs)
    sorted_average_b_factors = centered_window_average_b_factors(
        sorted_inputs,
        window_size=window_size,
    )

    sorted_results = tuple(
        BDamageAtomResult(
            bdamage_atom_index=atom_input.bdamage_atom_index,
            source_atom_index=atom_input.source_atom_index,
            packing_density=atom_input.packing_density,
            average_b_factor=average_b_factor,
            bdamage=atom_input.b_factor / average_b_factor,
        )
        for atom_input, average_b_factor in zip(
            sorted_inputs, sorted_average_b_factors, strict=True
        )
    )

    return tuple(sorted(sorted_results, key=lambda result: result.bdamage_atom_index))


def sort_atom_inputs_for_bdamage(
    atom_inputs: Iterable[BDamageAtomInput],
) -> tuple[BDamageAtomInput, ...]:
    """Sort atom inputs deterministically by increasing packing density."""
    # The tie-breakers fix which atoms fall in each window when many share a
    # packing density.
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

    # A file can leave an atom's serial blank. Its position in the file is used
    # instead, so every atom still has a number to sort by.
    return atom_input.source_atom_index


def centered_window_average_b_factors(
    sorted_atom_inputs: Iterable[BDamageAtomInput],
    *,
    window_size: int,
) -> tuple[float, ...]:
    """Return the average B-factor of the window around each sorted atom.

    An atom's window is the `window_size` atoms centred on it in the sorted
    order. Near either end of the list a centred window would run past the
    end, so those atoms use the first or last window that fits. Every average
    is therefore taken over exactly `window_size` B-factors. The averages are
    returned in the same order as the atoms.

    The pipeline entry point has already checked `window_size` against the
    atom count, and `_bdamage_input_from_prepared_atom` each B-factor.
    """
    b_factors = np.asarray(
        [atom_input.b_factor for atom_input in sorted_atom_inputs],
        dtype=np.float64,
    )
    atom_count = len(b_factors)

    # With running totals, the sum of any window is one subtraction, so the
    # B-factors are not added up again for every atom.
    prefix_sums = np.empty(atom_count + 1, dtype=np.float64)
    prefix_sums[0] = 0.0
    np.cumsum(b_factors, out=prefix_sums[1:])
    complete_window_averages = (
        prefix_sums[window_size:] - prefix_sums[:-window_size]
    ) / window_size

    # An atom's window starts half a window before it. Near the ends that
    # start is moved so the whole window stays inside the list.
    window_starts = np.clip(
        np.arange(atom_count) - window_size // 2,
        0,
        atom_count - window_size,
    )
    return tuple(complete_window_averages[window_starts].tolist())


def bdamage_scores_as_tuple(
    results: Iterable[BDamageAtomResult],
) -> tuple[float, ...]:
    """Return the BDamage value of each result, in the same order."""
    return tuple(result.bdamage for result in results)
