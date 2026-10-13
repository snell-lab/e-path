"""Select the atoms with the lowest BDamage as E-path candidates.

`rabdam.pipeline` supplies the scored atoms and their BDamage results, which
are used as they are. Each atom is given a Z-score against all of them here,
and the lowest percentile by BDamage is selected. `epath.analysis.clustering` then groups the selected
atoms, and `epath.outputs` reports the statistics kept on the result.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from rabdam.bdamage import BDamageAtomResult
from rabdam.structure.models import PreparedAtom
from xtal.reader import AtomRecord


class EpathSelectionError(ValueError):
    """Raised when E-path atom selection cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathScoredAtom:
    """One scored atom with its BDamage Z-score.

    `record` is the atom as `xtal.reader` read it from the structure file, and
    `bdamage_result` holds the packing density, the window-averaged B-factor
    and the BDamage that `rabdam.pipeline` calculated for it.
    """

    record: AtomRecord
    bdamage_result: BDamageAtomResult
    bdamage_z_score: float


@dataclass(frozen=True, slots=True)
class EpathSelectionResult:
    """Result of selecting candidate E-path atoms.

    `scored_atoms` holds every atom, in input order, and `selected_atoms` the
    ones selected. `selected_percent` is the share that was selected, which
    can differ from the percentile asked for. `bdamage_mean` and
    `bdamage_standard_deviation` describe all the atoms, and the two
    `selected_` values the selected ones.

    `bdamage_threshold` is the BDamage at the percentile asked for and
    `z_threshold` the same value as a Z-score. `boundary_z_score` is the
    highest Z-score among the selected atoms. It and the two `selected_`
    statistics are NaN when nothing is selected.
    """

    scored_atoms: tuple[EpathScoredAtom, ...]
    selected_atoms: tuple[EpathScoredAtom, ...]
    description: str
    selected_percent: float
    bdamage_mean: float
    bdamage_standard_deviation: float
    z_threshold: float
    bdamage_threshold: float
    boundary_z_score: float
    selected_bdamage_mean: float
    selected_bdamage_standard_deviation: float


def score_bdamage_atoms(
    atoms: Iterable[PreparedAtom],
    bdamage_results: Iterable[BDamageAtomResult],
) -> tuple[EpathScoredAtom, ...]:
    """Return the atoms, in input order, each with its BDamage Z-score.

    `atoms` are the atoms that were scored and `bdamage_results` their
    results, paired by position, which is how `rabdam.pipeline` returns the
    two. A different number of each raises `ValueError`.

    The Z-score is the atom's BDamage minus the mean of all the atoms given,
    divided by their population standard deviation. When every BDamage is the
    same, every Z-score is zero. A BDamage that is not finite raises
    `EpathSelectionError`.
    """
    results = tuple(bdamage_results)
    bdamage_values = np.asarray([result.bdamage for result in results], dtype=float)
    if not np.all(np.isfinite(bdamage_values)):
        raise EpathSelectionError("BDamage values must be finite.")

    standard_deviation = np.std(bdamage_values)
    if standard_deviation == 0.0:
        z_scores = np.zeros_like(bdamage_values)
    else:
        z_scores = (bdamage_values - np.mean(bdamage_values)) / standard_deviation

    return tuple(
        EpathScoredAtom(
            record=atom.record,
            bdamage_result=result,
            bdamage_z_score=float(z_score),
        )
        for atom, result, z_score in zip(atoms, results, z_scores, strict=True)
    )


def select_atoms_by_percentile(
    atoms: Iterable[PreparedAtom],
    bdamage_results: Iterable[BDamageAtomResult],
    *,
    percentile: float,
) -> EpathSelectionResult:
    """Select the atoms in the lowest `percentile` percent by BDamage.

    `atoms` and `bdamage_results` are the scored atoms and their results,
    paired by position as `score_bdamage_atoms` describes.

    The threshold is the BDamage at that percentile of all the atoms,
    interpolated between neighbouring values as `numpy.percentile` does, and
    an atom at exactly the threshold is a candidate. The candidates are then
    cut to that percentage of the atom count, rounded to a whole number of
    atoms, by dropping the ones of highest BDamage. Of candidates with the
    same BDamage, the later ones in input order are dropped first. A count
    that falls exactly halfway between two whole numbers is rounded to the
    even one, as `round` does, so a small percentile of a few atoms can select
    nothing. The selected atoms are returned in input order.

    A percentile of 0 selects nothing, with the lowest BDamage as its
    threshold, and 100 selects every atom, with the highest.

    `percentile` must be from 0 to 100, both included, and every BDamage must
    be finite, or `EpathSelectionError` is raised. There must be at least one
    atom, which the BDamage pipeline guarantees.
    """
    if not 0.0 <= percentile <= 100.0:
        raise EpathSelectionError(
            f"percentile must be in the range 0 to 100, got {percentile!r}."
        )

    scored_atoms = score_bdamage_atoms(atoms, bdamage_results)
    selected_atoms, bdamage_threshold = _select_lowest_percentile(
        scored_atoms,
        percentile=percentile,
    )
    bdamage_mean = _mean(scored_atoms)
    bdamage_standard_deviation = _standard_deviation(scored_atoms)

    return EpathSelectionResult(
        scored_atoms=scored_atoms,
        selected_atoms=selected_atoms,
        description=(
            f"lowest {percentile:.3f}% by BDamage "
            f"(bdamage_threshold={bdamage_threshold:.6f}; "
            f"selected {len(selected_atoms)} of {len(scored_atoms)})"
        ),
        selected_percent=100.0 * len(selected_atoms) / len(scored_atoms),
        bdamage_mean=bdamage_mean,
        bdamage_standard_deviation=bdamage_standard_deviation,
        # Identical BDamage values have no spread to divide by. The threshold
        # is then the mean, so its Z-score is zero, as every atom's is.
        z_threshold=(
            (bdamage_threshold - bdamage_mean) / (bdamage_standard_deviation or 1.0)
        ),
        bdamage_threshold=bdamage_threshold,
        boundary_z_score=max(
            (atom.bdamage_z_score for atom in selected_atoms),
            default=float("nan"),
        ),
        selected_bdamage_mean=_mean(selected_atoms),
        selected_bdamage_standard_deviation=_standard_deviation(selected_atoms),
    )


def _select_lowest_percentile(
    scored_atoms: tuple[EpathScoredAtom, ...],
    *,
    percentile: float,
) -> tuple[tuple[EpathScoredAtom, ...], float]:
    """Return the selected atoms, in input order, and the BDamage threshold."""
    bdamage_threshold = float(
        np.percentile(
            [atom.bdamage_result.bdamage for atom in scored_atoms], percentile
        )
    )
    candidates = [
        index
        for index, atom in enumerate(scored_atoms)
        if atom.bdamage_result.bdamage <= bdamage_threshold
    ]
    # The threshold can admit more atoms than the percentage of the atom
    # count, when it falls just above an atom's BDamage or several atoms share
    # a value at it. The candidates of highest BDamage are dropped to make up
    # the difference. The sort is stable, so equal values keep input order.
    target_count = round((percentile / 100.0) * len(scored_atoms))
    candidates.sort(key=lambda index: scored_atoms[index].bdamage_result.bdamage)
    selected_indices = sorted(candidates[:target_count])

    return tuple(scored_atoms[index] for index in selected_indices), bdamage_threshold


def _mean(scored_atoms: Sequence[EpathScoredAtom]) -> float:
    if not scored_atoms:
        return float("nan")

    return float(np.mean([atom.bdamage_result.bdamage for atom in scored_atoms]))


def _standard_deviation(scored_atoms: Sequence[EpathScoredAtom]) -> float:
    if not scored_atoms:
        return float("nan")

    return float(np.std([atom.bdamage_result.bdamage for atom in scored_atoms]))
