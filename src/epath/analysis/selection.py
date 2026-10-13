"""Select candidate E-path atoms from atom-level BDamage records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from epath.analysis.atoms import EpathAtom


class EpathSelectionError(ValueError):
    """Raised when E-path atom selection cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathScoredAtom:
    """One E-path atom with its BDamage Z-score."""

    atom: EpathAtom
    bdamage_z_score: float

    @property
    def bdamage(self) -> float:
        """Return the atom's BDamage value."""
        return self.atom.bdamage

    @property
    def chain_id(self) -> str:
        """Return the atom's chain identifier."""
        return self.atom.chain_id


@dataclass(frozen=True, slots=True)
class EpathSelectionResult:
    """Result of selecting candidate E-path atoms."""

    scored_atoms: tuple[EpathScoredAtom, ...]
    selected_atoms: tuple[EpathScoredAtom, ...]
    method: str
    description: str
    selected_percent: float
    bdamage_mean: float
    bdamage_standard_deviation: float
    z_threshold: float
    bdamage_threshold: float | None = None
    boundary_z_score: float | None = None
    selected_bdamage_mean: float = float("nan")
    selected_bdamage_standard_deviation: float = float("nan")


def score_bdamage_atoms(
    atoms: Iterable[EpathAtom],
) -> tuple[EpathScoredAtom, ...]:
    """Return atoms annotated with population BDamage Z-scores."""
    atom_tuple = tuple(atoms)
    bdamage_values = _bdamage_values(atom_tuple)
    z_scores = _z_scores(bdamage_values)

    return tuple(
        EpathScoredAtom(atom=atom, bdamage_z_score=float(z_score))
        for atom, z_score in zip(atom_tuple, z_scores, strict=True)
    )


def select_atoms_by_percentile(
    atoms: Iterable[EpathAtom],
    *,
    percentile: float,
) -> EpathSelectionResult:
    """Select the lowest percentile of atoms by BDamage."""
    # NaN fails every comparison, so the range test rejects it too.
    if not 0.0 <= percentile <= 100.0:
        raise EpathSelectionError(
            f"percentile must be in the range 0 to 100, got {percentile!r}."
        )

    scored_atoms = score_bdamage_atoms(atoms)
    selected_atoms, bdamage_threshold = _select_lowest_percentile(
        scored_atoms,
        percentile=percentile,
    )
    selected_percent = 100.0 * len(selected_atoms) / len(scored_atoms)
    boundary_z_score = max(
        (atom.bdamage_z_score for atom in selected_atoms),
        default=float("nan"),
    )
    z_threshold = _z_score_for_bdamage_value(
        scored_atoms,
        bdamage_threshold,
    )

    return EpathSelectionResult(
        scored_atoms=scored_atoms,
        selected_atoms=selected_atoms,
        method="percentile",
        description=(
            f"lowest {percentile:.3f}% by BDamage "
            f"(bdamage_threshold={bdamage_threshold:.6f}; "
            f"selected {len(selected_atoms)} of {len(scored_atoms)})"
        ),
        selected_percent=selected_percent,
        bdamage_mean=_mean(scored_atoms),
        bdamage_standard_deviation=_standard_deviation(scored_atoms),
        z_threshold=z_threshold,
        bdamage_threshold=bdamage_threshold,
        boundary_z_score=boundary_z_score,
        selected_bdamage_mean=_mean(selected_atoms),
        selected_bdamage_standard_deviation=_standard_deviation(selected_atoms),
    )


def _select_lowest_percentile(
    scored_atoms: Sequence[EpathScoredAtom],
    *,
    percentile: float,
) -> tuple[tuple[EpathScoredAtom, ...], float]:
    atom_count = len(scored_atoms)
    if percentile <= 0.0:
        return (), float("nan")

    if percentile >= 100.0:
        return tuple(scored_atoms), float("nan")

    bdamage_values = np.asarray(
        [scored_atom.bdamage for scored_atom in scored_atoms],
        dtype=float,
    )
    bdamage_threshold = float(np.percentile(bdamage_values, percentile))
    selected_atoms = [
        scored_atom
        for scored_atom in scored_atoms
        if scored_atom.bdamage <= bdamage_threshold
    ]
    target_count = round((percentile / 100.0) * atom_count)
    if len(selected_atoms) > target_count:
        selected_atoms.sort(
            key=lambda scored_atom: (
                scored_atom.bdamage,
                scored_atom.bdamage_z_score,
            )
        )
        selected_atoms = selected_atoms[:target_count]

    return tuple(selected_atoms), bdamage_threshold


def _bdamage_values(atoms: Sequence[EpathAtom]) -> NDArray[np.float64]:
    values = np.asarray([atom.bdamage for atom in atoms], dtype=float)
    if not np.all(np.isfinite(values)):
        raise EpathSelectionError("BDamage values must be finite.")
    return values


def _z_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation == 0.0:
        return np.zeros_like(values, dtype=float)

    return (values - mean) / standard_deviation


def _z_score_for_bdamage_value(
    scored_atoms: Sequence[EpathScoredAtom],
    bdamage: float,
) -> float:
    if not math.isfinite(bdamage):
        return float("nan")

    values = np.asarray(
        [scored_atom.bdamage for scored_atom in scored_atoms],
        dtype=float,
    )
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation == 0.0:
        standard_deviation = 1.0

    return float((bdamage - mean) / standard_deviation)


def _mean(scored_atoms: Sequence[EpathScoredAtom]) -> float:
    if not scored_atoms:
        return float("nan")

    return float(np.mean([scored_atom.bdamage for scored_atom in scored_atoms]))


def _standard_deviation(scored_atoms: Sequence[EpathScoredAtom]) -> float:
    if not scored_atoms:
        return float("nan")

    return float(np.std([scored_atom.bdamage for scored_atom in scored_atoms]))
