"""Select candidate E-path atoms from atom-level BDamage records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from epath.atoms import EpathAtom


DEFAULT_HISTOGRAM_BINS = 60


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
class ChainPercentileThreshold:
    """Per-chain diagnostics for stratified percentile selection."""

    chain_id: str
    atom_count: int
    selected_count: int
    bdamage_threshold: float
    z_threshold: float


@dataclass(frozen=True, slots=True)
class EpathSelectionResult:
    """Result of selecting candidate E-path atoms."""

    scored_atoms: tuple[EpathScoredAtom, ...]
    selected_atoms: tuple[EpathScoredAtom, ...]
    method: str
    description: str
    selected_percent: float
    z_mode: float
    z_threshold: float
    bdamage_threshold: float | None = None
    boundary_z_score: float | None = None
    chain_thresholds: tuple[ChainPercentileThreshold, ...] = ()


def score_bdamage_atoms(
    atoms: Iterable[EpathAtom],
) -> tuple[EpathScoredAtom, ...]:
    """Return atoms annotated with population BDamage Z-scores."""

    atom_tuple = tuple(atoms)
    if not atom_tuple:
        return ()

    bdamage_values = _bdamage_values(atom_tuple)
    z_scores = _z_scores(bdamage_values)

    return tuple(
        EpathScoredAtom(atom=atom, bdamage_z_score=float(z_score))
        for atom, z_score in zip(atom_tuple, z_scores, strict=True)
    )


def histogram_mode(
    values: Sequence[float] | np.ndarray,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> float:
    """Return the midpoint of the most populated histogram bin."""

    if type(bins) is not int or bins <= 0:
        raise EpathSelectionError(f"bins must be a positive integer, got {bins!r}.")

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise EpathSelectionError("values must be one-dimensional.")

    if array.size == 0:
        return float("nan")

    if not np.all(np.isfinite(array)):
        raise EpathSelectionError("values must contain only finite numbers.")

    counts, edges = np.histogram(array, bins=bins)
    max_index = int(np.argmax(counts))
    return float(0.5 * (edges[max_index] + edges[max_index + 1]))


def select_atoms_by_z(
    atoms: Iterable[EpathAtom],
    *,
    z_cutoff: float,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> EpathSelectionResult:
    """Select atoms whose BDamage Z-score is less than or equal to z_cutoff."""

    z_cutoff = _finite_float(z_cutoff, name="z_cutoff")
    scored_atoms = score_bdamage_atoms(atoms)
    selected_atoms = tuple(
        scored_atom
        for scored_atom in scored_atoms
        if scored_atom.bdamage_z_score <= z_cutoff
    )
    selected_percent = _selected_percent(
        selected_count=len(selected_atoms),
        total_count=len(scored_atoms),
    )

    return EpathSelectionResult(
        scored_atoms=scored_atoms,
        selected_atoms=selected_atoms,
        method="z_cutoff",
        description=(
            f"Z <= {z_cutoff:.3f} "
            f"(selected {selected_percent:.3f}% of atoms)"
        ),
        selected_percent=selected_percent,
        z_mode=_z_mode(scored_atoms, bins=histogram_bins),
        z_threshold=z_cutoff,
    )


def select_atoms_by_percentile(
    atoms: Iterable[EpathAtom],
    *,
    percentile: float,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> EpathSelectionResult:
    """Select the lowest percentile of atoms by BDamage."""

    percentile = _percentile_value(percentile)
    scored_atoms = score_bdamage_atoms(atoms)
    selected_atoms, bdamage_threshold = _select_lowest_percentile(
        scored_atoms,
        percentile=percentile,
    )
    selected_percent = _selected_percent(
        selected_count=len(selected_atoms),
        total_count=len(scored_atoms),
    )
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
        z_mode=_z_mode(scored_atoms, bins=histogram_bins),
        z_threshold=z_threshold,
        bdamage_threshold=bdamage_threshold,
        boundary_z_score=boundary_z_score,
    )


def select_atoms_by_percentile_per_chain(
    atoms: Iterable[EpathAtom],
    *,
    percentile: float,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> EpathSelectionResult:
    """Select the lowest BDamage percentile separately within each chain."""

    percentile = _percentile_value(percentile)
    scored_atoms = score_bdamage_atoms(atoms)
    selected_atoms: list[EpathScoredAtom] = []
    chain_thresholds: list[ChainPercentileThreshold] = []

    for chain_id, chain_atoms in _atoms_by_chain(scored_atoms):
        chain_selected_atoms, bdamage_threshold = _select_lowest_percentile(
            chain_atoms,
            percentile=percentile,
            use_max_threshold_for_full_percentile=True,
        )
        selected_atoms.extend(chain_selected_atoms)
        chain_thresholds.append(
            ChainPercentileThreshold(
                chain_id=chain_id,
                atom_count=len(chain_atoms),
                selected_count=len(chain_selected_atoms),
                bdamage_threshold=bdamage_threshold,
                z_threshold=_z_score_for_bdamage_value(
                    scored_atoms,
                    bdamage_threshold,
                ),
            )
        )

    selected_atom_tuple = tuple(selected_atoms)
    valid_chain_z_thresholds = tuple(
        threshold.z_threshold
        for threshold in chain_thresholds
        if math.isfinite(threshold.z_threshold)
    )
    z_threshold = (
        float(np.median(valid_chain_z_thresholds))
        if valid_chain_z_thresholds
        else float("nan")
    )
    selected_percent = _selected_percent(
        selected_count=len(selected_atom_tuple),
        total_count=len(scored_atoms),
    )

    return EpathSelectionResult(
        scored_atoms=scored_atoms,
        selected_atoms=selected_atom_tuple,
        method="percentile_per_chain",
        description=(
            f"lowest {percentile:.3f}% by BDamage per chain "
            f"(selected {len(selected_atom_tuple)} of {len(scored_atoms)})"
        ),
        selected_percent=selected_percent,
        z_mode=_z_mode(scored_atoms, bins=histogram_bins),
        z_threshold=z_threshold,
        chain_thresholds=tuple(chain_thresholds),
    )


def _select_lowest_percentile(
    scored_atoms: Sequence[EpathScoredAtom],
    *,
    percentile: float,
    use_max_threshold_for_full_percentile: bool = False,
) -> tuple[tuple[EpathScoredAtom, ...], float]:
    atom_count = len(scored_atoms)
    if atom_count == 0 or percentile <= 0.0:
        return (), float("nan")

    if percentile >= 100.0 and not use_max_threshold_for_full_percentile:
        return tuple(scored_atoms), float("nan")

    bdamage_values = np.asarray(
        [scored_atom.bdamage for scored_atom in scored_atoms],
        dtype=float,
    )
    bdamage_threshold = float(
        np.percentile(
            bdamage_values,
            percentile if percentile < 100.0 else 100.0,
        )
    )
    selected_atoms = [
        scored_atom
        for scored_atom in scored_atoms
        if scored_atom.bdamage <= bdamage_threshold
    ]
    target_count = int(round((percentile / 100.0) * atom_count))
    target_count = max(0, min(atom_count, target_count))
    if len(selected_atoms) > target_count:
        selected_atoms.sort(
            key=lambda scored_atom: (
                scored_atom.bdamage,
                scored_atom.bdamage_z_score,
            )
        )
        selected_atoms = selected_atoms[:target_count]

    return tuple(selected_atoms), bdamage_threshold


def _atoms_by_chain(
    scored_atoms: Sequence[EpathScoredAtom],
) -> tuple[tuple[str, tuple[EpathScoredAtom, ...]], ...]:
    grouped_atoms: dict[str, list[EpathScoredAtom]] = {}
    for scored_atom in scored_atoms:
        grouped_atoms.setdefault(scored_atom.chain_id, []).append(scored_atom)

    return tuple(
        (chain_id, tuple(chain_atoms))
        for chain_id, chain_atoms in grouped_atoms.items()
    )


def _bdamage_values(atoms: Sequence[EpathAtom]) -> np.ndarray:
    values = np.asarray([atom.bdamage for atom in atoms], dtype=float)
    if not np.all(np.isfinite(values)):
        raise EpathSelectionError("BDamage values must be finite.")
    return values


def _z_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

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
    if values.size == 0:
        return float("nan")

    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation == 0.0:
        standard_deviation = 1.0

    return float((bdamage - mean) / standard_deviation)


def _z_mode(
    scored_atoms: Sequence[EpathScoredAtom],
    *,
    bins: int,
) -> float:
    return histogram_mode(
        [scored_atom.bdamage_z_score for scored_atom in scored_atoms],
        bins=bins,
    )


def _selected_percent(*, selected_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return 100.0 * selected_count / total_count


def _percentile_value(value: float) -> float:
    percentile = _finite_float(value, name="percentile")
    if percentile < 0.0 or percentile > 100.0:
        raise EpathSelectionError(
            f"percentile must be in the range 0 to 100, got {percentile!r}."
        )
    return percentile


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathSelectionError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathSelectionError(
            f"{name} must be numeric, got {value!r}."
        ) from error

    if not math.isfinite(number):
        raise EpathSelectionError(f"{name} must be finite, got {number!r}.")

    return number
