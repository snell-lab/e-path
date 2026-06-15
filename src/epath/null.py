"""Monte Carlo null model for E-path atom clustering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import operator

import numpy as np

from epath.clustering import (
    DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    EpathCluster,
    cluster_selected_atoms,
)
from epath.selection import EpathScoredAtom


DEFAULT_NULL_ITERATIONS = 200
DEFAULT_NULL_SEED = 1


class EpathNullModelError(ValueError):
    """Raised when an E-path null model cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathNullModelResult:
    """Monte Carlo samples for random atom clustering."""

    iterations: int
    sample_size: int
    population_size: int
    distance_cutoff: float
    seed: int
    largest_cluster_sizes: tuple[int, ...]
    component_counts: tuple[int, ...]
    mst_total_lengths: tuple[float, ...]

    @property
    def largest_mean(self) -> float:
        """Return the mean random largest-cluster size."""

        return _mean(self.largest_cluster_sizes)

    @property
    def largest_std(self) -> float:
        """Return the standard deviation of random largest-cluster sizes."""

        return _std(self.largest_cluster_sizes)

    @property
    def ncomp_mean(self) -> float:
        """Return the mean random component count."""

        return _mean(self.component_counts)

    @property
    def ncomp_std(self) -> float:
        """Return the standard deviation of random component counts."""

        return _std(self.component_counts)

    @property
    def mst_mean(self) -> float:
        """Return the mean random MST total length."""

        return _mean(self.mst_total_lengths)

    @property
    def mst_std(self) -> float:
        """Return the standard deviation of random MST total lengths."""

        return _std(self.mst_total_lengths)

    @property
    def largest_vec(self) -> tuple[int, ...]:
        """Return largest-cluster samples."""

        return self.largest_cluster_sizes

    @property
    def ncomp_vec(self) -> tuple[int, ...]:
        """Return component-count samples."""

        return self.component_counts

    @property
    def mst_vec(self) -> tuple[float, ...]:
        """Return MST total-length samples."""

        return self.mst_total_lengths

@dataclass(frozen=True, slots=True)
class EpathNullModelComparison:
    """Observed cluster metrics compared with a Monte Carlo null model."""

    observed_largest_cluster_size: int
    observed_mst_total_length: float
    largest_z_score: float
    largest_p_value_greater: float
    mst_z_score: float
    mst_p_value_less: float


def null_model_metrics(
    atoms: Iterable[EpathScoredAtom],
    *,
    sample_size: int,
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    iterations: int = DEFAULT_NULL_ITERATIONS,
    seed: int = DEFAULT_NULL_SEED,
) -> EpathNullModelResult:
    """Run the E-path Monte Carlo null model."""

    return run_null_model(
        atoms,
        sample_size=sample_size,
        distance_cutoff=distance_cutoff,
        iterations=iterations,
        seed=seed,
    )


def run_null_model(
    atoms: Iterable[EpathScoredAtom],
    *,
    sample_size: int,
    distance_cutoff: float = DEFAULT_CLUSTER_DISTANCE_ANGSTROM,
    iterations: int = DEFAULT_NULL_ITERATIONS,
    seed: int = DEFAULT_NULL_SEED,
) -> EpathNullModelResult:
    """
    Randomly sample atoms and measure clustering under the same distance cutoff.

    Each iteration draws sample_size atoms without replacement, clusters them,
    and records the largest component size, number of components, and MST total
    length of the largest component.
    """

    atom_tuple = tuple(atoms)
    sample_size = _non_negative_int(sample_size, name="sample_size")
    iterations = _non_negative_int(iterations, name="iterations")
    seed = _int_value(seed, name="seed")
    distance_cutoff = _positive_finite_float(
        distance_cutoff,
        name="distance_cutoff",
    )

    if sample_size <= 1 or not atom_tuple or iterations == 0:
        return EpathNullModelResult(
            iterations=0,
            sample_size=sample_size,
            population_size=len(atom_tuple),
            distance_cutoff=distance_cutoff,
            seed=seed,
            largest_cluster_sizes=(),
            component_counts=(),
            mst_total_lengths=(),
        )

    population_size = len(atom_tuple)
    effective_sample_size = min(sample_size, population_size)
    random_generator = np.random.default_rng(seed)

    largest_cluster_sizes: list[int] = []
    component_counts: list[int] = []
    mst_total_lengths: list[float] = []
    for _ in range(iterations):
        sampled_indices = tuple(
            int(index)
            for index in random_generator.choice(
                population_size,
                size=effective_sample_size,
                replace=False,
            )
        )
        largest_size, component_count, mst_total_length = _metrics_for_sample(
            atom_tuple,
            sampled_indices,
            distance_cutoff=distance_cutoff,
        )
        largest_cluster_sizes.append(largest_size)
        component_counts.append(component_count)
        mst_total_lengths.append(mst_total_length)

    return EpathNullModelResult(
        iterations=iterations,
        sample_size=effective_sample_size,
        population_size=population_size,
        distance_cutoff=distance_cutoff,
        seed=seed,
        largest_cluster_sizes=tuple(largest_cluster_sizes),
        component_counts=tuple(component_counts),
        mst_total_lengths=tuple(mst_total_lengths),
    )


def compare_cluster_to_null(
    cluster: EpathCluster,
    null_model: EpathNullModelResult,
) -> EpathNullModelComparison:
    """Compare one observed cluster against a null model."""

    return compare_observed_to_null(
        observed_largest_cluster_size=cluster.atom_count,
        observed_mst_total_length=cluster.metrics.mst_total_length,
        null_model=null_model,
    )


def compare_observed_to_null(
    *,
    observed_largest_cluster_size: int,
    observed_mst_total_length: float,
    null_model: EpathNullModelResult,
) -> EpathNullModelComparison:
    """Return empirical z-scores and p-values for observed cluster metrics."""

    observed_largest_cluster_size = _non_negative_int(
        observed_largest_cluster_size,
        name="observed_largest_cluster_size",
    )
    observed_mst_total_length = _non_negative_finite_float(
        observed_mst_total_length,
        name="observed_mst_total_length",
    )

    return EpathNullModelComparison(
        observed_largest_cluster_size=observed_largest_cluster_size,
        observed_mst_total_length=observed_mst_total_length,
        largest_z_score=zscore_empirical(
            observed_largest_cluster_size,
            null_model.largest_mean,
            null_model.largest_std,
        ),
        largest_p_value_greater=p_value_empirical_greater(
            observed_largest_cluster_size,
            null_model.largest_cluster_sizes,
        ),
        mst_z_score=zscore_empirical(
            observed_mst_total_length,
            null_model.mst_mean,
            null_model.mst_std,
        ),
        mst_p_value_less=p_value_empirical_less(
            observed_mst_total_length,
            null_model.mst_total_lengths,
        ),
    )


def zscore_empirical(value: float, mean: float, standard_deviation: float) -> float:
    """Return an empirical z-score, or NaN when spread is zero."""

    value = _finite_float(value, name="value")
    mean = _finite_float(mean, name="mean")
    standard_deviation = _non_negative_finite_float(
        standard_deviation,
        name="standard_deviation",
    )
    if standard_deviation <= 0.0:
        return float("nan")

    return float((value - mean) / standard_deviation)


def p_value_empirical_greater(
    value: float,
    samples: Sequence[float] | Sequence[int],
) -> float:
    """Return P(X >= value) with +1 pseudocounts."""

    sample_tuple = tuple(samples)
    if not sample_tuple:
        return float("nan")

    value = _finite_float(value, name="value")
    count = sum(1 for sample in sample_tuple if float(sample) >= value)
    return float((count + 1.0) / (len(sample_tuple) + 1.0))


def p_value_empirical_less(
    value: float,
    samples: Sequence[float] | Sequence[int],
) -> float:
    """Return P(X <= value) with +1 pseudocounts."""

    sample_tuple = tuple(samples)
    if not sample_tuple:
        return float("nan")

    value = _finite_float(value, name="value")
    count = sum(1 for sample in sample_tuple if float(sample) <= value)
    return float((count + 1.0) / (len(sample_tuple) + 1.0))


def _metrics_for_sample(
    atoms: Sequence[EpathScoredAtom],
    sampled_indices: Sequence[int],
    *,
    distance_cutoff: float,
) -> tuple[int, int, float]:
    sampled_atoms = tuple(atoms[int(index)] for index in sampled_indices)
    clusters = cluster_selected_atoms(
        sampled_atoms,
        distance_cutoff=distance_cutoff,
    )
    if not clusters:
        return 0, 0, 0.0

    largest_cluster = max(clusters, key=lambda cluster: cluster.atom_count)
    return (
        largest_cluster.atom_count,
        len(clusters),
        largest_cluster.metrics.mst_total_length,
    )


def _mean(samples: Sequence[float] | Sequence[int]) -> float:
    if not samples:
        return float("nan")

    return float(np.mean(np.asarray(samples, dtype=float)))


def _std(samples: Sequence[float] | Sequence[int]) -> float:
    if not samples:
        return float("nan")

    return float(np.std(np.asarray(samples, dtype=float)))


def _non_negative_int(value: int, *, name: str) -> int:
    number = _int_value(value, name=name)
    if number < 0:
        raise EpathNullModelError(f"{name} must be zero or positive, got {number!r}.")

    return number


def _int_value(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise EpathNullModelError(f"{name} must be an integer, got {value!r}.")

    try:
        return operator.index(value)
    except TypeError as error:
        raise EpathNullModelError(
            f"{name} must be an integer, got {value!r}."
        ) from error


def _positive_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise EpathNullModelError(f"{name} must be positive, got {number!r}.")

    return number


def _non_negative_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise EpathNullModelError(f"{name} must be zero or positive, got {number!r}.")

    return number


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathNullModelError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathNullModelError(f"{name} must be numeric, got {value!r}.") from error

    if not math.isfinite(number):
        raise EpathNullModelError(f"{name} must be finite, got {number!r}.")

    return number
