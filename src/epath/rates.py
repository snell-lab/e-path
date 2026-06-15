"""Marcus-style and geometric rate proxies for E-path results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from epath.clustering import EpathCluster
from epath.paths import EpathPath


BOLTZMANN_EV_PER_K = 8.617333262145e-5
DEFAULT_REORGANIZATION_ENERGY_EV = 0.7
DEFAULT_DELTA_G_EV = -0.1
DEFAULT_TEMPERATURE_K = 298.0
DEFAULT_BETA_PER_ANGSTROM = 1.0


class EpathRateError(ValueError):
    """Raised when an E-path rate proxy cannot be calculated."""


@dataclass(frozen=True, slots=True)
class EpathMarcusRate:
    """Marcus-style geometric rate calculation for one path."""

    path_length_angstrom: float
    rate: float
    reorganization_energy_ev: float
    delta_g_ev: float
    temperature_k: float
    beta_per_angstrom: float
    reorganization_exponent: float
    distance_exponent: float


@dataclass(frozen=True, slots=True)
class EpathClusterRate:
    """Simple geometric decay rate for ranking one cluster."""

    cluster_id: int
    mst_total_length_angstrom: float
    rate: float
    normalized_rate: float


def marcus_geom_rate(
    R_A: float,
    reorg: float = DEFAULT_REORGANIZATION_ENERGY_EV,
    dG: float = DEFAULT_DELTA_G_EV,
    T: float = DEFAULT_TEMPERATURE_K,
    beta: float = DEFAULT_BETA_PER_ANGSTROM,
) -> float:
    """Return the E-path Marcus-like geometric rate proxy."""

    return marcus_rate_details(
        path_length_angstrom=R_A,
        reorganization_energy_ev=reorg,
        delta_g_ev=dG,
        temperature_k=T,
        beta_per_angstrom=beta,
    ).rate


def marcus_rate_details(
    *,
    path_length_angstrom: float,
    reorganization_energy_ev: float = DEFAULT_REORGANIZATION_ENERGY_EV,
    delta_g_ev: float = DEFAULT_DELTA_G_EV,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    beta_per_angstrom: float = DEFAULT_BETA_PER_ANGSTROM,
) -> EpathMarcusRate:
    """Return the Marcus-style rate proxy and its exponent components."""

    path_length_angstrom = _non_negative_finite_float(
        path_length_angstrom,
        name="path_length_angstrom",
    )
    reorganization_energy_ev = _finite_float(
        reorganization_energy_ev,
        name="reorganization_energy_ev",
    )
    delta_g_ev = _finite_float(delta_g_ev, name="delta_g_ev")
    temperature_k = _positive_finite_float(temperature_k, name="temperature_k")
    beta_per_angstrom = _finite_float(
        beta_per_angstrom,
        name="beta_per_angstrom",
    )

    if reorganization_energy_ev > 0.0:
        thermal_energy = BOLTZMANN_EV_PER_K * temperature_k
        reorganization_exponent = -(
            (delta_g_ev + reorganization_energy_ev) ** 2
        ) / (4.0 * reorganization_energy_ev * thermal_energy)
    else:
        reorganization_exponent = 0.0

    distance_exponent = -beta_per_angstrom * path_length_angstrom
    rate = float(math.exp(reorganization_exponent + distance_exponent))

    return EpathMarcusRate(
        path_length_angstrom=path_length_angstrom,
        rate=rate,
        reorganization_energy_ev=reorganization_energy_ev,
        delta_g_ev=delta_g_ev,
        temperature_k=temperature_k,
        beta_per_angstrom=beta_per_angstrom,
        reorganization_exponent=float(reorganization_exponent),
        distance_exponent=float(distance_exponent),
    )


def marcus_rate_for_path(
    path: EpathPath,
    *,
    reorganization_energy_ev: float = DEFAULT_REORGANIZATION_ENERGY_EV,
    delta_g_ev: float = DEFAULT_DELTA_G_EV,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    beta_per_angstrom: float = DEFAULT_BETA_PER_ANGSTROM,
) -> EpathMarcusRate:
    """Return the Marcus-style rate proxy for an E-path shortest path."""

    return marcus_rate_details(
        path_length_angstrom=path.total_length,
        reorganization_energy_ev=reorganization_energy_ev,
        delta_g_ev=delta_g_ev,
        temperature_k=temperature_k,
        beta_per_angstrom=beta_per_angstrom,
    )


def geometric_decay_rate(
    length_angstrom: float,
    *,
    beta_per_angstrom: float = DEFAULT_BETA_PER_ANGSTROM,
) -> float:
    """Return the cluster ranking proxy exp(-beta * length)."""

    length_angstrom = _non_negative_finite_float(
        length_angstrom,
        name="length_angstrom",
    )
    beta_per_angstrom = _finite_float(
        beta_per_angstrom,
        name="beta_per_angstrom",
    )
    return float(math.exp(-beta_per_angstrom * length_angstrom))


def cluster_geometric_rates(
    clusters: Iterable[EpathCluster],
    *,
    beta_per_angstrom: float = DEFAULT_BETA_PER_ANGSTROM,
) -> tuple[EpathClusterRate, ...]:
    """Return normalized geometric decay rates for clusters by MST length."""

    cluster_tuple = tuple(clusters)
    beta_per_angstrom = _finite_float(
        beta_per_angstrom,
        name="beta_per_angstrom",
    )
    raw_rates = tuple(
        geometric_decay_rate(
            cluster.metrics.mst_total_length,
            beta_per_angstrom=beta_per_angstrom,
        )
        for cluster in cluster_tuple
    )
    max_rate = max(raw_rates, default=1.0)

    return tuple(
        EpathClusterRate(
            cluster_id=cluster.cluster_id,
            mst_total_length_angstrom=cluster.metrics.mst_total_length,
            rate=raw_rate,
            normalized_rate=(raw_rate / max_rate if max_rate > 0.0 else 0.0),
        )
        for cluster, raw_rate in zip(cluster_tuple, raw_rates, strict=True)
    )


def _positive_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise EpathRateError(f"{name} must be positive, got {number!r}.")

    return number


def _non_negative_finite_float(value: float, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise EpathRateError(f"{name} must be zero or positive, got {number!r}.")

    return number


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathRateError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathRateError(f"{name} must be numeric, got {value!r}.") from error

    if not math.isfinite(number):
        raise EpathRateError(f"{name} must be finite, got {number!r}.")

    return number
