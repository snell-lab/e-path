"""Classify residues by side-chain chemistry for E-path pathway description.

The pathway framework in the PNAS analysis describes each path by its residue
composition: enrichment in aromatic, sulfur-containing, polar/charged, or
hydrophobic residues. Every residue is assigned exactly one class here, so the
reported fractions sum to one and no residue is double counted.

Histidine and tyrosine are counted as aromatic rather than polar, because the
analysis treats aromatic side chains as the redox-active set that supports
electron delocalization and transient radical states.

Enrichment is measured against the composition of the analysed model rather
than a fixed reference table, so a class is only called enriched when the path
is unusual for the structure it came from.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

ResidueClass = Literal[
    "aromatic",
    "sulfur",
    "charged",
    "polar",
    "hydrophobic",
    "other",
]

# Reported in this order wherever several classes are listed together.
RESIDUE_CLASS_ORDER: tuple[ResidueClass, ...] = (
    "aromatic",
    "sulfur",
    "charged",
    "polar",
    "hydrophobic",
    "other",
)

# Aromatic and sulfur side chains are the residue types the pathway model
# credits with efficient electronic coupling.
REDOX_ACTIVE_CLASSES: tuple[ResidueClass, ...] = ("aromatic", "sulfur")

# A class is enriched when it is at least this many times more common in the
# path than in the scored population of the same model.
ENRICHMENT_RATIO = 1.5

# A lone residue in a long path is not enrichment, so an enriched class must
# also account for at least one residue in fourteen. The ratio test above does
# most of the work; this floor keeps a single residue from carrying a class
# whose background is very small.
MINIMUM_ENRICHED_FRACTION = 0.07

# Classes within this fraction of the largest class share dominance, which is
# what produces a mixed descriptor such as "hydrophobic-charged".
DOMINANT_FRACTION_RATIO = 0.8
MAXIMUM_DOMINANT_CLASSES = 2

_RESIDUE_CLASSES: Mapping[str, ResidueClass] = {
    # Aromatic side chains, including the imidazole of histidine.
    "PHE": "aromatic",
    "TYR": "aromatic",
    "TRP": "aromatic",
    "HIS": "aromatic",
    "HID": "aromatic",
    "HIE": "aromatic",
    "HIP": "aromatic",
    "HSD": "aromatic",
    "HSE": "aromatic",
    "HSP": "aromatic",
    "PTR": "aromatic",
    "TPO": "polar",
    "SEP": "polar",
    # Sulfur-containing side chains, including selenomethionine, which is
    # substituted for methionine during phasing.
    "MET": "sulfur",
    "CYS": "sulfur",
    "MSE": "sulfur",
    "SEC": "sulfur",
    "CYM": "sulfur",
    "CYX": "sulfur",
    "CSO": "sulfur",
    "CME": "sulfur",
    "CSD": "sulfur",
    "FME": "sulfur",
    # Charged side chains, including protonation variants and common
    # modifications that retain the parent charge.
    "ASP": "charged",
    "GLU": "charged",
    "LYS": "charged",
    "ARG": "charged",
    "ASH": "charged",
    "GLH": "charged",
    "MLY": "charged",
    "KCX": "charged",
    "LLP": "charged",
    "PCA": "charged",
    # Neutral polar side chains.
    "SER": "polar",
    "THR": "polar",
    "ASN": "polar",
    "GLN": "polar",
    # Aliphatic side chains, plus glycine and proline, which contribute a
    # low-dielectric backbone environment rather than a functional group.
    "ALA": "hydrophobic",
    "GLY": "hydrophobic",
    "VAL": "hydrophobic",
    "LEU": "hydrophobic",
    "ILE": "hydrophobic",
    "PRO": "hydrophobic",
    "HYP": "hydrophobic",
}


@dataclass(frozen=True, slots=True)
class ResidueComposition:
    """Side-chain chemistry composition of one group of residues."""

    residue_count: int
    counts: Mapping[ResidueClass, int]
    fractions: Mapping[ResidueClass, float]
    dominant_classes: tuple[ResidueClass, ...]
    enriched_classes: tuple[ResidueClass, ...]

    @property
    def redox_fraction(self) -> float:
        """Return the fraction of residues with aromatic or sulfur chemistry."""
        return sum(self.fractions[name] for name in REDOX_ACTIVE_CLASSES)

    @property
    def charged_fraction(self) -> float:
        """Return the fraction of charged residues."""
        return self.fractions["charged"]


def residue_class(residue_name: str) -> ResidueClass:
    """Return the side-chain chemistry class of one residue name."""
    return _RESIDUE_CLASSES.get(residue_name.strip().upper(), "other")


def class_counts(residue_names: Iterable[str]) -> dict[ResidueClass, int]:
    """Count residues by side-chain chemistry class."""
    counts: dict[ResidueClass, int] = dict.fromkeys(RESIDUE_CLASS_ORDER, 0)
    for name in residue_names:
        counts[residue_class(name)] += 1

    return counts


def class_fractions(residue_names: Iterable[str]) -> dict[ResidueClass, float]:
    """Return the fraction of residues in each side-chain chemistry class."""
    counts = class_counts(residue_names)
    total = sum(counts.values())
    if total == 0:
        return dict.fromkeys(RESIDUE_CLASS_ORDER, 0.0)

    return {name: counts[name] / total for name in RESIDUE_CLASS_ORDER}


def analyze_composition(
    residue_names: Iterable[str],
    *,
    background: Mapping[ResidueClass, float] | None = None,
) -> ResidueComposition:
    """Describe the chemistry of a group of residues.

    Args:
        residue_names: Residue names, one per residue in the path.
        background: Class fractions of the scored population of the same model,
            used as the reference for enrichment. Enrichment is not reported
            when this is omitted.

    Returns:
        Class counts and fractions, the dominant classes, and the classes the
        path is enriched in relative to the model.
    """
    counts = class_counts(residue_names)
    total = sum(counts.values())
    fractions: dict[ResidueClass, float] = (
        {name: counts[name] / total for name in RESIDUE_CLASS_ORDER}
        if total
        else dict.fromkeys(RESIDUE_CLASS_ORDER, 0.0)
    )

    return ResidueComposition(
        residue_count=total,
        counts=counts,
        fractions=fractions,
        dominant_classes=_dominant_classes(fractions),
        enriched_classes=_enriched_classes(fractions, background=background),
    )


def _dominant_classes(
    fractions: Mapping[ResidueClass, float],
) -> tuple[ResidueClass, ...]:
    ranked = _ranked_classes(fractions)
    if not ranked:
        return ()

    highest_fraction = fractions[ranked[0]]
    if highest_fraction <= 0.0:
        return ()

    dominant: list[ResidueClass] = [
        name
        for name in ranked
        if fractions[name] >= DOMINANT_FRACTION_RATIO * highest_fraction
    ]
    return tuple(dominant[:MAXIMUM_DOMINANT_CLASSES])


def _enriched_classes(
    fractions: Mapping[ResidueClass, float],
    *,
    background: Mapping[ResidueClass, float] | None,
) -> tuple[ResidueClass, ...]:
    if background is None:
        return ()

    enriched: list[ResidueClass] = [
        name
        for name in _ranked_classes(fractions)
        if fractions[name] >= MINIMUM_ENRICHED_FRACTION
        and fractions[name] >= ENRICHMENT_RATIO * background.get(name, 0.0)
    ]
    return tuple(enriched)


def _ranked_classes(
    fractions: Mapping[ResidueClass, float],
) -> tuple[ResidueClass, ...]:
    named_classes: list[ResidueClass] = [
        name for name in RESIDUE_CLASS_ORDER if name != "other"
    ]
    named_classes.sort(
        key=lambda name: (-fractions[name], RESIDUE_CLASS_ORDER.index(name))
    )
    return tuple(name for name in named_classes if fractions[name] > 0.0)
