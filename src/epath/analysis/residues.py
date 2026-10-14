"""Classify residues by side-chain chemistry for E-path pathway description.

E-path describes each path by its residue composition, as the share of
aromatic, sulfur-containing, charged, polar, and hydrophobic residues and the
classes the path is enriched in. Every residue is assigned exactly one class
here, so the reported fractions sum to one and no residue is double counted.

Histidine and tyrosine are counted as aromatic rather than polar. A modified
residue is classed with its parent residue, whatever its own chemistry, so
phosphoserine is polar and pyroglutamate is charged.

Enrichment is measured against the composition of the analysed model rather
than a fixed reference table, so a class is only called enriched when the path
is unusual for the structure it came from.

`epath.analysis.pathways` calls this for every path and builds the structural
descriptor from the dominant and enriched classes. `epath.outputs` writes the
counts and fractions.
"""

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

# A class is enriched when it is at least this many times more common in the
# path than in the scored population of the same model.
ENRICHMENT_RATIO = 1.5

# A lone residue in a long path is not enrichment, so an enriched class must
# also account for at least one residue in fourteen. The ratio test above does
# most of the work; this floor keeps a single residue from carrying a class
# whose background is very small.
MINIMUM_ENRICHED_FRACTION = 0.07

# A class at least this large relative to the largest class shares dominance
# with it, which is what produces a mixed descriptor such as
# "hydrophobic-charged". A descriptor names at most two classes.
DOMINANT_FRACTION_RATIO = 0.8
MAXIMUM_DOMINANT_CLASSES = 2

_RESIDUE_CLASSES: Mapping[str, ResidueClass] = {
    # Aromatic side chains, including the imidazole of histidine.
    "PHE": "aromatic",
    "TYR": "aromatic",
    "TRP": "aromatic",
    "HIS": "aromatic",
    "PTR": "aromatic",
    # Sulfur-containing side chains, including selenomethionine, which is
    # substituted for methionine during phasing.
    "MET": "sulfur",
    "CYS": "sulfur",
    "MSE": "sulfur",
    "SEC": "sulfur",
    "CSO": "sulfur",
    "CME": "sulfur",
    "CSD": "sulfur",
    "FME": "sulfur",
    # Charged side chains. Their modified forms are classed with the parent
    # residue, whatever charge they carry themselves.
    "ASP": "charged",
    "GLU": "charged",
    "LYS": "charged",
    "ARG": "charged",
    "MLY": "charged",
    "KCX": "charged",
    "LLP": "charged",
    "PCA": "charged",
    # Neutral polar side chains, plus phosphoserine and phosphothreonine,
    # which are classed with their parent residues.
    "SER": "polar",
    "THR": "polar",
    "ASN": "polar",
    "GLN": "polar",
    "SEP": "polar",
    "TPO": "polar",
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
    """Side-chain chemistry composition of one group of residues.

    `counts` and `fractions` have an entry for every class, "other" included,
    and the fractions sum to one. `dominant_classes` holds the most common
    class and, when another is at least `DOMINANT_FRACTION_RATIO` times as
    common, that one too. `enriched_classes` holds the classes that are
    over-represented relative to the model. Both are ordered from the most
    common class down, never name "other", and are empty for a group in which
    no residue has a named class.
    """

    counts: Mapping[ResidueClass, int]
    fractions: Mapping[ResidueClass, float]
    dominant_classes: tuple[ResidueClass, ...]
    enriched_classes: tuple[ResidueClass, ...]


def residue_class(residue_name: str) -> ResidueClass:
    """Return the side-chain chemistry class of one residue name.

    Case and surrounding spaces are ignored. A name that is not an amino acid
    listed above, such as a water, a ligand or a nucleotide, is "other".
    """
    return _RESIDUE_CLASSES.get(residue_name.strip().upper(), "other")


def class_counts(residue_names: Iterable[str]) -> dict[ResidueClass, int]:
    """Count residues by side-chain chemistry class.

    Every class has an entry, zero when no residue falls in it, in
    `RESIDUE_CLASS_ORDER`.
    """
    counts: dict[ResidueClass, int] = dict.fromkeys(RESIDUE_CLASS_ORDER, 0)
    for name in residue_names:
        counts[residue_class(name)] += 1

    return counts


def class_fractions(residue_names: Iterable[str]) -> dict[ResidueClass, float]:
    """Return the fraction of residues in each side-chain chemistry class.

    "other" residues count towards the total, so the fractions sum to one.
    There must be at least one residue, which is not checked here.
    """
    counts = class_counts(residue_names)
    total = sum(counts.values())
    return {name: counts[name] / total for name in RESIDUE_CLASS_ORDER}


def analyze_composition(
    residue_names: Iterable[str],
    *,
    background: Mapping[ResidueClass, float],
) -> ResidueComposition:
    """Describe the chemistry of a group of residues, such as one path.

    `residue_names` holds one name per residue and must not be empty.
    `background` holds the class fractions of the scored population of the same
    model, as `class_fractions` returns them, and is the reference for
    enrichment.
    """
    names = tuple(residue_names)
    fractions = class_fractions(names)

    return ResidueComposition(
        counts=class_counts(names),
        fractions=fractions,
        dominant_classes=_dominant_classes(fractions),
        enriched_classes=_enriched_classes(fractions, background),
    )


def _dominant_classes(
    fractions: Mapping[ResidueClass, float],
) -> tuple[ResidueClass, ...]:
    ranked = _ranked_classes(fractions)
    if not ranked:
        return ()

    threshold = DOMINANT_FRACTION_RATIO * fractions[ranked[0]]
    dominant: list[ResidueClass] = [
        name for name in ranked if fractions[name] >= threshold
    ]
    return tuple(dominant[:MAXIMUM_DOMINANT_CLASSES])


def _enriched_classes(
    fractions: Mapping[ResidueClass, float],
    background: Mapping[ResidueClass, float],
) -> tuple[ResidueClass, ...]:
    return tuple(
        name
        for name in _ranked_classes(fractions)
        if fractions[name] >= MINIMUM_ENRICHED_FRACTION
        and fractions[name] >= ENRICHMENT_RATIO * background[name]
    )


def _ranked_classes(
    fractions: Mapping[ResidueClass, float],
) -> list[ResidueClass]:
    """Return the named classes that are present, most common first.

    "other" is never ranked, so the result is empty when every residue is
    "other".
    """
    present: list[ResidueClass] = [
        name
        for name in RESIDUE_CLASS_ORDER
        if name != "other" and fractions[name] > 0.0
    ]
    # The sort is stable, so equally common classes stay in RESIDUE_CLASS_ORDER.
    return sorted(present, key=lambda name: -fractions[name])
