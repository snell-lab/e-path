"""Classify residues by side-chain chemistry for E-path pathway description.

E-path describes each path by its residue composition, as the share of
aromatic, sulfur-containing, charged, polar, and hydrophobic residues. Every
residue is assigned exactly one class here, so the reported fractions sum to
one and no residue is double counted.

Histidine and tyrosine are counted as aromatic rather than polar. A modified
residue is classed with its parent residue, whatever its own chemistry, so
phosphoserine is polar and pyroglutamate is charged. The parent is the one the
Chemical Component Dictionary records, so every modified amino acid that
`rabdam.structure` scores as protein has one.

`epath.analysis.pathways` calls this for every path, and `epath.outputs`
writes the counts and fractions.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

import gemmi

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

# Keyed by the one-letter code of the parent amino acid, which is what a
# modified residue is classed by.
_PARENT_CLASSES: Mapping[str, ResidueClass] = {
    # Aromatic side chains, including the imidazole of histidine.
    "F": "aromatic",
    "Y": "aromatic",
    "W": "aromatic",
    "H": "aromatic",
    # Sulfur-containing side chains, and selenocysteine, whose selenium takes
    # the place of the sulfur of cysteine.
    "M": "sulfur",
    "C": "sulfur",
    "U": "sulfur",
    # Charged side chains, and pyrrolysine, which is built on lysine.
    "D": "charged",
    "E": "charged",
    "K": "charged",
    "R": "charged",
    "O": "charged",
    # Neutral polar side chains.
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    # Aliphatic side chains, plus glycine and proline, which contribute a
    # low-dielectric backbone environment rather than a functional group.
    "A": "hydrophobic",
    "G": "hydrophobic",
    "V": "hydrophobic",
    "L": "hydrophobic",
    "I": "hydrophobic",
    "P": "hydrophobic",
}


@dataclass(frozen=True, slots=True)
class ResidueComposition:
    """Side-chain chemistry composition of one group of residues.

    `counts` and `fractions` have an entry for every class, "other" included,
    and the fractions sum to one.
    """

    counts: Mapping[ResidueClass, int]
    fractions: Mapping[ResidueClass, float]


@cache
def residue_class(residue_name: str) -> ResidueClass:
    """Return the side-chain chemistry class of one residue name.

    The parent of a residue is the one-letter code the Chemical Component
    Dictionary gives it, as Gemmi tabulates it: upper case for a standard
    amino acid and lower case for a modified one. Case and surrounding spaces
    in the name are ignored. A name that is not an amino acid, such as a
    water, a ligand or a nucleotide, is "other", and so is an amino acid with
    no known parent, such as UNK.
    """
    residue = gemmi.find_tabulated_residue(residue_name.strip().upper())
    if not residue.is_amino_acid():
        return "other"

    return _PARENT_CLASSES.get(residue.one_letter_code.upper(), "other")


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


def analyze_composition(residue_names: Iterable[str]) -> ResidueComposition:
    """Describe the chemistry of a group of residues, such as one path.

    `residue_names` holds one name per residue and must not be empty.
    """
    names = tuple(residue_names)

    return ResidueComposition(
        counts=class_counts(names),
        fractions=class_fractions(names),
    )
