"""Classify residues by side-chain chemistry for E-path pathway description.

E-path describes each path by its residue composition, as the share of
aromatic, sulfur-containing, charged, polar, and hydrophobic residues. Every
residue is assigned exactly one class here, so the reported fractions sum to
one and no residue is double counted.

Histidine and tyrosine are counted as aromatic rather than polar. A modified
residue is classed with its parent residue, whatever its own chemistry, so
phosphoserine is polar and pyroglutamate is charged.

`epath.analysis.pathways` calls this for every path, and `epath.outputs`
writes the counts and fractions.
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
    and the fractions sum to one.
    """

    counts: Mapping[ResidueClass, int]
    fractions: Mapping[ResidueClass, float]


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


def analyze_composition(residue_names: Iterable[str]) -> ResidueComposition:
    """Describe the chemistry of a group of residues, such as one path.

    `residue_names` holds one name per residue and must not be empty.
    """
    names = tuple(residue_names)

    return ResidueComposition(
        counts=class_counts(names),
        fractions=class_fractions(names),
    )
