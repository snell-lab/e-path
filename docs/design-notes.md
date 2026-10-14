# Design Notes

This page explains why E-path makes some of its fixed choices. The
[user guide](user-guide.md) describes what it does.

## Why Only Protein Atoms Are Scored

The scored population is fixed rather than configurable:

- BDamage compares an atom's B-factor against atoms in a similar packing
  density environment. Protein, nucleic-acid, and heteroatom B-factors relate
  to packing density differently, so scoring them together would make an atom's
  BDamage partly a statement about which population it belongs to.
- The share of a model that is modelled solvent scales with resolution, from
  under 2% to over 14% of scored atoms across ordinary structures. Including
  solvent would make the selection percentile mean something different in every
  structure, so cluster statistics could not be compared between models.
- Bnet is defined on Asp/Glu carboxyl oxygens and is protein-only, so a mixed
  scored population would be inconsistent with the Bnet reported beside it.

## Why the Analysis Covers Every Chain

Restricting the analysis to some chains would change both the Z-score
distribution the selection percentile is taken from and the cross-chain
connectivity that produces interface pathways. `--chains` therefore filters
only the coordinate outputs, after the analysis has finished.
