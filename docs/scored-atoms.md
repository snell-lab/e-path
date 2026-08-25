# Scored Atom Population

E-path analyses electron-transfer pathways through proteins, so only protein
atoms receive BDamage scores. Hydrogens are removed and alternate atom sites
are resolved by highest occupancy, matching RABDAM. Modified amino acids
deposited as HETATM stay in the protein selection.

Solvent, ions, ligands, cofactors, and nucleic acids are not scored. They are
not ignored: packing density counts every non-hydrogen atom in the crystal
environment whether or not it is scored, so they still shape the BDamage value
of every protein atom near them.

This population is fixed rather than configurable, because it is the single
choice the whole analysis rests on:

- BDamage compares an atom's B-factor against atoms in a similar packing
  density environment. Protein, nucleic-acid, and heteroatom B-factors relate
  to packing density differently, so scoring them together would make an
  atom's BDamage partly a statement about which population it belongs to.
- The share of a model that is modelled solvent scales with resolution, from
  under 2% to over 14% of scored atoms across ordinary structures. Including
  solvent would make the selection percentile mean something different in
  every structure, so cluster statistics could not be compared between models.
- Bnet is defined on Asp/Glu carboxyl oxygens and is protein-only, so a mixed
  scored population would be inconsistent with the Bnet reported beside it.

There is no option anywhere in the program to score anything else.

The default E-path selection is the lowest 2% of BDamage values. Connected
components use a 4 A atom-to-atom cutoff and only components containing at least
eight atoms are reported as clusters. `--minimum-cluster-size` changes that
reporting threshold. Each reported cluster includes the atom count, MST total
length, mean step, largest edge, and span used in the E-path analysis
described in [pathways.md](pathways.md).

E-path analysis always runs over the complete prepared model. Restricting the
analysis to a subset of chains would change both the BDamage Z-score
distribution the percentile is taken from and the cross-chain connectivity that
produces interface clusters, so no chain-restriction option is provided. Use
`--per-chain-percentile` to stratify the percentile cutoff within each chain
while keeping every atom in the connectivity graph, and restrict chains at the
visualization stage when preparing figures.
