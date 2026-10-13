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

The default E-path selection is the lowest 2% of BDamage values. The threshold
is the BDamage at that percentile of the scored atoms, and the atoms at or
below it are then cut to 2% of the atom count by dropping the highest values.
The count is rounded to a whole number of atoms, and one that falls exactly
halfway rounds to the even number. Connected
components use a 4 A atom-to-atom cutoff and only components containing at least
eight atoms are reported as clusters. `--minimum-cluster-size` changes that
reporting threshold. Each reported cluster includes the atom count, MST total
length, mean step, largest edge, and span used in the E-path analysis
described in [pathways.md](pathways.md).

E-path analysis always runs over the complete prepared model. Restricting the
analysis to a subset of chains would change both the BDamage Z-score
distribution the percentile is taken from and the cross-chain connectivity that
produces interface clusters, so no analysis-level chain restriction is
provided. `--chains A,B` restricts only the coordinate outputs (the selected
atom, cleaned model, and per-pathway mmCIF files) to the named chains, after the
analysis has finished, so figures can focus on one chain while every result
still reflects the complete model. A pathway with no atoms in the kept chains
gets no coordinate file, and the CSVs and summary always describe every chain.
A chain that is not in the model is reported as an error before anything is
written.
