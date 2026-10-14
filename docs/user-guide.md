# E-path User Guide

The [README](../README.md) covers installing and running E-path. This guide
explains what it expects as input, how it selects atoms and describes pathways,
and how to read what it writes.

## Input

E-path expects the deposited asymmetric-unit file of a crystal structure, or
the refined model it came from. Files with no unit cell, such as RCSB
biological-assembly mmCIF files and NMR or cryo-EM models, are rejected, and so
is a model too large to fit in its cell. A Matthews coefficient outside roughly
1.5 to 4.5 Å³/Da is reported as a warning that chains may be missing or
duplicated. [Input validation](input-validation.md) explains why and how far
the check can be trusted.

## Scored Atoms

E-path analyses electron-transfer pathways through proteins, so only protein
atoms receive BDamage scores. Hydrogens are removed and alternate atom sites
are resolved by highest occupancy, matching RABDAM. Modified amino acids
deposited as HETATM stay in the protein selection.

Solvent, ions, ligands, cofactors, and nucleic acids are not scored. They are
not ignored, though. Packing density counts every non-hydrogen atom in the
crystal environment whether or not it is scored, so they still shape the
BDamage value of every protein atom near them.

This population is fixed; the [design notes](design-notes.md) explain why.

## Selection and Clustering

The default E-path selection is the lowest 2% of BDamage values. The threshold
is the BDamage at that percentile of the scored atoms, and the atoms at or
below it are then cut to 2% of the atom count by dropping the highest values.
The count is rounded to a whole number of atoms, and one that falls exactly
halfway rounds to the even number. When atoms share the BDamage value at the
cut, the ones later in the input file are dropped first.

Each atom's Z-score is its BDamage minus the mean BDamage of the scored atoms,
divided by their population standard deviation (dividing by the atom count,
not by one less).

`--percentile` takes several comma-separated values, such as `1,2,3`. BDamage
and Bnet are calculated once, and the selection, clustering, and pathway
analysis then run at each cutoff, with the full outputs of each written to a
`percentile_<value>` directory of its own. Their run rows are collected in
`*_runs.csv` beside those directories, so whether a pathway persists as the
cutoff changes can be read from one run.

Selected atoms within 4.0 Å of each other are connected, and every connected
component of at least eight atoms is reported as a pathway.
`--minimum-cluster-size` changes that reporting threshold. Each pathway's
geometry is measured on its minimum spanning tree (MST), giving the atom count,
total length, mean step, and largest step.

The analysis always runs over every chain. `--chains` filters only the
coordinate outputs.

## Pathways

Every reported pathway is described in `*_clusters.csv` and in the run summary.
The pathway description and the step provenance below follow the E-path
analysis framework of a manuscript that is currently in preparation; the
citation will be added to the README and to `--citations` when it is published.
Each pathway is reported with:

- The residues the path runs through and the protected atoms in each of them,
  written `Chain A - Trp76 (CA, O, CB) -> His87 (CB)`. Chains appear in the
  order of the first residue each contributes, residues in ascending residue
  number, and atoms in the order the connectivity graph reached them.
- The residue count, the node (atom) count, and the total path length.
- The chains the path runs through.
- The cofactors within reach of the path, and their distances.
- The step provenance and residue-chemistry measurements.

A path that crosses between chains lists each of them in `chains`, and its
crossings are counted as interchain steps (see Step Provenance below).

The packing density of every path atom, the number of atoms within 7.0 Å that
BDamage counts (crystal neighbours included), is the `packing_density` column
of `*_epath.csv`, where `cluster_id` gives the path each atom belongs to. A low
count suggests an atom near the surface and a typical one an atom inside the
protein, but packing density is not a solvent-accessibility calculation; use a
dedicated tool when the distinction matters.

### Nearby Cofactors

Only redox-capable groups are treated as cofactors. A bound group qualifies if
it contains a redox-active transition metal (V, Mn, Fe, Co, Ni, Cu, Mo, or W),
which covers haems, iron-sulfur clusters, copper centres, and molybdenum or
tungsten cofactors under whatever component name they were deposited, or if it
is an organic redox cofactor, meaning a flavin, nicotinamide, quinone,
chlorophyll, pheophytin, metal-free or zinc porphyrin, or pterin. Everything
else bound to the protein, including lipids, detergents, sugars, buffer and
cryoprotectant molecules, substrates and inhibitors, and redox-inert ions such
as Na, K, Mg, Ca, and Zn, is still in the model and still contributes to
packing density, but is not reported as a cofactor a path lies near.

Some groups hold one of those metals without being redox cofactors, and are
left out too: the free vanadate, molybdate, and tungstate oxyanions (`VO4`,
`MOO`, `WO4`), cobalt hexammine (`NCO`), and a lone cobalt ion, which stands in
for another metal or acts as a Lewis acid. Cobalt still counts inside
cobalamin.

A lone nickel ion counts only when at least two sulfur atoms lie within 2.7 Å
of it, as cysteine thiolates or cluster sulfides. Redox-active nickel is held
by sulfur, as in nickel superoxide dismutase, nickel-iron hydrogenase, acetyl-CoA
synthase, and carbon monoxide dehydrogenase, while nickel held by histidine,
carboxylates, or water, as in urease, glyoxalase I, or a His-tag site, is not
redox-active. Nickel inside a larger cofactor such as F430 always counts.

Otherwise the rule cannot tell a functional metal from the same metal added for
crystallization, so a manganese ion from the mother liquor is still listed as a
cofactor and should be discounted by eye.

Sites sharing a residue name are numbered by chain, with the chains in the
order they first appear in the input, and then by residue number, which is what
produces `SF4-1` to `SF4-7`.

Every cofactor that lies within 8.0 Å of any atom of a path is reported as
near it, wherever along the path that atom is, so a path running between two
cofactors is reported with both. This flags paths worth inspecting alongside a
cofactor; it does not show that the two are connected. In a protein with many
cofactors, many paths will lie near one.

`nearby_cofactors`, `nearby_cofactor_distances`, and `nearby_cofactor_atoms` in
`*_clusters.csv` list the sites, their distances, and the path atom closest to
each, nearest first. Each column separates its entries with `/`, in the same
order in all three, and all three are empty when no cofactor is in range. The
summaries list every cofactor site found, so a path with no cofactor nearby can
be told apart from a model that has no cofactors.

### Step Provenance

Each step of a path is one edge of its minimum spanning tree, so the steps
counted here are the same ones the reported length and mean step describe. They
are classified by how they connect residues:

- `intra_residue_edges`: both ends in the same residue, which is side-chain
  shape rather than progress along the path.
- `peptide_edges`: consecutive residues of one chain.
- `tertiary_edges`: the same chain, but residues far apart in sequence.
- `interchain_edges`: different chains.

`peptide_edge_fraction` is the share of between-residue steps that follow the
sequence. It separates a rigid stretch of backbone from a path of the same size
whose steps jump between residues that are distant in sequence, which geometry
alone reports as identical. It is blank for a path that never leaves one
residue, which has no between-residue steps.

Peptide steps are identified by a modelled peptide bond, the carbonyl C of one
residue within 2.0 Å of the amide N of the other, not by residue numbers. A
chain break is never a peptide step, however the residues on either side are
numbered, and neither is a step to a residue whose backbone C or N is not
modelled. Residues joined across a jump in numbering or an insertion code are
peptide steps as long as the bond is there.

`*_steps.csv` lists every step individually, with its length, its kind, and the
two atoms it joins. Steps are listed in the order the spanning tree was built,
which is deterministic but is not a walk from one end of a path to the other.

### Composition

Composition assigns each residue exactly one chemistry class, so the reported
fractions sum to one. Histidine and tyrosine count as aromatic rather than
polar. A modified residue is classed with its parent residue, as the Chemical
Component Dictionary records it, so phosphoserine counts as polar and
trimethyllysine as charged.

Paths are often only a few residues long, so the fractions of one path move in
large steps. A single tryptophan in a four-residue path makes it 25% aromatic.

## Output Files

The files a run writes are listed in the [README](../README.md#output).

### Summaries and Plot

The terminal and persistent summaries report the mean and population standard
deviation of BDamage for the whole model, which are the values the Z-scores are
derived from, and again for the selected atoms alone. They record the BDamage
calculation parameters actually used, so a run made with non-default values is
identifiable from its output alone. The window size is reported as well as the
fraction it came from, since it is derived from the scored atom count rather
than set directly.

The summaries also include protein Bnet and, when the input contains resolution
metadata, its percentile against the bundled resolution-matched reference
database. If a structure has no usable Asp/Glu carboxyl-oxygen sites or no
resolution, the summaries explain why the value is unavailable without stopping
E-path.

`*_zfalloff.png` ranks atoms by ascending BDamage Z-score and plots the
negative tail, which is the region the selection is drawn from. The selection
threshold is marked, so the plot shows where the cutoff falls on the falloff.

### Coordinate Files

Every coordinate file is mmCIF. Run notes are written as an `_epath_remark`
loop and machine-readable values (the E-path version as `_epath.version`, atom
counts, the threshold, the pathway number, its cofactors and the cluster
distance) as an `_epath` category; viewers ignore both. Every
file carries the unit cell and space group of the input, so a viewer can
generate symmetry mates around a pathway. E-path works in the asymmetric unit,
so this is how to check whether a pathway sits at a crystal contact.
`_cell.Z_PDB` is written as `?` because the input's value is not kept.

`*_model.cif` holds every atom that survived cleaning, so solvent, ions,
ligands, and cofactors are retained for inspection alongside the E-path results
even though they are never scored. Scored atoms carry their BDamage value in
`_atom_site.B_iso_or_equiv` and unscored atoms carry `0.00`, so colouring by
B-factor shows the BDamage landscape without the excluded atoms distorting the
scale. Every selected atom has a B-factor above zero and at or below the
BDamage threshold, which the file records as `_epath.bdamage_threshold`,
written to the same precision as the B-factor column. A few
unselected atoms can lie in that range too, because the selection is cut to a
whole number of atoms, so `*_selected.cif` and the `selected` column of
`*_epath.csv` are what identify the selection exactly.

`*_selected.cif` holds every selected atom, including those in components too
small to be reported as pathways. `*_path01.cif`, `*_path02.cif`, and so on
each hold one pathway's atoms, with `_epath.path_id`,
`_epath.nearby_cofactors` (the nearby sites separated by `/`, or `none`), and
`_epath.cluster_distance` (the `--cluster-distance` the atoms were connected
within, in angstroms) recorded. Loading them separately is how to colour by pathway.
A run removes the pathway files an earlier run left under the same prefix, so
rerunning with `-o` never leaves stale pathways beside the new ones.
