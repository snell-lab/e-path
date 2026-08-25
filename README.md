# E-path

E-path calculates per-atom BDamage scores and analyzes spatially connected
low-BDamage atom clusters in protein crystal structures. Its
calculation core is a from-scratch implementation of the BDamage workflow from
RABDAM 2, created by the Garman Group at the University of Oxford
(https://github.com/GarmanGroup/RABDAM).

Use this near the end of refinement, or on deposited models, where per-atom
B-factors are meaningful.

## What It Calculates

BDamage is a per-atom score that compares an atom's B-factor with atoms in a
similar local packing-density environment. By default, packing density is the
number of non-hydrogen crystal-environment atoms within 7.0 A.

The E-path pipeline builds on these per-atom BDamage values with lowest-percentile
selection, 4 A connectivity clustering, and minimum-spanning-tree
(MST) geometry metrics. The complete RABDAM stage also calculates raw protein
Bnet and a resolution-matched Bnet percentile. ePath consumes BDamage only;
Bnet and its percentile are reported alongside the ePath results but do not
change ePath selection or clustering.

## Install

E-path runs directly from a clone of this repository; there is no package to
install. It requires Python 3.10 or newer and four libraries: Gemmi, NumPy,
SciPy, and Matplotlib (for the default Z-score plot).

```bash
git clone https://github.com/Aspartyl/e-path.git
cd e-path
python3 -m pip install -r requirements.txt
./epath --help
```

The `./epath` launcher adds `src/` to the Python path and runs the `epath`
package, so edits to the source take effect immediately. Run it from any
directory by its full path, or add the repository root to your `PATH`.

## Run

Local structure file:

```bash
./epath model.cif
```

RCSB/PDB ID:

```bash
./epath 1LYZ
```

Batch run:

```bash
./epath 1LYZ 2BLP model.cif
```

Accepted inputs are `.cif`, `.mmcif`, `.pdb`, and classic 4-character PDB IDs.
Pass one input for a single run or multiple inputs for a batch run.

## Output

Single and batch E-path runs write results in `output/` by default, using the
PDB ID or input filename stem as the output prefix. A normal run writes:

- `*_epath.csv`: one row per scored atom, with its BDamage, Z-score, whether
  it was selected, and its cluster membership. Run-level statistics are in the
  summary and cluster geometry is in `*_clusters.csv`, joinable on
  `cluster_id`, so neither is repeated per atom.
- `*_clusters.csv`: one row per reported pathway (eight or more atoms by
  default), holding the pathway table described below.
- `*_steps.csv`: one row per step of every reported pathway.
- `*_run.csv`: a single row of run-level results, described under Comparing
  Models.
- `*_selected.pdb`: selected E-path atoms.
- `*_path01.pdb`, `*_path02.pdb`, ...: one file per reported pathway.
- `*_model.pdb`: the complete cleaned model with BDamage in the B-factor
  column.
- `*_zfalloff.png`: rank-ordered BDamage Z-score falloff.
- `*_summary.txt`: persistent human-readable analysis summary.

`*_zfalloff.png` ranks atoms by ascending BDamage Z-score and plots the
negative tail, which is the region the selection is drawn from. The selection
threshold is marked, as is any detected plateau: a run of atoms packed more
densely in Z than the runs on either side of it. Atom density always rises
toward the threshold, so a plateau is only reported where the falloff has a
genuine local shelf rather than merely flattening as it approaches the bulk. A
plateau indicates that many atoms share a protection level.

The summaries report the mean and standard deviation of BDamage for the whole
model, which are the values the Z-scores are derived from, and again for the
selected atoms alone.

A batch run additionally writes `epath_runs.csv` in the output directory, with
one row per structure.

`*_model.pdb` holds every atom that survived cleaning, so solvent, ions,
ligands, and cofactors are retained for inspection alongside the E-path results
even though they are never scored. Scored atoms carry their BDamage value in
the B-factor column and unscored atoms carry `0.00`, so colouring by B-factor
shows the BDamage landscape without the excluded atoms distorting the scale.
The selected atoms are those with a B-factor above zero and at or below the
reported BDamage threshold, which the file records in a `REMARK 250` line.

Both summaries record the BDamage calculation parameters actually used, so a
run made with non-default values is identifiable from its output alone. The
window size is reported as well as the fraction it came from, since it is
derived from the scored atom count rather than set directly.

The terminal and persistent summaries include raw protein Bnet and, when the
input contains resolution metadata, its percentile against the bundled
resolution-matched reference database. If a structure has no usable Asp/Glu
carboxyl-oxygen sites or no resolution, the summaries explain why the
corresponding RABDAM value is unavailable without stopping ePath.

Use `--output-dir` to choose another output directory or `--output-prefix` for
a single structure. `--no-csv`, `--no-pdb`, `--no-plot`, and `--no-reports`
disable the corresponding output groups. Per-atom BDamage, packing density, and
window-averaged B-factors are columns of the combined `*_epath.csv` output.

## Scored Atom Population

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
length, mean step, largest edge, and span used in the PNAS analysis.

E-path analysis always runs over the complete prepared model. Restricting the
analysis to a subset of chains would change both the BDamage Z-score
distribution the percentile is taken from and the cross-chain connectivity that
produces interface clusters, so no chain-restriction option is provided. Use
`--per-chain-percentile` to stratify the percentile cutoff within each chain
while keeping every atom in the connectivity graph, and restrict chains at the
visualization stage when preparing figures.

## Pathway Table

Every reported cluster is described as a pathway in `*_clusters.csv` and in the
run summary, giving the columns of the pathway table in the PNAS analysis:

- the residues the path runs through and the protected atoms in each of them,
  written `Chain A - Trp76 (CA, O, CB) -> His87 (CB)`. Chains appear in the
  order of the first residue each contributes, residues in ascending residue
  number, and atoms in the order the connectivity graph reached them.
- the residue count, the node (atom) count, and the total path length.
- the environments the two ends of the path terminate in.
- the class and a structural descriptor.
- the topology, step provenance, and residue-chemistry measurements the class
  was assigned from.

### Endpoints

The endpoints are the two atoms that define the cluster span, which is already
reported as the straight-line end-to-end distance of the path. Each is labelled
with the environment it terminates in, tested in this order:

- `SF4-1`, `FMN`, `HEM-2`: in contact with that cofactor, within 4.0 A.
- `FMN region`: within 8.0 A of that cofactor, so in its environment without
  touching it.
- `Interface region`: within 4.0 A of a different chain.
- `Surface region`: in the least densely packed quarter of the scored atoms.
  Packing density is already calculated for every scored atom, so no separate
  solvent-accessibility calculation is involved.
- `Buried region`: none of the above.

Cofactor sites are taken from the cleaned model, since cofactors are never
scored and so can never lie on a path. Sites sharing a residue name are
numbered by chain and residue number, which is what produces `SF4-1` to
`SF4-7`. Common crystallization and cryoprotection additives are not treated as
cofactors, because terminating beside a glycerol does not make a path an
electron conduit; they are still in the model and still contribute to packing
density. The summaries list every cofactor site found, so an endpoint that
reports no cofactor can be told apart from a model that has none.

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
sequence. It separates two paths that geometry alone reports as identical: a
rigid stretch of backbone, and a path of the same size whose steps jump between
residues that are distant in sequence. The second is the stronger coupling
candidate, and the analysis in the published supplement repeatedly rests on
this distinction. Peptide steps are identified by sequence adjacency within a
chain rather than by a modelled backbone bond, so a chain break numbered
consecutively counts as peptide-contiguous.

Step provenance is reported but does not feed the classification.

`*_steps.csv` lists every step individually, with its length, its kind, and the
two atoms it joins. The summary reports the total, mean, and largest step of a
path, which says that a long step exists but not where it falls or how many
others come close to it. Because tunnelling efficiency falls away sharply with
distance, a single long step can limit a whole path, so the individual lengths
are worth having. Steps are listed in the order the spanning tree was built,
which is deterministic but is not a walk from one end of a path to the other.

### Pathway Coordinate Files

`*_selected.pdb` holds every selected atom, including those in components too
small to be reported as pathways. Alongside it, one file per reported pathway
is written as `*_path01.pdb`, `*_path02.pdb`, and so on, each holding only that
pathway's atoms, with its class, descriptor, size, and endpoints recorded in
`REMARK` lines.

The two forms answer different questions. The combined file shows what the
cutoff selected. The per-pathway files let a viewer load and colour each
pathway separately, which is what colouring by class requires. Keeping them
separate avoids encoding a pathway label into the occupancy or B-factor column,
both of which already carry real values. Path numbers are zero padded so the
files sort in path order, and they are suppressed by `--no-pdb` along with the
other coordinate outputs.

### Classification

Paths are classified on the three tiers used in the PNAS analysis: graph
topology, residue composition, and the functional role inferred from them.

Topology comes from the span-to-length ratio and node connectivity. A path
whose span is at least 20% of its total MST length is `linear`; connectivity is
counted between residues rather than atoms, and a path where at least 25% of
residues contact three or more other residues of the same cluster is
`branched`; anything else is `clustered`. Connectivity is deliberately not
counted between atoms: the atoms of one side chain are all mutually close and
an aromatic ring contains junctions of its own, so atom connectivity would
mostly report the shape of the residues rather than a fork in the path.

Composition assigns each residue exactly one chemistry class, so the reported
fractions sum to one. Histidine and tyrosine count as aromatic rather than
polar, because the analysis treats aromatic side chains as the redox-active
set. A class is enriched when it is at least 1.5 times more common in the path
than in the scored population of the same model, so enrichment is measured
against the structure being analysed rather than a fixed reference.

The role is then assigned conservatively:

- `conductor`: anchored at a cofactor, linear, at least 10% aromatic or sulfur
  residues, and less than 30% charged residues.
- `modulator`: anchored at a cofactor but failing one of those tests, which is
  the case for compact, branched, or strongly charged networks.
- `reservoir`: no cofactor anchor, but at least 20% aromatic or sulfur
  residues. An unanchored path needs stronger chemical evidence than an
  anchored one before it is called redox-relevant, because nothing else ties it
  to a redox centre.
- `structural`: no cofactor anchor and no redox-active enrichment.

The published framework fixes these criteria but not the numeric thresholds
that separate the classes. The defaults above were chosen so that the published
pathway assignments are reproduced: replaying the 22 published paths through
this classifier returns the published class for 19 of them. The three that
differ are all cases where the published endpoint label carries information
that is not in the coordinates, such as a cofactor that was not modelled, a
hand-named residue hub, or adjacency to another reported path. The thresholds
are fields of `PathAnalysisOptions` and can be overridden per run.

Structural descriptors are generated from the same measurements, naming what
the path is mostly made of, its topology, and the classes it is enriched in,
which gives phrasings such as `Hydrophobic corridor with aromatic/sulfur
enrichment` or `Single aromatic hotspot`.

## Comparing Models

Comparing many structures means reading the same run-level values back out
again, so each run records them once in `*_run.csv`: the input, selection
method and cutoff, scored atom count, atoms selected, the equivalent Z-score
and BDamage threshold, Bnet and its percentile, the connected-component counts,
and the geometry of the largest component. A batch run writes the same rows
combined into `epath_runs.csv`, which is the comparison table for that cohort.
Rows from separate single-structure runs concatenate to the same thing.

Two of those columns describe components that are never reported as pathways:

- `total_components` counts every connected component the selected atoms form,
  not just the ones large enough to report.
- `largest_component_*` describes the largest component whether or not it
  reached the reporting threshold, with `largest_component_reported` recording
  which. A model where nothing qualified still reports what its connectivity
  analysis found, so a largest component of seven atoms is distinguishable from
  one of two.

The run summary reports the same component count, and names the largest
component when no pathway reached the threshold.

Grouping those rows into cohorts, and taking medians within a group, is left to
whoever assembles the comparison. Which group a structure belongs to is a prior
judgement about its biology rather than something the analysis can determine,
so the software emits the per-structure rows and stops there.

## Differences From RABDAM 2

E-path currently contains a calculation-compatible rewrite of RABDAM 2, not a
feature-for-feature clone. Its purpose is to keep the scientific calculation
stable while making the program easier to install, faster to run, easier to
test, and better suited to scripted use.

The core BDamage calculation is preserved. Under normal default use, `PD`,
`AVRG_BF`, and `BDAM` are intended to match RABDAM 2. The rewrite keeps the
same scientific model: crystallographic symmetry expansion, neighbouring
unit-cell translation, strict 7.0 A packing-density counts, and BDamage sliding
windows.

The rewrite improves practical use in several ways:

- Faster calculations: packing-density neighbour searches use `cKDTree`,
  translation and trimming are fused by default to avoid materializing the full
  3x3x3 crystal block, and BDamage window averages use cumulative sums instead
  of pandas rolling means.
- Larger-structure speedups: in large-structure runs, these changes can make
  E-path complete more than 80 times faster than RABDAM 2.
- Lighter installation: the core runtime stack is Gemmi, NumPy, SciPy, and
  Matplotlib. It does not require cctbx, CCP4/PDBCUR, pandas, or requests.
- Clearer structure: the code is split into small modules for input, structure
  preparation, crystal expansion, packing density, BDamage, and later E-path
  analysis.
- Cleaner automation: the CLI uses
  `epath STRUCTURE_INPUT [STRUCTURE_INPUT ...] [options]`, supports simple
  batch runs, and writes compact per-atom CSVs plus terminal summaries.
- Full damage context: raw protein Bnet and its resolution-matched percentile
  are calculated with the bundled reference database and reported alongside
  E-path results. Their scientific defaults are fixed by the integrated RABDAM
  workflow rather than exposed as additional E-path settings.

These improvements intentionally remove some RABDAM 2 application features.
E-path does not support the legacy text input file, `-f`, `-i`, `-r`
dataframe restart modes, interactive overwrite prompts, HTML reports, plots,
BDamage-colored coordinate files, or restartable dataframe outputs.

E-path also does not expose RABDAM's atom-selection controls. RABDAM allows
HETATM, nucleic-acid, and per-atom or per-component selection overrides;
E-path fixes the population at protein atoms, because a pathway analysis that
compares clusters between structures requires the scored population to be
defined the same way in every model.

## Input Validation

BDamage compares an atom's B-factor with atoms at a similar local packing
density, and packing density is counted in a crystal environment rebuilt from
the model by symmetry expansion and unit-cell translation. That rebuilt
environment is only correct when the model fills exactly one asymmetric unit.

Symmetry expansion cannot detect a violation on its own, because it faithfully
rebuilds a crystal from whatever the file contains. E-path therefore validates
the asymmetric-unit content against the unit-cell volume using the Matthews
coefficient before any calculation runs:

- A model that cannot physically fit in the cell is rejected. This usually
  means a biological-assembly file was supplied instead of the asymmetric unit.
- A Matthews coefficient outside roughly 1.5 to 4.5 A^3/Da is reported as a
  warning. A high value suggests chains are missing, which makes packing
  densities too low at the missing interfaces; a low value suggests the model
  holds more than one asymmetric unit.

The space group, unit-cell volume, Matthews coefficient, and estimated solvent
content are recorded in every run summary. Only protein and nucleic-acid atoms
contribute to the modelled mass, as for a conventional Matthews calculation.
Hydrogens are removed before scoring, so the reported solvent content is a
slight overestimate.

## Package Layout

- `xtal`: crystallographic intake and validation. Input resolution, RCSB
  download, file reading, unit-cell and space-group accessors, and
  asymmetric-unit validation. Contains no damage calculation.
- `rabdam`: the from-scratch reimplementation of the RABDAM 2 BDamage and Bnet
  calculations, and nothing else, so it can be compared against RABDAM 2
  directly.
- `epath`: the pathway analysis built on per-atom BDamage values. Selection,
  clustering, and MST geometry produce the clusters; residue chemistry,
  topology, and endpoint labelling then describe each cluster as a classified
  pathway.

Dependencies run in one direction only: `epath` uses `rabdam`, and `rabdam`
uses `xtal`.

## Limits And Assumptions

The CLI expects a single-model crystallographic structure with unit-cell and
space-group metadata. BDamage and Bnet rely on per-atom B-factors; structures
with grouped or otherwise non-comparable B-factors should be interpreted with
care. Bnet and Bnet percentiles are intended for protein crystal structures
collected at cryogenic temperatures.

## Citations

RABDAM: Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
J Appl Cryst 51, 552-559. https://doi.org/10.1107/S1600576718002509

BDamage: Gerstel M, Deane CM & Garman EF (2015).
J Synchrotron Radiat 22, 201-212. https://doi.org/10.1107/S1600577515002131

Bnet and Bnet percentile: Shelley KL & Garman EF (2022).
Nat Commun 13, 1314. https://doi.org/10.1038/s41467-022-28934-0
