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

E-path requires Python 3.10 or newer. Install the package and its normal
runtime dependencies, including Matplotlib for the default Z-score plot, with:

```bash
python3 -m pip install .
```

From the repository root:

```bash
./epath --help
```

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

- `*_epath.csv`: combined per-atom BDamage and E-path results.
- `*_clusters.csv`: one row per reported selected-atom cluster (eight or more
  atoms by default).
- `*_selected.pdb`: selected E-path atoms.
- `*_model.pdb`: the complete cleaned model with BDamage in the B-factor
  column.
- `*_zscores.png`: BDamage Z-score histogram.
- `*_summary.txt`: persistent human-readable analysis summary.

`*_model.pdb` holds every atom that survived cleaning, so solvent, ions,
ligands, and cofactors are retained for inspection alongside the E-path results
even though they are never scored. Scored atoms carry their BDamage value in
the B-factor column and unscored atoms carry `0.00`, so colouring by B-factor
shows the BDamage landscape without the excluded atoms distorting the scale.
The selected atoms are those with a B-factor above zero and at or below the
reported BDamage threshold, which the file records in a `REMARK 250` line.

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
- `epath`: the pathway analysis built on per-atom BDamage values.

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
