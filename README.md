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

The E-path pipeline builds on these per-atom BDamage values with Z-score or
percentile selection, 4 A connectivity clustering, and minimum-spanning-tree
(MST) geometry metrics. Bnet code remains in the repository because Bnet and
Bnet percentile are reported in the PNAS analysis, but it is not currently
hooked into the default CLI pipeline.

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
- `*_zscores.png`: BDamage Z-score histogram.
- `*_summary.txt`: persistent human-readable analysis summary.

Use `--output-dir` to choose another output directory or `--output-prefix` for
a single structure. `--no-csv`, `--no-pdb`, `--no-plot`, and `--no-reports`
disable the corresponding output groups. The `epath bdamage` subcommand writes
the standalone per-atom BDamage CSV.

## Selection Defaults

By default the full E-path command removes hydrogens, resolves alternate atom
sites by highest occupancy, and scores protein and HETATM records, including
water. This is the manuscript-oriented pathway-analysis population and allows
waters and cofactors to participate without extra flags. The standalone
`epath bdamage` command retains the RABDAM-style default of excluding
non-protein HETATM records.

Useful selection options:

```bash
./epath model.cif --include-hetatm
./epath model.cif --exclude-hetatm
./epath model.cif --include-nucleic-acid
./epath model.cif --chains A,C
./epath model.cif --remove-component HOH
./epath model.cif --add-component LIG
./epath model.cif --remove-atom-serial 123 --add-atom-serial 456
./epath model.cif --minimum-cluster-size 1
```

Changing the selected atoms changes the BDamage distribution. Prefer the
documented manuscript-style default for comparable E-path runs, and record any
selection overrides used for a particular structure.

The default E-path selection is the lowest 2% of BDamage values. Connected
components use a 4 A atom-to-atom cutoff and only components containing at least
eight atoms are reported as clusters. `--minimum-cluster-size` changes that
reporting threshold. Each reported cluster includes the atom count, MST total
length, mean step, largest edge, and span used in the PNAS analysis.

`--chains` restricts the downstream E-path Z-score distribution, selection,
clustering, and outputs. BDamage itself is still calculated first from the
complete prepared crystallographic structure.

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
- Bnet retained for later: Bnet and reference-database modules remain under
  `src/rabdam/bnet/`, but are not currently hooked into the default CLI.

These improvements intentionally remove some RABDAM 2 application features.
E-path does not support the legacy text input file, `-f`, `-i`, `-r`
dataframe restart modes, interactive overwrite prompts, HTML reports, plots,
BDamage-colored coordinate files, or restartable dataframe outputs.

## Limits And Assumptions

The CLI expects a single-model crystallographic structure with unit-cell and
space-group metadata. BDamage relies on per-atom B-factors; structures with
grouped or otherwise non-comparable B-factors should be interpreted with care.

## Citations

RABDAM: Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
J Appl Cryst 51, 552-559. https://doi.org/10.1107/S1600576718002509

BDamage: Gerstel M, Deane CM & Garman EF (2015).
J Synchrotron Radiat 22, 201-212. https://doi.org/10.1107/S1600577515002131
