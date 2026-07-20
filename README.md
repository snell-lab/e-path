# E-path

E-path currently calculates per-atom BDamage scores for protein crystal
structures. Its calculation core is a from-scratch implementation of the
BDamage workflow from RABDAM 2, created by the Garman Group at the University
of Oxford (https://github.com/GarmanGroup/RABDAM).

Use this near the end of refinement, or on deposited models, where per-atom
B-factors are meaningful.

## What It Calculates

BDamage is a per-atom score that compares an atom's B-factor with atoms in a
similar local packing-density environment. By default, packing density is the
number of non-hydrogen crystal-environment atoms within 7.0 A.

The E-path pipeline will build on these per-atom BDamage values for pathway
analysis. Bnet code remains in the repository for later optional reporting, but
it is not part of the current default CLI pipeline.

## Install

E-path requires Python 3.10 or newer and the dependencies listed in
`pyproject.toml`: `gemmi`, `numpy`, and `scipy`.

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

Single and batch runs write CSVs in `output/` by default, using the PDB ID or
input filename stem: for example `output/1LYZ_BDamage.csv` or
`output/model_BDamage.csv`. Use `--output-dir` to choose another output
directory; use `--output-csv` only when running one structure.

Each CSV contains one row per selected atom. The main result columns are:

- `PD`: packing-density neighbour count.
- `AVRG_BF`: local average B-factor for atoms with similar packing density.
- `BDAM`: per-atom BDamage score.

The terminal summary reports the BDamage calculation and crystal-neighbour
diagnostics.

## Selection Defaults

By default E-path removes hydrogens, resolves alternate atom sites by
highest occupancy, scores protein atoms, and requires a protein selection.

Useful selection options:

```bash
./epath model.cif --include-hetatm
./epath model.cif --include-nucleic-acid
./epath model.cif --chains A,C
./epath model.cif --remove-component HOH
./epath model.cif --add-component LIG
./epath model.cif --remove-atom-serial 123 --add-atom-serial 456
```

Changing the selected atoms changes the BDamage distribution. Prefer the
default protein selection unless you have a specific reason to change it.

`--chains` restricts the downstream E-path Z-score distribution, selection,
clustering, path search, null model, annotations, and outputs. BDamage itself is
still calculated first from the complete prepared crystallographic structure.

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
- Lighter installation: the runtime stack is Gemmi, NumPy, and SciPy. It does
  not require cctbx, CCP4/PDBCUR, pandas, matplotlib, or requests.
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
