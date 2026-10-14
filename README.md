# E-path

E-path finds spatially connected clusters of low-BDamage atoms in protein
crystal structures and describes each as a pathway. BDamage compares an atom's
B-factor with atoms in a similar local packing-density environment. Its
calculation is a from-scratch implementation of the BDamage pipeline from
[RABDAM 2](https://github.com/GarmanGroup/RABDAM), by the Garman Group at the
University of Oxford. Like RABDAM, E-path also reports protein Bnet and its
resolution-matched percentile.

Use it near the end of refinement, or on deposited models, where per-atom
B-factors are meaningful. Bnet and its percentile are intended for structures
collected at cryogenic temperatures.

## Install

E-path runs from a clone of this repository. It needs Python 3.14 or newer and
the libraries in `requirements.txt`.

```bash
git clone https://github.com/snell-lab/e-path.git
cd e-path
python3 -m pip install -r requirements.txt
```

On Windows, run `python epath` wherever this README says `./epath`.

## Run

```bash
./epath model.cif               # a local file
./epath 1ABC                    # a PDB ID, downloaded from RCSB and cached
./epath 1ABC 2XYZ model.cif     # a batch run
./epath 1ABC --percentile 1,2,3 # several selection cutoffs in one run
```

Inputs can be `.cif`, `.mmcif`, `.pdb`, or `.ent` files, optionally gzipped, or
4-character PDB IDs. `./epath --help` lists every option.

## Output

Each run writes into its own directory under `output/`, such as `output/1ABC/`:

- `*_epath.csv`: one row per scored atom, with its BDamage, Z-score, and
  cluster membership.
- `*_clusters.csv`: one row per reported pathway.
- `*_steps.csv`: one row per step of every reported pathway.
- `*_run.csv`: a single row of run-level results.
- `*_selected.cif`: the selected atoms.
- `*_path01.cif`, `*_path02.cif`, ...: one file per reported pathway.
- `*_model.cif`: the cleaned model with BDamage in the B-factor column.
- `*_zfalloff.png`: the BDamage Z-score falloff plot.
- `*_summary.txt`: the analysis summary.

With several percentiles, BDamage is calculated once, the files above for each
cutoff go in a directory of their own, such as `output/1ABC/percentile_2/`, and
`*_runs.csv` beside them has one row per percentile.

A batch run also writes `epath_runs.csv`, with one row per structure and
percentile. Running a structure again writes a new numbered directory rather
than overwriting the old one.

## Documentation

- [User guide](docs/user-guide.md): accepted input, which atoms are scored, how
  pathways are described, and how to read each output file.
- [Input validation](docs/input-validation.md): the Matthews-coefficient check
  and how far it can be trusted.
- [Design notes](docs/design-notes.md): why some choices are fixed.
- [Differences from RABDAM 2](docs/rabdam-differences.md): what matches and
  what was dropped.

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

## Citations

RABDAM: Shelley KL, Dixon TPE, Brooks-Bartlett JC & Garman EF (2018).
J Appl Cryst 51, 552-559. https://doi.org/10.1107/S1600576718002509

BDamage: Gerstel M, Deane CM & Garman EF (2015).
J Synchrotron Radiat 22, 201-212. https://doi.org/10.1107/S1600577515002131

Bnet and Bnet percentile: Shelley KL & Garman EF (2022).
Nat Commun 13, 1314. https://doi.org/10.1038/s41467-022-28934-0

## License

MIT. See [LICENSE](LICENSE).
