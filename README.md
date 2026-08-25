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
Bnet and a resolution-matched Bnet percentile. E-path consumes BDamage only;
Bnet and its percentile are reported alongside the E-path results but do not
change E-path selection or clustering.

## Install

E-path runs directly from a clone of this repository; there is no package to
install. It requires Python 3.10 or newer and the four libraries listed in
`requirements.txt`: Gemmi, NumPy, SciPy, and Matplotlib (for the default
Z-score plot). Clone the repository, then install the libraries however you
prefer — pip, your system package manager, conda — for example:

```bash
git clone https://github.com/Aspartyl/e-path.git
cd e-path
python3 -m pip install -r requirements.txt
./epath --help
```

`./epath` is a short Python script: it checks that the four libraries are
importable (and names any that are missing), adds `src/` to `sys.path`, and
calls the package's `main`, so edits to the source take effect immediately.
Run it from any directory by its full path, add the repository root to your
`PATH`, or run it with an explicit interpreter as `python3 epath ...`.

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

PDB IDs are downloaded from RCSB as mmCIF files and cached in
`downloads/rcsb/` by default, so a repeated run of the same ID works offline.
`--download-dir` changes the cache directory and `--overwrite-download`
fetches the file again even when a cached copy exists.

`--version` and `--citations` print the program version and the papers to
cite, and `-q`/`--quiet` suppresses progress and summary output. Two options
exist only for debugging the BDamage calculation:
`--materialize-translated-block` builds the full 3x3x3 translated crystal
block instead of the fused translate-and-trim path, and `--preview-count N`
prints the first N packing-density and BDamage values to the terminal.

## Output

Single and batch E-path runs write results in `output/` by default, using the
PDB ID or input filename stem as the output prefix. A normal run writes:

- `*_epath.csv`: one row per scored atom, with its BDamage, Z-score, whether
  it was selected, and its cluster membership. Run-level statistics are in the
  summary and cluster geometry is in `*_clusters.csv`, joinable on
  `cluster_id`, so neither is repeated per atom.
- `*_clusters.csv`: one row per reported pathway (eight or more atoms by
  default), holding the pathway table described in
  [docs/pathways.md](docs/pathways.md).
- `*_steps.csv`: one row per step of every reported pathway.
- `*_run.csv`: a single row of run-level results, described in
  [docs/comparing-models.md](docs/comparing-models.md).
- `*_selected.pdb`: selected E-path atoms.
- `*_path01.pdb`, `*_path02.pdb`, ...: one file per reported pathway.
- `*_model.pdb`: the complete cleaned model with BDamage in the B-factor
  column.
- `*_zfalloff.png`: rank-ordered BDamage Z-score falloff.
- `*_summary.txt`: persistent human-readable analysis summary.

A batch run additionally writes `epath_runs.csv` in the output directory, with
one row per structure. `--output-dir` chooses another output directory and
`--output-prefix` sets the prefix for a single structure; `--no-csv`,
`--no-pdb`, `--no-plot`, and `--no-reports` disable the corresponding output
groups. How to read each file is described in [docs/outputs.md](docs/outputs.md).

## Tests

The test suite needs `pytest` in addition to the runtime libraries. Run it
from the repository root:

```bash
python3 -m pytest
```

The tests build a small synthetic crystal rather than downloading structures,
so they run offline in a few seconds. The BDamage stages are checked against
brute-force references built directly on Gemmi's symmetry and cell handling,
and the E-path stages against hand-constructed clusters with known geometry,
chemistry, and endpoints.

## Documentation

The sections above are enough to run E-path. The reasoning behind its choices
and the details of its outputs are in `docs/`:

- [Reading the output files](docs/outputs.md): what each file contains and how
  to read the plot, the summaries, and the coordinate files.
- [Scored atom population](docs/scored-atoms.md): why only protein atoms are
  scored and why that is not configurable.
- [Pathway table](docs/pathways.md): endpoints, step provenance, per-pathway
  coordinate files, and the three-tier classification.
- [Comparing models](docs/comparing-models.md): the run-level table for
  comparing structures across a cohort.
- [Input validation](docs/input-validation.md): the Matthews-coefficient check
  that rejects biological assemblies and warns about truncated models.
- [Differences from RABDAM 2](docs/rabdam-differences.md): what is preserved,
  what is faster, and what was deliberately dropped.
- [Package layout](docs/package-layout.md): the `xtal`, `rabdam`, `epath`, and
  `spatial` packages and their dependency direction.
- [Python API](docs/python-api.md): running the workflow and overriding
  thresholds from Python.

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

## License

MIT. See [LICENSE](LICENSE).
