# Package Layout

- `xtal`: crystallographic intake and validation. Input resolution, RCSB
  download, file reading, unit-cell and space-group accessors, and
  asymmetric-unit validation. Contains no damage calculation.
- `rabdam`: the from-scratch reimplementation of the RABDAM 2 BDamage and Bnet
  calculations, and nothing else, so it can be compared against RABDAM 2
  directly. `structure` prepares the model, `crystal` rebuilds the crystal
  environment, `packing` counts packing density, `bdamage` scores the atoms,
  and `bnet` calculates Bnet and its resolution-matched percentile.
- `epath`: the pathway analysis built on per-atom BDamage values.
  `epath.analysis` holds the science: selection, clustering, and MST geometry
  produce the clusters, then residue chemistry, topology, and endpoint
  labelling describe each cluster as a classified pathway. `epath.outputs`
  writes the CSV, coordinate, plot, and report files. `epath.cli` parses
  arguments and prints terminal summaries, and `epath.workflow` is the entry
  point that ties the stages together.
- `spatial`: radius neighbour searches backed by SciPy's KD-tree, shared by
  the packing-density and clustering steps.

Dependencies run in one direction only: `epath` uses `rabdam`, `rabdam` uses
`xtal`, and both damage layers use `spatial`. Within `epath`, `outputs`
depends on `analysis` and never the other way round.
