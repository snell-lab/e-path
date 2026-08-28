# Differences From RABDAM 2

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
dataframe restart modes, interactive overwrite prompts, or restartable
dataframe outputs, and it does not produce RABDAM's HTML report, its
per-structure BDamage plots, or its BDamage-coloured coordinate files. E-path
writes its own equivalents instead: the `*_zfalloff.png` plot and the
`*_model.cif` file with BDamage in the B-factor column.

E-path also does not expose RABDAM's atom-selection controls. RABDAM allows
HETATM, nucleic-acid, and per-atom or per-component selection overrides;
E-path fixes the population at protein atoms, because a pathway analysis that
compares clusters between structures requires the scored population to be
defined the same way in every model.
