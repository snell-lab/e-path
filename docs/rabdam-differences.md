# Differences from RABDAM 2

E-path contains a calculation-compatible rewrite of RABDAM 2, not a
feature-for-feature clone. Under normal default use, `PD`, `AVRG_BF`, and
`BDAM` are intended to match RABDAM 2, because the rewrite keeps the same
symmetry expansion, neighbouring unit-cell translation, strict 7.0 Å
packing-density counts, and BDamage sliding windows. On large structures it can
finish more than 80 times faster, and it does not need cctbx, CCP4/PDBCUR,
pandas, or requests.

It drops some RABDAM 2 application features:

- The legacy text input file, the `-f`, `-i`, and `-r` dataframe restart modes,
  interactive overwrite prompts, and restartable dataframe outputs.
- RABDAM's HTML report, per-structure BDamage plots, and BDamage-coloured
  coordinate files. `*_zfalloff.png` and `*_model.cif` take their place.
- The optional `filter` setting, which stops a run when the model fails the
  Bnet eligibility criteria of Shelley & Garman (2022). As in a default RABDAM
  run, Bnet and its percentile are reported without those checks.
- The atom-selection controls. E-path always scores protein atoms, for
  reasons given in the
  [design notes](design-notes.md#why-only-protein-atoms-are-scored).
