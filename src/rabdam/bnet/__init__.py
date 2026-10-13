"""Bnet, a whole-protein damage metric built from BDamage scores.

Bnet and its percentile are from Shelley & Garman (2022). The README lists the
full reference.

- `pipeline` runs the Bnet steps and records why any were unavailable.
- `metric` selects the damage-sensitive protein atoms and computes protein
  Bnet from their BDamage values.
- `reference` loads the reference database of Bnet values for PDB entries.
- `percentile` ranks Bnet against resolution-matched references.
"""
