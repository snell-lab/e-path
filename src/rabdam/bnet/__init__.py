"""Bnet, a whole-protein damage metric built from BDamage scores.

- `workflow` runs the Bnet steps and records why any were unavailable.
- `calculate` computes raw protein Bnet.
- `sites` selects the damage-sensitive protein atoms.
- `metric` computes Bnet from their BDamage values.
- `reference` loads the reference database of Bnet values for PDB entries.
- `percentile` ranks Bnet against resolution-matched references.
"""
