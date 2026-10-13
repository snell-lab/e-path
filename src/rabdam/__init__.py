"""Reimplementation of the published RABDAM 2 BDamage calculation.

BDamage is from Gerstel et al. (2015), RABDAM from Shelley et al. (2018), and
Bnet from Shelley & Garman (2022). The README lists the full references.

- `pipeline` runs the BDamage stages in order.
- `structure` cleans the model and selects the atoms to score.
- `crystal` rebuilds the surrounding crystal by symmetry and translation.
- `packing` counts each scored atom's packing-density neighbours.
- `bdamage` scores each atom against others of similar packing density.
- `bnet` summarises damage to the whole protein as Bnet.
"""
