"""Reimplementation of the published RABDAM BDamage calculation.

- `workflow` runs the BDamage stages in order.
- `structure` cleans the model and selects the atoms to score.
- `crystal` rebuilds the surrounding crystal by symmetry and translation.
- `packing` counts each scored atom's packing-density neighbours.
- `bdamage` scores each atom against others of similar packing density.
- `bnet` summarises damage to the whole protein as Bnet.
"""
