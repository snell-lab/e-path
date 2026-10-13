"""Crystallographic structure intake and validation.

- `resolver` decides whether an input is a local file or a PDB ID.
- `rcsb` downloads and caches an RCSB entry as mmCIF.
- `reader` reads a coordinate file into atom records and crystal metadata.
- `cell` builds the checked unit cell, space group, and lattice vectors.
- `validate` checks the cell and, with the Matthews coefficient, that the
  model plausibly fills one asymmetric unit.
"""
