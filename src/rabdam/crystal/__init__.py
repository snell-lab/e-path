"""Crystal environment construction for packing density.

- `symmetry` expands the model by the space-group operations.
- `translate` copies the expanded atoms into neighbouring unit cells.
- `trim` keeps only atoms within reach of the scored atoms.
"""
