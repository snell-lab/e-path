"""E-path output layer.

- `runs` collects one run-level record per structure.
- `write` coordinates every file written for a run.
- `tables` writes the atom, cluster, step, and run CSVs.
- `structures` writes the mmCIF coordinate outputs.
- `plots` draws the Z-score falloff plot.
- `summary` renders the human-readable report.
- `files` holds the options and path records the others share.
"""
