"""E-path analysis layer, from scored atoms to described pathways.

- `selection` picks the candidate atoms, those with the lowest BDamage.
- `clustering` groups selected atoms by spatial connectivity.
- `pathways` describes each cluster as a path.
- `residues` classifies path residues by side-chain chemistry.
- `cofactors` finds the redox cofactors and the one nearest to each path.
- `geometry` holds the distance and spanning-tree helpers they share.
- `labels` names atoms and residues for the tables and reports.
"""
