"""E-path analysis layer, from scored atoms to described pathways.

- `atoms` builds E-path atom records from BDamage output.
- `selection` picks candidate atoms from the negative Z-score tail.
- `clustering` groups selected atoms by spatial connectivity.
- `pathways` describes each cluster as a path.
- `residues` classifies path residues by side-chain chemistry.
- `topology` measures whether a path is linear, branched, or clustered.
- `endpoints` relates path ends to cofactors, interfaces, and the surface.
- `geometry` holds the distance and spanning-tree helpers they share.
"""
