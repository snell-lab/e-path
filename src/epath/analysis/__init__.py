"""E-path analysis layer.

These modules turn atom-level BDamage output into described pathways, in
dependency order: `atoms`, `geometry`, and `residues` hold the primitives;
`selection` picks candidate atoms from the negative Z-score tail;
`clustering` groups them by spatial connectivity; and `topology`,
`endpoints`, and `pathways` describe each cluster as a classified path.
Nothing here writes files — that is the `epath.outputs` layer.
"""
