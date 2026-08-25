# Input Validation

BDamage compares an atom's B-factor with atoms at a similar local packing
density, and packing density is counted in a crystal environment rebuilt from
the model by symmetry expansion and unit-cell translation. That rebuilt
environment is only correct when the model fills exactly one asymmetric unit.

Symmetry expansion cannot detect a violation on its own, because it faithfully
rebuilds a crystal from whatever the file contains. E-path therefore validates
the asymmetric-unit content against the unit-cell volume using the Matthews
coefficient before any calculation runs:

- A model that cannot physically fit in the cell is rejected. This usually
  means a biological-assembly file was supplied instead of the asymmetric unit.
- A Matthews coefficient outside roughly 1.5 to 4.5 A^3/Da is reported as a
  warning. A high value suggests chains are missing, which makes packing
  densities too low at the missing interfaces; a low value suggests the model
  holds more than one asymmetric unit.

The space group, unit-cell volume, Matthews coefficient, and estimated solvent
content are recorded in every run summary. Only protein and nucleic-acid atoms
contribute to the modelled mass, as for a conventional Matthews calculation.
Hydrogens are removed before scoring, so the reported solvent content is a
slight overestimate.
