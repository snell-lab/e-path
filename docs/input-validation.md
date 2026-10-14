# Input Validation

BDamage compares an atom's B-factor with atoms at a similar local packing
density, and packing density is counted in a crystal environment rebuilt from
the model by symmetry expansion and unit-cell translation. That environment is
only correct when the model fills exactly one asymmetric unit.

Symmetry expansion rebuilds a crystal from whatever the file contains without
complaint. Two inputs commonly break it:

- A file holding more than one asymmetric unit, such as a biological assembly.
  Symmetry expansion stacks the surplus copies on top of each other, so packing
  densities, and therefore BDamage, are too high wherever they meet.
- A truncated model, such as one chain extracted from an oligomer. Atoms buried
  against the removed subunit are counted as though they face solvent, so their
  packing density is too low at the missing interfaces.

## Matthews Coefficient

The Matthews coefficient compares the modelled mass with the unit-cell volume
and gives a coarse check for both surplus and missing chains.

- A model that cannot physically fit in the cell is rejected. This usually
  means a biological-assembly file was supplied instead of the asymmetric unit.
- A Matthews coefficient outside roughly 1.5 to 4.5 Å³/Da is reported as a
  warning. A high value suggests chains are missing; a low value suggests the
  model holds more than one asymmetric unit.

The check is coarse. A model missing one chain of four can still fall inside
the advisory range, and so can an assembly saved with its crystal cell when the
crystal has a high solvent content. A run without warnings does not prove the
model is one complete asymmetric unit.

The space group, unit-cell volume, Matthews coefficient, and estimated solvent
content are recorded in every run summary. The mass counts protein and
nucleic-acid atoms only, weighted by occupancy, with missing hydrogens added
back. Residues that were never modelled, such as disordered loops, are missing
from it, so the solvent content of a model with large gaps is an overestimate.
