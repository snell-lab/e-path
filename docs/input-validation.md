# Input Validation

BDamage compares an atom's B-factor with atoms at a similar local packing
density, and packing density is counted in a crystal environment rebuilt from
the model by symmetry expansion and unit-cell translation. That rebuilt
environment is only correct when the model fills exactly one asymmetric unit,
so E-path expects the deposited asymmetric-unit file of a crystal structure,
or the refined model it came from.

Symmetry expansion rebuilds a crystal from whatever the file contains without
complaint. Two inputs commonly break it.

- A file holding more than one asymmetric unit, such as a biological assembly.
  Symmetry expansion stacks the surplus copies on top of each other, so
  packing densities, and therefore BDamage, are too high wherever they meet.
- A truncated model, such as one chain extracted from an oligomer. Atoms
  buried against the removed subunit are counted as though they face solvent,
  so their packing density is too low at the missing interfaces.

## Files without a cell

Biological-assembly files downloaded from the RCSB as mmCIF carry no unit cell
or space group, and nor do NMR and cryo-EM models. They are rejected for
missing crystal metadata, and the error names these as the usual causes and
asks for the deposited asymmetric-unit file of a crystal structure. RCSB
assembly files in PDB format hold each copy as a separate model, and the
reader accepts only single-model files.

## Matthews coefficient

The Matthews coefficient compares the modelled mass with the unit-cell volume
and gives a coarse check for both surplus and missing chains.

- A model that cannot physically fit in the cell is rejected. This usually
  means a biological-assembly file was supplied instead of the asymmetric unit.
- A Matthews coefficient outside roughly 1.5 to 4.5 A^3/Da is reported as a
  warning. A high value suggests chains are missing; a low value suggests the
  model holds more than one asymmetric unit.

The check is coarse. A model missing one chain of four can still fall inside
the advisory range, and so can an assembly saved with its crystal cell when
the crystal has a high solvent content. A run without warnings does not prove
the model is one complete asymmetric unit.

The space group, unit-cell volume, Matthews coefficient, and estimated solvent
content are recorded in every run summary. Only protein and nucleic-acid atoms
contribute to the modelled mass, as for a conventional Matthews calculation.
The mass is taken from the model as read, before the atoms that cannot be scored
are removed. Each atom counts in proportion to its occupancy, so the alternate
conformers of a site add up to one atom. Hydrogens the model lacks are added
back from each residue's tabulated hydrogen count. The solvent content uses a
partial specific volume of 0.74 cm^3/g for protein and 0.50 cm^3/g for nucleic
acid (Kantardjieff & Rupp, 2003). Residues that were never modelled, such as
disordered loops, are still missing from the mass, so the solvent content of a
model with large gaps is an overestimate.
