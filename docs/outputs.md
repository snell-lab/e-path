# Reading The Output Files

The files a run writes are listed in the [README](../README.md#output). This
page explains what each one contains.

`*_zfalloff.png` ranks atoms by ascending BDamage Z-score and plots the
negative tail, which is the region the selection is drawn from. The selection
threshold is marked, as is any detected plateau: a run of atoms packed more
densely in Z than the runs on either side of it. Atom density always rises
toward the threshold, so a plateau is only reported where the falloff has a
genuine local shelf rather than merely flattening as it approaches the bulk. A
plateau indicates that many atoms share a protection level.

The summaries report the mean and standard deviation of BDamage for the whole
model, which are the values the Z-scores are derived from, and again for the
selected atoms alone.

`*_model.cif` holds every atom that survived cleaning, so solvent, ions,
ligands, and cofactors are retained for inspection alongside the E-path results
even though they are never scored. Scored atoms carry their BDamage value in
`_atom_site.B_iso_or_equiv` and unscored atoms carry `0.00`, so colouring by
B-factor shows the BDamage landscape without the excluded atoms distorting the
scale. The selected atoms are those with a B-factor above zero and at or below
the reported BDamage threshold, which the file records as
`_epath.bdamage_threshold`.

Every coordinate file is mmCIF. Run notes are written as an `_epath_remark`
loop and machine-readable values (atom counts, the threshold, the pathway
number and class) as an `_epath` category; viewers ignore both.

Both summaries record the BDamage calculation parameters actually used, so a
run made with non-default values is identifiable from its output alone. The
window size is reported as well as the fraction it came from, since it is
derived from the scored atom count rather than set directly.

The terminal and persistent summaries include raw protein Bnet and, when the
input contains resolution metadata, its percentile against the bundled
resolution-matched reference database. If a structure has no usable Asp/Glu
carboxyl-oxygen sites or no resolution, the summaries explain why the
corresponding RABDAM value is unavailable without stopping E-path.

Per-atom BDamage, packing density, and window-averaged B-factors are columns
of the combined `*_epath.csv` output.
