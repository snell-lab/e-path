# Pathway Table

Every reported cluster is described as a pathway in `*_clusters.csv` and in the
run summary. The pathway description, the step provenance, and the three-tier
classification below follow the E-path analysis framework of a manuscript that
is currently in preparation; the citation will be added to the README and to
`--citations` when it is published. Each pathway is reported with:

- the residues the path runs through and the protected atoms in each of them,
  written `Chain A - Trp76 (CA, O, CB) -> His87 (CB)`. Chains appear in the
  order of the first residue each contributes, residues in ascending residue
  number, and atoms in the order the connectivity graph reached them.
- the residue count, the node (atom) count, and the total path length.
- the environments the two ends of the path terminate in.
- the class and a structural descriptor.
- the topology, step provenance, and residue-chemistry measurements the class
  was assigned from.

## Endpoints

The endpoints are the two atoms that define the cluster span, which is already
reported as the straight-line end-to-end distance of the path. Each is labelled
with the environment it terminates in, tested in this order:

- `SF4-1`, `FMN`, `HEM-2`: in contact with that cofactor, within 4.0 A.
- `FMN region`: within 8.0 A of that cofactor, so in its environment without
  touching it.
- `Interface region`: within 4.0 A of a different chain.
- `Surface region`: in the least densely packed quarter of the scored atoms.
  Packing density is already calculated for every scored atom, so no separate
  solvent-accessibility calculation is involved.
- `Buried region`: none of the above.

Cofactor sites are taken from the cleaned model, since cofactors are never
scored and so can never lie on a path. Sites sharing a residue name are
numbered by chain and residue number, which is what produces `SF4-1` to
`SF4-7`. Common crystallization and cryoprotection additives are not treated as
cofactors, because terminating beside a glycerol does not make a path an
electron conduit; they are still in the model and still contribute to packing
density. The summaries list every cofactor site found, so an endpoint that
reports no cofactor can be told apart from a model that has none.

## Step Provenance

Each step of a path is one edge of its minimum spanning tree, so the steps
counted here are the same ones the reported length and mean step describe. They
are classified by how they connect residues:

- `intra_residue_edges`: both ends in the same residue, which is side-chain
  shape rather than progress along the path.
- `peptide_edges`: consecutive residues of one chain.
- `tertiary_edges`: the same chain, but residues far apart in sequence.
- `interchain_edges`: different chains.

`peptide_edge_fraction` is the share of between-residue steps that follow the
sequence. It separates two paths that geometry alone reports as identical: a
rigid stretch of backbone, and a path of the same size whose steps jump between
residues that are distant in sequence. The second is the stronger coupling
candidate, and the E-path analysis repeatedly rests on this distinction. Peptide steps are identified by sequence adjacency within a
chain rather than by a modelled backbone bond, so a chain break numbered
consecutively counts as peptide-contiguous.

Step provenance is reported but does not feed the classification.

`*_steps.csv` lists every step individually, with its length, its kind, and the
two atoms it joins. The summary reports the total, mean, and largest step of a
path, which says that a long step exists but not where it falls or how many
others come close to it. Because tunnelling efficiency falls away sharply with
distance, a single long step can limit a whole path, so the individual lengths
are worth having. Steps are listed in the order the spanning tree was built,
which is deterministic but is not a walk from one end of a path to the other.

## Pathway Coordinate Files

`*_selected.pdb` holds every selected atom, including those in components too
small to be reported as pathways. Alongside it, one file per reported pathway
is written as `*_path01.pdb`, `*_path02.pdb`, and so on, each holding only that
pathway's atoms, with its class, descriptor, size, and endpoints recorded in
`REMARK` lines.

Both are sparse subsets, so residues that are not sequential in the source
chain are separated by `TER` records. Without them, viewers such as ChimeraX
treat the subset as one polymer with gaps and draw long dashed
"missing structure" pseudobonds between every selected residue.

The two forms answer different questions. The combined file shows what the
cutoff selected. The per-pathway files let a viewer load and colour each
pathway separately, which is what colouring by class requires. Keeping them
separate avoids encoding a pathway label into the occupancy or B-factor column,
both of which already carry real values. Path numbers are zero padded so the
files sort in path order, and they are suppressed by `--no-pdb` along with the
other coordinate outputs.

## Classification

Paths are classified on the three tiers of the E-path framework: graph
topology, residue composition, and the functional role inferred from them.

Topology comes from the span-to-length ratio and node connectivity. A path
whose span is at least 20% of its total MST length is `linear`; connectivity is
counted between residues rather than atoms, and a path where at least 25% of
residues contact three or more other residues of the same cluster is
`branched`; anything else is `clustered`. Connectivity is deliberately not
counted between atoms: the atoms of one side chain are all mutually close and
an aromatic ring contains junctions of its own, so atom connectivity would
mostly report the shape of the residues rather than a fork in the path.

Composition assigns each residue exactly one chemistry class, so the reported
fractions sum to one. Histidine and tyrosine count as aromatic rather than
polar, because the analysis treats aromatic side chains as the redox-active
set. A class is enriched when it is at least 1.5 times more common in the path
than in the scored population of the same model, so enrichment is measured
against the structure being analysed rather than a fixed reference.

The role is then assigned conservatively:

- `conductor`: anchored at a cofactor, linear, at least 10% aromatic or sulfur
  residues, and less than 30% charged residues.
- `modulator`: anchored at a cofactor but failing one of those tests, which is
  the case for compact, branched, or strongly charged networks.
- `reservoir`: no cofactor anchor, but at least 20% aromatic or sulfur
  residues. An unanchored path needs stronger chemical evidence than an
  anchored one before it is called redox-relevant, because nothing else ties it
  to a redox centre.
- `structural`: no cofactor anchor and no redox-active enrichment.

The framework fixes these criteria but not the numeric thresholds that
separate the classes. The defaults above were chosen so that the manuscript's
pathway assignments are reproduced: replaying its 22 paths through this
classifier returns the assigned class for 19 of them. The three that differ
are all cases where the manuscript's endpoint label carries information that
is not in the coordinates, such as a cofactor that was not modelled, a
hand-named residue hub, or adjacency to another reported path. The thresholds
are not exposed on the command line; they are fields of `PathAnalysisOptions`
and can be overridden from Python, as shown in [python-api.md](python-api.md).

Structural descriptors are generated from the same measurements, naming what
the path is mostly made of, its topology, and the classes it is enriched in,
which gives phrasings such as `Hydrophobic corridor with aromatic/sulfur
enrichment` or `Single aromatic hotspot`.
