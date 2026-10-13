# Comparing Models

Comparing many structures means reading the same run-level values back out
again, so each run records them once in `*_run.csv`: the input, selection
cutoff, scored atom count, atoms selected, the equivalent Z-score and BDamage
threshold, Bnet and its percentile, the connected-component counts, and the
geometry of the largest component. A batch run writes the same rows
combined into `epath_runs.csv`, which is the comparison table for that cohort.
A later batch into the same output directory writes `epath_runs_1.csv`, then
`epath_runs_2.csv`, so the table of an earlier batch is never overwritten.
Rows from separate single-structure runs concatenate to the same thing.

Two of those columns describe components that are never reported as pathways:

- `total_components` counts every connected component the selected atoms form,
  not just the ones large enough to report.
- `largest_component_*` describes the largest component whether or not it
  reached the reporting threshold, with `largest_component_reported` recording
  which. A model where nothing qualified still reports what its connectivity
  analysis found, so a largest component of seven atoms is distinguishable from
  one of two.

The run summary reports the same component count, and names the largest
component when no pathway reached the threshold.

Grouping those rows into cohorts, and taking medians within a group, is left to
whoever assembles the comparison. Which group a structure belongs to is a prior
judgement about its biology rather than something the analysis can determine,
so the software emits the per-structure rows and stops there.
