# Python API

The `./epath` launcher is a thin wrapper; everything it does is available from
Python once `src/` is on `sys.path`. A run is three calls: resolve and read
the input, calculate BDamage, then run the E-path analysis:

```python
from pathlib import Path

from epath.pipeline import (
    EpathClusterPipelineOptions,
    EpathPipelineOptions,
    run_epath_pipeline,
)
from epath.outputs.files import EpathOutputPipelineOptions
from rabdam.pipeline import run_bdamage_pipeline
from xtal.rcsb import download_rcsb_mmcif
from xtal.reader import read_structure
from xtal.resolver import StructureSourceType, resolve_structure_input

resolved = resolve_structure_input("1LYZ")
if resolved.source_type is StructureSourceType.RCSB_ID:
    resolved = download_rcsb_mmcif(resolved)
structure = read_structure(resolved)

bdamage = run_bdamage_pipeline(structure)

options = EpathPipelineOptions(
    clustering=EpathClusterPipelineOptions(minimum_cluster_size=10),
    output=EpathOutputPipelineOptions(output_prefix=Path("output") / "1LYZ"),
)
result, files = run_epath_pipeline(bdamage, options=options)

for report in result.path_reports:
    print(report.endpoint_summary, report.anchor_summary)
```

`EpathPipelineResult` holds the selection, with every scored atom and the ones
selected from them, the clusters and their statistics, one `PathReport` per
reported pathway, the cofactor sites, the run record that becomes `*_run.csv`,
and the BDamage and Bnet results the analysis rests on. A scored atom is the
atom record from `xtal.reader` and the BDamage result from `rabdam`, with its
Z-score. `EpathPipelineOutputFiles` lists the files written. Every option
group has a frozen dataclass with the same defaults as the CLI:
`BDamagePipelineOptions` for the calculation, `EpathSelectionPipelineOptions`
and `EpathClusterPipelineOptions` for the selection and clustering, and
`EpathOutputPipelineOptions` for which files to write.
