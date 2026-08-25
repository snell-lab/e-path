# Python API

The `./epath` launcher is a thin wrapper; everything it does is available from
Python once `src/` is on `sys.path`. A run is three calls: resolve and read
the input, calculate BDamage, then run the E-path analysis:

```python
from pathlib import Path

from epath.workflow import EpathWorkflowOptions, run_epath_workflow
from epath.analysis.pathways import PathAnalysisOptions
from epath.outputs.files import EpathOutputWorkflowOptions
from rabdam.workflow import calculate_bdamage_for_structure_data
from xtal.rcsb import download_rcsb_mmcif
from xtal.reader import read_structure
from xtal.resolver import StructureSourceType, resolve_structure_input

resolved = resolve_structure_input("1LYZ")
if resolved.source_type is StructureSourceType.RCSB_ID:
    resolved = download_rcsb_mmcif(resolved)
structure = read_structure(resolved)

bdamage = calculate_bdamage_for_structure_data(structure)

options = EpathWorkflowOptions(
    paths=PathAnalysisOptions(linear_ratio=0.25),
    output=EpathOutputWorkflowOptions(output_prefix=Path("output") / "1LYZ"),
)
result, files = run_epath_workflow(bdamage, options=options)

for report in result.path_reports:
    print(report.path_class, report.start_endpoint, report.end_endpoint)
```

`EpathWorkflowResult` holds the scored atoms, the selection, the clusters and
their statistics, one `PathReport` per reported pathway, the cofactor sites,
and the run record that becomes `*_run.csv`; `EpathWorkflowOutputFiles` lists
the files written. Every option group has a frozen dataclass with the same
defaults as the CLI: `BDamageWorkflowOptions` for the calculation,
`EpathSelectionWorkflowOptions` and `EpathClusterWorkflowOptions` for the
selection and clustering, `PathAnalysisOptions` for the classification
thresholds, and `EpathOutputWorkflowOptions` for which files to write.
