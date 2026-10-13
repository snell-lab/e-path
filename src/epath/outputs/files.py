"""Options and path records shared by the E-path output modules.

`epath.pipeline` carries the options as the `output` group of its own, and
`epath.outputs.write` reads them and returns the record of files written,
which the summary report and the command line then list. The two classes live
here, apart from `write`, because `write` imports `summary` and `summary`
needs the record.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpathOutputPipelineOptions:
    """Choose which files an E-path run writes, and where.

    Each file is named by adding a suffix such as `_epath.csv` to the last
    component of `output_prefix`. Without a prefix nothing is written, whatever
    the `write_*` flags say. `write_structure_files` covers every mmCIF file:
    the selected atoms, the cleaned model, and one file per pathway.
    `write_reports` covers the cluster, step and run CSVs and the summary
    report.

    `display_chains` limits the mmCIF files to those chains. The analysis
    always runs over every chain, so the tables and the plot are the same with
    or without it. `epath.pipeline` rejects a chain that is not in the model
    before anything is written.
    """

    output_prefix: Path | None = None
    write_combined_csv: bool = True
    write_structure_files: bool = True
    write_z_score_plot: bool = True
    write_reports: bool = True
    display_chains: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EpathPipelineOutputFiles:
    """The paths of the files an E-path run wrote.

    A field is `None` when its file was not written because its option was
    off. `path_cifs` holds the file of each reported pathway by path number,
    in report order, without the pathways that have no atoms in the displayed
    chains.
    """

    combined_csv: Path | None = None
    selected_cif: Path | None = None
    model_cif: Path | None = None
    path_cifs: dict[int, Path] = field(default_factory=dict[int, Path])
    z_score_falloff: Path | None = None
    cluster_csv: Path | None = None
    step_csv: Path | None = None
    run_csv: Path | None = None
    summary_report: Path | None = None
