"""Options and path records shared by the E-path output modules."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpathOutputPipelineOptions:
    """Options controlling E-path output files."""

    output_prefix: str | Path | None = None
    write_combined_csv: bool = True
    write_structure_files: bool = True
    write_z_score_plot: bool = True
    write_cluster_csv: bool = True
    write_step_csv: bool = True
    write_run_csv: bool = True
    write_summary_report: bool = True
    # Chains to keep in the coordinate outputs. The analysis itself always
    # runs over every chain; this only trims what the coordinate files show.
    display_chains: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class EpathPipelineOutputFiles:
    """Files written by an E-path pipeline run."""

    combined_csv: Path | None = None
    selected_cif: Path | None = None
    model_cif: Path | None = None
    path_cifs: tuple[Path, ...] = ()
    z_score_falloff: Path | None = None
    cluster_csv: Path | None = None
    step_csv: Path | None = None
    run_csv: Path | None = None
    summary_report: Path | None = None
