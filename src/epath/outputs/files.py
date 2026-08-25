"""Options and path records shared by the E-path output modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpathOutputWorkflowOptions:
    """Options controlling E-path output files."""

    output_prefix: str | Path | None = None
    write_combined_csv: bool = True
    write_structure_files: bool = True
    write_z_score_plot: bool = True
    write_cluster_csv: bool = True
    write_step_csv: bool = True
    write_run_csv: bool = True
    write_summary_report: bool = True


@dataclass(frozen=True, slots=True)
class EpathWorkflowOutputFiles:
    """Files written by an E-path workflow run."""

    combined_csv: Path | None = None
    selected_pdb: Path | None = None
    model_pdb: Path | None = None
    path_pdbs: tuple[Path, ...] = ()
    z_score_falloff: Path | None = None
    cluster_csv: Path | None = None
    step_csv: Path | None = None
    run_csv: Path | None = None
    summary_report: Path | None = None
