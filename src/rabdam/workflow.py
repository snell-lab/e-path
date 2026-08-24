"""Orchestrate the end-to-end BDamage workflow."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from rabdam.bdamage.score import (
    BDamageScoreResult,
    calculate_bdamage_scores_for_structure,
)
from rabdam.crystal.symmetry import (
    SymmetryExpandedStructure,
    expand_prepared_structure_by_symmetry,
)
from rabdam.crystal.translate import (
    TranslatedCrystalBlock,
    translate_expanded_unit_cell,
)
from rabdam.crystal.trim import (
    TrimmedNeighbourBlock,
    trim_expanded_unit_cell_for_bdamage,
)
from rabdam.packing.density import (
    PackingDensityResult,
    calculate_bdamage_packing_density,
)
from rabdam.structure.models import PreparedStructure, StructurePreparationOptions
from rabdam.structure.prepare import prepare_structure
from xtal.reader import StructureData
from xtal.validate import ValidatedCrystal, validate_crystal

T = TypeVar("T")


class WorkflowStageRunner(Protocol):
    """Run a labelled workflow callback with caller-defined instrumentation."""

    def __call__(self, label: str, callback: Callable[[], T]) -> T:
        """Run a labelled callback and return its result."""
        ...


class BDamageWorkflowError(ValueError):
    """Report invalid input to the end-to-end BDamage workflow."""


@dataclass(frozen=True, slots=True)
class BDamageWorkflowOptions:
    """Configure the scientific and memory behavior of BDamage calculation."""

    # These scientific defaults match RABDAM 2 so the resulting scores remain
    # directly comparable.
    packing_density_threshold: float = 7.0
    window_size_fraction: float = 0.02
    minimum_window_size: int = 11
    translation_range: int = 1
    materialize_translated_block: bool = False


@dataclass(frozen=True, slots=True)
class BDamageWorkflowResult:
    """Store every materialized result from the BDamage workflow."""

    prepared_structure: PreparedStructure
    crystal: ValidatedCrystal
    symmetry_expanded_structure: SymmetryExpandedStructure
    translated_block: TranslatedCrystalBlock | None
    trimmed_block: TrimmedNeighbourBlock
    packing_density_result: PackingDensityResult
    bdamage_score_result: BDamageScoreResult
    options: BDamageWorkflowOptions
    window_size: int


def calculate_bdamage_for_structure_data(
    structure_data: StructureData,
    *,
    workflow_options: BDamageWorkflowOptions | None = None,
    preparation_options: StructurePreparationOptions | None = None,
    stage_runner: WorkflowStageRunner | None = None,
) -> BDamageWorkflowResult:
    """Prepare structure data and calculate per-atom BDamage scores.

    Args:
        structure_data: Parsed atoms and crystallographic metadata.
        workflow_options: Scientific calculation settings.
        preparation_options: Atom cleaning and selection settings.
        stage_runner: Optional wrapper for timing or reporting named stages.

    Returns:
        Prepared atoms, crystal-neighbour data, packing densities, and BDamage
        scores with the effective options and window size.
    """
    prepared_structure = _run_workflow_stage(
        "Preparing structure",
        lambda: prepare_structure(
            structure_data,
            options=preparation_options,
        ),
        stage_runner=stage_runner,
    )

    return calculate_bdamage_for_prepared_structure(
        prepared_structure=prepared_structure,
        options=workflow_options,
        stage_runner=stage_runner,
    )


def calculate_bdamage_for_prepared_structure(
    *,
    prepared_structure: PreparedStructure,
    options: BDamageWorkflowOptions | None = None,
    stage_runner: WorkflowStageRunner | None = None,
) -> BDamageWorkflowResult:
    """Calculate per-atom BDamage scores for a prepared structure.

    The workflow expands crystallographic symmetry, builds a trimmed translated
    neighbour cloud, counts neighbours within the configured radius, and
    normalizes B-factors within packing-density windows.

    Raises:
        BDamageWorkflowError: If options are invalid, no atoms are selected, or
            the calculated window is larger than the selection.
    """
    if options is None:
        options = BDamageWorkflowOptions()

    validate_workflow_options(options)

    selected_atom_count = len(prepared_structure.selected_atoms)
    if selected_atom_count == 0:
        raise BDamageWorkflowError(
            "Cannot calculate BDamage for a prepared structure with no selected atoms."
        )

    window_size = bdamage_window_size_from_fraction(
        atom_count=selected_atom_count,
        window_size_fraction=options.window_size_fraction,
        minimum_window_size=options.minimum_window_size,
    )

    if window_size > selected_atom_count:
        raise BDamageWorkflowError(
            "Calculated BDamage window size is larger than the number of selected atoms: "
            f"window_size={window_size!r}, selected_atom_count={selected_atom_count!r}."
        )

    crystal = _run_workflow_stage(
        "Validating crystal",
        lambda: validate_crystal_for_prepared_structure(prepared_structure),
        stage_runner=stage_runner,
    )
    symmetry_expanded_structure = _run_workflow_stage(
        "Expanding crystallographic symmetry",
        lambda: expand_prepared_structure_by_symmetry(
            prepared_structure,
            crystal=crystal,
        ),
        stage_runner=stage_runner,
    )
    # The fused trimming path avoids allocating all 27 translated cells;
    # materialization exists only for diagnostics.
    translated_block = (
        _run_workflow_stage(
            "Translating neighbouring unit cells",
            lambda: translate_expanded_unit_cell(
                symmetry_expanded_structure,
                translation_range=options.translation_range,
            ),
            stage_runner=stage_runner,
        )
        if options.materialize_translated_block
        else None
    )
    trimmed_block = _run_workflow_stage(
        "Building trimmed neighbour block",
        lambda: trim_expanded_unit_cell_for_bdamage(
            expanded_structure=symmetry_expanded_structure,
            prepared_structure=prepared_structure,
            padding=options.packing_density_threshold,
            translation_range=options.translation_range,
        ),
        stage_runner=stage_runner,
    )
    packing_density_result = _run_workflow_stage(
        "Calculating packing density",
        lambda: calculate_bdamage_packing_density(
            prepared_structure=prepared_structure,
            trimmed_block=trimmed_block,
            packing_density_threshold=options.packing_density_threshold,
        ),
        stage_runner=stage_runner,
    )
    bdamage_score_result = _run_workflow_stage(
        "Calculating BDamage scores",
        lambda: calculate_bdamage_scores_for_structure(
            prepared_structure=prepared_structure,
            packing_density_result=packing_density_result,
            window_size=window_size,
        ),
        stage_runner=stage_runner,
    )

    return BDamageWorkflowResult(
        prepared_structure=prepared_structure,
        crystal=crystal,
        symmetry_expanded_structure=symmetry_expanded_structure,
        translated_block=translated_block,
        trimmed_block=trimmed_block,
        packing_density_result=packing_density_result,
        bdamage_score_result=bdamage_score_result,
        options=options,
        window_size=window_size,
    )


def validate_crystal_for_prepared_structure(
    prepared_structure: PreparedStructure,
) -> ValidatedCrystal:
    """Validate the crystal metadata and asymmetric-unit content of a model.

    Only protein and nucleic-acid atoms contribute to the modelled mass, as
    they do for a conventional Matthews calculation. Solvent, ions, and other
    heteroatoms are excluded even when they are scored.
    """
    macromolecule_elements = tuple(
        atom.record.element
        for atom in prepared_structure.cleaned_atoms
        if atom.is_protein or atom.is_nucleic_acid
    )
    hydrogens_excluded = not any(
        atom.is_hydrogen for atom in prepared_structure.cleaned_atoms
    )

    return validate_crystal(
        prepared_structure.metadata,
        macromolecule_elements=macromolecule_elements,
        hydrogens_excluded=hydrogens_excluded,
    )


def bdamage_window_size_from_fraction(
    *,
    atom_count: int,
    window_size_fraction: float = 0.02,
    minimum_window_size: int = 11,
) -> int:
    """Calculate an odd BDamage window size from an atom-count fraction.

    The rounded fractional size is increased to the next odd integer and then
    bounded below by `minimum_window_size`.

    Raises:
        BDamageWorkflowError: If the atom count, fraction, or minimum is invalid.
    """
    if type(atom_count) is not int or atom_count <= 0:
        raise BDamageWorkflowError(
            f"atom_count must be a positive integer, got {atom_count!r}."
        )

    if not math.isfinite(window_size_fraction) or not 0 < window_size_fraction < 1:
        raise BDamageWorkflowError(
            "window_size_fraction must be a finite float in the range 0 to 1, "
            f"got {window_size_fraction!r}."
        )

    if type(minimum_window_size) is not int or minimum_window_size <= 0:
        raise BDamageWorkflowError(
            "minimum_window_size must be a positive integer, "
            f"got {minimum_window_size!r}."
        )

    window_size = int(round(atom_count * window_size_fraction, 0))
    # A centered window needs a single middle atom; rounding upward preserves
    # at least the requested fraction.
    if window_size % 2 == 0:
        window_size += 1

    window_size = max(window_size, minimum_window_size)

    return window_size


def validate_workflow_options(options: BDamageWorkflowOptions) -> None:
    """Validate all BDamage workflow options.

    Raises:
        BDamageWorkflowError: If a numeric range or option type is invalid.
    """
    if (
        not math.isfinite(options.packing_density_threshold)
        or options.packing_density_threshold <= 0
    ):
        raise BDamageWorkflowError(
            "packing_density_threshold must be a finite positive number, "
            f"got {options.packing_density_threshold!r}."
        )

    if (
        not math.isfinite(options.window_size_fraction)
        or not 0 < options.window_size_fraction < 1
    ):
        raise BDamageWorkflowError(
            "window_size_fraction must be a finite float in the range 0 to 1, "
            f"got {options.window_size_fraction!r}."
        )

    if type(options.minimum_window_size) is not int or options.minimum_window_size <= 0:
        raise BDamageWorkflowError(
            "minimum_window_size must be a positive integer, "
            f"got {options.minimum_window_size!r}."
        )

    if type(options.translation_range) is not int or options.translation_range < 0:
        raise BDamageWorkflowError(
            "translation_range must be a non-negative integer, "
            f"got {options.translation_range!r}."
        )

    if type(options.materialize_translated_block) is not bool:
        raise BDamageWorkflowError(
            "materialize_translated_block must be a boolean, "
            f"got {options.materialize_translated_block!r}."
        )


def _run_workflow_stage(
    label: str,
    callback: Callable[[], T],
    *,
    stage_runner: WorkflowStageRunner | None,
) -> T:
    if stage_runner is None:
        return callback()

    return stage_runner(label, callback)
