"""The complete BDamage workflow checked against brute-force references."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from support import (
    approx,
    brute_force_packing_density,
    make_metadata,
    make_prepared_atom,
    reference_bdamage,
    write_synthetic_pdb,
)

from rabdam.bdamage import bdamage_scores_as_tuple
from rabdam.packing import packing_density_counts_as_tuple
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
)
from rabdam.structure.prepare import prepare_structure
from rabdam.workflow import (
    BDamageWorkflowError,
    BDamageWorkflowOptions,
    BDamageWorkflowResult,
    calculate_bdamage_for_prepared_structure,
    calculate_bdamage_for_structure_data,
    validate_crystal_for_prepared_structure,
)
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input

EXPECTED_STAGES = [
    "Preparing structure",
    "Validating crystal",
    "Expanding crystallographic symmetry",
    "Building trimmed neighbour block",
    "Calculating packing density",
    "Calculating BDamage scores",
]


def _check_against_references(
    result: BDamageWorkflowResult, *, translation_range: int = 1
) -> None:
    """Assert the workflow's densities and BDamage match the brute-force references."""
    prepared = result.prepared_structure
    expected_density = brute_force_packing_density(
        cleaned_atoms=prepared.cleaned_atoms,
        selected_atoms=prepared.selected_atoms,
        cell=result.crystal.gemmi_cell,
        space_group=result.crystal.space_group,
        translation_range=translation_range,
    )
    assert (
        list(packing_density_counts_as_tuple(result.packing_density_result))
        == expected_density
    )

    selected = prepared.selected_atoms
    expected_bdamage = reference_bdamage(
        b_factors=[a.record.b_factor for a in selected],
        packing_densities=expected_density,
        serials=[a.record.atom_serial or 0 for a in selected],
        window_size=result.window_size,
    )
    assert bdamage_scores_as_tuple(result.bdamage_score_result) == approx(
        expected_bdamage
    )


def test_workflow_matches_brute_force_symmetry_expansion(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))
    stages: list[str] = []

    def stage_runner(label: str, callback: Callable[[], Any]) -> Any:
        stages.append(label)
        return callback()

    result = calculate_bdamage_for_structure_data(data, stage_runner=stage_runner)

    assert stages == EXPECTED_STAGES
    assert result.window_size == 11
    assert result.translated_block is None
    assert result.options == BDamageWorkflowOptions()
    assert result.crystal.operation_count == 2
    assert result.crystal.content is not None and result.crystal.warnings == ()
    assert (
        len(result.bdamage_score_result.atom_results)
        == len(result.prepared_structure.selected_atoms)
        == 77
    )
    for atom, score in zip(
        result.prepared_structure.selected_atoms,
        result.bdamage_score_result.atom_results,
        strict=True,
    ):
        assert atom.record.source_atom_index == score.source_atom_index
        assert score.b_factor == atom.record.b_factor
        assert score.bdamage == approx(score.b_factor / score.average_b_factor)
    _check_against_references(result)


def test_packing_density_is_independent_of_the_trimming_box(
    synthetic_pdb: Path,
) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))
    fused = calculate_bdamage_for_structure_data(data)
    materialized = calculate_bdamage_for_structure_data(
        data, workflow_options=BDamageWorkflowOptions(materialize_translated_block=True)
    )

    assert materialized.translated_block is not None
    assert len(materialized.translated_block.atoms) == 27 * len(
        fused.symmetry_expanded_structure.atoms
    )
    assert packing_density_counts_as_tuple(
        materialized.packing_density_result
    ) == packing_density_counts_as_tuple(fused.packing_density_result)


def test_translation_range_zero_and_wider_threshold(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))
    options = BDamageWorkflowOptions(
        translation_range=0,
        packing_density_threshold=5.0,
        window_size_fraction=0.2,
        minimum_window_size=3,
    )

    result = calculate_bdamage_for_structure_data(data, workflow_options=options)

    assert result.window_size == 15
    prepared = result.prepared_structure
    expected = brute_force_packing_density(
        cleaned_atoms=prepared.cleaned_atoms,
        selected_atoms=prepared.selected_atoms,
        cell=result.crystal.gemmi_cell,
        space_group=result.crystal.space_group,
        threshold=5.0,
        translation_range=0,
    )
    assert (
        list(packing_density_counts_as_tuple(result.packing_density_result)) == expected
    )


def test_file_specific_frame_gives_correct_neighbours(tmp_path: Path) -> None:
    # With these SCALE records Gemmi keeps a rotated Cartesian frame. Lattice
    # translations must be taken in that frame, or the crystal is rebuilt from
    # mismatched axes and the neighbour counts are wrong.
    scale = (
        "SCALE1      0.000000  0.066667  0.000000        0.00000",
        "SCALE2      0.000000  0.000000  0.062500        0.00000",
        "SCALE3      0.058824  0.000000  0.000000        0.00000",
    )
    path = write_synthetic_pdb(
        tmp_path / "rotated.pdb",
        cell=(15.0, 16.0, 17.0, 90.0, 90.0, 90.0),
        space_group="P 1 21 1",
        scale_lines=scale,
    )
    data = read_structure(resolve_structure_input(str(path)))

    result = calculate_bdamage_for_structure_data(data)

    assert result.crystal.operation_count == 2
    _check_against_references(result)


def test_workflow_rejects_structures_that_cannot_be_scored() -> None:
    def prepared_with(selected: tuple[PreparedAtom, ...]) -> PreparedStructure:
        return PreparedStructure(
            cleaned_atoms=selected,
            selected_atoms=selected,
            metadata=make_metadata(),
            removed=RemovedAtomCounts(0, 0, 0, 0, 0),
        )

    with pytest.raises(BDamageWorkflowError, match="no selected atoms"):
        calculate_bdamage_for_prepared_structure(prepared_structure=prepared_with(()))

    five = tuple(
        make_prepared_atom(source_atom_index=i, atom_serial=i + 1, x=float(i))
        for i in range(5)
    )
    with pytest.raises(BDamageWorkflowError, match="window size is larger"):
        calculate_bdamage_for_prepared_structure(prepared_structure=prepared_with(five))

    with pytest.raises(BDamageWorkflowError):
        calculate_bdamage_for_prepared_structure(
            prepared_structure=prepared_with(five),
            options=BDamageWorkflowOptions(translation_range=-1),
        )


def test_only_macromolecule_atoms_count_toward_the_matthews_mass(
    synthetic_pdb: Path,
) -> None:
    prepared = prepare_structure(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )

    crystal = validate_crystal_for_prepared_structure(prepared)

    assert crystal.content is not None
    assert (
        crystal.content.macromolecule_atom_count == len(prepared.selected_atoms) == 77
    )
    # The synthetic model has no hydrogens, so every residue gains its
    # tabulated chain hydrogens.
    assert crystal.content.implicit_hydrogen_mass_dalton > 0.0
    assert crystal.content.nucleic_acid_mass_dalton == 0.0
