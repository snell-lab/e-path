"""The complete BDamage pipeline checked against brute-force references."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from support import (
    PdbAtom,
    approx,
    brute_force_packing_density,
    make_metadata,
    make_prepared_atom,
    reference_bdamage,
    synthetic_atoms,
    write_synthetic_pdb,
)

from rabdam.bdamage import bdamage_scores_as_tuple
from rabdam.pipeline import (
    BDamagePipelineError,
    BDamagePipelineOptions,
    BDamagePipelineResult,
    run_bdamage_pipeline,
    run_bdamage_pipeline_from_prepared,
    validate_crystal_for_prepared_structure,
)
from rabdam.structure.models import (
    PreparedAtom,
    PreparedStructure,
    RemovedAtomCounts,
)
from rabdam.structure.prepare import classify_atom, prepare_structure
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
    result: BDamagePipelineResult, *, translation_range: int = 1
) -> None:
    """Assert the pipeline's densities and BDamage match the brute-force references."""
    prepared = result.prepared_structure
    expected_density = brute_force_packing_density(
        cleaned_atoms=prepared.cleaned_atoms,
        selected_atoms=prepared.selected_atoms,
        cell=result.crystal.gemmi_cell,
        space_group=result.crystal.space_group,
        translation_range=translation_range,
    )
    assert list(result.packing_densities) == expected_density

    selected = prepared.selected_atoms
    expected_bdamage = reference_bdamage(
        b_factors=[a.record.b_factor for a in selected],
        packing_densities=expected_density,
        serials=[a.record.atom_serial or 0 for a in selected],
        window_size=result.window_size,
    )
    assert bdamage_scores_as_tuple(result.bdamage_results) == approx(expected_bdamage)


def test_pipeline_matches_brute_force_symmetry_expansion(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))
    stages: list[str] = []

    def stage_runner(label: str, callback: Callable[[], Any]) -> Any:
        stages.append(label)
        return callback()

    result = run_bdamage_pipeline(data, stage_runner=stage_runner)

    assert stages == EXPECTED_STAGES
    assert result.window_size == 11
    assert result.options == BDamagePipelineOptions()
    assert result.crystal.operation_count == 2
    assert result.crystal.content is not None and result.crystal.warnings == ()
    assert (
        len(result.bdamage_results)
        == len(result.prepared_structure.selected_atoms)
        == 77
    )
    for atom, score in zip(
        result.prepared_structure.selected_atoms,
        result.bdamage_results,
        strict=True,
    ):
        assert atom.record.source_atom_index == score.source_atom_index
        assert score.bdamage == approx(atom.record.b_factor / score.average_b_factor)
    _check_against_references(result)


def test_translation_range_zero_and_wider_threshold(synthetic_pdb: Path) -> None:
    data = read_structure(resolve_structure_input(str(synthetic_pdb)))
    options = BDamagePipelineOptions(
        translation_range=0,
        packing_density_threshold=5.0,
        window_size_fraction=0.2,
        minimum_window_size=3,
    )

    result = run_bdamage_pipeline(data, options=options)

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
    assert list(result.packing_densities) == expected


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

    result = run_bdamage_pipeline(data)

    assert result.crystal.operation_count == 2
    _check_against_references(result)


def test_pipeline_rejects_structures_that_cannot_be_scored() -> None:
    def prepared_with(selected: tuple[PreparedAtom, ...]) -> PreparedStructure:
        return PreparedStructure(
            source_atoms=tuple(atom.record for atom in selected),
            cleaned_atoms=selected,
            selected_atoms=selected,
            metadata=make_metadata(),
            removed=RemovedAtomCounts(0, 0, 0, 0, 0),
        )

    five = tuple(
        make_prepared_atom(source_atom_index=i, atom_serial=i + 1, x=float(i))
        for i in range(5)
    )
    with pytest.raises(BDamagePipelineError, match="window size is larger"):
        run_bdamage_pipeline_from_prepared(prepared_structure=prepared_with(five))


def test_only_macromolecule_atoms_count_toward_the_matthews_mass(
    synthetic_pdb: Path,
) -> None:
    prepared = prepare_structure(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )

    crystal = validate_crystal_for_prepared_structure(prepared)

    assert crystal.content is not None
    protein_records = [
        record for record in prepared.source_atoms if classify_atom(record).is_protein
    ]
    # The mass is taken from the model as read. That is four records more than
    # are scored: a hydrogen, the minor conformer of one site, and the two
    # records cleaning removes for an unusable occupancy or B-factor.
    assert crystal.content.macromolecule_atom_count == len(protein_records) == 81
    assert len(prepared.selected_atoms) == 77
    # The synthetic model has one hydrogen, so the residues gain all but one
    # of their tabulated chain hydrogens.
    assert crystal.content.implicit_hydrogen_mass_dalton > 0.0
    assert crystal.content.nucleic_acid_mass_dalton == 0.0


def test_alternate_conformers_weigh_one_site_in_the_matthews_mass(
    tmp_path: Path,
) -> None:
    atoms = synthetic_atoms()
    major, minor = (atom for atom in atoms if atom.altloc)
    assert major.occupancy + minor.occupancy == approx(1.0)
    merged = [
        replace(atom, altloc="", occupancy=1.0) if atom is major else atom
        for atom in atoms
        if atom is not minor
    ]

    def protein_mass(name: str, variant: list[PdbAtom]) -> float:
        path = write_synthetic_pdb(tmp_path / name, atoms=variant)
        prepared = prepare_structure(read_structure(resolve_structure_input(str(path))))
        content = validate_crystal_for_prepared_structure(prepared).content
        assert content is not None
        return content.protein_mass_dalton

    # Scoring keeps only the major conformer, but the site is one whole atom.
    assert protein_mass("split.pdb", atoms) == approx(
        protein_mass("merged.pdb", merged)
    )
