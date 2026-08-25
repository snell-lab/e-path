"""Symmetry expansion, unit-cell translation, and neighbour-block trimming."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import gemmi
import numpy as np
import pytest
from support import approx, make_metadata, make_prepared_atom, write_synthetic_pdb

from rabdam.crystal.symmetry import (
    CrystalSymmetryError,
    expand_atoms_by_symmetry,
    expand_prepared_structure_by_symmetry,
)
from rabdam.crystal.translate import (
    CrystalTranslationError,
    translate_atoms,
    translate_expanded_unit_cell,
    translation_vector_for_offsets,
    unit_cell_translation_vectors,
)
from rabdam.crystal.trim import (
    CrystalTrimError,
    bounds_from_prepared_atoms,
    expand_bounds,
    trim_expanded_unit_cell_to_reference_atoms,
)
from rabdam.structure.models import PreparedAtom
from rabdam.structure.prepare import prepare_structure
from xtal.cell import UnitCellParameters, translation_vectors_from_gemmi_cell
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input
from xtal.validate import ValidatedCrystal, validate_crystal

ATOMS = (
    make_prepared_atom(source_atom_index=0, x=1.0, y=2.0, z=3.0),
    make_prepared_atom(source_atom_index=1, atom_name="CB", x=-4.5, y=7.25, z=12.0),
    make_prepared_atom(source_atom_index=2, residue_number=2, x=20.0, y=-1.0, z=5.5),
)


def _gemmi_images(
    crystal: ValidatedCrystal, atom: PreparedAtom
) -> Iterator[tuple[str, tuple[float, float, float]]]:
    """Every symmetry image of an atom, computed by Gemmi alone."""
    fractional = crystal.gemmi_cell.fractionalize(
        gemmi.Position(atom.record.x, atom.record.y, atom.record.z)
    )
    for operation in crystal.space_group.operations():
        moved = operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])
        position = crystal.gemmi_cell.orthogonalize(gemmi.Fractional(*moved))
        yield operation.triplet(), (position.x, position.y, position.z)


def test_p1_expansion_is_the_identity() -> None:
    crystal = validate_crystal(make_metadata(space_group="P 1"))

    expanded = expand_atoms_by_symmetry(atoms=ATOMS, crystal=crystal)

    assert expanded.operation_count == 1 and expanded.space_group_name == "P 1"
    assert expanded.translation_vectors == crystal.translation_vectors
    assert [atom.unit_cell_atom_index for atom in expanded.atoms] == [1, 2, 3]
    for image, atom in zip(expanded.atoms, ATOMS, strict=True):
        assert image.is_identity_symmetry_operation
        assert image.symmetry_operation_index == 1
        assert image.source_atom_index == atom.record.source_atom_index
        assert (image.x, image.y, image.z) == approx(
            (atom.record.x, atom.record.y, atom.record.z), abs=1e-9
        )


@pytest.mark.parametrize(
    "space_group", ["P 1 21 1", "P 21 21 21", "C 1 2 1", "P 41 3 2"]
)
def test_images_match_gemmi_operations_without_wrapping(space_group: str) -> None:
    cell = (
        (30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
        if space_group == "P 41 3 2"
        else (15.0, 16.0, 17.0, 90.0, 95.0, 90.0)
    )
    crystal = validate_crystal(make_metadata(cell=cell, space_group=space_group))

    expanded = expand_atoms_by_symmetry(atoms=ATOMS, crystal=crystal)

    assert expanded.operation_count == len(crystal.operations)
    assert len(expanded.atoms) == len(ATOMS) * expanded.operation_count
    images = iter(expanded.atoms)
    for atom in ATOMS:
        for operation_index, (triplet, position) in enumerate(
            _gemmi_images(crystal, atom), start=1
        ):
            image = next(images)
            assert image.source_atom_index == atom.record.source_atom_index
            assert image.symmetry_operation_index == operation_index
            assert image.symmetry_operation == triplet
            assert image.is_identity_symmetry_operation is (
                triplet.replace(" ", "") == "x,y,z"
            )
            assert (image.x, image.y, image.z) == approx(position, abs=1e-9)
    assert sum(atom.is_identity_symmetry_operation for atom in expanded.atoms) == len(
        ATOMS
    )


def test_empty_expansion_is_rejected() -> None:
    with pytest.raises(CrystalSymmetryError):
        expand_atoms_by_symmetry(atoms=(), crystal=validate_crystal(make_metadata()))


def test_prepared_structure_expansion_uses_all_cleaned_atoms(
    synthetic_pdb: Path,
) -> None:
    prepared = prepare_structure(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )

    expanded = expand_prepared_structure_by_symmetry(prepared)

    assert len(expanded.atoms) == 2 * len(prepared.cleaned_atoms)
    assert expanded.translation_vectors is not None


def test_standard_translation_vectors() -> None:
    ortho = unit_cell_translation_vectors(UnitCellParameters(10, 20, 30, 90, 90, 90))
    assert (ortho.a.x, ortho.a.y, ortho.a.z) == approx((10, 0, 0))
    assert (ortho.b.x, ortho.b.y, ortho.b.z) == approx((0, 20, 0), abs=1e-12)
    assert (ortho.c.x, ortho.c.y, ortho.c.z) == approx((0, 0, 30), abs=1e-12)

    mono = unit_cell_translation_vectors(UnitCellParameters(10, 20, 30, 90, 120, 90))
    assert (mono.c.x, mono.c.y, mono.c.z) == approx(
        (-15, 0, 30 * np.sin(np.radians(120))), abs=1e-12
    )

    shift = translation_vector_for_offsets(ortho, a_offset=1, b_offset=-1, c_offset=2)
    assert (shift.x, shift.y, shift.z) == approx((10, -20, 60), abs=1e-12)

    with pytest.raises(CrystalTranslationError):
        unit_cell_translation_vectors(UnitCellParameters(10, 20, 30, 90, 90, 180))
    with pytest.raises(CrystalTranslationError):
        unit_cell_translation_vectors(UnitCellParameters(10, 20, 30, 10, 10, 170))


def test_translation_generates_every_neighbouring_cell() -> None:
    crystal = validate_crystal(make_metadata(space_group="P 1"))
    expanded = expand_atoms_by_symmetry(atoms=ATOMS, crystal=crystal)

    block = translate_expanded_unit_cell(expanded, translation_range=1)

    assert len(block.atoms) == 27 * len(ATOMS)
    assert block.source_unit_cell_atom_count == len(ATOMS)
    assert block.translation_vectors == crystal.translation_vectors
    offsets = {
        (atom.translation_a, atom.translation_b, atom.translation_c)
        for atom in block.atoms
    }
    assert offsets == set(itertools.product((-1, 0, 1), repeat=3))
    vectors = block.translation_vectors
    for atom in block.atoms:
        base = expanded.atoms[atom.unit_cell_atom_index - 1]
        shift = translation_vector_for_offsets(
            vectors,
            a_offset=atom.translation_a,
            b_offset=atom.translation_b,
            c_offset=atom.translation_c,
        )
        assert (atom.x, atom.y, atom.z) == approx(
            (base.x + shift.x, base.y + shift.y, base.z + shift.z), abs=1e-9
        )
    assert [atom.translated_atom_index for atom in block.atoms] == list(
        range(1, len(block.atoms) + 1)
    )

    single = translate_atoms(
        atoms=expanded.atoms, unit_cell=expanded.unit_cell, translation_range=0
    )
    assert len(single.atoms) == len(ATOMS)

    with pytest.raises(CrystalTranslationError):
        translate_atoms(
            atoms=expanded.atoms, unit_cell=expanded.unit_cell, translation_range=-1
        )
    with pytest.raises(CrystalTranslationError):
        translate_atoms(atoms=(), unit_cell=expanded.unit_cell)


def test_bounds_and_padding() -> None:
    bounds = bounds_from_prepared_atoms(ATOMS)
    assert (
        bounds.x_min,
        bounds.x_max,
        bounds.y_min,
        bounds.y_max,
        bounds.z_min,
        bounds.z_max,
    ) == (-4.5, 20.0, -1.0, 7.25, 3.0, 12.0)
    padded = expand_bounds(bounds, 7.0)
    assert (padded.x_min, padded.x_max, padded.z_max) == (-11.5, 27.0, 19.0)
    with pytest.raises(CrystalTrimError):
        expand_bounds(bounds, -1.0)
    with pytest.raises(CrystalTrimError):
        bounds_from_prepared_atoms(())


def test_trimmed_block_matches_brute_force_translation() -> None:
    crystal = validate_crystal(make_metadata(space_group="P 1 21 1"))
    expanded = expand_atoms_by_symmetry(atoms=ATOMS, crystal=crystal)
    reference = ATOMS[:2]
    padding = 7.0

    trimmed = trim_expanded_unit_cell_to_reference_atoms(
        expanded_structure=expanded, reference_atoms=reference, padding=padding
    )

    full = translate_expanded_unit_cell(expanded, translation_range=1)
    bounds = expand_bounds(bounds_from_prepared_atoms(reference), padding)
    expected = {
        (
            atom.source_atom_index,
            atom.symmetry_operation_index,
            atom.translation_a,
            atom.translation_b,
            atom.translation_c,
            round(atom.x, 6),
            round(atom.y, 6),
            round(atom.z, 6),
        )
        for atom in full.atoms
        if bounds.x_min <= atom.x <= bounds.x_max
        and bounds.y_min <= atom.y <= bounds.y_max
        and bounds.z_min <= atom.z <= bounds.z_max
    }
    actual: set[tuple[int, int, int, int, int, float, float, float]] = set()
    identity_by_source = {
        a.source_atom_index: a.symmetry_operation_index
        for a in expanded.atoms
        if a.is_identity_symmetry_operation
    }
    for row in range(trimmed.atom_count):
        source = int(trimmed.source_atom_indices[row])
        x, y, z = trimmed.coordinates[row]
        # Recover the operation index by matching the unshifted image.
        shift = translation_vector_for_offsets(
            trimmed.translation_vectors,
            a_offset=int(trimmed.translation_offsets[row][0]),
            b_offset=int(trimmed.translation_offsets[row][1]),
            c_offset=int(trimmed.translation_offsets[row][2]),
        )
        base = (x - shift.x, y - shift.y, z - shift.z)
        operation_index = next(
            a.symmetry_operation_index
            for a in expanded.atoms
            if a.source_atom_index == source
            and np.allclose((a.x, a.y, a.z), base, atol=1e-6)
        )
        assert bool(trimmed.is_identity_symmetry_operation[row]) is (
            operation_index == identity_by_source[source]
        )
        actual.add(
            (
                source,
                operation_index,
                int(trimmed.translation_offsets[row][0]),
                int(trimmed.translation_offsets[row][1]),
                int(trimmed.translation_offsets[row][2]),
                round(x, 6),
                round(y, 6),
                round(z, 6),
            )
        )

    assert actual == expected
    assert 0 < trimmed.atom_count < len(full.atoms)
    assert trimmed.original_atom_count == len(full.atoms)
    assert trimmed.padding == padding and trimmed.trim_bounds == bounds


def test_trim_rejects_invalid_input() -> None:
    crystal = validate_crystal(make_metadata(space_group="P 1"))
    expanded = expand_atoms_by_symmetry(atoms=ATOMS, crystal=crystal)
    with pytest.raises(CrystalTrimError):
        trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded, reference_atoms=ATOMS, padding=-1.0
        )
    with pytest.raises(CrystalTrimError):
        trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded,
            reference_atoms=ATOMS,
            padding=1.0,
            translation_range=1.0,  # type: ignore[arg-type]
        )
    with pytest.raises(CrystalTrimError):
        trim_expanded_unit_cell_to_reference_atoms(
            expanded_structure=expanded, reference_atoms=(), padding=1.0
        )


def test_file_specific_orthogonalization_frame_is_kept(tmp_path: Path) -> None:
    # The SCALE records put a along y, b along z, and c along x. Gemmi keeps
    # that frame for the atoms, so lattice translations must use it too.
    scale = (
        "SCALE1      0.000000  0.066667  0.000000        0.00000",
        "SCALE2      0.000000  0.000000  0.062500        0.00000",
        "SCALE3      0.058824  0.000000  0.000000        0.00000",
    )
    path = write_synthetic_pdb(
        tmp_path / "rotated.pdb",
        cell=(15.0, 16.0, 17.0, 90.0, 90.0, 90.0),
        space_group="P 1",
        scale_lines=scale,
    )
    metadata = read_structure(resolve_structure_input(str(path))).metadata

    crystal = validate_crystal(metadata)

    assert metadata.gemmi_cell is not None
    vectors = crystal.translation_vectors
    # The SCALE records carry six decimals, so the axes match to about 1e-4.
    assert (vectors.a.x, vectors.a.y, vectors.a.z) == approx((0, 15, 0), abs=1e-3)
    assert (vectors.b.x, vectors.b.y, vectors.b.z) == approx((0, 0, 16), abs=1e-3)
    assert (vectors.c.x, vectors.c.y, vectors.c.z) == approx((17, 0, 0), abs=1e-3)
    assert crystal.translation_vectors == translation_vectors_from_gemmi_cell(
        metadata.gemmi_cell
    )
    standard = unit_cell_translation_vectors(crystal.unit_cell)
    assert (standard.a.x, standard.a.y, standard.a.z) == approx((15, 0, 0))
