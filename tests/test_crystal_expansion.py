"""Symmetry expansion and the trimmed block of neighbouring cells."""

import itertools
from pathlib import Path

import gemmi
import numpy as np
import pytest
from numpy.typing import NDArray
from support import approx, make_metadata, make_prepared_atom, write_synthetic_pdb

from rabdam.crystal.symmetry import symmetry_image_coordinates
from rabdam.crystal.trim import trim_crystal_to_reference_atoms
from rabdam.structure.models import PreparedAtom
from rabdam.structure.prepare import prepare_structure
from xtal.cell import translation_vectors_from_gemmi_cell
from xtal.reader import read_structure
from xtal.resolver import resolve_structure_input
from xtal.validate import ValidatedCrystal, validate_crystal

ATOMS = (
    make_prepared_atom(source_atom_index=0, x=1.0, y=2.0, z=3.0),
    make_prepared_atom(source_atom_index=1, atom_name="CB", x=-4.5, y=7.25, z=12.0),
    make_prepared_atom(source_atom_index=2, residue_number=2, x=20.0, y=-1.0, z=5.5),
)


def _gemmi_images(
    crystal: ValidatedCrystal,
    atoms: tuple[PreparedAtom, ...],
    *,
    translation_range: int = 0,
) -> list[tuple[float, float, float]]:
    """Every image of `atoms` in the block of cells, computed by Gemmi alone.

    The images are ordered by atom, then by operation, then by cell offset,
    and the offsets are added in fractional space, so no code is shared with
    the implementation.
    """
    cell = crystal.gemmi_cell
    offsets = list(
        itertools.product(range(-translation_range, translation_range + 1), repeat=3)
    )
    images: list[tuple[float, float, float]] = []
    for atom in atoms:
        fractional = cell.fractionalize(
            gemmi.Position(atom.record.x, atom.record.y, atom.record.z)
        )
        for operation in crystal.space_group.operations():
            moved = operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])
            for a_offset, b_offset, c_offset in offsets:
                position = cell.orthogonalize(
                    gemmi.Fractional(
                        moved[0] + a_offset, moved[1] + b_offset, moved[2] + c_offset
                    )
                )
                images.append((position.x, position.y, position.z))
    return images


def _rows_in_canonical_order(rows: object) -> NDArray[np.float64]:
    """Return `(n, 3)` coordinate rows sorted by x, then y, then z."""
    array = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    rounded = np.round(array, 6)
    return array[np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))]


def test_p1_expansion_is_the_identity() -> None:
    crystal = validate_crystal(make_metadata(space_group="P 1"), macromolecule_atoms=())

    images = symmetry_image_coordinates(ATOMS, crystal=crystal)

    assert images.shape == (len(ATOMS), 3)
    assert np.allclose(
        images,
        [[atom.record.x, atom.record.y, atom.record.z] for atom in ATOMS],
        atol=1e-9,
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
    crystal = validate_crystal(
        make_metadata(cell=cell, space_group=space_group), macromolecule_atoms=()
    )

    images = symmetry_image_coordinates(ATOMS, crystal=crystal)

    # Rows follow the input atoms and then Gemmi's operation order.
    expected = _gemmi_images(crystal, ATOMS)
    assert images.shape == (len(ATOMS) * len(crystal.operations), 3)
    assert np.allclose(images, np.asarray(expected), atol=1e-9)


def test_no_atoms_give_no_images() -> None:
    crystal = validate_crystal(
        make_metadata(space_group="P 1 21 1"), macromolecule_atoms=()
    )

    assert symmetry_image_coordinates((), crystal=crystal).shape == (0, 3)


def _standard_frame_vectors(*cell: float) -> NDArray[np.float64]:
    return translation_vectors_from_gemmi_cell(gemmi.UnitCell(*cell))


def test_standard_translation_vectors() -> None:
    ortho = _standard_frame_vectors(10, 20, 30, 90, 90, 90)
    assert np.allclose(ortho, [[10, 0, 0], [0, 20, 0], [0, 0, 30]], atol=1e-9)

    mono = _standard_frame_vectors(10, 20, 30, 90, 120, 90)
    assert mono[2].tolist() == approx([-15, 0, 30 * np.sin(np.radians(120))], abs=1e-9)


def test_an_untrimmed_block_holds_every_neighbouring_cell() -> None:
    crystal = validate_crystal(
        make_metadata(space_group="P 1 21 1"), macromolecule_atoms=()
    )
    images = symmetry_image_coordinates(ATOMS, crystal=crystal)

    # A padding far larger than the block keeps every image of every cell.
    for translation_range, cell_count in ((0, 1), (1, 27), (2, 125)):
        block = trim_crystal_to_reference_atoms(
            image_coordinates=images,
            translation_vectors=crystal.translation_vectors,
            reference_atoms=ATOMS,
            padding=1000.0,
            translation_range=translation_range,
        )

        expected = _gemmi_images(crystal, ATOMS, translation_range=translation_range)
        assert block.shape == (cell_count * len(images), 3)
        assert np.allclose(
            _rows_in_canonical_order(block),
            _rows_in_canonical_order(expected),
            atol=1e-9,
        )

    # Rows are ordered by cell, with a slowest and c fastest, and the images
    # of one cell keep their order.
    vectors = crystal.translation_vectors
    first_cell_shift = -vectors.sum(axis=0)
    block = trim_crystal_to_reference_atoms(
        image_coordinates=images,
        translation_vectors=vectors,
        reference_atoms=ATOMS,
        padding=1000.0,
    )
    assert np.allclose(block[: len(images)], images + first_cell_shift, atol=1e-9)
    assert np.allclose(block[13 * len(images) : 14 * len(images)], images, atol=1e-9)


def test_trimmed_block_matches_brute_force_translation() -> None:
    crystal = validate_crystal(
        make_metadata(space_group="P 1 21 1"), macromolecule_atoms=()
    )
    images = symmetry_image_coordinates(ATOMS, crystal=crystal)
    reference = ATOMS[:2]
    padding = 7.0

    trimmed = trim_crystal_to_reference_atoms(
        image_coordinates=images,
        translation_vectors=crystal.translation_vectors,
        reference_atoms=reference,
        padding=padding,
    )

    full = _gemmi_images(crystal, ATOMS, translation_range=1)
    positions = [(atom.record.x, atom.record.y, atom.record.z) for atom in reference]
    lower = [min(axis) - padding for axis in zip(*positions, strict=True)]
    upper = [max(axis) + padding for axis in zip(*positions, strict=True)]
    expected = [
        image
        for image in full
        if all(
            low <= value <= high
            for value, low, high in zip(image, lower, upper, strict=True)
        )
    ]

    # The trimmed block must hold exactly the images of the full block that
    # lie in the box, each once, whatever their order.
    assert 0 < len(trimmed) == len(expected) < len(full)
    assert np.allclose(
        _rows_in_canonical_order(trimmed),
        _rows_in_canonical_order(expected),
        atol=1e-9,
    )


def test_every_cleaned_atom_of_a_structure_is_in_the_block(
    synthetic_pdb: Path,
) -> None:
    prepared = prepare_structure(
        read_structure(resolve_structure_input(str(synthetic_pdb)))
    )
    crystal = validate_crystal(prepared.metadata, macromolecule_atoms=())

    images = symmetry_image_coordinates(prepared.cleaned_atoms, crystal=crystal)

    assert len(images) == 2 * len(prepared.cleaned_atoms)


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

    crystal = validate_crystal(metadata, macromolecule_atoms=())

    assert metadata.gemmi_cell is not None
    vectors = crystal.translation_vectors
    # The SCALE records carry six decimals, so the axes match to about 1e-4.
    assert np.allclose(vectors, [[0, 15, 0], [0, 0, 16], [17, 0, 0]], atol=1e-3)
    assert np.array_equal(
        vectors, translation_vectors_from_gemmi_cell(metadata.gemmi_cell)
    )
    standard = translation_vectors_from_gemmi_cell(
        gemmi.UnitCell(15.0, 16.0, 17.0, 90.0, 90.0, 90.0)
    )
    assert standard[0].tolist() == approx([15, 0, 0])
