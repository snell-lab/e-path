"""Run the BDamage calculation from parsed structure data to per-atom scores.

The stages run in order, each from its own module: structure preparation,
crystal validation, symmetry expansion, translation and trimming of the
neighbouring cells, packing density, and the BDamage scores. `rabdam.bnet` and
`epath` start from the result returned here.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from rabdam.bdamage import BDamageAtomResult, calculate_bdamage_scores_for_structure
from rabdam.crystal.symmetry import symmetry_image_coordinates
from rabdam.crystal.trim import trim_crystal_to_reference_atoms
from rabdam.packing import calculate_packing_density
from rabdam.structure.models import PreparedStructure
from rabdam.structure.prepare import classify_atom, prepare_structure
from xtal.reader import AtomRecord, StructureData
from xtal.validate import (
    MacromoleculeAtom,
    MacromoleculeKind,
    ValidatedCrystal,
    validate_crystal,
)


class PipelineStageRunner(Protocol):
    """Wrap each named pipeline stage, for example to time or report it.

    A runner must call the callback exactly once and return what it returns.
    """

    def __call__[T](self, label: str, callback: Callable[[], T]) -> T:
        """Run a labelled callback and return its result."""
        ...


class BDamagePipelineError(ValueError):
    """Report invalid input to the end-to-end BDamage pipeline."""


@dataclass(frozen=True, slots=True)
class BDamagePipelineOptions:
    """Set the parameters of a BDamage calculation.

    `packing_density_threshold` is the radius, in angstroms, within which
    neighbours are counted. The sliding window holds `window_size_fraction` of
    the selected atoms, made odd and never smaller than `minimum_window_size`.
    `translation_range` is how many unit cells the crystal is extended by in
    each direction along every axis, so 1 gives a 3x3x3 block of cells.

    Nothing checks the values here or in the pipeline. The threshold and the
    minimum window size must be positive, the fraction between 0 and 1, and
    the translation range not negative, which the command line enforces when
    it parses its arguments.
    """

    packing_density_threshold: float = 7.0
    window_size_fraction: float = 0.02
    minimum_window_size: int = 11
    translation_range: int = 1


@dataclass(frozen=True, slots=True)
class BDamagePipelineResult:
    """Store the results of the BDamage pipeline.

    `packing_densities` holds the neighbour count of each selected atom and
    `bdamage_results` its score, both in selected-atom order. `options` holds the settings the run used, defaults
    included, and `window_size` the odd window size derived from them.
    """

    prepared_structure: PreparedStructure
    crystal: ValidatedCrystal
    packing_densities: tuple[int, ...]
    bdamage_results: tuple[BDamageAtomResult, ...]
    options: BDamagePipelineOptions
    window_size: int


def run_bdamage_pipeline(
    structure_data: StructureData,
    *,
    options: BDamagePipelineOptions | None = None,
    stage_runner: PipelineStageRunner | None = None,
) -> BDamagePipelineResult:
    """Prepare parsed structure data and calculate per-atom BDamage scores.

    Preparation cleans the atoms and selects the ones to score, and
    `run_bdamage_pipeline_from_prepared` does the rest. Without `options` the
    defaults of `BDamagePipelineOptions` apply. A `stage_runner` is handed each
    stage as a label and a callback, so a caller can time or report the stages.

    A structure with nothing left to score raises `StructurePreparationError`.
    The other failures are those of `run_bdamage_pipeline_from_prepared`.
    """
    prepared_structure = _run_pipeline_stage(
        "Preparing structure",
        lambda: prepare_structure(structure_data),
        stage_runner=stage_runner,
    )

    return run_bdamage_pipeline_from_prepared(
        prepared_structure=prepared_structure,
        options=options,
        stage_runner=stage_runner,
    )


def run_bdamage_pipeline_from_prepared(
    *,
    prepared_structure: PreparedStructure,
    options: BDamagePipelineOptions | None = None,
    stage_runner: PipelineStageRunner | None = None,
) -> BDamagePipelineResult:
    """Calculate per-atom BDamage scores for a prepared structure.

    The window size is checked before any stage runs. The crystal is then
    validated and expanded by symmetry, the neighbouring cells are translated
    and trimmed to the region around the selected atoms, neighbours within the
    packing-density threshold are counted, and each B-factor is divided by the
    mean over a window of atoms with similar packing density.

    A `BDamagePipelineError` is raised when the window would be larger than
    the selection. A missing or invalid unit cell or space group raises
    `xtal.cell.CrystalMetadataError` from crystal validation.
    """
    if options is None:
        options = BDamagePipelineOptions()

    selected_atom_count = len(prepared_structure.selected_atoms)
    window_size = bdamage_window_size_from_fraction(
        atom_count=selected_atom_count,
        window_size_fraction=options.window_size_fraction,
        minimum_window_size=options.minimum_window_size,
    )

    if window_size > selected_atom_count:
        raise BDamagePipelineError(
            "Calculated BDamage window size is larger than the number of selected atoms: "
            f"window_size={window_size!r}, selected_atom_count={selected_atom_count!r}."
        )

    crystal = _run_pipeline_stage(
        "Validating crystal",
        lambda: validate_crystal_for_prepared_structure(prepared_structure),
        stage_runner=stage_runner,
    )
    # Every cleaned atom is expanded, not only the selected ones, because the
    # unselected atoms are still neighbours of the atoms being scored.
    image_coordinates = _run_pipeline_stage(
        "Expanding crystallographic symmetry",
        lambda: symmetry_image_coordinates(
            prepared_structure.cleaned_atoms,
            crystal=crystal,
        ),
        stage_runner=stage_runner,
    )
    # Translating and trimming in one pass avoids allocating every atom of the
    # translated cells, most of which lie outside the trim box.
    neighbour_coordinates = _run_pipeline_stage(
        "Building trimmed neighbour block",
        lambda: trim_crystal_to_reference_atoms(
            image_coordinates=image_coordinates,
            translation_vectors=crystal.translation_vectors,
            reference_atoms=prepared_structure.selected_atoms,
            padding=options.packing_density_threshold,
            translation_range=options.translation_range,
        ),
        stage_runner=stage_runner,
    )
    packing_densities = _run_pipeline_stage(
        "Calculating packing density",
        lambda: calculate_packing_density(
            selected_atoms=prepared_structure.selected_atoms,
            neighbour_coordinates=neighbour_coordinates,
            packing_density_threshold=options.packing_density_threshold,
        ),
        stage_runner=stage_runner,
    )
    bdamage_results = _run_pipeline_stage(
        "Calculating BDamage scores",
        lambda: calculate_bdamage_scores_for_structure(
            prepared_structure=prepared_structure,
            packing_densities=packing_densities,
            window_size=window_size,
        ),
        stage_runner=stage_runner,
    )

    return BDamagePipelineResult(
        prepared_structure=prepared_structure,
        crystal=crystal,
        packing_densities=packing_densities,
        bdamage_results=bdamage_results,
        options=options,
        window_size=window_size,
    )


def bdamage_window_size_from_fraction(
    *,
    atom_count: int,
    window_size_fraction: float,
    minimum_window_size: int,
) -> int:
    """Return the odd BDamage window size for `atom_count` selected atoms.

    The size is `window_size_fraction` of the atoms, rounded to the nearest
    integer and never less than `minimum_window_size`. An even size, whether
    it came from the fraction or from the minimum, is raised to the next odd
    one. The result can exceed `atom_count`, which the caller has to reject.
    """
    window_size = max(round(atom_count * window_size_fraction), minimum_window_size)
    # A centered window needs a single middle atom, and going up rather than
    # down keeps at least the requested fraction.
    if window_size % 2 == 0:
        window_size += 1

    return window_size


def validate_crystal_for_prepared_structure(
    prepared_structure: PreparedStructure,
) -> ValidatedCrystal:
    """Validate the crystal metadata and asymmetric-unit content of a model.

    Only protein and nucleic-acid atoms contribute to the modelled mass, as
    they do for a conventional Matthews calculation. Solvent, ions, and other
    heteroatoms are excluded even when they are scored. The protein and
    nucleic-acid split comes from the same residue classification that
    structure preparation uses.

    The atoms are taken as read, before cleaning, because the model's mass
    does not depend on which atoms can be scored. Every conformer of an atom
    site counts at its own occupancy, so the site adds up to its full weight,
    and modelled hydrogens and atoms with an unusable B-factor count too.
    """
    return validate_crystal(
        prepared_structure.metadata,
        macromolecule_atoms=_macromolecule_atoms(prepared_structure.source_atoms),
    )


def _macromolecule_atoms(
    source_atoms: Iterable[AtomRecord],
) -> tuple[MacromoleculeAtom, ...]:
    """Return the protein and nucleic-acid atoms as asymmetric-unit check input."""
    atoms = (classify_atom(atom) for atom in source_atoms)
    return tuple(
        MacromoleculeAtom(
            element=atom.record.element,
            occupancy=atom.record.occupancy,
            kind=(
                MacromoleculeKind.PROTEIN
                if atom.is_protein
                else MacromoleculeKind.NUCLEIC_ACID
            ),
            residue_name=atom.record.residue_name,
            residue_id=(
                atom.record.chain_id,
                atom.record.residue_number,
                atom.record.insertion_code,
            ),
            atom_name=atom.record.atom_name,
        )
        for atom in atoms
        if atom.is_protein or atom.is_nucleic_acid
    )


def _run_pipeline_stage[T](
    label: str,
    callback: Callable[[], T],
    *,
    stage_runner: PipelineStageRunner | None,
) -> T:
    if stage_runner is None:
        return callback()

    return stage_runner(label, callback)
