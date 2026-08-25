"""Validate that a model is a usable crystallographic asymmetric unit.

BDamage compares an atom's B-factor with atoms at a similar local packing
density, and packing density is counted in a crystal environment rebuilt by
symmetry expansion and unit-cell translation. That rebuilt environment is only
correct when the deposited model actually fills one asymmetric unit.

Two common inputs break that assumption, and neither is detectable from the
coordinates alone, because symmetry expansion faithfully reproduces a crystal
built from whatever the file contains:

- A truncated model, for example a single chain extracted from an oligomer.
  Atoms that were buried against a removed subunit are counted as though they
  sit in solvent, so their packing density, and therefore their BDamage, is
  wrong exactly at the interfaces.
- A biological-assembly file rather than the asymmetric unit. Symmetry
  expansion then stacks copies of an already-complete assembly.

Both are caught by comparing the modelled macromolecular mass against the
unit-cell volume through the Matthews coefficient.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import gemmi

from xtal.cell import (
    CrystalMetadataError,
    UnitCellParameters,
    make_gemmi_unit_cell,
    space_group_from_metadata,
    unit_cell_from_metadata,
)
from xtal.reader import StructureMetadata

# Matthews' constant: the reciprocal of the partial specific volume commonly
# assumed for protein, 0.74 cm3/g, expressed in A^3/Da.
MATTHEWS_PROTEIN_CONSTANT = 1.23

# Advisory band for the Matthews coefficient in A^3/Da. Most protein crystals
# fall near 1.7-3.5. The bounds here are deliberately generous so that only
# clearly unusual asymmetric-unit contents are reported.
MINIMUM_PLAUSIBLE_MATTHEWS = 1.5
MAXIMUM_PLAUSIBLE_MATTHEWS = 4.5


class CrystalValidationError(CrystalMetadataError):
    """Report a model that cannot be a valid crystallographic asymmetric unit."""


@dataclass(frozen=True, slots=True)
class AsymmetricUnitContent:
    """Describe how much macromolecule the modelled asymmetric unit holds."""

    macromolecule_atom_count: int
    macromolecule_mass_dalton: float
    matthews_coefficient: float
    solvent_fraction: float
    hydrogens_excluded: bool


@dataclass(frozen=True, slots=True)
class ValidatedCrystal:
    """Store the validated crystal description shared by later calculations.

    Symmetry expansion, unit-cell translation, and packing density all consume
    this single validated description instead of re-reading and re-checking the
    raw metadata.
    """

    unit_cell: UnitCellParameters
    gemmi_cell: gemmi.UnitCell
    space_group: gemmi.SpaceGroup
    space_group_name: str
    operations: tuple[gemmi.Op, ...]
    unit_cell_volume_angstrom3: float
    content: AsymmetricUnitContent | None = None
    warnings: tuple[str, ...] = ()

    @property
    def operation_count(self) -> int:
        """Return the number of asymmetric units in the unit cell."""
        return len(self.operations)


def validate_crystal(
    metadata: StructureMetadata,
    *,
    macromolecule_elements: Sequence[str] = (),
    hydrogens_excluded: bool = True,
) -> ValidatedCrystal:
    """Validate crystallographic metadata and asymmetric-unit content.

    Args:
        metadata: Crystallographic metadata read from the structure file.
        macromolecule_elements: Element symbols of the modelled protein and
            nucleic-acid atoms. Solvent, ions, and ligands are excluded, as
            they are for a conventional Matthews calculation. Pass an empty
            sequence to validate the cell and space group only.
        hydrogens_excluded: Whether hydrogens were removed before the element
            symbols were collected, which is the default for BDamage.

    Returns:
        The validated cell, space group, symmetry operations, and, when
        element symbols were supplied, the asymmetric-unit content with any
        advisory warnings.

    Raises:
        CrystalMetadataError: If the unit cell or space group is missing or
            invalid.
        CrystalValidationError: If the space group has no symmetry operations,
            or the modelled macromolecule cannot physically fit in the cell.
    """
    unit_cell = unit_cell_from_metadata(metadata)
    gemmi_cell = make_gemmi_unit_cell(unit_cell)
    space_group = space_group_from_metadata(metadata)
    operations = tuple(space_group.operations())

    if not operations:
        raise CrystalValidationError(
            f"Space group {metadata.space_group!r} has no symmetry operations."
        )

    unit_cell_volume = float(gemmi_cell.volume)
    if not math.isfinite(unit_cell_volume) or unit_cell_volume <= 0.0:
        raise CrystalValidationError(
            f"Unit cell has a non-physical volume: {unit_cell_volume!r} A^3."
        )

    content, warnings = _asymmetric_unit_content(
        macromolecule_elements=macromolecule_elements,
        unit_cell_volume_angstrom3=unit_cell_volume,
        operation_count=len(operations),
        hydrogens_excluded=hydrogens_excluded,
    )

    return ValidatedCrystal(
        unit_cell=unit_cell,
        gemmi_cell=gemmi_cell,
        space_group=space_group,
        space_group_name=space_group.hm,
        operations=operations,
        unit_cell_volume_angstrom3=unit_cell_volume,
        content=content,
        warnings=warnings,
    )


def _asymmetric_unit_content(
    *,
    macromolecule_elements: Sequence[str],
    unit_cell_volume_angstrom3: float,
    operation_count: int,
    hydrogens_excluded: bool,
) -> tuple[AsymmetricUnitContent | None, tuple[str, ...]]:
    if not macromolecule_elements:
        return None, ()

    mass = macromolecule_mass_dalton(macromolecule_elements)
    if mass <= 0.0:
        return None, ()

    matthews = unit_cell_volume_angstrom3 / (operation_count * mass)
    solvent_fraction = 1.0 - (MATTHEWS_PROTEIN_CONSTANT / matthews)

    if solvent_fraction <= 0.0:
        raise CrystalValidationError(
            "The modelled macromolecule cannot fit in the unit cell "
            f"(Matthews coefficient {matthews:.2f} A^3/Da, solvent content "
            f"{100.0 * solvent_fraction:.1f}%). This usually means the input is "
            "a biological assembly rather than the asymmetric unit. BDamage "
            "requires the asymmetric unit, because packing density is counted "
            "in a crystal rebuilt from it by symmetry expansion."
        )

    content = AsymmetricUnitContent(
        macromolecule_atom_count=len(macromolecule_elements),
        macromolecule_mass_dalton=mass,
        matthews_coefficient=matthews,
        solvent_fraction=solvent_fraction,
        hydrogens_excluded=hydrogens_excluded,
    )

    return content, _content_warnings(content)


def _content_warnings(content: AsymmetricUnitContent) -> tuple[str, ...]:
    matthews = content.matthews_coefficient
    solvent_percent = 100.0 * content.solvent_fraction

    if matthews > MAXIMUM_PLAUSIBLE_MATTHEWS:
        return (
            (
                f"Matthews coefficient {matthews:.2f} A^3/Da (solvent content "
                f"{solvent_percent:.1f}%) is higher than expected. The model may "
                "not fill the asymmetric unit, for example if chains were removed. "
                "Packing densities, and therefore BDamage, would then be too low "
                "for atoms at the missing interfaces."
            ),
        )

    if matthews < MINIMUM_PLAUSIBLE_MATTHEWS:
        return (
            (
                f"Matthews coefficient {matthews:.2f} A^3/Da (solvent content "
                f"{solvent_percent:.1f}%) is lower than expected. The model may "
                "contain more than one asymmetric unit, for example a biological "
                "assembly, which would make the rebuilt crystal environment and "
                "the resulting packing densities too dense."
            ),
        )

    return ()


def macromolecule_mass_dalton(elements: Sequence[str]) -> float:
    """Return the summed atomic mass in daltons for the given element symbols.

    Unrecognized element symbols contribute no mass.
    """
    return float(
        sum(gemmi.Element(str(element).strip()).weight for element in elements)
    )
