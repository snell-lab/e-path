"""Check that a model is a usable crystallographic asymmetric unit.

BDamage compares each atom's B-factor with those of atoms at a similar packing
density, and packing density is counted in a crystal rebuilt from the model by
symmetry expansion and unit-cell translation. That rebuilt crystal is only
correct when the model fills exactly one asymmetric unit.

Two common inputs break this, and symmetry expansion builds a crystal from
either one without complaint.

- A truncated model, such as one chain extracted from an oligomer. Atoms
  buried against the removed subunit are counted as though they face solvent,
  so their packing density, and therefore their BDamage, is wrong exactly at
  the interfaces.
- A file holding more than one asymmetric unit, such as a biological
  assembly. Symmetry expansion then stacks the surplus copies onto images of
  the copies they duplicate.

The Matthews coefficient, the unit-cell volume per dalton of macromolecule in
one asymmetric unit, gives a coarse check for both. A model too massive to fit
the cell is rejected, and an implausibly high or low coefficient produces an
advisory warning. A model that passes is not proven complete. Losing one chain
of four, for example, can leave the coefficient inside the plausible band.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache

import gemmi

from xtal.cell import (
    CrystalMetadataError,
    UnitCellParameters,
    UnitCellTranslationVectors,
    make_gemmi_unit_cell,
    space_group_from_metadata,
    translation_vectors_from_gemmi_cell,
    unit_cell_from_metadata,
)
from xtal.reader import StructureMetadata


class CrystalValidationError(CrystalMetadataError):
    """Report a cell or model that cannot form a valid asymmetric unit.

    It subclasses `CrystalMetadataError`, so callers that handle bad metadata
    also handle these failures.
    """


class MacromoleculeKind(StrEnum):
    """Name the macromolecule classes that differ in partial specific volume."""

    PROTEIN = "protein"
    NUCLEIC_ACID = "nucleic acid"


# The partial specific volume conventionally assumed for protein, 0.74 cm^3/g
# (Matthews 1968), converted to A^3/Da (0.74 x 1e24 / N_A, about 1.23). It is
# the volume that one dalton of protein occupies in the crystal.
MATTHEWS_PROTEIN_CONSTANT = 1.23

# The partial specific volume Kantardjieff and Rupp used for nucleic acid,
# 0.5 cm^3/g, converted the same way (0.5 x 1e24 / N_A, about 0.83). They apply
# each constituent's own value to protein-nucleic acid complexes, as
# `validate_crystal` does. Kantardjieff, K. A. and Rupp, B. (2003), Matthews
# coefficient probabilities, Protein Science 12, 1865-1871.
MATTHEWS_NUCLEIC_ACID_CONSTANT = 0.83

# Most protein crystals have a Matthews coefficient between about 1.7 and
# 3.5 A^3/Da. This advisory band is deliberately wider, so that only clearly
# unusual asymmetric-unit contents produce a warning. It is a protein band and
# is applied unchanged to nucleic-acid models.
MINIMUM_PLAUSIBLE_MATTHEWS = 1.5
MAXIMUM_PLAUSIBLE_MATTHEWS = 4.5

# Implicit hydrogens per modelled heavy atom, for residues Gemmi does not
# tabulate. The protein value is the averagine residue of Senko, Beu and
# McLafferty (1995), J. Am. Soc. Mass Spectrom. 6, 229-233, which has 7.7583
# hydrogens per 7.8151 heavy atoms (C4.9384 H7.7583 N1.3577 O1.4773 S0.0417).
# The nucleic-acid value averages the four DNA and four RNA nucleotides as
# neutral chain residues, 96 hydrogens per 167 heavy atoms. Their formulas
# match the residue weights Gemmi tabulates.
PROTEIN_HYDROGENS_PER_HEAVY_ATOM = 0.9927
NUCLEIC_ACID_HYDROGENS_PER_HEAVY_ATOM = 0.5749

# Gemmi tabulates each residue as a free monomer. A residue inside a peptide or
# nucleic-acid chain has lost one water to each of its two links, so it carries
# two hydrogens fewer. The water left over at the ends of each chain adds only
# two hydrogens per chain, which is too little to track.
_LINKAGE_HYDROGEN_LOSS = 2

_HYDROGEN_WEIGHT = gemmi.Element("H").weight

# How many distinct unrecognized symbols a warning quotes, so that one bad
# file cannot flood the report.
_QUOTED_SYMBOL_LIMIT = 5


@dataclass(frozen=True, slots=True)
class MacromoleculeAtom:
    """Describe one modelled macromolecule atom for the asymmetric-unit checks.

    Atoms that share `residue_id` and `residue_name` form one residue, which
    is the unit implicit hydrogens are counted for. `residue_id` is normally
    the chain, residue number, and insertion code. `atom_name` tells
    alternate conformers of one site apart from distinct sites, so conformers
    sharing a name add their occupancies.
    """

    element: str
    occupancy: float
    kind: MacromoleculeKind
    residue_name: str
    residue_id: tuple[str, int | None, str]
    atom_name: str


@dataclass(frozen=True, slots=True)
class MacromoleculeMass:
    """Break down the occupancy-weighted macromolecule mass in daltons.

    The protein and nucleic-acid masses each include their share of the
    implicit-hydrogen mass. `atom_count` counts the atoms whose element was
    recognized, and `unrecognized_symbols` lists the others, which contribute
    nothing. `adjusted_occupancy_count` counts occupancies that were clamped
    into the range 0 to 1.
    """

    protein_dalton: float
    nucleic_acid_dalton: float
    implicit_hydrogen_dalton: float
    atom_count: int
    unrecognized_symbols: tuple[str, ...]
    adjusted_occupancy_count: int

    @property
    def total_dalton(self) -> float:
        """Return the protein and nucleic-acid mass together."""
        return self.protein_dalton + self.nucleic_acid_dalton


@dataclass(frozen=True, slots=True)
class AsymmetricUnitContent:
    """Describe the macromolecule modelled in one asymmetric unit.

    Masses are in daltons and weighted by occupancy, and they include
    implicit hydrogens, as `macromolecule_mass` explains. The Matthews
    coefficient (A^3/Da) uses the total mass, as is conventional. The solvent
    fraction gives protein and nucleic acid their own partial specific volumes,
    lies strictly between 0 and 1, and is not a percentage. Residues that were
    never modelled remain missing, so the mass can still fall short of the
    sequence mass.
    """

    macromolecule_atom_count: int
    macromolecule_mass_dalton: float
    protein_mass_dalton: float
    nucleic_acid_mass_dalton: float
    implicit_hydrogen_mass_dalton: float
    matthews_coefficient: float
    solvent_fraction: float


@dataclass(frozen=True, slots=True)
class ValidatedCrystal:
    """Hold the validated crystal description shared by later calculations.

    Symmetry expansion, unit-cell translation, and packing density all read
    this one description instead of re-reading and re-checking the raw
    metadata. Build it with `validate_crystal`, which checks that `gemmi_cell`
    has a finite, positive volume. Everything else about the cell and space
    group is derived from `gemmi_cell` and `space_group` on access, so it
    cannot drift out of step with them, and the volume and translation vectors
    share the frame in which the atoms are positioned.

    `content` is `None` when no asymmetric-unit content was measured. Warnings
    can still be present then, for example when no element symbol was
    recognized.
    """

    unit_cell: UnitCellParameters
    gemmi_cell: gemmi.UnitCell
    space_group: gemmi.SpaceGroup
    content: AsymmetricUnitContent | None = None
    warnings: tuple[str, ...] = ()

    @property
    def space_group_name(self) -> str:
        """Return the extended Hermann-Mauguin name of the space group.

        The extended form names the setting where it is ambiguous, such as
        "R 3:H" or "R 3:R", so reports show which axes the operations assume.
        """
        return str(self.space_group.xhm())

    @property
    def operations(self) -> tuple[gemmi.Op, ...]:
        """Return every symmetry operation of the space group, including centering."""
        return tuple(self.space_group.operations())

    @property
    def operation_count(self) -> int:
        """Return the number of asymmetric units in the unit cell.

        Centering translations count, so a C-centered cell has twice as many
        asymmetric units as its point-group operations alone would give.
        """
        return len(self.space_group.operations())

    @property
    def unit_cell_volume_angstrom3(self) -> float:
        """Return the unit-cell volume in cubic angstroms."""
        return float(self.gemmi_cell.volume)

    @property
    def translation_vectors(self) -> UnitCellTranslationVectors:
        """Return the unit-cell axes in the Cartesian frame of `gemmi_cell`."""
        return translation_vectors_from_gemmi_cell(self.gemmi_cell)


@dataclass(slots=True)
class _ResidueSites:
    """Collect the atom sites of one residue while the mass is summed."""

    kind: MacromoleculeKind
    heavy_occupancy: dict[str, float] = field(default_factory=dict[str, float])
    hydrogen_names: set[str] = field(default_factory=set[str])

    def occupancy(self) -> float:
        """Return the mean heavy-atom site occupancy, or 0 with no heavy atoms."""
        if not self.heavy_occupancy:
            return 0.0
        # Conformers that over-fill a site in a malformed file still describe
        # at most one full site.
        site_occupancies = [min(1.0, value) for value in self.heavy_occupancy.values()]
        return sum(site_occupancies) / len(site_occupancies)


def validate_crystal(
    metadata: StructureMetadata,
    *,
    macromolecule_atoms: Sequence[MacromoleculeAtom] = (),
) -> ValidatedCrystal:
    """Validate crystal metadata and, optionally, the asymmetric-unit content.

    When `metadata` carries the Gemmi cell read from the file, that cell is
    used unchanged, so a file-specific orthogonalization frame from `SCALE` or
    `_atom_sites.fract_transf_matrix` is preserved. Otherwise a cell in the
    standard frame is built from the parameters.

    `macromolecule_atoms` holds the modelled protein and nucleic-acid atoms
    only. Solvent, ions, and ligands are left out, as in a conventional
    Matthews calculation. With no atoms, only the cell and space group are
    validated. Atoms with unrecognized elements or out-of-range occupancies
    are handled as `macromolecule_mass` describes and reported as warnings.

    A missing or invalid cell or space group raises `CrystalMetadataError`. A
    cell with a non-physical volume, or a model whose protein and nucleic acid
    would leave no room for solvent, raises its subclass
    `CrystalValidationError`. An implausible Matthews coefficient only adds a
    warning.
    """
    unit_cell = unit_cell_from_metadata(metadata)
    gemmi_cell = (
        metadata.gemmi_cell
        if metadata.gemmi_cell is not None
        else make_gemmi_unit_cell(unit_cell)
    )
    space_group = space_group_from_metadata(metadata)

    # The cell read from the file is used as it is, without passing through the
    # parameter checks above, so its volume is the one guarantee it has to meet.
    unit_cell_volume = float(gemmi_cell.volume)
    if not math.isfinite(unit_cell_volume) or unit_cell_volume <= 0.0:
        raise CrystalValidationError(
            f"Unit cell has a non-physical volume: {unit_cell_volume!r} A^3."
        )

    content, content_warnings = _asymmetric_unit_content(
        macromolecule_atoms=macromolecule_atoms,
        asymmetric_unit_volume_angstrom3=(
            unit_cell_volume / len(space_group.operations())
        ),
    )

    return ValidatedCrystal(
        unit_cell=unit_cell,
        gemmi_cell=gemmi_cell,
        space_group=space_group,
        content=content,
        warnings=content_warnings,
    )


def _asymmetric_unit_content(
    *,
    macromolecule_atoms: Sequence[MacromoleculeAtom],
    asymmetric_unit_volume_angstrom3: float,
) -> tuple[AsymmetricUnitContent | None, tuple[str, ...]]:
    """Return the asymmetric-unit content, if any is measurable, and warnings."""
    if not macromolecule_atoms:
        return None, ()

    mass = macromolecule_mass(macromolecule_atoms)
    warnings = list(_input_warnings(mass))
    if mass.total_dalton <= 0.0:
        if mass.atom_count:
            warnings.append(
                "Every macromolecule atom has zero occupancy, so the "
                "asymmetric-unit content was not checked."
            )
        return None, tuple(warnings)

    matthews = asymmetric_unit_volume_angstrom3 / mass.total_dalton
    occupied_volume = (
        MATTHEWS_PROTEIN_CONSTANT * mass.protein_dalton
        + MATTHEWS_NUCLEIC_ACID_CONSTANT * mass.nucleic_acid_dalton
    )
    solvent_fraction = 1.0 - occupied_volume / asymmetric_unit_volume_angstrom3

    # With no room left for solvent, the macromolecule alone would fill more
    # than the asymmetric unit, which no real crystal allows. That is an error
    # rather than a warning. The test uses the occupied volume, not the
    # Matthews coefficient, because nucleic acid packs more mass into the same
    # volume and a fixed coefficient limit would reject dense nucleic-acid
    # crystals that are physically possible.
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
        macromolecule_atom_count=mass.atom_count,
        macromolecule_mass_dalton=mass.total_dalton,
        protein_mass_dalton=mass.protein_dalton,
        nucleic_acid_mass_dalton=mass.nucleic_acid_dalton,
        implicit_hydrogen_mass_dalton=mass.implicit_hydrogen_dalton,
        matthews_coefficient=matthews,
        solvent_fraction=solvent_fraction,
    )
    warnings.extend(_content_warnings(content))
    return content, tuple(warnings)


def macromolecule_mass(atoms: Sequence[MacromoleculeAtom]) -> MacromoleculeMass:
    """Return the occupancy-weighted macromolecule mass, with implicit hydrogens.

    Each atom contributes its atomic weight times its occupancy, so alternate
    conformers of one site add up to that site's full weight. An occupancy
    outside 0 to 1 is clamped into that range, and a NaN counts as 0, because
    neither describes a real fraction of unit cells. Element symbols match
    case-insensitively after surrounding whitespace is stripped, and an
    unrecognized symbol contributes nothing.

    Hydrogens a residue lacks are added back. Its expected count comes from
    Gemmi's tabulated residues, reduced by two for the chain links. A residue
    Gemmi does not tabulate instead gets `PROTEIN_HYDROGENS_PER_HEAVY_ATOM` or
    `NUCLEIC_ACID_HYDROGENS_PER_HEAVY_ATOM` per modelled heavy-atom site. A
    tabulated residue gets its full count even when some of its heavy atoms
    were not modelled, which partly offsets their missing mass. Hydrogens
    already modelled in the residue, counted by distinct atom name, are
    subtracted, so a fully protonated residue gains nothing and a partly
    protonated one gains only the remainder. The added mass is weighted by the
    residue's occupancy, the mean occupancy of its heavy-atom sites. The
    expected counts are for neutral residues and ignore chain ends, so they are
    approximate.
    """
    dalton = dict.fromkeys(MacromoleculeKind, 0.0)
    implicit_dalton = 0.0
    residues: dict[tuple[tuple[str, int | None, str], str], _ResidueSites] = {}
    unrecognized: list[str] = []
    adjusted_occupancy_count = 0
    atom_count = 0

    for atom in atoms:
        element = gemmi.Element(atom.element.strip())
        # Gemmi maps an unknown symbol to a placeholder element with atomic
        # number 0 and a nominal weight of 1 Da, which must not count as mass.
        if element.atomic_number == 0:
            unrecognized.append(atom.element)
            continue

        occupancy = _physical_occupancy(atom.occupancy)
        if occupancy != atom.occupancy:
            adjusted_occupancy_count += 1
        atom_count += 1
        dalton[atom.kind] += element.weight * occupancy

        sites = residues.setdefault(
            (atom.residue_id, atom.residue_name), _ResidueSites(kind=atom.kind)
        )
        site_name = atom.atom_name.strip()
        if element.atomic_number == 1:
            sites.hydrogen_names.add(site_name)
        else:
            sites.heavy_occupancy[site_name] = (
                sites.heavy_occupancy.get(site_name, 0.0) + occupancy
            )

    for (_, residue_name), sites in residues.items():
        missing = max(
            0.0,
            _expected_hydrogens(residue_name, sites) - len(sites.hydrogen_names),
        )
        added = missing * sites.occupancy() * _HYDROGEN_WEIGHT
        dalton[sites.kind] += added
        implicit_dalton += added

    return MacromoleculeMass(
        protein_dalton=dalton[MacromoleculeKind.PROTEIN],
        nucleic_acid_dalton=dalton[MacromoleculeKind.NUCLEIC_ACID],
        implicit_hydrogen_dalton=implicit_dalton,
        atom_count=atom_count,
        unrecognized_symbols=tuple(unrecognized),
        adjusted_occupancy_count=adjusted_occupancy_count,
    )


def _physical_occupancy(occupancy: float) -> float:
    if math.isnan(occupancy):
        return 0.0
    return min(1.0, max(0.0, occupancy))


def _expected_hydrogens(residue_name: str, sites: _ResidueSites) -> float:
    """Return how many hydrogens a complete, neutral chain residue would carry."""
    tabulated = _tabulated_chain_hydrogens(residue_name.strip().upper())
    if tabulated is not None:
        return float(tabulated)

    per_heavy_atom = (
        PROTEIN_HYDROGENS_PER_HEAVY_ATOM
        if sites.kind is MacromoleculeKind.PROTEIN
        else NUCLEIC_ACID_HYDROGENS_PER_HEAVY_ATOM
    )
    return per_heavy_atom * len(sites.heavy_occupancy)


@cache
def _tabulated_chain_hydrogens(residue_name: str) -> int | None:
    residue = gemmi.find_tabulated_residue(residue_name)
    if not residue.found():
        return None
    return max(0, residue.hydrogen_count - _LINKAGE_HYDROGEN_LOSS)


def _input_warnings(mass: MacromoleculeMass) -> tuple[str, ...]:
    """Return warnings for atoms the mass had to drop or adjust."""
    warnings: list[str] = []
    unrecognized = mass.unrecognized_symbols
    if unrecognized and not mass.atom_count:
        warnings.append(
            f"None of the {len(unrecognized)} macromolecule element symbols was "
            f"recognized ({_quoted_symbols(unrecognized)}), so the asymmetric-unit "
            "content was not checked."
        )
    elif unrecognized:
        warnings.append(
            f"{len(unrecognized)} of {mass.atom_count + len(unrecognized)} "
            "macromolecule atoms have unrecognized element symbols "
            f"({_quoted_symbols(unrecognized)}) and were left out of the mass, so "
            "the Matthews coefficient is overestimated."
        )

    if mass.adjusted_occupancy_count:
        warnings.append(
            f"{mass.adjusted_occupancy_count} macromolecule atoms have an "
            "occupancy outside 0 to 1, which was clamped into that range for "
            "the macromolecule mass."
        )

    return tuple(warnings)


def _quoted_symbols(symbols: Sequence[str]) -> str:
    distinct = sorted(set(symbols))
    quoted = ", ".join(repr(symbol) for symbol in distinct[:_QUOTED_SYMBOL_LIMIT])
    if len(distinct) > _QUOTED_SYMBOL_LIMIT:
        quoted += ", ..."
    return quoted


def _content_warnings(content: AsymmetricUnitContent) -> tuple[str, ...]:
    """Return a warning when the Matthews coefficient leaves the advisory band."""
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
