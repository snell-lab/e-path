"""Select protein atoms used in the Bnet calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rabdam.bdamage.score import BDamageAtomResult, BDamageScoreResult
from rabdam.structure.models import PreparedAtom, PreparedStructure

# Asp/Glu carboxyl oxygens are used because decarboxylation makes them
# established protein radiation-damage markers.
ASPARTATE_BNET_RESIDUES = frozenset({"ASP", "DAS"})
GLUTAMATE_BNET_RESIDUES = frozenset({"GLU", "DGL"})

ASPARTATE_BNET_ATOMS = frozenset({"OD1", "OD2"})
GLUTAMATE_BNET_ATOMS = frozenset({"OE1", "OE2"})


class BnetSiteSelectionError(ValueError):
    """Report invalid input to protein Bnet-site selection."""


@dataclass(frozen=True, slots=True)
class BnetSite:
    """Store the identity and BDamage value of one protein Bnet site."""

    source_atom_index: int
    bdamage_atom_index: int
    atom_serial: int | None
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_name: str
    bdamage: float


@dataclass(frozen=True, slots=True)
class ProteinBnetSiteSelection:
    """Store selected protein Bnet sites in prepared-atom order."""

    sites: tuple[BnetSite, ...]

    @property
    def bdamage_values(self) -> tuple[float, ...]:
        """Return site BDamage values in selection order."""
        return tuple(site.bdamage for site in self.sites)


def select_protein_bnet_sites(
    *,
    prepared_structure: PreparedStructure,
    bdamage_score_result: BDamageScoreResult,
) -> ProteinBnetSiteSelection:
    """Select Asp/Glu carboxyl oxygens from aligned BDamage results.

    Raises:
        BnetSiteSelectionError: If selected atoms and results differ in count or
            identity, or if a selected site has a non-finite BDamage value.
    """
    selected_atoms = tuple(prepared_structure.selected_atoms)
    bdamage_results = tuple(bdamage_score_result.atom_results)

    if len(selected_atoms) != len(bdamage_results):
        raise BnetSiteSelectionError(
            "Selected atom count does not match BDamage result count: "
            f"{len(selected_atoms)} selected atoms, "
            f"{len(bdamage_results)} BDamage results."
        )

    sites: list[BnetSite] = []

    for atom, bdamage_result in zip(selected_atoms, bdamage_results, strict=True):
        _validate_atom_matches_bdamage_result(
            atom=atom,
            bdamage_result=bdamage_result,
        )

        if not is_protein_bnet_site(
            residue_name=atom.record.residue_name,
            atom_name=atom.record.atom_name,
        ):
            continue

        bdamage = float(bdamage_result.bdamage)
        if not math.isfinite(bdamage):
            raise BnetSiteSelectionError(
                "Bnet site has a non-finite BDamage value: "
                f"source_atom_index={atom.record.source_atom_index!r}, "
                f"atom_serial={atom.record.atom_serial!r}, "
                f"bdamage={bdamage!r}."
            )

        sites.append(
            BnetSite(
                source_atom_index=atom.record.source_atom_index,
                bdamage_atom_index=bdamage_result.bdamage_atom_index,
                atom_serial=atom.record.atom_serial,
                chain_id=atom.record.chain_id,
                residue_name=atom.record.residue_name,
                residue_number=atom.record.residue_number,
                insertion_code=atom.record.insertion_code,
                atom_name=atom.record.atom_name,
                bdamage=bdamage,
            )
        )

    return ProteinBnetSiteSelection(
        sites=tuple(sites),
    )


def all_selected_bdamage_values(
    bdamage_score_result: BDamageScoreResult,
) -> tuple[float, ...]:
    """Return all finite BDamage values in result order.

    Raises:
        BnetSiteSelectionError: If the result is empty or contains a non-finite
            BDamage value.
    """
    values: list[float] = []

    for atom_result in bdamage_score_result.atom_results:
        bdamage = float(atom_result.bdamage)
        if not math.isfinite(bdamage):
            raise BnetSiteSelectionError(
                "BDamage result contains a non-finite BDamage value: "
                f"source_atom_index={atom_result.source_atom_index!r}, "
                f"atom_serial={atom_result.atom_serial!r}, "
                f"bdamage={bdamage!r}."
            )

        values.append(bdamage)

    if not values:
        raise BnetSiteSelectionError(
            "Cannot collect BDamage values from an empty BDamage result."
        )

    return tuple(values)


def is_protein_bnet_site(*, residue_name: str, atom_name: str) -> bool:
    """Return whether an atom is a recognized Asp/Glu carboxyl oxygen."""
    residue_name = residue_name.strip().upper()
    atom_name = atom_name.strip().upper()

    if residue_name in ASPARTATE_BNET_RESIDUES:
        return atom_name in ASPARTATE_BNET_ATOMS

    if residue_name in GLUTAMATE_BNET_RESIDUES:
        return atom_name in GLUTAMATE_BNET_ATOMS

    return False


def _validate_atom_matches_bdamage_result(
    *,
    atom: PreparedAtom,
    bdamage_result: BDamageAtomResult,
) -> None:
    if atom.record.source_atom_index != bdamage_result.source_atom_index:
        raise BnetSiteSelectionError(
            "Selected atom and BDamage result refer to different source atoms: "
            f"{atom.record.source_atom_index!r} != "
            f"{bdamage_result.source_atom_index!r}."
        )

    if atom.record.atom_serial != bdamage_result.atom_serial:
        raise BnetSiteSelectionError(
            "Selected atom and BDamage result refer to different atom serials: "
            f"{atom.record.atom_serial!r} != "
            f"{bdamage_result.atom_serial!r}."
        )
