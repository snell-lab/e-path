"""Selection of atom sites used for protein Bnet calculations.

Protein Bnet is calculated from the BDamage values of Asp/Glu side-chain
carboxyl oxygen atoms. This module connects RABDAM's prepared atoms and
per-atom BDamage results to the raw Bnet metric calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from bdamage.score import BDamageAtomResult, BDamageScoreResult
from structure.models import PreparedAtom, PreparedStructure


ASPARTATE_BNET_RESIDUES = frozenset({"ASP", "DAS"})
GLUTAMATE_BNET_RESIDUES = frozenset({"GLU", "DGL"})

ASPARTATE_BNET_ATOMS = frozenset({"OD1", "OD2"})
GLUTAMATE_BNET_ATOMS = frozenset({"OE1", "OE2"})


class BnetSiteSelectionError(ValueError):
    """Raised when RABDAM cannot select Bnet sites from BDamage results."""


@dataclass(frozen=True, slots=True)
class BnetSite:
    """One atom used in the protein Bnet calculation.

    source_atom_index:
        Zero-based reader index of the selected asymmetric-unit atom.

    bdamage_atom_index:
        One-based position of this atom in the BDamage selected-atom order.

    atom_serial:
        Atom serial number from the input structure, when available.

    chain_id, residue_name, residue_number, insertion_code, atom_name:
        Structure identifiers copied from the input atom record.

    bdamage:
        Per-atom BDamage value for this Bnet site.
    """

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
    """Selected protein Bnet sites and their BDamage values."""

    sites: tuple[BnetSite, ...]

    @property
    def bdamage_values(self) -> tuple[float, ...]:
        """Return the BDamage values for the selected Bnet sites."""

        return tuple(site.bdamage for site in self.sites)


def select_protein_bnet_sites(
    *,
    prepared_structure: PreparedStructure,
    bdamage_score_result: BDamageScoreResult,
) -> ProteinBnetSiteSelection:
    """Select Asp/Glu carboxyl oxygen atoms for protein Bnet.

    The selected atoms and BDamage results are expected to be in the same
    selected-atom order. Identity is checked by source atom index and atom
    serial before selecting Bnet sites.
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
    """Return all selected-atom BDamage values from a BDamage score result."""

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
    """Return whether an atom is an Asp/Glu carboxyl oxygen Bnet site."""

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
    """Validate that a prepared atom and BDamage result refer to the same atom."""

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
