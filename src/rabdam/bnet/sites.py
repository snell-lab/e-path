"""Select protein atoms used in the Bnet calculation."""

from __future__ import annotations

from dataclasses import dataclass

from rabdam.bdamage import BDamageScoreResult
from rabdam.structure.models import PreparedStructure

# Asp/Glu carboxyl oxygens are used because decarboxylation makes them
# established protein radiation-damage markers.
ASPARTATE_BNET_RESIDUES = frozenset({"ASP", "DAS"})
GLUTAMATE_BNET_RESIDUES = frozenset({"GLU", "DGL"})

ASPARTATE_BNET_ATOMS = frozenset({"OD1", "OD2"})
GLUTAMATE_BNET_ATOMS = frozenset({"OE1", "OE2"})


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
    """Select Asp/Glu carboxyl oxygens from aligned BDamage results."""
    sites: list[BnetSite] = []

    for atom, bdamage_result in zip(
        prepared_structure.selected_atoms,
        bdamage_score_result.atom_results,
        strict=True,
    ):
        if not is_protein_bnet_site(
            residue_name=atom.record.residue_name,
            atom_name=atom.record.atom_name,
        ):
            continue

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
                bdamage=bdamage_result.bdamage,
            )
        )

    return ProteinBnetSiteSelection(
        sites=tuple(sites),
    )


def is_protein_bnet_site(*, residue_name: str, atom_name: str) -> bool:
    """Return whether an atom is a recognized Asp/Glu carboxyl oxygen."""
    residue_name = residue_name.strip().upper()
    atom_name = atom_name.strip().upper()

    if residue_name in ASPARTATE_BNET_RESIDUES:
        return atom_name in ASPARTATE_BNET_ATOMS

    if residue_name in GLUTAMATE_BNET_RESIDUES:
        return atom_name in GLUTAMATE_BNET_ATOMS

    return False
