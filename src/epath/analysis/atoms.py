"""Build E-path atom records from RABDAM atom-level BDamage output."""

from __future__ import annotations

from dataclasses import dataclass

from rabdam.workflow import BDamageWorkflowResult


class EpathAtomError(ValueError):
    """Raised when E-path atom records cannot be built from RABDAM output."""


@dataclass(frozen=True, slots=True)
class EpathAtom:
    """One atom with the BDamage data needed by E-path analysis."""

    source_atom_index: int
    bdamage_atom_index: int
    atom_serial: int | None
    record_type: str
    chain_id: str
    residue_name: str
    residue_number: int | None
    insertion_code: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    packing_density: int
    average_b_factor: float
    bdamage: float


def atoms_from_bdamage_workflow(
    workflow_result: BDamageWorkflowResult,
) -> tuple[EpathAtom, ...]:
    """Build E-path atom records from a completed RABDAM BDamage workflow.

    BDamage results pair with selected atoms by position, so each pair must
    refer to the same source atom.
    """
    selected_atoms = workflow_result.prepared_structure.selected_atoms
    bdamage_results = workflow_result.bdamage_score_result.atom_results

    epath_atoms: list[EpathAtom] = []
    for atom, bdamage_result in zip(selected_atoms, bdamage_results, strict=True):
        record = atom.record
        if record.source_atom_index != bdamage_result.source_atom_index:
            raise EpathAtomError(
                "Selected atom and BDamage result refer to different source atoms: "
                f"{record.source_atom_index!r} != "
                f"{bdamage_result.source_atom_index!r}."
            )

        epath_atoms.append(
            EpathAtom(
                source_atom_index=record.source_atom_index,
                bdamage_atom_index=bdamage_result.bdamage_atom_index,
                atom_serial=record.atom_serial,
                record_type="HETATM" if record.is_hetatm else "ATOM",
                chain_id=record.chain_id,
                residue_name=record.residue_name,
                residue_number=record.residue_number,
                insertion_code=record.insertion_code,
                atom_name=record.atom_name,
                element=record.element,
                x=record.x,
                y=record.y,
                z=record.z,
                occupancy=record.occupancy,
                b_factor=record.b_factor,
                packing_density=bdamage_result.packing_density,
                average_b_factor=bdamage_result.average_b_factor,
                bdamage=bdamage_result.bdamage,
            )
        )

    return tuple(epath_atoms)


def atom_token(atom: EpathAtom) -> str:
    """Return an atom token in Chain:Resnum[icode]:Atom form."""
    residue_number = "" if atom.residue_number is None else str(atom.residue_number)
    insertion_code = atom.insertion_code.strip()
    return f"{atom.chain_id}:{residue_number}{insertion_code}:{atom.atom_name.strip()}"


def residue_label(atom: EpathAtom) -> str:
    """Return a residue label in Trp76 form."""
    return format_residue_label(
        atom.residue_name,
        atom.residue_number,
        atom.insertion_code,
    )


def format_residue_label(
    residue_name: str,
    residue_number: int | None,
    insertion_code: str = "",
) -> str:
    """Return a residue label built from its name, number, and insertion code."""
    name = residue_name.strip().capitalize()
    number = "" if residue_number is None else str(residue_number)
    return f"{name}{number}{insertion_code.strip()}"


def residue_key(atom: EpathAtom) -> tuple[str, int | None, str]:
    """Return the identity of the residue an atom belongs to."""
    return atom.chain_id, atom.residue_number, atom.insertion_code.strip()
