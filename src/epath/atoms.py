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
    """Build E-path atom records from a completed RABDAM BDamage workflow."""

    selected_atoms = tuple(workflow_result.prepared_structure.selected_atoms)
    bdamage_results = tuple(workflow_result.bdamage_score_result.atom_results)

    if len(selected_atoms) != len(bdamage_results):
        raise EpathAtomError(
            "Selected atom count does not match BDamage result count: "
            f"{len(selected_atoms)} selected atoms, "
            f"{len(bdamage_results)} BDamage results."
        )

    epath_atoms: list[EpathAtom] = []
    for atom, bdamage_result in zip(selected_atoms, bdamage_results, strict=True):
        record = atom.record
        if record.source_atom_index != bdamage_result.source_atom_index:
            raise EpathAtomError(
                "Selected atom and BDamage result refer to different source atoms: "
                f"{record.source_atom_index!r} != "
                f"{bdamage_result.source_atom_index!r}."
            )

        if record.atom_serial != bdamage_result.atom_serial:
            raise EpathAtomError(
                "Selected atom and BDamage result refer to different atom serials: "
                f"{record.atom_serial!r} != {bdamage_result.atom_serial!r}."
            )

        epath_atoms.append(
            EpathAtom(
                source_atom_index=record.source_atom_index,
                bdamage_atom_index=bdamage_result.bdamage_atom_index,
                atom_serial=record.atom_serial,
                record_type=record.record_type,
                chain_id=record.chain_id,
                residue_name=record.residue_name,
                residue_number=record.residue_number,
                insertion_code=record.insertion_code,
                atom_name=record.atom_name,
                element=record.element,
                x=float(record.x),
                y=float(record.y),
                z=float(record.z),
                occupancy=float(record.occupancy),
                b_factor=float(record.b_factor),
                packing_density=bdamage_result.packing_density,
                average_b_factor=float(bdamage_result.average_b_factor),
                bdamage=float(bdamage_result.bdamage),
            )
        )

    return tuple(epath_atoms)
