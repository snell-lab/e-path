"""Name atoms and residues the way the E-path tables and reports do.

An atom is written as a token such as A:87A:CB and a residue as a label such
as Trp76. `pathways`, `cofactors` and `epath.outputs.tables` all write them,
so the two forms are defined once, here, with the key that tells which atoms
belong to one residue. The atoms are the records `xtal.reader` read from the
structure file.
"""

from xtal.reader import AtomRecord


def atom_token(record: AtomRecord) -> str:
    """Return an atom token in Chain:Resnum[icode]:Atom form, such as A:87A:CB.

    A missing residue number is left out.
    """
    residue_number = "" if record.residue_number is None else str(record.residue_number)
    insertion_code = record.insertion_code.strip()
    return (
        f"{record.chain_id}:{residue_number}{insertion_code}:{record.atom_name.strip()}"
    )


def residue_label(record: AtomRecord) -> str:
    """Return a residue label in Trp76 form."""
    return format_residue_label(
        record.residue_name,
        record.residue_number,
        record.insertion_code,
    )


def format_residue_label(
    residue_name: str,
    residue_number: int | None,
    insertion_code: str = "",
) -> str:
    """Return a residue label in Trp76 form from its parts.

    Only the first letter of the name is capitalized. A missing residue number
    is left out, and the insertion code follows the number, as in His87A.
    """
    name = residue_name.strip().capitalize()
    number = "" if residue_number is None else str(residue_number)
    return f"{name}{number}{insertion_code.strip()}"


def residue_key(record: AtomRecord) -> tuple[str, int | None, str]:
    """Return the chain, residue number and insertion code of an atom's residue.

    The residue name is not part of the key.
    """
    return record.chain_id, record.residue_number, record.insertion_code.strip()
