"""Resolve user-supplied E-path endpoint atom tokens."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from epath.atoms import EpathAtom
from epath.clustering import euclidean_distance
from epath.selection import EpathScoredAtom


DEFAULT_SNAP_RADIUS_ANGSTROM = 2.0


class EpathEndpointError(ValueError):
    """Raised when an E-path endpoint token cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class EpathAtomToken:
    """Parsed atom token in Chain:Resnum[icode]:Atom form."""

    chain_id: str
    residue_number: int
    insertion_code: str
    atom_name: str


@dataclass(frozen=True, slots=True)
class EpathResolvedEndpoint:
    """Endpoint token resolved to a selected E-path atom."""

    token: str
    parsed_token: EpathAtomToken
    target_atom: EpathAtom
    selected_index: int
    selected_atom: EpathScoredAtom
    exact_match: bool
    snap_distance: float


def atom_token(atom: EpathAtom) -> str:
    """Return an atom token in Chain:Resnum[icode]:Atom form."""

    return (
        f"{atom.chain_id}:"
        f"{residue_number_string(atom)}:"
        f"{atom.atom_name.strip()}"
    )


def residue_number_string(atom: EpathAtom) -> str:
    """Return the residue number plus insertion code as a compact string."""

    residue_number = atom.residue_number
    residue_text = "" if residue_number is None else str(residue_number)
    insertion_code = atom.insertion_code.strip()
    if insertion_code:
        return f"{residue_text}{insertion_code}"

    return residue_text


def residue_tuple(atom: EpathAtom) -> tuple[str, str, str]:
    """Return a compact residue description for tabular output."""

    return atom.chain_id, residue_number_string(atom), atom.residue_name.strip()


def parse_atom_token(token: str) -> EpathAtomToken:
    """Parse Chain:Resnum[icode]:Atom into a structured atom token."""

    if not isinstance(token, str):
        raise EpathEndpointError(f"atom token must be a string, got {token!r}.")

    parts = token.split(":")
    if len(parts) != 3:
        raise EpathEndpointError(
            "atom token must have the form Chain:Resnum[icode]:Atom, "
            f"got {token!r}."
        )

    chain_id, residue_part, atom_name = (part.strip() for part in parts)
    if not chain_id:
        raise EpathEndpointError(f"atom token has no chain identifier: {token!r}.")

    if not residue_part:
        raise EpathEndpointError(f"atom token has no residue number: {token!r}.")

    if not atom_name:
        raise EpathEndpointError(f"atom token has no atom name: {token!r}.")

    residue_digits = ""
    insertion_code = ""
    for character in residue_part:
        if character.isdigit():
            residue_digits += character
        else:
            insertion_code += character

    if not residue_digits:
        raise EpathEndpointError(f"atom token has no residue number: {token!r}.")

    return EpathAtomToken(
        chain_id=chain_id,
        residue_number=int(residue_digits),
        insertion_code=insertion_code.strip(),
        atom_name=atom_name,
    )


def find_atom_by_token(
    atoms: Iterable[EpathAtom],
    token: str | EpathAtomToken,
    *,
    allowed_chains: Iterable[str] | None = None,
) -> EpathAtom | None:
    """Return the atom matching a parsed endpoint token, if present."""

    parsed_token = _parsed_token(token)
    allowed_chain_set = (
        {chain_id.strip() for chain_id in allowed_chains}
        if allowed_chains is not None
        else None
    )

    if allowed_chain_set is not None and parsed_token.chain_id not in allowed_chain_set:
        return None

    atom_name = parsed_token.atom_name.upper()
    insertion_code = parsed_token.insertion_code.strip()
    for atom in atoms:
        if allowed_chain_set is not None and atom.chain_id not in allowed_chain_set:
            continue

        if atom.chain_id != parsed_token.chain_id:
            continue

        if atom.residue_number != parsed_token.residue_number:
            continue

        if atom.insertion_code.strip() != insertion_code:
            continue

        if atom.atom_name.strip().upper() == atom_name:
            return atom

    return None


def resolve_endpoint(
    token: str,
    *,
    all_atoms: Iterable[EpathAtom],
    selected_atoms: Sequence[EpathScoredAtom],
    snap_radius: float = DEFAULT_SNAP_RADIUS_ANGSTROM,
    allowed_chains: Iterable[str] | None = None,
) -> EpathResolvedEndpoint | None:
    """Resolve an endpoint token to an exact or nearby selected atom."""

    parsed_token = parse_atom_token(token)
    snap_radius = _non_negative_finite_float(snap_radius, name="snap_radius")
    target_atom = find_atom_by_token(
        all_atoms,
        parsed_token,
        allowed_chains=allowed_chains,
    )
    if target_atom is None:
        return None

    exact_index = selected_atom_index(target_atom, selected_atoms)
    if exact_index is not None:
        return EpathResolvedEndpoint(
            token=token,
            parsed_token=parsed_token,
            target_atom=target_atom,
            selected_index=exact_index,
            selected_atom=selected_atoms[exact_index],
            exact_match=True,
            snap_distance=0.0,
        )

    nearest = nearest_selected_atom(
        target_atom,
        selected_atoms,
    )
    if nearest is None:
        return None

    nearest_index, nearest_atom, distance = nearest
    if distance > snap_radius:
        return None

    return EpathResolvedEndpoint(
        token=token,
        parsed_token=parsed_token,
        target_atom=target_atom,
        selected_index=nearest_index,
        selected_atom=nearest_atom,
        exact_match=False,
        snap_distance=distance,
    )


def selected_atom_index(
    target_atom: EpathAtom,
    selected_atoms: Sequence[EpathScoredAtom],
) -> int | None:
    """Return the selected atom index for target_atom, if it was selected."""

    for index, scored_atom in enumerate(selected_atoms):
        if same_atom(scored_atom.atom, target_atom):
            return index

    return None


def nearest_selected_atom(
    target_atom: EpathAtom,
    selected_atoms: Sequence[EpathScoredAtom],
) -> tuple[int, EpathScoredAtom, float] | None:
    """Return the nearest selected atom to target_atom."""

    if not selected_atoms:
        return None

    best_index = -1
    best_atom: EpathScoredAtom | None = None
    best_distance = math.inf
    target_coordinate = atom_coordinate(target_atom)

    for index, scored_atom in enumerate(selected_atoms):
        distance = euclidean_distance(
            target_coordinate,
            atom_coordinate(scored_atom.atom),
        )
        if distance < best_distance:
            best_index = index
            best_atom = scored_atom
            best_distance = distance

    if best_atom is None:
        return None

    return best_index, best_atom, float(best_distance)


def same_atom(first_atom: EpathAtom, second_atom: EpathAtom) -> bool:
    """Return whether two E-path atom records identify the same source atom."""

    return (
        first_atom.source_atom_index == second_atom.source_atom_index
        and first_atom.atom_serial == second_atom.atom_serial
    )


def atom_coordinate(atom: EpathAtom) -> tuple[float, float, float]:
    """Return an atom coordinate tuple."""

    return float(atom.x), float(atom.y), float(atom.z)


def atom_token_help_text() -> str:
    """Return user-facing documentation for endpoint atom tokens."""

    return (
        "Atom tokens must be supplied as 'Chain:Resnum[icode]:Atom'.\n"
        "Examples:\n"
        "  A:45:FE\n"
        "  B:102:ND1\n"
        "  A:145A:OXT\n\n"
        "Field meaning:\n"
        "  Chain   = PDB chain identifier\n"
        "  Resnum  = residue sequence number\n"
        "  icode   = optional insertion code written immediately after Resnum\n"
        "  Atom    = atom name as used in the PDB, parsed case-insensitively"
    )


def _parsed_token(token: str | EpathAtomToken) -> EpathAtomToken:
    if isinstance(token, EpathAtomToken):
        return token

    return parse_atom_token(token)


def _non_negative_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise EpathEndpointError(f"{name} must be numeric, got {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EpathEndpointError(
            f"{name} must be numeric, got {value!r}."
        ) from error

    if not math.isfinite(number):
        raise EpathEndpointError(f"{name} must be finite, got {number!r}.")

    if number < 0.0:
        raise EpathEndpointError(
            f"{name} must be zero or positive, got {number!r}."
        )

    return number
