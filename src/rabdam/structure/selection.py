"""Select atoms that receive BDamage scores."""

from rabdam.structure.models import PreparedAtom, StructurePreparationOptions


def select_bdamage_atoms(
    atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> tuple[PreparedAtom, ...]:
    """Select atoms that should receive BDamage scores.

    Returns:
        Selected atoms in input order, without duplicate source atoms.
    """
    selected_atoms: list[PreparedAtom] = []
    selected_keys: set[int] = set()

    remove_component_names = _normalize_component_name_set(
        options.remove_component_names
    )
    add_component_names = _normalize_component_name_set(options.add_component_names)

    for atom in atoms:
        if should_select_bdamage_atom(
            atom,
            options,
            remove_component_names,
            add_component_names,
        ):
            _append_selected_atom(atom, selected_atoms, selected_keys)

    return tuple(selected_atoms)


def should_select_bdamage_atom(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
    remove_component_names: frozenset[str],
    add_component_names: frozenset[str],
) -> bool:
    """Return whether an atom passes explicit overrides or default selection.

    Atom-serial overrides take precedence over component-name overrides, and
    removal takes precedence over addition at either level.
    """
    atom_serial = atom.record.atom_serial
    component_name = atom.record.residue_name.strip().upper()

    # Atom-level overrides are more specific than component-level overrides;
    # removals win within each level so exclusions remain reliable.
    if atom_serial is not None and atom_serial in options.remove_atom_serials:
        return False

    if atom_serial is not None and atom_serial in options.add_atom_serials:
        return True

    if component_name in remove_component_names:
        return False

    if component_name in add_component_names:
        return True

    return is_default_bdamage_selection(atom, options)


def is_default_bdamage_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
) -> bool:
    """Return whether an atom passes the configured default selection."""
    if atom.is_protein and (
        not atom.is_hetatm or options.include_protein_like_hetatm_in_selection
    ):
        return True

    if options.include_hetatm_in_selection and atom.is_hetatm and not atom.is_solvent:
        return True

    return options.include_nucleic_acid_in_selection and atom.is_nucleic_acid


def _append_selected_atom(
    atom: PreparedAtom,
    selected_atoms: list[PreparedAtom],
    selected_keys: set[int],
) -> None:
    key = _prepared_atom_key(atom)

    if key in selected_keys:
        return

    selected_atoms.append(atom)
    selected_keys.add(key)


def _normalize_component_name_set(component_names: frozenset[str]) -> frozenset[str]:
    return frozenset(
        component_name.strip().upper() for component_name in component_names
    )


def _prepared_atom_key(atom: PreparedAtom) -> int:
    return atom.record.source_atom_index
