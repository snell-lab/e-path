"""Residue chemistry, cofactor sites and proximity, and pathway description."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
from support import (
    approx,
    make_atom_record,
    make_prepared_atom,
    scored_atoms_from_coordinates,
)

from epath.analysis.clustering import cluster_selected_atoms
from epath.analysis.cofactors import build_cofactor_context, cofactor_site_summary
from epath.analysis.labels import atom_token, format_residue_label, residue_label
from epath.analysis.pathways import PathReport, analyze_paths
from epath.analysis.residues import (
    analyze_composition,
    class_counts,
    class_fractions,
    residue_class,
)
from rabdam.structure.models import PreparedAtom


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("TRP", "aromatic"),
        ("HIS", "aromatic"),
        ("tyr", "aromatic"),
        ("MSE", "sulfur"),
        ("CYS", "sulfur"),
        ("ASP", "charged"),
        ("LYS", "charged"),
        ("SER", "polar"),
        ("GLY", "hydrophobic"),
        ("PRO", "hydrophobic"),
        # Modified residues take the class of their parent.
        ("SEP", "polar"),
        ("PCA", "charged"),
        ("M3L", "charged"),
        ("CGU", "charged"),
        ("TYS", "aromatic"),
        ("NEP", "aromatic"),
        ("OCS", "sulfur"),
        ("DAL", "hydrophobic"),
        ("SEC", "sulfur"),
        ("XYZ", "other"),
        ("UNK", "other"),
        ("HOH", "other"),
        ("DA", "other"),  # a nucleotide, though its one-letter code is A
    ],
)
def test_residue_classes(name: str, expected: str) -> None:
    assert residue_class(name) == expected


def test_composition_counts_and_fractions_sum_to_one() -> None:
    names = ["TRP", "TYR", "CYS", "ALA", "ALA", "GLY", "LYS", "SER", "HOH", "ALA"]
    counts = class_counts(names)
    assert counts == {
        "aromatic": 2,
        "sulfur": 1,
        "charged": 1,
        "polar": 1,
        "hydrophobic": 4,
        "other": 1,
    }
    assert sum(class_fractions(names).values()) == approx(1.0)

    composition = analyze_composition(names)

    assert composition.counts == counts
    assert composition.fractions["aromatic"] == approx(0.2)
    assert composition.fractions["charged"] == approx(0.1)


def _cleaned_model() -> tuple[PreparedAtom, ...]:
    """Protein atoms of two chains plus cofactors and an additive."""
    return (
        make_prepared_atom(source_atom_index=0, chain_id="A", residue_number=1, x=0.0),
        make_prepared_atom(source_atom_index=1, chain_id="A", residue_number=2, x=3.0),
        make_prepared_atom(
            source_atom_index=2, chain_id="B", residue_number=1, x=30.0, y=3.0
        ),
        make_prepared_atom(
            source_atom_index=3,
            chain_id="A",
            residue_name="FMN",
            residue_number=201,
            atom_name="N1",
            element="N",
            is_hetatm=True,
            x=12.0,
        ),
        make_prepared_atom(
            source_atom_index=4,
            chain_id="A",
            residue_name="SF4",
            residue_number=301,
            atom_name="FE1",
            element="FE",
            is_hetatm=True,
            x=40.0,
        ),
        make_prepared_atom(
            source_atom_index=5,
            chain_id="B",
            residue_name="SF4",
            residue_number=302,
            atom_name="FE1",
            element="FE",
            is_hetatm=True,
            x=60.0,
        ),
        make_prepared_atom(
            source_atom_index=6,
            chain_id="A",
            residue_name="GOL",
            residue_number=401,
            atom_name="C1",
            is_hetatm=True,
            x=80.0,
        ),
        make_prepared_atom(
            source_atom_index=7,
            chain_id="A",
            residue_name="HOH",
            residue_number=501,
            atom_name="O",
            element="O",
            is_hetatm=True,
            x=100.0,
        ),
    )


def test_only_redox_capable_groups_are_cofactors() -> None:
    def group(index: int, name: str, element: str) -> PreparedAtom:
        return make_prepared_atom(
            source_atom_index=index,
            residue_name=name,
            residue_number=100 + index,
            atom_name=element,
            element=element,
            is_hetatm=True,
            x=20.0 * index,
        )

    cleaned = (
        make_prepared_atom(source_atom_index=0, residue_number=1),
        group(1, "CDN", "C"),  # a lipid
        group(2, "NAG", "C"),  # a sugar
        group(3, "ZN", "ZN"),  # a redox-inert ion
        group(4, "CA", "CA"),
        group(5, "HG", "HG"),  # a heavy-atom derivative
        group(6, "ZNH", "ZN"),  # zinc porphyrin, listed by name
        group(7, "U10", "C"),  # a quinone, listed by name
        group(8, "CU", "CU"),  # a redox-active metal ion
        group(9, "XYZ", "FE"),  # an unfamiliar name that holds iron
        group(10, "VO4", "V"),  # free oxyanions
        group(11, "MOO", "MO"),
        group(12, "WO4", "W"),
        group(13, "NCO", "CO"),  # cobalt hexammine
        group(14, "NI", "NI"),  # a lone nickel ion that no sulfur holds
        group(15, "CO", "CO"),  # a lone cobalt ion
        group(16, "MO", "MO"),  # the metal of a molybdenum cofactor
        group(17, "F43", "NI"),  # nickel in a larger group, as in F430
        group(17, "F43", "C"),
    )
    context = build_cofactor_context(cleaned)

    assert [site.label for site in context.sites] == [
        "ZNH",
        "U10",
        "CU",
        "XYZ",
        "MO",
        "F43",
    ]


def test_a_lone_nickel_ion_is_a_cofactor_only_when_sulfur_holds_it() -> None:
    def atom(
        index: int, name: str, atom_name: str, element: str, x: float, y: float
    ) -> PreparedAtom:
        return make_prepared_atom(
            source_atom_index=index,
            residue_name=name,
            residue_number=index,
            atom_name=atom_name,
            element=element,
            is_hetatm=name not in {"CYS", "HIS"},
            x=x,
            y=y,
        )

    def nickel(index: int, x: float) -> PreparedAtom:
        return atom(index, "NI", "NI", "NI", x, 0.0)

    cleaned = (
        # Two cysteine thiolates, as in nickel superoxide dismutase.
        nickel(1, 0.0),
        atom(2, "CYS", "SG", "S", 0.0, 2.2),
        atom(3, "CYS", "SG", "S", 0.0, -2.3),
        # One thiolate and a cluster sulfide, as in the C-cluster.
        nickel(4, 100.0),
        atom(5, "CYS", "SG", "S", 100.0, 2.2),
        atom(6, "S", "S", "S", 100.0, -2.3),
        # A single thiolate.
        nickel(7, 200.0),
        atom(8, "CYS", "SG", "S", 200.0, 2.2),
        # Two sulfur atoms beyond bonding distance.
        nickel(9, 300.0),
        atom(10, "CYS", "SG", "S", 300.0, 2.8),
        atom(11, "CYS", "SG", "S", 300.0, -2.8),
        # Histidine, as in urease or a His-tag.
        nickel(12, 400.0),
        atom(13, "HIS", "NE2", "N", 400.0, 2.1),
        atom(14, "HIS", "NE2", "N", 400.0, -2.1),
    )

    sites = build_cofactor_context(cleaned).sites

    assert [site.residue_number for site in sites] == [1, 4]


def test_cofactor_sites_follow_the_chain_order_of_the_model() -> None:
    def site(index: int, chain_id: str) -> PreparedAtom:
        return make_prepared_atom(
            source_atom_index=index,
            chain_id=chain_id,
            residue_name="SF4",
            residue_number=301,
            atom_name="FE1",
            element="FE",
            is_hetatm=True,
            x=20.0 * index,
        )

    cleaned = (
        make_prepared_atom(source_atom_index=0, chain_id="2"),
        make_prepared_atom(source_atom_index=1, chain_id="10"),
        site(2, "10"),
        site(3, "2"),
    )

    sites = build_cofactor_context(cleaned).sites

    assert [(s.label, s.chain_id) for s in sites] == [("SF4-1", "2"), ("SF4-2", "10")]


def test_cofactor_sites_are_numbered_and_other_groups_ignored() -> None:
    context = build_cofactor_context(_cleaned_model())

    assert [site.label for site in context.sites] == ["FMN", "SF4-1", "SF4-2"]
    assert [(site.chain_id, site.residue_number) for site in context.sites] == [
        ("A", 201),
        ("A", 301),
        ("B", 302),
    ]
    assert (
        cofactor_site_summary(context.sites)
        == "FMN (Fmn201 chain A), SF4-1 (Sf4301 chain A), SF4-2 (Sf4302 chain B)"
    )
    assert cofactor_site_summary(()) == "none detected"
    assert build_cofactor_context(()).sites == ()


def test_labels_and_tokens() -> None:
    atom = make_atom_record(
        chain_id="B",
        residue_name="his",
        residue_number=87,
        insertion_code="A",
        atom_name=" CB ",
    )
    assert atom_token(atom) == "B:87A:CB"
    assert residue_label(atom) == "His87A"
    assert format_residue_label("TRP", None) == "Trp"


def _path(
    residue_names: Sequence[str],
    *,
    cofactors: Sequence[tuple[str, str, tuple[float, float, float]]] = (),
) -> PathReport:
    """A straight path with one residue per atom, 3.5 A apart along x.

    Each cofactor is one atom, given as its residue name, element and position.
    """
    scored = scored_atoms_from_coordinates(
        [(3.5 * i, 0.0, 0.0) for i in range(len(residue_names))],
        residue_names=list(residue_names),
        residue_numbers=list(range(1, len(residue_names) + 1)),
    )
    cleaned = [
        make_prepared_atom(
            source_atom_index=200 + i, residue_number=90 + i, x=100.0 + i
        )
        for i in range(4)
    ]
    for index, (name, element, (x, y, z)) in enumerate(cofactors):
        cleaned.append(
            make_prepared_atom(
                source_atom_index=300 + index,
                residue_name=name,
                residue_number=201 + 100 * index,
                atom_name=element,
                element=element,
                is_hetatm=True,
                x=x,
                y=y,
                z=z,
            )
        )
    clusters = cluster_selected_atoms(scored, distance_cutoff=4.0)
    assert len(clusters) == 1
    return analyze_paths(clusters, cleaned_atoms=cleaned).reports[0]


def test_a_cofactor_in_range_of_any_atom_is_near_the_path() -> None:
    names = ["TRP", "ALA", "LEU", "TYR", "VAL", "GLY"]
    at_start = _path(names, cofactors=[("FMN", "N", (-3.0, 0.0, 0.0))])
    assert at_start.is_near_cofactor and len(at_start.nearby_cofactors) == 1
    cofactor = at_start.nearby_cofactors[0]
    assert (cofactor.label, cofactor.residue, cofactor.atom_token) == (
        "FMN",
        "Trp1",
        "A:1:CA",
    )
    assert cofactor.distance == approx(3.0)
    assert at_start.nearby_cofactor_summary == "FMN at 3.0 A"

    # A cofactor beside the middle of a long path, out of range of both ends,
    # is still near it. Gly4 and Trp5 are equally near, and Gly4 comes first.
    names = ["ALA", "LEU", "VAL", "GLY", "TRP", "ILE", "PRO", "ALA"]
    midway = _path(names, cofactors=[("FMN", "N", (12.25, 3.0, 0.0))])
    assert [cofactor.label for cofactor in midway.nearby_cofactors] == ["FMN"]
    assert midway.nearby_cofactors[0].distance == approx(np.hypot(1.75, 3.0))
    assert midway.nearby_cofactors[0].residue == "Gly4"

    # Just beyond the proximity range of every atom.
    distant = _path(names, cofactors=[("FMN", "N", (12.25, 8.5, 0.0))])
    assert distant.nearby_cofactors == () and not distant.is_near_cofactor
    assert distant.nearby_cofactor_summary == "none"

    assert _path(names).nearby_cofactors == ()


def test_a_path_between_two_cofactors_is_near_both() -> None:
    names = ["ALA", "LEU", "VAL", "GLY", "TRP", "ILE", "PRO", "ALA"]
    bridging = _path(
        names,
        cofactors=[
            ("FMN", "N", (-3.0, 0.0, 0.0)),  # 3.0 A from Ala1
            ("SF4", "FE", (27.0, 0.0, 0.0)),  # 2.5 A from Ala8
            ("HEM", "FE", (12.25, 9.0, 0.0)),  # out of range
        ],
    )

    # The nearest comes first, whatever the site order.
    assert [cofactor.label for cofactor in bridging.nearby_cofactors] == ["SF4", "FMN"]
    assert [cofactor.distance for cofactor in bridging.nearby_cofactors] == approx(
        [2.5, 3.0]
    )
    assert [cofactor.residue for cofactor in bridging.nearby_cofactors] == [
        "Ala8",
        "Ala1",
    ]
    assert bridging.nearby_cofactor_summary == "SF4 at 2.5 A, FMN at 3.0 A"


def test_residue_path_listing_and_step_provenance() -> None:
    coordinates = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2.5, 1.0, 0.0),  # Trp76: CA, O, CB
        (5.5, 1.0, 0.0),  # His87 CB, a tertiary step
        (5.5, 4.5, 0.0),  # His88 CA, a peptide step
        (5.5, 8.0, 0.0),  # chain B Ala5, an interchain step
    ]
    scored = scored_atoms_from_coordinates(
        coordinates,
        residue_names=["TRP", "TRP", "TRP", "HIS", "HIS", "ALA"],
        residue_numbers=[76, 76, 76, 87, 88, 5],
        chain_ids=["A", "A", "A", "A", "A", "B"],
        atom_names=["CA", "O", "CB", "CB", "CA", "CA"],
    )
    clusters = cluster_selected_atoms(scored, distance_cutoff=4.0)
    # The peptide bond of His87 to His88, which the model holds though the
    # path does not.
    cleaned = _backbone_bonds([(("HIS", 87, ""), ("HIS", 88, ""), True)])

    report = analyze_paths(clusters, cleaned_atoms=cleaned).reports[0]

    # Chains are ordered by the lowest residue number each contributes, so
    # chain B's Ala5 leads; atoms keep the order the graph reached them in.
    assert (
        report.residue_path
        == "Chain B - Ala5 (CA) -> Chain A - Trp76 (CA, CB, O) -> His87 (CB) -> His88 (CA)"
    )
    assert report.chains == ("B", "A")
    assert (report.residue_count, report.node_count) == (4, 6)
    assert [r.label for r in report.residues] == ["Ala5", "Trp76", "His87", "His88"]
    assert set(report.residues[1].atom_names) == {"CA", "O", "CB"}
    assert report.path_id == 1 and report.total_length == approx(
        report.cluster.metrics.mst_total_length
    )
    edges = report.edges
    assert (edges.intra_residue, edges.peptide, edges.tertiary, edges.interchain) == (
        2,
        1,
        1,
        1,
    )
    assert edges.between_residues == 3 and edges.peptide_fraction == approx(1 / 3)
    assert [s.step_index for s in report.steps] == [1, 2, 3, 4, 5]
    assert {s.kind for s in report.steps} == {
        "intra_residue",
        "peptide",
        "tertiary",
        "interchain",
    }
    interchain = next(s for s in report.steps if s.kind == "interchain")
    assert {interchain.first_atom, interchain.second_atom} == {"A:88:CA", "B:5:CA"}
    assert report.composition.fractions["aromatic"] == approx(0.75)


def test_single_residue_path() -> None:
    scored = scored_atoms_from_coordinates(
        [(0, 0, 0), (1.4, 0, 0), (2.8, 0, 0)],
        residue_names=["PHE"] * 3,
        residue_numbers=[9] * 3,
        atom_names=["CG", "CD1", "CE1"],
    )
    report = analyze_paths(
        cluster_selected_atoms(scored),
        cleaned_atoms=(),
    ).reports[0]

    assert report.residue_path.startswith("Chain A - Phe9 (CG, ")
    assert set(report.residues[0].atom_names) == {"CG", "CD1", "CE1"}
    assert report.edges.between_residues == 0 and report.edges.peptide_fraction == 0.0


def _backbone_bonds(
    pairs: Sequence[tuple[tuple[str, int, str], tuple[str, int, str], bool]],
) -> list[PreparedAtom]:
    """The C of the first residue and the N of the second, for each pair.

    Each residue is its name, number and insertion code, in chain A. The two
    atoms are 1.33 A apart, a peptide bond, when the pair is bonded, and
    3.5 A apart, a chain break, when it is not.
    """
    atoms: list[PreparedAtom] = []
    for index, (first, second, bonded) in enumerate(pairs):
        x = 100.0 + 10.0 * index
        for (name, number, icode), atom_name, atom_x in (
            (first, "C", x),
            (second, "N", x + (1.33 if bonded else 3.5)),
        ):
            atoms.append(
                make_prepared_atom(
                    source_atom_index=1000 + len(atoms),
                    residue_name=name,
                    residue_number=number,
                    insertion_code=icode,
                    atom_name=atom_name,
                    element=atom_name,
                    x=atom_x,
                )
            )
    return atoms


def test_peptide_steps_follow_modelled_bonds_not_numbers() -> None:
    residues = [(10, ""), (11, ""), (52, ""), (52, "B"), (60, ""), (70, "")]
    scored = tuple(
        replace(atom, record=replace(atom.record, insertion_code=icode))
        for atom, (_, icode) in zip(
            scored_atoms_from_coordinates(
                [(3.5 * i, 0.0, 0.0) for i in range(len(residues))],
                residue_numbers=[number for number, _ in residues],
            ),
            residues,
            strict=True,
        )
    )
    cleaned = _backbone_bonds(
        [
            # Consecutive numbers across a chain break.
            (("ALA", 10, ""), ("ALA", 11, ""), False),
            # One number, but 52A lies between them.
            (("ALA", 52, ""), ("ALA", 52, "A"), True),
            (("ALA", 52, "A"), ("ALA", 52, "B"), True),
            # A jump in numbering across an unbroken chain.
            (("ALA", 60, ""), ("ALA", 70, ""), True),
        ]
    )

    report = analyze_paths(
        cluster_selected_atoms(scored, distance_cutoff=4.0), cleaned_atoms=cleaned
    ).reports[0]

    kinds = {
        frozenset((step.first_residue, step.second_residue)): step.kind
        for step in report.steps
    }
    assert kinds == {
        frozenset(("Ala10", "Ala11")): "tertiary",
        frozenset(("Ala11", "Ala52")): "tertiary",
        frozenset(("Ala52", "Ala52B")): "tertiary",
        frozenset(("Ala52B", "Ala60")): "tertiary",
        frozenset(("Ala60", "Ala70")): "peptide",
    }
    assert report.edges.peptide_fraction == approx(1 / 5)
