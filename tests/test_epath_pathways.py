"""Residue chemistry, topology, endpoints, cofactor anchors, and pathway description."""

from collections.abc import Sequence

import numpy as np
import pytest
from support import (
    approx,
    make_atom_record,
    make_prepared_atom,
    make_scored_atom,
    scored_atoms_from_coordinates,
)

from epath.analysis.clustering import cluster_selected_atoms, mst_length_stats
from epath.analysis.endpoints import (
    annotate_endpoint,
    build_endpoint_context,
    cofactor_site_summary,
)
from epath.analysis.labels import atom_token, format_residue_label, residue_label
from epath.analysis.pathways import PathReport, analyze_paths
from epath.analysis.residues import (
    ResidueClass,
    analyze_composition,
    class_counts,
    class_fractions,
    residue_class,
)
from epath.analysis.topology import PathTopology, analyze_topology
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
        ("XYZ", "other"),
    ],
)
def test_residue_classes(name: str, expected: str) -> None:
    assert residue_class(name) == expected


def test_composition_fractions_sum_to_one_and_report_dominance_and_enrichment() -> None:
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

    background: dict[ResidueClass, float] = {
        "aromatic": 0.08,
        "sulfur": 0.03,
        "charged": 0.25,
        "polar": 0.25,
        "hydrophobic": 0.39,
        "other": 0.0,
    }
    composition = analyze_composition(names, background=background)

    assert composition.counts == counts
    assert composition.fractions["aromatic"] == approx(0.2)
    assert composition.fractions["charged"] == approx(0.1)
    assert composition.dominant_classes == ("hydrophobic",)
    assert composition.enriched_classes == (
        "aromatic",
        "sulfur",
    )  # 0.2 >= 1.5*0.08, 0.1 >= 1.5*0.03

    mixed = analyze_composition(["ALA"] * 5 + ["ASP"] * 6, background=background)
    assert mixed.dominant_classes == ("charged", "hydrophobic")  # 5/11 >= 0.8 * 6/11


def _topology(
    coordinates: Sequence[tuple[float, float, float]],
    residue_indices: Sequence[int],
    cutoff: float = 4.0,
) -> PathTopology:
    """Classify one-atom-per-residue coordinates as a cluster topology."""
    array = np.asarray(coordinates, dtype=float)
    return analyze_topology(
        array, residue_indices, mst_length_stats(array), distance_cutoff=cutoff
    )


def test_topology_classes() -> None:
    line = _topology([(3.5 * i, 0, 0) for i in range(6)], list(range(6)))
    assert (
        line.topology == "linear"
        and line.linearity == approx(1.0)
        and line.branch_fraction == 0.0
    )

    # A hub residue in contact with three arms: one junction in four residues.
    arms = [(3.5, 0, 0), (-3.5, 0, 0), (0, 3.5, 0)]
    star = _topology([(0, 0, 0), *arms], [0, 1, 2, 3])
    assert star.topology == "branched" and star.branch_fraction == approx(0.25)
    # Atoms at exactly the cutoff are in contact, and atoms just beyond are not.
    assert _topology([(0, 0, 0), *arms], [0, 1, 2, 3], cutoff=3.5).branch_fraction > 0
    assert _topology([(0, 0, 0), *arms], [0, 1, 2, 3], cutoff=3.4).branch_fraction == 0
    # A hub residue of two atoms touches each arm twice, which is still one
    # contact per arm, and its own two atoms are not a contact at all.
    doubled = _topology([(0, 0, 0), (0, 0, 1), *arms], [0, 0, 1, 2, 3])
    assert doubled.branch_fraction == approx(0.25)

    # A tightly wound coil: a long path packed into a short span, and only
    # consecutive residues in contact, so neither linear nor branched.
    steps = np.arange(40)
    coil = [
        (3.0 * np.cos(np.radians(30 * i)), 3.0 * np.sin(np.radians(30 * i)), 0.2 * i)
        for i in steps
    ]
    compact = _topology(coil, list(range(40)), cutoff=1.6)
    assert compact.topology == "clustered"
    assert compact.linearity < 0.2 and compact.branch_fraction == 0.0


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
    )
    context = build_endpoint_context(
        scored_atoms=[make_scored_atom()], cleaned_atoms=cleaned
    )

    assert [site.label for site in context.sites] == ["ZNH", "U10", "CU", "XYZ"]


def test_cofactor_sites_are_numbered_and_other_groups_ignored() -> None:
    context = build_endpoint_context(
        scored_atoms=[make_scored_atom()], cleaned_atoms=_cleaned_model()
    )

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
    assert (
        build_endpoint_context(
            scored_atoms=[make_scored_atom()], cleaned_atoms=()
        ).sites
        == ()
    )


def test_endpoints_are_labelled_in_order_of_mechanistic_weight() -> None:
    scored = [
        make_scored_atom(packing_density=d) for d in (10, 20, 30, 40, 50, 60, 70, 80)
    ]
    context = build_endpoint_context(
        scored_atoms=scored, cleaned_atoms=_cleaned_model()
    )
    assert context.low_packing_density == approx(
        np.percentile([10, 20, 30, 40, 50, 60, 70, 80], 25)
    )

    contact = annotate_endpoint(
        make_scored_atom(
            chain_id="A", residue_name="TRP", residue_number=7, x=9.0, packing_density=1
        ),
        context,
    )
    assert (contact.kind, contact.label) == ("cofactor", "FMN")
    assert contact.residue == "Trp7" and contact.atom_token == "A:7:CA"

    region = annotate_endpoint(
        make_scored_atom(chain_id="A", x=5.0, packing_density=1), context
    )
    assert (region.kind, region.label) == ("cofactor_region", "FMN region")

    # Within 4 A of chain B's protein atom, but more than 8 A from any cofactor.
    interface = annotate_endpoint(
        make_scored_atom(chain_id="A", x=30.0, y=0.0, packing_density=1), context
    )
    assert (interface.kind, interface.label) == ("interface", "Interface region")

    low = annotate_endpoint(
        make_scored_atom(chain_id="A", x=-20.0, packing_density=20), context
    )
    assert (low.kind, low.label) == ("low_packing", "Low-packing region")

    typical = annotate_endpoint(
        make_scored_atom(chain_id="A", x=-20.0, packing_density=70), context
    )
    assert (typical.kind, typical.label) == (
        "typical_packing",
        "Typical-packing region",
    )

    # An atom of the same chain does not make an interface.
    assert (
        annotate_endpoint(
            make_scored_atom(chain_id="B", x=30.0, packing_density=70), context
        ).kind
        == "typical_packing"
    )


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
    cofactor_at_start: bool,
    cofactor_position: tuple[float, float, float] = (-3.0, 0.0, 0.0),
    spacing: float = 3.5,
    extra_atoms: Sequence[tuple[float, float, float]] = (),
) -> PathReport:
    """A straight path with one residue per atom, optionally with an FMN beside it."""
    count = len(residue_names)
    coordinates = [(spacing * i, 0.0, 0.0) for i in range(count)] + list(extra_atoms)
    scored = scored_atoms_from_coordinates(
        coordinates,
        residue_names=list(residue_names) + ["ALA"] * len(extra_atoms),
        residue_numbers=list(range(1, count + 1)) + [count] * len(extra_atoms),
    )
    cleaned = [
        make_prepared_atom(
            source_atom_index=200 + i, residue_number=90 + i, x=100.0 + i
        )
        for i in range(4)
    ]
    if cofactor_at_start:
        cleaned.append(
            make_prepared_atom(
                source_atom_index=300,
                residue_name="FMN",
                residue_number=201,
                atom_name="N1",
                element="N",
                is_hetatm=True,
                x=cofactor_position[0],
                y=cofactor_position[1],
                z=cofactor_position[2],
            )
        )
    clusters = cluster_selected_atoms(scored, distance_cutoff=4.0)
    assert len(clusters) == 1
    return analyze_paths(clusters, scored_atoms=scored, cleaned_atoms=cleaned).reports[
        0
    ]


def test_path_anchor_is_the_nearest_cofactor_to_any_atom() -> None:
    at_start = _path(["TRP", "ALA", "LEU", "TYR", "VAL", "GLY"], cofactor_at_start=True)
    assert at_start.topology.topology == "linear"
    assert (
        at_start.start_endpoint.kind == "cofactor"
        and at_start.end_endpoint.kind == "low_packing"
    )
    assert at_start.endpoint_summary == "FMN -> Low-packing region"
    assert at_start.anchor is not None and at_start.is_cofactor_anchored
    assert (at_start.anchor.label, at_start.anchor.residue) == ("FMN", "Trp1")
    assert at_start.anchor.distance == approx(3.0)
    assert at_start.anchor_summary == "FMN at 3.0 A"

    # A cofactor beside the middle of a long path is out of range of both ends,
    # so neither endpoint is labelled with it, but the path is still anchored.
    names = ["ALA", "LEU", "VAL", "GLY", "TRP", "ILE", "PRO", "ALA"]
    midway = _path(names, cofactor_at_start=True, cofactor_position=(12.25, 3.0, 0.0))
    assert midway.start_endpoint.kind == "low_packing"
    assert midway.end_endpoint.kind == "low_packing"
    assert midway.anchor is not None and midway.anchor.label == "FMN"
    assert midway.anchor.distance == approx(np.hypot(1.75, 3.0))
    assert midway.anchor.residue in {"Gly4", "Trp5"}

    # Just beyond the proximity range of every atom.
    distant = _path(names, cofactor_at_start=True, cofactor_position=(12.25, 8.5, 0.0))
    assert distant.anchor is None and not distant.is_cofactor_anchored

    unanchored = _path(
        ["ALA", "LEU", "VAL", "GLY", "ILE", "PRO"], cofactor_at_start=False
    )
    assert unanchored.anchor is None
    assert unanchored.anchor_summary == "no cofactor in range"
    assert unanchored.start_endpoint.kind == "low_packing"
    assert unanchored.structural_descriptor == "Hydrophobic corridor"


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

    report = analyze_paths(clusters, scored_atoms=scored, cleaned_atoms=()).reports[0]

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
    assert report.start_endpoint.atom_token in {"A:76:CA", "B:5:CA"}


def test_single_residue_hotspot_descriptor() -> None:
    scored = scored_atoms_from_coordinates(
        [(0, 0, 0), (1.4, 0, 0), (2.8, 0, 0)],
        residue_names=["PHE"] * 3,
        residue_numbers=[9] * 3,
        atom_names=["CG", "CD1", "CE1"],
    )
    report = analyze_paths(
        cluster_selected_atoms(scored),
        scored_atoms=scored,
        cleaned_atoms=(),
    ).reports[0]

    assert report.structural_descriptor == "Single aromatic hotspot"
    assert report.residue_path.startswith("Chain A - Phe9 (CG, ")
    assert set(report.residues[0].atom_names) == {"CG", "CD1", "CE1"}
    assert report.edges.between_residues == 0 and report.edges.peptide_fraction == 0.0
