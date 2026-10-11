"""Residue chemistry, topology, endpoints, and pathway classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from support import (
    approx,
    make_epath_atom,
    make_prepared_atom,
    scored_atoms_from_coordinates,
)

from epath.analysis.atoms import atom_token, format_residue_label, residue_label
from epath.analysis.clustering import cluster_selected_atoms, mst_length_stats
from epath.analysis.endpoints import (
    annotate_endpoint,
    build_endpoint_context,
    cofactor_site_summary,
)
from epath.analysis.pathways import PathAnalysisOptions, PathReport, analyze_paths
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

    assert composition.residue_count == 10
    assert composition.redox_fraction == approx(0.3)
    assert composition.charged_fraction == approx(0.1)
    assert composition.dominant_classes == ("hydrophobic",)
    assert composition.enriched_classes == (
        "aromatic",
        "sulfur",
    )  # 0.2 >= 1.5*0.08, 0.1 >= 1.5*0.03
    assert analyze_composition(names).enriched_classes == ()

    mixed = analyze_composition(["ALA"] * 5 + ["ASP"] * 6)
    assert mixed.dominant_classes == ("charged", "hydrophobic")  # 5/11 >= 0.8 * 6/11
    assert analyze_composition([]).dominant_classes == ()


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
    star = _topology([(0, 0, 0), (3.5, 0, 0), (-3.5, 0, 0), (0, 3.5, 0)], [0, 1, 2, 3])
    assert star.topology == "branched" and star.branch_fraction == approx(0.25)

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

    empty = analyze_topology(
        np.zeros((0, 3)), [], mst_length_stats(np.zeros((0, 3))), distance_cutoff=4.0
    )
    assert (empty.topology, empty.linearity, empty.branch_fraction) == (
        "clustered",
        0.0,
        0.0,
    )


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
        scored_atoms=[make_epath_atom()], cleaned_atoms=cleaned
    )

    assert [site.label for site in context.sites] == ["ZNH", "U10", "CU", "XYZ"]


def test_cofactor_sites_are_numbered_and_other_groups_ignored() -> None:
    context = build_endpoint_context(
        scored_atoms=[make_epath_atom()], cleaned_atoms=_cleaned_model()
    )

    assert [site.label for site in context.sites] == ["FMN", "SF4-1", "SF4-2"]
    assert [
        (site.chain_id, site.residue_number, site.atom_count) for site in context.sites
    ] == [("A", 201, 1), ("A", 301, 1), ("B", 302, 1)]
    assert context.chain_ids == ("A", "B")
    assert (
        cofactor_site_summary(context.sites)
        == "FMN (Fmn201 chain A), SF4-1 (Sf4301 chain A), SF4-2 (Sf4302 chain B)"
    )
    assert cofactor_site_summary(()) == "none detected"
    assert build_endpoint_context(scored_atoms=[make_epath_atom()]).sites == ()


def test_endpoints_are_labelled_in_order_of_mechanistic_weight() -> None:
    scored = [
        make_epath_atom(packing_density=d) for d in (10, 20, 30, 40, 50, 60, 70, 80)
    ]
    context = build_endpoint_context(
        scored_atoms=scored, cleaned_atoms=_cleaned_model()
    )
    assert context.surface_packing_density == approx(
        np.percentile([10, 20, 30, 40, 50, 60, 70, 80], 25)
    )

    contact = annotate_endpoint(
        make_epath_atom(
            chain_id="A", residue_name="TRP", residue_number=7, x=9.0, packing_density=1
        ),
        context,
    )
    assert (contact.kind, contact.label) == ("cofactor", "FMN")
    assert contact.cofactor is not None and contact.cofactor.label == "FMN"
    assert contact.cofactor_distance == approx(3.0)
    assert (
        contact.is_cofactor_anchored
        and contact.residue == "Trp7"
        and contact.atom_token == "A:7:CA"
    )

    region = annotate_endpoint(
        make_epath_atom(chain_id="A", x=5.0, packing_density=1), context
    )
    assert (region.kind, region.label) == ("cofactor_region", "FMN region")

    # Within 4 A of chain B's protein atom, but more than 8 A from any cofactor.
    interface = annotate_endpoint(
        make_epath_atom(chain_id="A", x=30.0, y=0.0, packing_density=1), context
    )
    assert (interface.kind, interface.label) == ("interface", "Interface region")
    assert not interface.is_cofactor_anchored

    surface = annotate_endpoint(
        make_epath_atom(chain_id="A", x=-20.0, packing_density=20), context
    )
    assert (surface.kind, surface.label) == ("surface", "Surface region")

    buried = annotate_endpoint(
        make_epath_atom(chain_id="A", x=-20.0, packing_density=70), context
    )
    assert (buried.kind, buried.label) == ("buried", "Buried region")

    # An atom of the same chain does not make an interface.
    assert (
        annotate_endpoint(
            make_epath_atom(chain_id="B", x=30.0, packing_density=70), context
        ).kind
        == "buried"
    )


def test_labels_and_tokens() -> None:
    atom = make_epath_atom(
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
    spacing: float = 3.5,
    extra_atoms: Sequence[tuple[float, float, float]] = (),
) -> PathReport:
    """A straight path with one residue per atom, optionally anchored on FMN."""
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
                x=-3.0,
            )
        )
    clusters = cluster_selected_atoms(scored, distance_cutoff=4.0)
    assert len(clusters) == 1
    return analyze_paths(
        clusters, scored_atoms=[a.atom for a in scored], cleaned_atoms=cleaned
    ).reports[0]


def test_conductor_modulator_reservoir_and_structural_classes() -> None:
    conductor = _path(
        ["TRP", "ALA", "LEU", "TYR", "VAL", "GLY"], cofactor_at_start=True
    )
    assert conductor.path_class == "conductor" and conductor.class_label == "Cond."
    assert conductor.topology.topology == "linear"
    assert (
        conductor.start_endpoint.kind == "cofactor"
        and conductor.end_endpoint.kind == "surface"
    )
    assert conductor.endpoint_summary == "FMN -> Surface region"

    charged = _path(["TRP", "ASP", "GLU", "LYS", "ARG", "GLY"], cofactor_at_start=True)
    assert charged.path_class == "modulator"

    aliphatic = _path(
        ["ALA", "LEU", "VAL", "GLY", "ILE", "PRO"], cofactor_at_start=True
    )
    assert aliphatic.path_class == "modulator"

    reservoir = _path(
        ["TRP", "TYR", "ALA", "LEU", "VAL", "GLY"], cofactor_at_start=False
    )
    assert (
        reservoir.path_class == "reservoir"
        and reservoir.start_endpoint.kind == "surface"
    )

    structural = _path(
        ["ALA", "LEU", "VAL", "GLY", "ILE", "PRO"], cofactor_at_start=False
    )
    assert structural.path_class == "structural"
    assert structural.structural_descriptor == "Hydrophobic corridor"

    # Thresholds are options: a stricter redox requirement demotes the reservoir.
    strict = analyze_paths(
        [reservoir.cluster],
        scored_atoms=[a.atom for a in reservoir.cluster.atoms],
        options=PathAnalysisOptions(reservoir_redox_fraction=0.5),
    ).reports[0]
    assert strict.path_class == "structural"


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

    report = analyze_paths(clusters, scored_atoms=[a.atom for a in scored]).reports[0]

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
        cluster_selected_atoms(scored), scored_atoms=[a.atom for a in scored]
    ).reports[0]

    assert report.structural_descriptor == "Single aromatic hotspot"
    assert report.residue_path.startswith("Chain A - Phe9 (CG, ")
    assert set(report.residues[0].atom_names) == {"CG", "CD1", "CE1"}
    assert report.edges.between_residues == 0 and report.edges.peptide_fraction == 0.0
