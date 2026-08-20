# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Integration tests for GBMaker's exact integer grain-construction path."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import CSLExactSpec, PQSpec
from GBOpt.GBMaker import GBMaker
from tests.data.zhang_2022_uo2_ceo2_gb_energies import BOUNDARIES

# --------------------------------------------------------------------------------------
# Shared boundary specifications and material data
# --------------------------------------------------------------------------------------

SIGMA5_TILT_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SIGMA5_TILT_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    quat=[3, 0, 0, 1],
)
SIGMA5_TILT_PQ_SPEC = PQSpec(
    P=SIGMA5_TILT_P,
    Q=SIGMA5_TILT_Q,
    basis_mode="supplied",
)

A0_FCC = 3.615
STRUCTURE_FCC = "fcc"
ATOM_TYPES_FCC = "Cu"
COINCIDENCE_TOLERANCE_ANGSTROM = 1e-4

EXACT_BOUNDARY_SPECS = [
    pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
    pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
]

EXACT_BOX_CASES = [
    pytest.param(
        SIGMA5_TILT_PQ_SPEC,
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        id="pq-fcc",
    ),
    pytest.param(
        SIGMA5_TILT_EXACT_SPEC,
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        id="csl-exact-fcc",
    ),
    pytest.param(
        SIGMA5_TILT_PQ_SPEC,
        5.47,
        "fluorite",
        ("U", "O"),
        id="pq-fluorite",
    ),
]

VACUUM_ZERO_BOX_CASES = [
    pytest.param(
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        {"interaction_distance": A0_FCC, "repeat_factor": 2},
        id="fcc-default-thickness",
    ),
    pytest.param(
        5.47,
        "fluorite",
        ("U", "O"),
        {"interaction_distance": 5.47, "repeat_factor": 2},
        id="fluorite-default-thickness",
    ),
    pytest.param(
        5.47,
        "fluorite",
        ("U", "O"),
        {"x_dim_min": 20.0, "interaction_distance": 1.0, "repeat_factor": 2},
        id="fluorite-reduced-thickness",
    ),
]

REPRESENTATIVE_EXACT_CASES = [
    pytest.param(
        [[0, 18, -1], [0, 1, 18], [1, 0, 0]],
        [[0, 1, -18], [0, 18, 1], [1, 0, 0]],
        19_500,
        19_500,
        id="zhang-001-ST-100",
    ),
    pytest.param(
        [[0, -5, 14], [1, 0, 0], [0, 14, 5]],
        [[0, 10, 11], [1, 0, 0], [0, 11, -10]],
        13_260,
        13_260,
        id="zhang-031-AT-100",
    ),
    pytest.param(
        [[0, 0, 1], [4, 1, 0], [-1, 4, 0]],
        [[0, 0, 1], [4, -1, 0], [1, 4, 0]],
        2_448,
        2_448,
        id="zhang-041-TW-100",
    ),
    pytest.param(
        [[-1, -1, 6], [1, -1, 0], [3, 3, 1]],
        [[1, 1, 12], [1, -1, 0], [6, 6, -1]],
        112_176,
        220_752,
        marks=pytest.mark.slow(
            reason="large asymmetric representative contains more than 330,000 atoms"
        ),
        id="zhang-086-AT-110",
    ),
]


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def build_gb():
    """Return a function-scoped GBMaker factory with compact exact-path defaults."""

    def _build(
        boundary=SIGMA5_TILT_PQ_SPEC,
        *,
        a0=A0_FCC,
        structure=STRUCTURE_FCC,
        atom_types=ATOM_TYPES_FCC,
        mode="exact",
        **overrides,
    ):
        kwargs = {
            "gb_thickness": 0.0,
            "repeat_factor": 2,
            "interaction_distance": a0,
        }
        kwargs.update(overrides)
        return GBMaker.from_boundary_spec(
            a0,
            structure,
            atom_types,
            boundary,
            mode=mode,
            **kwargs,
        )

    return _build


def _positions(atoms):
    """Return structured atom coordinates as an ``(N, 3)`` float array."""
    return np.column_stack((atoms["x"], atoms["y"], atoms["z"]))


def _assert_fluorite_stoichiometry(atoms, *, label):
    """Assert a nonempty atom collection contains only the expected UO2 ratio."""
    uranium_count = int(np.count_nonzero(atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(atoms["name"] == "O"))

    assert uranium_count > 0, f"{label} contains no U atoms"
    assert oxygen_count == 2 * uranium_count, (
        f"{label} stoichiometry is {uranium_count} U to {oxygen_count} O; "
        "expected UO2"
    )
    assert uranium_count + oxygen_count == len(atoms), (
        f"{label} contains atom species other than U and O"
    )


def _assert_rocksalt_stoichiometry(atoms, *, label):
    """Assert a nonempty atom collection contains only the expected NaCl ratio."""
    sodium_count = int(np.count_nonzero(atoms["name"] == "Na"))
    chlorine_count = int(np.count_nonzero(atoms["name"] == "Cl"))

    assert sodium_count > 0, f"{label} contains no Na atoms"
    assert chlorine_count == sodium_count, (
        f"{label} stoichiometry is {sodium_count} Na to {chlorine_count} Cl; "
        "expected NaCl"
    )
    assert sodium_count + chlorine_count == len(atoms), (
        f"{label} contains atom species other than Na and Cl"
    )


def _assert_complete_fluorite_population(atoms, expected):
    """Assert exact total and species populations for complete fluorite sites."""
    assert len(atoms) == expected
    cell_count = expected // 12
    uranium_count = int(np.count_nonzero(atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(atoms["name"] == "O"))

    assert uranium_count == 4 * cell_count
    assert oxygen_count == 8 * cell_count
    assert uranium_count + oxygen_count == len(atoms)


def _assert_atoms_within_box(gb, atoms):
    """Assert finite atom coordinates lie inside the GBMaker half-open box."""
    positions = _positions(atoms)
    tolerance = max(1e-8, 100.0 * gb.epsilon)
    upper_bounds = np.array([gb.x_dim, gb.y_dim, gb.z_dim], dtype=float)

    assert np.all(np.isfinite(positions))
    assert np.all(np.min(positions, axis=0) >= -tolerance)
    assert np.all(np.max(positions, axis=0) < upper_bounds + tolerance)


def _assert_no_coincident_periodic_sites(gb):
    """Assert the final periodic whole system contains no coincident representatives."""
    box_lengths = np.array([gb.x_dim, gb.y_dim, gb.z_dim], dtype=float)
    positions = np.mod(_positions(gb.whole_system), box_lengths)
    tree = KDTree(positions, boxsize=box_lengths)
    nearest_distances, _ = tree.query(positions, k=2)
    coincident_count = int(
        np.count_nonzero(
            nearest_distances[:, 1] <= COINCIDENCE_TOLERANCE_ANGSTROM
        )
    )

    assert coincident_count == 0, (
        f"detected {coincident_count} coincident atom representatives under periodic "
        "boundary conditions"
    )


def _build_campaign_style_exact_boundary(P, Q):
    """Build one representative boundary using the campaign's exact-path settings."""
    boundary = PQSpec(P=P, Q=Q, basis_mode="supplied")
    common = {
        "a0": 5.454,
        "structure": "fluorite",
        "atom_types": ("U", "O"),
        "boundary": boundary,
        "mode": "exact",
        "repeat_factor": (1, 1),
        "x_dim_min": 60.0,
        "vacuum": 0.0,
        "interaction_distance": 11.0,
        "mismatch_tol": 0.005,
        "mismatch_max_cells": 50,
        "strain_grain": "both",
    }
    probe = GBMaker.from_boundary_spec(gb_thickness=5.454, **common)
    thickness = 2.0 * max(
        float(probe.spacing["x"]["left"]),
        float(probe.spacing["x"]["right"]),
    )
    return GBMaker.from_boundary_spec(gb_thickness=thickness, **common)


# --------------------------------------------------------------------------------------
# Exact-path dispatch and commensurability
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grain_rows", "row_index", "dimension_name"),
    [
        pytest.param(SIGMA5_TILT_P, 1, "y_dim", id="left-y"),
        pytest.param(SIGMA5_TILT_P, 2, "z_dim", id="left-z"),
        pytest.param(SIGMA5_TILT_Q, 1, "y_dim", id="right-y"),
        pytest.param(SIGMA5_TILT_Q, 2, "z_dim", id="right-z"),
    ],
)
def test_exact_inplane_dimensions_are_integer_multiples_of_both_grain_periods(
    build_gb,
    grain_rows,
    row_index,
    dimension_name,
):
    gb = build_gb()

    period = A0_FCC * np.linalg.norm(
        np.asarray(grain_rows[row_index], dtype=float)
    )
    repeat_count = getattr(gb, dimension_name) / period

    assert repeat_count == pytest.approx(round(repeat_count), abs=1e-6, rel=0.0)


def test_exact_mode_reports_exact_construction(build_gb):
    gb = build_gb()

    assert gb.uses_exact_construction is True
    assert gb.whole_system.size > 0


# --------------------------------------------------------------------------------------
# Exact grain assembly
# --------------------------------------------------------------------------------------


def test_exact_builder_returns_atom_dtype_and_combines_grain_populations(build_gb):
    gb = build_gb()

    assert gb.whole_system.dtype == Atom.atom_dtype
    assert gb.left_grain.size > 0
    assert gb.right_grain.size > 0
    assert gb.whole_system.size == gb.left_grain.size + gb.right_grain.size


# --------------------------------------------------------------------------------------
# Cartesian box and central-interface bounds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "a0", "structure", "atom_types"), EXACT_BOX_CASES)
def test_exact_atoms_are_within_periodic_yz_box(
    build_gb,
    spec,
    a0,
    structure,
    atom_types,
):
    gb = build_gb(
        spec,
        a0=a0,
        structure=structure,
        atom_types=atom_types,
    )
    atoms = gb.whole_system
    tolerance = max(1e-8, 100.0 * gb.epsilon)

    assert np.min(atoms["y"]) >= -tolerance
    assert np.max(atoms["y"]) < gb.y_dim + tolerance
    assert np.min(atoms["z"]) >= -tolerance
    assert np.max(atoms["z"]) < gb.z_dim + tolerance


def test_exact_grains_do_not_cross_central_boundary_plane():
    spec = PQSpec(
        P=[[3, 1, 0], [0, 0, 2], [1, -3, 0]],
        Q=[[3, 1, 0], [0, 0, -2], [-1, 3, 0]],
    )

    with pytest.warns(UserWarning, match=r"Recommended repeat factor is at least 2\."):
        gb = GBMaker.from_boundary_spec(
            3.52,
            "fcc",
            "Ni",
            spec,
            mode="exact",
            gb_thickness=0.0,
            repeat_factor=(1, 3),
            x_dim_min=20.0,
            vacuum=0.0,
            interaction_distance=5.0,
        )

    tolerance = 1e-4 * gb.a0
    assert np.max(gb.left_grain["x"]) <= gb.gb_plane_x + tolerance
    assert np.min(gb.right_grain["x"]) >= gb.gb_plane_x - tolerance


@pytest.mark.parametrize(
    ("a0", "structure", "atom_types", "kwargs"),
    VACUUM_ZERO_BOX_CASES,
)
def test_vacuum_zero_exact_atoms_are_within_x_box(
    build_gb,
    a0,
    structure,
    atom_types,
    kwargs,
):
    gb = build_gb(
        a0=a0,
        structure=structure,
        atom_types=atom_types,
        vacuum=0.0,
        **kwargs,
    )
    atoms = gb.whole_system
    tolerance = max(1e-8, 100.0 * gb.epsilon)

    assert np.min(atoms["x"]) >= -tolerance
    assert np.max(atoms["x"]) < gb.x_dim + tolerance


# --------------------------------------------------------------------------------------
# Vacuum-zero periodic-interface regressions
# --------------------------------------------------------------------------------------


def test_vacuum_zero_exact_gaps_are_nonnegative_without_deleting_grain_layers(build_gb):
    gb = build_gb(vacuum=0.0)

    central_gap = float(
        np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"])
    )
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"]))
        + np.min(gb.left_grain["x"])
    )

    assert central_gap >= -gb.epsilon
    assert periodic_gap >= -gb.epsilon
    assert len(gb.left_grain) == 1_200
    assert len(gb.right_grain) == 1_200
    assert len(gb.whole_system) == 2_400


@pytest.mark.parametrize("spec", EXACT_BOUNDARY_SPECS)
def test_vacuum_zero_has_no_coincident_atoms_across_periodic_images(build_gb, spec):
    gb = build_gb(
        spec,
        a0=5.47,
        structure="fluorite",
        atom_types=("U", "O"),
        vacuum=0.0,
    )
    box_lengths = np.array([gb.x_dim, gb.y_dim, gb.z_dim], dtype=float)
    left_positions = np.mod(_positions(gb.left_grain), box_lengths)
    right_positions = np.mod(_positions(gb.right_grain), box_lengths)

    tree = KDTree(right_positions, boxsize=box_lengths)
    nearest_distances, _ = tree.query(left_positions, k=1)
    coincident_count = int(
        np.count_nonzero(nearest_distances <= COINCIDENCE_TOLERANCE_ANGSTROM)
    )

    assert coincident_count == 0, (
        f"detected {coincident_count} coincident left/right atom pairs under periodic "
        "boundary conditions"
    )


@pytest.mark.parametrize("spec", EXACT_BOUNDARY_SPECS)
def test_vacuum_zero_preserves_fluorite_stoichiometry_in_each_grain_and_system(
    build_gb,
    spec,
):
    gb = build_gb(
        spec,
        a0=5.47,
        structure="fluorite",
        atom_types=("U", "O"),
        vacuum=0.0,
    )

    _assert_fluorite_stoichiometry(gb.left_grain, label="left grain")
    _assert_fluorite_stoichiometry(gb.right_grain, label="right grain")
    _assert_fluorite_stoichiometry(gb.whole_system, label="whole system")


@pytest.mark.parametrize("spec", EXACT_BOUNDARY_SPECS)
def test_vacuum_zero_preserves_rocksalt_stoichiometry_in_each_grain_and_system(
    build_gb,
    spec,
):
    gb = build_gb(
        spec,
        a0=4.0,
        structure="rocksalt",
        atom_types=("Na", "Cl"),
        vacuum=0.0,
    )

    _assert_rocksalt_stoichiometry(gb.left_grain, label="left grain")
    _assert_rocksalt_stoichiometry(gb.right_grain, label="right grain")
    _assert_rocksalt_stoichiometry(gb.whole_system, label="whole system")


def test_zhang_sigma53_vacuum_zero_preserves_bounds_nonnegative_gaps_and_stoichiometry():
    """Regression for the external fluorite case that leaked basis offsets in x."""
    entry = BOUNDARIES["sigma53_100_0_7_2bar_0_2bar_7_STGB"]
    spec = PQSpec(P=entry["P"], Q=entry["Q"])

    with pytest.warns(UserWarning, match=r"Recommended repeat factor is at least 2\."):
        gb = GBMaker.from_boundary_spec(
            5.454,
            "fluorite",
            ("U", "O"),
            spec,
            mode="exact",
            gb_thickness=0.0,
            vacuum=0.0,
            repeat_factor=[1, 1],
            x_dim_min=20.0,
            interaction_distance=1.0,
        )

    tolerance = max(1e-8, 100.0 * gb.epsilon)
    assert np.min(gb.whole_system["x"]) >= -tolerance
    assert np.max(gb.whole_system["x"]) < gb.x_dim + tolerance

    central_gap = float(
        np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"])
    )
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"]))
        + np.min(gb.left_grain["x"])
    )

    assert np.isfinite(central_gap)
    assert np.isfinite(periodic_gap)
    assert central_gap >= -gb.epsilon
    assert periodic_gap >= -gb.epsilon

    _assert_fluorite_stoichiometry(gb.left_grain, label="left grain")
    _assert_fluorite_stoichiometry(gb.right_grain, label="right grain")
    _assert_fluorite_stoichiometry(gb.whole_system, label="whole system")


# --------------------------------------------------------------------------------------
# Exact decorated-site construction regressions
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    r"ignore:Commensurate repeat pair in [yz] multiplied by \d+ to satisfy the minimum "
    r"in-plane dimension cutoff of .* A\.:UserWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)
@pytest.mark.parametrize(
    ("P", "Q", "left_expected", "right_expected"),
    REPRESENTATIVE_EXACT_CASES,
)
def test_campaign_representatives_preserve_complete_exact_populations(
    P,
    Q,
    left_expected,
    right_expected,
):
    gb = _build_campaign_style_exact_boundary(P, Q)

    _assert_complete_fluorite_population(gb.left_grain, left_expected)
    _assert_complete_fluorite_population(gb.right_grain, right_expected)
    _assert_complete_fluorite_population(
        gb.whole_system,
        left_expected + right_expected,
    )
    _assert_atoms_within_box(gb, gb.whole_system)
    _assert_no_coincident_periodic_sites(gb)


@pytest.mark.filterwarnings(
    r"ignore:Commensurate repeat pair in [yz] multiplied by \d+ to satisfy the minimum "
    r"in-plane dimension cutoff of .* A\.:UserWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)
def test_exact_decorated_site_order_is_deterministic():
    P = [[0, 0, 1], [4, 1, 0], [-1, 4, 0]]
    Q = [[0, 0, 1], [4, -1, 0], [1, 4, 0]]
    first = _build_campaign_style_exact_boundary(P, Q)
    second = _build_campaign_style_exact_boundary(P, Q)

    np.testing.assert_array_equal(first.left_grain, second.left_grain)
    np.testing.assert_array_equal(first.right_grain, second.right_grain)
    np.testing.assert_array_equal(first.whole_system, second.whole_system)
