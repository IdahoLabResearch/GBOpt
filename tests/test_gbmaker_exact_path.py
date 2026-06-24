# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Integration tests for GBMaker's exact integer grain-construction path."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import CSLExactSpec, PQSpec
from GBOpt.crystallography import pq_spec_to_embedding
from GBOpt.GBMaker import GBMaker

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
NONCOMMENSURATE_PQ_SPEC = PQSpec(
    P=SIGMA5_TILT_P,
    Q=[[1, 0, 0], [0, 1, 1], [0, -1, 1]],
    basis_mode="supplied",
)

A0_FCC = 3.615
STRUCTURE_FCC = "fcc"
ATOM_TYPES_FCC = "Cu"

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

FLUORITE_EXACT_SPECS = [
    pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
    pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
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
    """Assert a nonempty atom collection has the expected UO2 species ratio."""
    uranium_count = int(np.count_nonzero(atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(atoms["name"] == "O"))

    assert uranium_count > 0, f"{label} contains no U atoms"
    assert oxygen_count == 2 * uranium_count, (
        f"{label} stoichiometry is {uranium_count} U to {oxygen_count} O; "
        "expected UO2"
    )


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


def test_exact_embedding_uses_integer_rows_without_float_approximation(
    monkeypatch,
    build_gb,
):
    embedding = pq_spec_to_embedding(
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_P, basis_mode="supplied")
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "The exact embedding path must not approximate rotation rows as integers."
        )

    monkeypatch.setattr(
        GBMaker,
        "_GBMaker__approximate_rotation_matrix_as_int",
        fail_if_called,
    )

    gb = GBMaker._from_boundary_embedding(
        embedding,
        a0=A0_FCC,
        structure=STRUCTURE_FCC,
        atom_types=ATOM_TYPES_FCC,
        gb_thickness=0.0,
        repeat_factor=2,
        interaction_distance=A0_FCC,
    )

    assert gb.whole_system.size > 0


# --------------------------------------------------------------------------------------
# Complete-origin construction
# --------------------------------------------------------------------------------------


def test_exact_builder_returns_atom_dtype_and_complete_unit_cell_origins(build_gb):
    gb = build_gb()
    basis_size = len(gb.unit_cell.asarray())

    assert gb.whole_system.dtype == Atom.atom_dtype
    assert gb.whole_system.size == gb.left_grain.size + gb.right_grain.size

    for label, grain in (("left", gb.left_grain), ("right", gb.right_grain)):
        assert grain.size > 0, f"{label} grain is empty"
        assert grain.size % basis_size == 0, (
            f"{label} grain contains {grain.size} atoms, which is not divisible by "
            f"the conventional-cell basis size {basis_size}"
        )


# --------------------------------------------------------------------------------------
# Cartesian box and central-interface bounds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "a0", "structure", "atom_types"),
    EXACT_BOX_CASES,
)
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


def test_vacuum_zero_periodic_gap_is_not_smaller_than_central_gap(build_gb):
    gb = build_gb(vacuum=0.0)

    central_gap = float(
        np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"])
    )
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"]))
        + np.min(gb.left_grain["x"])
    )

    assert periodic_gap >= central_gap - gb.epsilon, (
        f"periodic gap {periodic_gap:.8f} is smaller than central gap "
        f"{central_gap:.8f}"
    )


@pytest.mark.parametrize("spec", FLUORITE_EXACT_SPECS)
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
    coincident_count = int(np.count_nonzero(nearest_distances <= 1e-4))

    assert coincident_count == 0, (
        f"detected {coincident_count} coincident left/right atom pairs under periodic "
        "boundary conditions"
    )


@pytest.mark.parametrize("spec", FLUORITE_EXACT_SPECS)
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


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
        pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
    ],
)
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

    for label, atoms in (
        ("left grain", gb.left_grain),
        ("right grain", gb.right_grain),
        ("whole system", gb.whole_system),
    ):
        sodium_count = int(np.count_nonzero(atoms["name"] == "Na"))
        chlorine_count = int(np.count_nonzero(atoms["name"] == "Cl"))

        assert sodium_count > 0, f"{label} contains no Na atoms"
        assert chlorine_count == sodium_count, (
            f"{label} stoichiometry is {sodium_count} Na to "
            f"{chlorine_count} Cl; expected NaCl"
        )


def test_zhang_sigma53_vacuum_zero_regression_preserves_box_gap_and_stoichiometry():
    """Cover the external fluorite case that previously leaked basis offsets in x."""
    zhang_boundaries = pytest.importorskip(
        "zhang2021_boundaries",
        reason="optional Zhang boundary dataset is not installed",
    )
    entry = zhang_boundaries.BOUNDARIES[
        "sigma53_100_0_7_2bar_0_2bar_7_STGB"
    ]
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
    assert periodic_gap >= central_gap - gb.epsilon

    _assert_fluorite_stoichiometry(gb.left_grain, label="left grain")
    _assert_fluorite_stoichiometry(gb.right_grain, label="right grain")
    _assert_fluorite_stoichiometry(gb.whole_system, label="whole system")
