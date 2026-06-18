# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

# NOTE: There may be overlap between tests here and test_gbmaker.py.
# Overlap will be resolved when GBMaker is split into its own module.

import math
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial import KDTree
from zhang2021_boundaries import BOUNDARIES

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import (
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.crystallography import pq_spec_to_embedding
from GBOpt.GBMaker import GBMaker

# ---------------------------------------------------------------------------
# Shared boundary specs
# ---------------------------------------------------------------------------

SIGMA5_TILT_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SIGMA5_TILT_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
SIGMA5_TILT_PQ_SPEC = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied")

A0_FCC = 3.615
STRUCTURE_FCC = "fcc"
ATOM_TYPES_FCC = "Cu"


# ---------------------------------------------------------------------------
# Exact grain repeats
# ---------------------------------------------------------------------------

def test_exact_grain_repeats_builds_and_dims_divisible_by_left_grain_periods():
    gb = GBMaker.from_boundary_spec(
        A0_FCC, STRUCTURE_FCC, ATOM_TYPES_FCC, SIGMA5_TILT_PQ_SPEC,
        mode="exact", gb_thickness=0.0, repeat_factor=2,
        interaction_distance=A0_FCC,
    )
    assert gb.whole_system.size > 0

    y_period_left = A0_FCC * np.linalg.norm(np.array(SIGMA5_TILT_P[1]))
    z_period_left = A0_FCC * np.linalg.norm(np.array(SIGMA5_TILT_P[2]))
    y_ratio = gb.y_dim / y_period_left
    z_ratio = gb.z_dim / z_period_left
    assert abs(y_ratio - round(y_ratio)) < 1e-6
    assert abs(z_ratio - round(z_ratio)) < 1e-6


# ---------------------------------------------------------------------------
# Exact grain builder
# ---------------------------------------------------------------------------

def test_exact_grain_builder_output_dtype_and_atom_count_are_valid():

    gb = GBMaker.from_boundary_spec(
        A0_FCC, STRUCTURE_FCC, ATOM_TYPES_FCC, SIGMA5_TILT_PQ_SPEC,
        mode="exact", gb_thickness=0.0, repeat_factor=2,
        interaction_distance=A0_FCC,
    )
    assert gb.whole_system.dtype == Atom.atom_dtype
    assert gb.whole_system.size > 0


def test_exact_grain_builder_pqspec_and_cslexactspec_produce_same_atoms():
    gb_pq = GBMaker.from_boundary_spec(
        A0_FCC, STRUCTURE_FCC, ATOM_TYPES_FCC, SIGMA5_TILT_PQ_SPEC,
        mode="exact", gb_thickness=0.0, repeat_factor=2,
        interaction_distance=A0_FCC,
    )
    gb_csl = GBMaker.from_boundary_spec(
        A0_FCC, STRUCTURE_FCC, ATOM_TYPES_FCC, SIGMA5_TILT_EXACT_SPEC,
        mode="exact", gb_thickness=0.0, repeat_factor=2,
        interaction_distance=A0_FCC,
    )
    np.testing.assert_array_equal(gb_pq.whole_system, gb_csl.whole_system)


def test_exact_grain_builder_approx_path_still_works():
    approx = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    gb = GBMaker.from_boundary_spec(
        A0_FCC, STRUCTURE_FCC, ATOM_TYPES_FCC, approx,
        mode="approximate", gb_thickness=0.0, repeat_factor=2,
        interaction_distance=A0_FCC,
    )
    assert gb.whole_system.size > 0


# ---------------------------------------------------------------------------
# _from_boundary_embedding
# ---------------------------------------------------------------------------

def test_from_boundary_embedding_sigma5_exact_builds_valid_fcc_bicrystal():
    theta = math.atan2(3, 4)
    misorientation = np.array([0.0, 0.0, theta, 0.0, 0.0])
    gb_legacy = GBMaker(
        A0_FCC, STRUCTURE_FCC, 0.0, misorientation, ATOM_TYPES_FCC,
        interaction_distance=A0_FCC, repeat_factor=2, x_dim_min=30.0,
    )

    P = gb_legacy._GBMaker__R_left_approx.astype(int).tolist()
    Q = gb_legacy._GBMaker__R_right_approx.astype(int).tolist()
    spec = PQSpec(P=P, Q=Q, basis_mode="supplied")
    emb = pq_spec_to_embedding(spec)
    gb_emb = GBMaker._from_boundary_embedding(
        emb, a0=A0_FCC, structure=STRUCTURE_FCC, atom_types=ATOM_TYPES_FCC,
        gb_thickness=0.0, repeat_factor=2, x_dim_min=30.0,
        interaction_distance=A0_FCC,
    )

    ws = gb_emb.whole_system
    assert ws.size > 0
    assert set(ws["name"]) == {"Cu"}
    for field in ("x", "y", "z"):
        assert np.all(np.isfinite(ws[field]))


def test_from_boundary_embedding_exact_path_skips_approximation():
    spec = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_P)  # identity
    emb = pq_spec_to_embedding(spec)
    assert emb.exact is True

    spy_target = "_GBMaker__approximate_rotation_matrix_as_int"
    with patch.object(GBMaker, spy_target) as spy:
        GBMaker._from_boundary_embedding(
            emb, a0=A0_FCC, structure=STRUCTURE_FCC, atom_types=ATOM_TYPES_FCC,
            gb_thickness=0.0, repeat_factor=2, interaction_distance=A0_FCC,
        )
        spy.assert_not_called()


def test_from_boundary_embedding_coherent_sets_inplane_periodic():
    spec = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_P)  # identity
    emb = pq_spec_to_embedding(spec)
    assert emb.coherent is True
    gb = GBMaker._from_boundary_embedding(
        emb, a0=A0_FCC, structure=STRUCTURE_FCC, atom_types=ATOM_TYPES_FCC,
        gb_thickness=0.0, repeat_factor=2, interaction_distance=A0_FCC,
    )
    assert gb.inplane_periodic == (True, True)


def test_from_boundary_embedding_misorientation_setter_clears_embedding():
    spec = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied")
    emb = pq_spec_to_embedding(spec)
    gb = GBMaker._from_boundary_embedding(
        emb, a0=A0_FCC, structure=STRUCTURE_FCC, atom_types=ATOM_TYPES_FCC,
        gb_thickness=0.0, repeat_factor=2, interaction_distance=A0_FCC,
    )
    assert gb._GBMaker__embedding is not None

    theta = math.atan2(3, 4)
    gb.misorientation = np.array([0.0, 0.0, theta, 0.0, 0.0])
    assert gb._GBMaker__embedding is None
    np.testing.assert_array_almost_equal(gb._GBMaker__R_left, np.eye(3))


# ---------------------------------------------------------------------------
# Box bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,a0,structure,atom_types", [
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        3.615, "fcc", "Cu",
    ),
    (
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
        3.615, "fcc", "Cu",
    ),
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        5.47, "fluorite", ("U", "O"),
    ),
])
def test_exact_path_atoms_within_yz_box(spec, a0, structure, atom_types):
    gb = GBMaker.from_boundary_spec(
        a0, structure, atom_types, spec, mode="exact",
        repeat_factor=2, interaction_distance=a0,
    )
    ws = gb.whole_system
    tol = 1e-4
    assert np.all(ws["y"] >= -tol), f"y underflow: min={ws['y'].min():.6f}"
    assert np.all(ws["y"] < gb.y_dim + tol), (
        f"y overflow: max={ws['y'].max():.6f} > y_dim={gb.y_dim:.6f}"
    )
    assert np.all(ws["z"] >= -tol), f"z underflow: min={ws['z'].min():.6f}"
    assert np.all(ws["z"] < gb.z_dim + tol), (
        f"z overflow: max={ws['z'].max():.6f} > z_dim={gb.z_dim:.6f}"
    )


def test_exact_left_grain_does_not_overflow_gb_plane():
    spec = PQSpec(
        P=[[3, 1, 0], [0, 0, 2], [1, -3, 0]],
        Q=[[3, 1, 0], [0, 0, -2], [-1, 3, 0]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gb = GBMaker.from_boundary_spec(
            3.52, "fcc", "Ni", spec, mode="exact",
            gb_thickness=0.0, repeat_factor=1, x_dim_min=20.0,
            vacuum=0.0, interaction_distance=5.0,
        )

    tol = 1e-4 * gb.a0
    left_max_x = float(np.max(gb.left_grain["x"]))
    right_min_x = float(np.min(gb.right_grain["x"]))
    assert left_max_x <= gb.gb_plane_x + tol, (
        f"left grain overflows GB plane: max_x={left_max_x:.6f}, "
        f"gb_plane_x={gb.gb_plane_x:.6f}"
    )
    assert right_min_x >= gb.gb_plane_x - tol, (
        f"right grain underflows GB plane: min_x={right_min_x:.6f}, "
        f"gb_plane_x={gb.gb_plane_x:.6f}"
    )


@pytest.mark.parametrize("spec,a0,structure,atom_types,kwargs", [
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        3.615, "fcc", "Cu",
        {"interaction_distance": 3.615, "repeat_factor": 2},
    ),
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        5.47, "fluorite", ("U", "O"),
        {"interaction_distance": 5.47, "repeat_factor": 2},
    ),
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        5.47, "fluorite", ("U", "O"),
        {"x_dim_min": 20, "interaction_distance": 1, "repeat_factor": 2},
    ),
])
def test_vacuum0_atoms_within_x_box(spec, a0, structure, atom_types, kwargs):
    gb = GBMaker.from_boundary_spec(
        a0, structure, atom_types, spec, mode="exact", vacuum=0, **kwargs,
    )
    ws = gb.whole_system
    tol = 1e-4
    x_dim = gb._GBMaker__x_dim
    assert np.all(ws["x"] >= -tol), f"x underflow: min={ws['x'].min():.6f}"
    assert np.all(ws["x"] < x_dim + tol), (
        f"x overflow: max={ws['x'].max():.6f} > x_dim={x_dim:.6f}"
    )


def test_vacuum0_zhang_sigma53_atoms_within_x_box():
    """Regression for a fluorite exact build with basis-offset x leakage."""

    entry = BOUNDARIES["sigma53_100_0_7_2bar_0_2bar_7_STGB"]
    spec = PQSpec(P=entry["P"], Q=entry["Q"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gb = GBMaker.from_boundary_spec(
            5.454, "fluorite", ("U", "O"), spec, mode="exact",
            gb_thickness=0.0, vacuum=0.0, repeat_factor=[1, 1],
            x_dim_min=20, interaction_distance=1.0,
        )

    ws = gb.whole_system
    tol = 1e-8
    assert np.min(ws["x"]) >= -tol
    assert np.max(ws["x"]) < gb.x_dim - tol

    central_gap = float(np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"]))
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"])) + np.min(gb.left_grain["x"])
    )
    assert periodic_gap >= central_gap - tol

    names, counts = np.unique(ws["name"], return_counts=True)
    species = {str(n): int(c) for n, c in zip(names, counts)}
    assert species["O"] == 2 * species["U"]


def test_vacuum0_periodic_gap_matches_central_gap():
    gb = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", SIGMA5_TILT_PQ_SPEC, mode="exact",
        vacuum=0, repeat_factor=2, interaction_distance=3.615,
    )
    x_dim = gb._GBMaker__x_dim
    rg = gb._GBMaker__right_grain
    lg = gb._GBMaker__left_grain
    left_max_x = np.max(lg["x"])
    right_min_x = np.min(rg["x"])
    central_gap = right_min_x - left_max_x
    right_max_x = np.max(rg["x"])
    left_min_x = np.min(lg["x"])
    periodic_gap = (x_dim - right_max_x) + left_min_x
    assert right_max_x < x_dim + 1e-4, (
        f"vacuum=0 right grain overflows box: max_x={right_max_x:.4f} "
        f"> x_dim={x_dim:.4f}"
    )
    assert abs(periodic_gap - central_gap) < 0.1, (
        f"vacuum=0 periodic_gap ({periodic_gap:.4f}) != central_gap "
        f"({central_gap:.4f})"
    )


# ---------------------------------------------------------------------------
# No coincident atoms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,a0,structure,atom_types", [
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        5.47, "fluorite", ("U", "O"),
    ),
    (
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
        5.47, "fluorite", ("U", "O"),
    ),
])
def test_no_coincident_interface_atoms(spec, a0, structure, atom_types):

    gb = GBMaker.from_boundary_spec(
        a0, structure, atom_types, spec, mode="exact", vacuum=0,
        repeat_factor=2, interaction_distance=a0,
    )
    lg = gb._GBMaker__left_grain
    rg = gb._GBMaker__right_grain
    L = np.column_stack([lg["x"], lg["y"], lg["z"]])
    R = np.column_stack([rg["x"], rg["y"], rg["z"]])
    tree = KDTree(R)
    dists, _ = tree.query(L, k=1)
    assert dists.min() > 1e-4, (
        f"Coincident left/right atoms detected: "
        f"{(dists < 1e-4).sum()} pairs at zero distance"
    )


@pytest.mark.parametrize("spec,a0,structure,atom_types", [
    (
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied"),
        5.47, "fluorite", ("U", "O"),
    ),
    (
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
        5.47, "fluorite", ("U", "O"),
    ),
])
def test_vacuum0_stoichiometry_preserved(spec, a0, structure, atom_types):
    gb = GBMaker.from_boundary_spec(
        a0, structure, atom_types, spec, mode="exact", vacuum=0,
        repeat_factor=2, interaction_distance=a0,
    )
    rg = gb._GBMaker__right_grain
    ws = gb.whole_system
    u_rg = (rg["name"] == "U").sum()
    o_rg = (rg["name"] == "O").sum()
    u_ws = (ws["name"] == "U").sum()
    o_ws = (ws["name"] == "O").sum()
    assert u_rg > 0, "Right grain has no U atoms"
    assert o_rg == 2 * u_rg, (
        f"Right grain stoichiometry broken: {u_rg} U, {o_rg} O (expected 2:1)"
    )
    assert o_ws == 2 * u_ws, (
        f"Whole-system stoichiometry broken: {u_ws} U, {o_ws} O (expected 2:1)"
    )
