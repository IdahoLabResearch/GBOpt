# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

# NOTE: There may be overlap between tests here and test_gbmaker.py.
# Overlap will be resolved when GBMaker is split into its own module.


import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.crystallography import (
    csl_spec_to_embedding,
    pq_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from GBOpt.GBMaker import GBMaker

# ---------------------------------------------------------------------------
# Shared boundary specs
# ---------------------------------------------------------------------------

SIGMA5_TILT_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SIGMA5_TILT_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
SIGMA5_TILT_PQ_SPEC = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied")
SIGMA5_TWIST_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])


# ---------------------------------------------------------------------------
# primitive_bicrystal_atom_count via GBMaker
# ---------------------------------------------------------------------------

def test_primitive_metadata_does_not_shrink_expanded_gbmaker_cell():
    emb = csl_spec_to_embedding(SIGMA5_TWIST_SPEC)
    primitive_atoms = primitive_bicrystal_atom_count(emb, 12)
    a0 = 5.47
    repeat_factor = [2, 3]
    x_dim_min = 30.0
    interaction_distance = 11.0

    gb = GBMaker.from_boundary_spec(
        a0,
        "fluorite",
        ("U", "O"),
        SIGMA5_TWIST_SPEC,
        mode="exact",
        gb_thickness=0.0,
        repeat_factor=repeat_factor,
        x_dim_min=x_dim_min,
        vacuum=0.0,
        interaction_distance=interaction_distance,
    )

    assert emb.P is not None
    y_period = a0 * np.linalg.norm(emb.P[1])
    z_period = a0 * np.linalg.norm(emb.P[2])
    assert gb.whole_system.size > primitive_atoms
    assert gb._GBMaker__left_x >= x_dim_min - 1e-9
    assert gb._GBMaker__right_x >= x_dim_min - 1e-9
    assert gb.y_dim >= repeat_factor[0] * y_period - 1e-9
    assert gb.z_dim >= repeat_factor[1] * z_period - 1e-9
    assert gb.y_dim >= 2.0 * interaction_distance - 1e-9
    assert gb.z_dim >= 2.0 * interaction_distance - 1e-9


# ---------------------------------------------------------------------------
# from_boundary_spec -- PQSpec exact path
# ---------------------------------------------------------------------------

def test_from_boundary_spec_pqspec_exact_matches_embedding_path():
    gb_spec = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", SIGMA5_TILT_PQ_SPEC, mode="exact",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    emb = pq_spec_to_embedding(SIGMA5_TILT_PQ_SPEC)
    gb_emb = GBMaker._from_boundary_embedding(
        emb, a0=3.615, structure="fcc", atom_types="Cu",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    np.testing.assert_array_equal(gb_spec.whole_system, gb_emb.whole_system)


# ---------------------------------------------------------------------------
# from_boundary_spec -- CSLExactSpec
# ---------------------------------------------------------------------------

def test_from_boundary_spec_cslexactspec_builds_monatomic_bicrystal():
    gb = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", SIGMA5_TILT_EXACT_SPEC, mode="exact",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    assert len(np.unique(gb.whole_system["name"])) == 1


def test_from_boundary_spec_cslexactspec_matches_equivalent_pqspec():
    gb_csl = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", SIGMA5_TILT_EXACT_SPEC, mode="exact",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    gb_pq = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", SIGMA5_TILT_PQ_SPEC, mode="exact",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    np.testing.assert_array_equal(gb_csl.whole_system, gb_pq.whole_system)


def test_from_boundary_spec_cslapproxspec_exact_raises():
    approx = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    with pytest.raises(BoundarySpecError):
        GBMaker.from_boundary_spec(
            3.615, "fcc", "Cu", approx, mode="exact",
            gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
        )


def test_from_boundary_spec_cslapproxspec_approximate_succeeds():
    approx = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    gb = GBMaker.from_boundary_spec(
        3.615, "fcc", "Cu", approx, mode="approximate",
        gb_thickness=5.0, repeat_factor=2, interaction_distance=3.615,
    )
    assert gb.whole_system.size > 0


@pytest.mark.parametrize("kwargs", [
    {"axis": [0, 0, 1], "plane": [1, 0, 0]},
    {"axis": [0, 0, 1], "plane": [1, 0, 0], "angle_deg": np.nan},
    {"axis": [0, 0, 1], "plane": [1, 0, 0], "angle_deg": np.inf},
    {"axis": [0, 0], "plane": [1, 0, 0], "angle_deg": 36.87},
    {"axis": [0, 0, 0], "plane": [1, 0, 0], "angle_deg": 36.87},
    {"axis": [0, 0, 1], "plane": [1, 0], "angle_deg": 36.87},
    {"axis": [0, 0, 1], "plane": [0, 0, 0], "angle_deg": 36.87},
])
def test_from_boundary_spec_cslapproxspec_invalid_inputs_raise(kwargs):
    with pytest.raises(BoundarySpecError):
        CSLApproxSpec(**kwargs)


# ---------------------------------------------------------------------------
# from_boundary_spec -- multispecies stoichiometry
# ---------------------------------------------------------------------------

def test_from_boundary_spec_rocksalt_pqspec_stoichiometric():
    spec = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied")
    gb = GBMaker.from_boundary_spec(
        4.0, "rocksalt", ("Na", "Cl"), spec, mode="exact",
        gb_thickness=2.0, repeat_factor=2, interaction_distance=4.0,
    )
    ws = gb.whole_system
    names, counts = np.unique(ws["name"], return_counts=True)
    species = {str(n): int(c) for n, c in zip(names, counts)}
    assert species["Na"] == species["Cl"], (
        f"Rocksalt bicrystal is not stoichiometric: {species}"
    )


def test_from_boundary_spec_rocksalt_cslexactspec_stoichiometric():
    gb = GBMaker.from_boundary_spec(
        4.0, "rocksalt", ("Na", "Cl"), SIGMA5_TILT_EXACT_SPEC, mode="exact",
        gb_thickness=2.0, repeat_factor=2, interaction_distance=4.0,
    )
    ws = gb.whole_system
    names, counts = np.unique(ws["name"], return_counts=True)
    species = {str(n): int(c) for n, c in zip(names, counts)}
    assert species["Na"] == species["Cl"], (
        f"Rocksalt bicrystal via CSLExactSpec is not stoichiometric: {species}"
    )


def test_from_boundary_spec_fluorite_pqspec_stoichiometric():
    spec = PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_Q, basis_mode="supplied")
    gb = GBMaker.from_boundary_spec(
        5.47, "fluorite", ("U", "O"), spec, mode="exact",
        gb_thickness=2.0, repeat_factor=2, interaction_distance=5.47,
    )
    ws = gb.whole_system
    names, counts = np.unique(ws["name"], return_counts=True)
    species = {str(n): int(c) for n, c in zip(names, counts)}
    assert species["O"] == 2 * species["U"], (
        f"Fluorite bicrystal is not stoichiometric: {species}"
    )


def test_from_boundary_spec_fluorite_cslexactspec_stoichiometric():
    gb = GBMaker.from_boundary_spec(
        5.47, "fluorite", ("U", "O"), SIGMA5_TILT_EXACT_SPEC, mode="exact",
        gb_thickness=2.0, repeat_factor=2, interaction_distance=5.47,
    )
    ws = gb.whole_system
    names, counts = np.unique(ws["name"], return_counts=True)
    species = {str(n): int(c) for n, c in zip(names, counts)}
    assert species["O"] == 2 * species["U"], (
        f"Fluorite bicrystal via CSLExactSpec is not stoichiometric: {species}"
    )
