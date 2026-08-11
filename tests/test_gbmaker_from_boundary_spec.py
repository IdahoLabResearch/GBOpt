# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Integration tests for ``GBMaker.from_boundary_spec`` dispatch and modes."""

import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.crystallography import (
    csl_exact_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from GBOpt.GBMaker import GBMaker, GBMakerValueError

# --------------------------------------------------------------------------------------
# Shared boundary specifications
# --------------------------------------------------------------------------------------

SIGMA5_TILT_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SIGMA5_TILT_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

SIGMA5_TILT_PQ_SPEC = PQSpec(
    P=SIGMA5_TILT_P,
    Q=SIGMA5_TILT_Q,
    basis_mode="supplied",
)
SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    quat=[3, 0, 0, 1],
)
SIGMA5_TILT_APPROX_SPEC = CSLApproxSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    angle_deg=36.87,
)
SIGMA5_TWIST_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1],
    plane=[0, 0, 1],
    quat=[3, 0, 0, 1],
)

EXACT_SPECS = [
    pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
    pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
]

_MISSING = object()


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def build_gb():
    """Return a function-scoped factory with compact boundary-spec defaults."""

    def _build(
        boundary=SIGMA5_TILT_PQ_SPEC,
        *,
        a0=3.615,
        structure="fcc",
        atom_types="Cu",
        mode=_MISSING,
        **overrides,
    ):
        kwargs = {
            "gb_thickness": 5.0,
            "repeat_factor": 2,
            "interaction_distance": a0,
        }
        kwargs.update(overrides)

        if mode is _MISSING:
            return GBMaker.from_boundary_spec(
                a0,
                structure,
                atom_types,
                boundary,
                **kwargs,
            )

        return GBMaker.from_boundary_spec(
            a0,
            structure,
            atom_types,
            boundary,
            mode=mode,  # type: ignore[ty:invalid-argument-type]
            **kwargs,
        )

    return _build


def _sorted_atoms(atoms):
    """Return atoms in deterministic coordinate/species order."""
    order = np.lexsort((atoms["name"], atoms["z"], atoms["y"], atoms["x"]))
    return atoms[order]


def _positions(atoms):
    """Return structured atom coordinates as an ``(N, 3)`` float array."""
    return np.column_stack((atoms["x"], atoms["y"], atoms["z"]))


# --------------------------------------------------------------------------------------
# Exact-mode dispatch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", EXACT_SPECS)
def test_from_boundary_spec_defaults_to_exact_mode(build_gb, spec):
    gb = build_gb(spec)

    assert gb.whole_system.size > 0
    assert set(gb.whole_system["name"]) == {"Cu"}


def test_from_boundary_spec_equivalent_exact_specs_build_equivalent_bicrystals(
    build_gb,
):
    gb_pq = build_gb(SIGMA5_TILT_PQ_SPEC, mode="exact")
    gb_csl = build_gb(SIGMA5_TILT_EXACT_SPEC, mode="exact")

    np.testing.assert_array_equal(
        _sorted_atoms(gb_csl.whole_system),
        _sorted_atoms(gb_pq.whole_system),
    )
    np.testing.assert_allclose(
        gb_csl.box_dims,
        gb_pq.box_dims,
        atol=1e-12,
        rtol=0.0,
    )


# --------------------------------------------------------------------------------------
# Approximate-mode dispatch
# --------------------------------------------------------------------------------------


def test_from_boundary_spec_approximate_csl_builds_finite_incoherent_bicrystal(
    build_gb
):
    gb = build_gb(SIGMA5_TILT_APPROX_SPEC, mode="approximate")

    assert gb.left_grain.size > 0
    assert gb.right_grain.size > 0
    assert gb.whole_system.size == gb.left_grain.size + gb.right_grain.size
    assert set(gb.whole_system["name"]) == {"Cu"}
    assert np.isfinite(_positions(gb.whole_system)).all()
    assert gb.inplane_periodic == (False, False)


# --------------------------------------------------------------------------------------
# Unsupported combinations and invalid inputs
# --------------------------------------------------------------------------------------


def test_from_boundary_spec_rejects_exact_mode_for_cslapproxspec(build_gb):
    with pytest.raises(
        BoundarySpecError,
        match=r"CSLApproxSpec cannot be used with mode='exact'",
    ):
        build_gb(SIGMA5_TILT_APPROX_SPEC, mode="exact")


@pytest.mark.parametrize("spec", EXACT_SPECS)
def test_from_boundary_spec_rejects_approximate_mode_for_exact_specs(build_gb, spec):
    with pytest.raises(
        NotImplementedError,
        match=r"mode 'approximate' is not yet supported.*mode='prefer_exact'",
    ):
        build_gb(spec, mode="approximate")


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("invalid", id="unknown-string"),
        pytest.param(None, id="none"),
        pytest.param(1, id="integer"),
    ],
)
def test_from_boundary_spec_rejects_invalid_mode(build_gb, mode):
    with pytest.raises(GBMakerValueError, match=r"mode must be one of"):
        build_gb(
            SIGMA5_TILT_PQ_SPEC,
            mode=mode,
        )


def test_from_boundary_spec_rejects_unsupported_boundary_type(build_gb):
    with pytest.raises(
        NotImplementedError,
        match=r"does not yet support boundary objects of type object",
    ):
        build_gb(
            object(),
            mode="exact",
        )


# --------------------------------------------------------------------------------------
# Primitive-metadata expansion regression
# --------------------------------------------------------------------------------------


def test_exact_twist_cell_expands_beyond_primitive_metadata_without_shrinking():
    embedding = csl_exact_spec_to_embedding(SIGMA5_TWIST_EXACT_SPEC)
    a0 = 5.47
    repeat_factor = (2, 3)
    x_dim_min = 30.0
    interaction_distance = 11.0

    gb = GBMaker.from_boundary_spec(
        a0,
        "fluorite",
        ("U", "O"),
        SIGMA5_TWIST_EXACT_SPEC,
        mode="exact",
        gb_thickness=0.0,
        repeat_factor=repeat_factor,
        x_dim_min=x_dim_min,
        vacuum=0.0,
        interaction_distance=interaction_distance,
    )

    assert embedding.P is not None
    basis_size = len(gb.unit_cell.asarray())
    primitive_atoms = primitive_bicrystal_atom_count(embedding, basis_size)
    y_period = a0 * np.linalg.norm(np.asarray(embedding.P[1], dtype=float))
    z_period = a0 * np.linalg.norm(np.asarray(embedding.P[2], dtype=float))
    left_width = gb.gb_plane_x
    right_width = gb.x_dim - gb.gb_plane_x

    assert gb.whole_system.size > primitive_atoms
    assert left_width >= x_dim_min - 1e-9
    assert right_width >= x_dim_min - 1e-9
    assert gb.y_dim >= repeat_factor[0] * y_period - 1e-9
    assert gb.z_dim >= repeat_factor[1] * z_period - 1e-9
    assert gb.y_dim >= 2.0 * interaction_distance - 1e-9
    assert gb.z_dim >= 2.0 * interaction_distance - 1e-9
