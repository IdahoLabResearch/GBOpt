# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest
from crystallography_fixtures import CSL_SCENARIO_DICTS

# NOTE: Tests that require csl_spec_to_embedding (row-rotation convention,
# round-trip P/Q pinning) are deferred to test_crystallography_boundary.py
# since csl_spec_to_embedding belongs to boundary.py, which sits above csl.py
# in the dependency hierarchy.
from GBOpt.crystallography.csl import (
    csl_from_scaled_rotation,
    dsc_basis,
    sigma_from_snf_diagonal,
    verify_coincidence_basis,
)
from GBOpt.crystallography.integer import integer_det3
from GBOpt.crystallography.plane import inplane_basis_from_csl
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.types import (
    CrystallographyBackendError,
    CrystallographyValueError,
)

# ---------------------------------------------------------------------------
# Shared scenarios
# ---------------------------------------------------------------------------

EXACT_CSL_SCENARIOS = [
    pytest.param(d, id=str(d["id"])) for d in CSL_SCENARIO_DICTS
]

INPLANE_EXACT_CSL_SCENARIOS = [
    pytest.param(d, id=str(d["id"]))
    for d in CSL_SCENARIO_DICTS
    if d["plane"] is not None
]


def _build_exact_csl_case(case):
    rot = quaternion_to_scaled_rotation(case["q"])
    csl = csl_from_scaled_rotation(rot)
    inplane = None
    if case["plane"] is not None:
        inplane = inplane_basis_from_csl(csl.basis_hnf, case["plane"])
    return rot, csl, inplane


# ---------------------------------------------------------------------------
# sigma_from_snf_diagonal
# ---------------------------------------------------------------------------

def test_sigma_from_snf_diagonal_sigma5_001():
    sigma, moduli = sigma_from_snf_diagonal(5, (1, 5, 25))
    assert sigma == 5
    assert moduli == (5, 1, 1)


def test_sigma_from_snf_diagonal_sigma1():
    sigma, moduli = sigma_from_snf_diagonal(4, (4, 4, 4))
    assert sigma == 1
    assert moduli == (1, 1, 1)


def test_sigma_from_snf_diagonal_zero_N_raises():
    with pytest.raises(CrystallographyValueError):
        sigma_from_snf_diagonal(0, (1, 1, 1))


def test_sigma_from_snf_diagonal_negative_N_raises():
    with pytest.raises(CrystallographyValueError):
        sigma_from_snf_diagonal(-5, (1, 1, 5))


# ---------------------------------------------------------------------------
# csl_from_scaled_rotation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_scaled_rotation_N(case):
    rot, _csl, _inplane = _build_exact_csl_case(case)
    assert rot.N == case["expected_N"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_scaled_rotation_M(case):
    rot, _csl, _inplane = _build_exact_csl_case(case)
    if "expected_M" in case:
        np.testing.assert_array_equal(rot.M, case["expected_M"])


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_scaled_rotation_orthogonal(case):
    rot, _csl, _inplane = _build_exact_csl_case(case)
    gram = rot.M @ rot.M.T
    np.testing.assert_array_equal(
        gram,
        (rot.N ** 2) * np.eye(3, dtype=object),
    )


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_basis_sigma(case):
    _rot, csl, _inplane = _build_exact_csl_case(case)
    assert csl.sigma == case["expected_sigma"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_basis_hnf_det(case):
    _rot, csl, _inplane = _build_exact_csl_case(case)
    assert abs(integer_det3(csl.basis_hnf)) == case["expected_hnf_det"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_basis_hnf_known(case):
    _rot, csl, _inplane = _build_exact_csl_case(case)
    if "expected_basis_hnf" in case:
        np.testing.assert_array_equal(csl.basis_hnf, case["expected_basis_hnf"])


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_kernel_moduli(case):
    _rot, csl, _inplane = _build_exact_csl_case(case)
    if "expected_kernel_moduli" in case:
        assert csl.diagnostics.kernel_moduli == case["expected_kernel_moduli"]


def test_csl_from_scaled_rotation_invalid_post_reduce_raises():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    with pytest.raises(CrystallographyValueError):
        csl_from_scaled_rotation(rot, post_reduce="invalid")  # type: ignore


def test_csl_from_scaled_rotation_backend_error_on_snf_failure():
    from unittest.mock import patch

    from GBOpt.Utils.integer_normal_forms import ExactNormalFormError
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    with patch(
        "GBOpt.crystallography.csl.smith_normal_form_3x3",
        side_effect=ExactNormalFormError("forced SNF failure"),
    ):
        with pytest.raises(CrystallographyBackendError):
            csl_from_scaled_rotation(rot)

# ---------------------------------------------------------------------------
# verify_coincidence_basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_verify_coincidence_basis_passes(case):
    rot, csl, _inplane = _build_exact_csl_case(case)
    check = verify_coincidence_basis(rot, csl.basis_hnf, sigma=case["expected_sigma"])
    assert check.ok
    np.testing.assert_array_equal(
        check.residual_mod_N,
        np.zeros((3, 3), dtype=object),
    )


def test_verify_coincidence_basis_non_csl_basis_fails():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    # Identity is not a CSL basis for Sigma5
    non_csl = np.eye(3, dtype=object)
    check = verify_coincidence_basis(rot, non_csl, sigma=5)
    assert not check.ok


def test_verify_coincidence_basis_sigma_mismatch_fails():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    # Correct basis but wrong sigma
    check = verify_coincidence_basis(rot, csl.basis_hnf, sigma=3)
    assert not check.ok


def test_verify_coincidence_basis_invalid_N_raises():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    bad_rot = rot.__class__(N=-1, M=rot.M, source=rot.source)
    with pytest.raises(CrystallographyValueError):
        verify_coincidence_basis(bad_rot, csl.basis_hnf)


# ---------------------------------------------------------------------------
# inplane_basis_from_csl (deferred from test_crystallography_plane.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_in_plane(case):
    _rot, _csl, inplane = _build_exact_csl_case(case)
    assert inplane is not None
    h = np.array(case["plane"], dtype=object)
    assert int(h @ inplane.basis[:, 0]) == 0
    assert int(h @ inplane.basis[:, 1]) == 0


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_known(case):
    _rot, _csl, inplane = _build_exact_csl_case(case)
    h = np.array(case["plane"], dtype=object)
    assert int(h @ inplane.basis[:, 0]) == 0
    assert int(h @ inplane.basis[:, 1]) == 0
    cross = np.cross(
        inplane.basis[:, 0].astype(int), inplane.basis[:, 1].astype(int)
    )
    assert any(v != 0 for v in cross), "basis vectors are linearly dependent"


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_cross_abs(case):
    _rot, _csl, inplane = _build_exact_csl_case(case)
    if "expected_inplane_cross_abs" in case:
        v1 = inplane.basis[:, 0].astype(int)
        v2 = inplane.basis[:, 1].astype(int)
        np.testing.assert_array_equal(
            np.abs(np.cross(v1, v2)),
            case["expected_inplane_cross_abs"],
        )


# ---------------------------------------------------------------------------
# dsc_basis
# ---------------------------------------------------------------------------

def test_dsc_basis_sigma5_001():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    result = dsc_basis(csl.basis_hnf, csl.sigma)
    expected_adj = np.array([[5, 0, 0], [-2, 1, 0], [0, 0, 5]], dtype=object)
    assert result.denominator == 5
    np.testing.assert_array_equal(result.numerator, expected_adj)
    assert integer_det3(result.numerator) == 25


def test_dsc_basis_sigma_mismatch_raises():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    with pytest.raises(CrystallographyValueError):
        dsc_basis(csl.basis_hnf, sigma=3)


def test_dsc_basis_invalid_sigma_raises():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    with pytest.raises(CrystallographyValueError):
        dsc_basis(csl.basis_hnf, sigma=-1)


def test_dsc_basis_lattice_basis_raises():
    from GBOpt.crystallography.types import CrystallographyNotImplementedError
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    with pytest.raises(CrystallographyNotImplementedError):
        dsc_basis(csl.basis_hnf, csl.sigma, lattice_basis=np.eye(3))


# ---------------------------------------------------------------------------
# Inplane CSL basis
# ---------------------------------------------------------------------------

def test_inplane_csl_sigma5_53deg_plane100_in_plane(sigma5_53deg_rotation):
    csl = csl_from_scaled_rotation(sigma5_53deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (1, 0, 0))
    plane = np.array([1, 0, 0], dtype=object)
    assert int(plane @ inplane.basis[:, 0]) == 0
    assert int(plane @ inplane.basis[:, 1]) == 0


def test_inplane_csl_sigma5_53deg_plane100_csl_membership(sigma5_53deg_rotation):
    csl = csl_from_scaled_rotation(sigma5_53deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (1, 0, 0))
    M = np.asarray(sigma5_53deg_rotation.M, dtype=object)
    N = sigma5_53deg_rotation.N
    for i in range(2):
        v = np.asarray(inplane.basis[:, i], dtype=object)
        residual = M @ v % N
        np.testing.assert_array_equal(residual, np.zeros(3, dtype=object))


def test_inplane_csl_sigma5_53deg_plane100_linearly_independent(sigma5_53deg_rotation):
    csl = csl_from_scaled_rotation(sigma5_53deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (1, 0, 0))
    v1 = inplane.basis[:, 0].astype(float)
    v2 = inplane.basis[:, 1].astype(float)
    assert np.linalg.norm(np.cross(v1, v2)) > 0.5


def test_inplane_csl_sigma5_36deg_plane100_csl_membership(sigma5_36deg_rotation):
    csl = csl_from_scaled_rotation(sigma5_36deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (1, 0, 0))
    M = np.asarray(sigma5_36deg_rotation.M, dtype=object)
    N = sigma5_36deg_rotation.N
    for i in range(2):
        v = np.asarray(inplane.basis[:, i], dtype=object)
        residual = M @ v % N
        np.testing.assert_array_equal(residual, np.zeros(3, dtype=object))


def test_inplane_csl_sigma5_36deg_plane523_primitive_basis(sigma5_36deg_rotation):
    # General planes must use the full primitive plane lattice so the
    # in-plane CSL area stays minimal
    csl = csl_from_scaled_rotation(sigma5_36deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (5, 2, 3))
    v1 = inplane.basis[:, 0].astype(float)
    v2 = inplane.basis[:, 1].astype(float)
    area = np.linalg.norm(np.cross(v1, v2))
    assert area < 40.0, (
        f"CSL cell area ({area:.4f}) too large; non-primitive null basis suspected"
    )
    plane = np.array([5, 2, 3], dtype=object)
    assert int(plane @ inplane.basis[:, 0]) == 0
    assert int(plane @ inplane.basis[:, 1]) == 0


def test_inplane_csl_sigma5_36deg_plane210_shortest_vector_length_1(
    sigma5_36deg_rotation,
):
    # The primitive [2,1,0] plane lattice contains [0,0,1], which is also a
    # CSL vector for the [001] rotation; shortest vector must have length 1
    from GBOpt.crystallography.reduction import gauss_reduce_2d
    csl = csl_from_scaled_rotation(sigma5_36deg_rotation)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (2, 1, 0))
    v1 = inplane.basis[:, 0].astype(float)
    v2 = inplane.basis[:, 1].astype(float)
    r1, _r2 = gauss_reduce_2d(v1, v2)
    assert abs(np.linalg.norm(r1) - 1.0) < 1e-9, (
        f"Expected shortest in-plane CSL vector of length 1, got {r1} "
        f"(norm={np.linalg.norm(r1):.4f})"
    )
    area = np.linalg.norm(np.cross(v1, v2))
    assert area < 15.0, (
        f"CSL cell area ({area:.4f}) too large; non-primitive basis suspected"
    )
