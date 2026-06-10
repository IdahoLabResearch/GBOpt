# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
from unittest.mock import patch

import numpy as np
import pytest
from crystallography_fixtures import CSL_SCENARIO_DICTS

from GBOpt.crystallography.csl import (
    csl_from_scaled_rotation,
    dsc_basis,
    sigma_from_snf_diagonal,
    verify_coincidence_basis,
)
from GBOpt.crystallography.integer import integer_det3
from GBOpt.crystallography.plane import inplane_area_index, inplane_basis_from_csl
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.reduction import gauss_reduce_2d
from GBOpt.crystallography.types import (
    CrystallographyBackendError,
    CrystallographyNotImplementedError,
    CrystallographyValueError,
)
from GBOpt.Utils.integer_normal_forms import ExactNormalFormError

# --------------------------------------------------------------------------------------
# Shared scenarios
# --------------------------------------------------------------------------------------

EXACT_CSL_SCENARIOS = [pytest.param(d, id=str(d["id"])) for d in CSL_SCENARIO_DICTS]

EXACT_CSL_SCENARIOS_WITH_EXPECTED_HNF = [
    pytest.param(d, id=str(d["id"]))
    for d in CSL_SCENARIO_DICTS
    if "expected_basis_hnf" in d
]

EXACT_CSL_SCENARIOS_WITH_EXPECTED_KERNEL_MODULI = [
    pytest.param(d, id=str(d["id"]))
    for d in CSL_SCENARIO_DICTS
    if "expected_kernel_moduli" in d
]

INPLANE_EXACT_CSL_SCENARIOS = [
    pytest.param(d, id=str(d["id"]))
    for d in CSL_SCENARIO_DICTS
    if d["plane"] is not None
]

INPLANE_EXACT_CSL_SCENARIOS_WITH_EXPECTED_CROSS_ABS = [
    pytest.param(d, id=str(d["id"]))
    for d in CSL_SCENARIO_DICTS
    if d["plane"] is not None and "expected_inplane_cross_abs" in d
]

SIGMA5_36DEG_QUAT = (3, 0, 0, 1)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _rotation_from_case(case):
    return quaternion_to_scaled_rotation(case["q"])


def _csl_from_case(case):
    return csl_from_scaled_rotation(_rotation_from_case(case))


def _inplane_from_case(case):
    csl = _csl_from_case(case)
    return inplane_basis_from_csl(csl.basis_hnf, case["plane"])


# --------------------------------------------------------------------------------------
# sigma_from_snf_diagonal
# --------------------------------------------------------------------------------------


def test_sigma_from_snf_diagonal_returns_sigma5_moduli():
    sigma, moduli = sigma_from_snf_diagonal(5, (1, 5, 25))
    assert sigma == 5
    assert moduli == (5, 1, 1)


def test_sigma_from_snf_diagonal_returns_sigma1_moduli():
    sigma, moduli = sigma_from_snf_diagonal(4, (4, 4, 4))
    assert sigma == 1
    assert moduli == (1, 1, 1)


@pytest.mark.parametrize(
    "denominator",
    [
        pytest.param(0, id="zero"),
        pytest.param(-5, id="negative"),
        pytest.param(5.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param("5", id="string"),
    ],
)
def test_sigma_from_snf_diagonal_rejects_invalid_denominator(denominator):
    with pytest.raises(
        CrystallographyValueError,
        match="denominator must be a positive integer",
    ):
        sigma_from_snf_diagonal(denominator, (1, 5, 25))


# --------------------------------------------------------------------------------------
# csl_from_scaled_rotation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_basis_sigma(case):
    csl = _csl_from_case(case)
    assert csl.sigma == case["expected_sigma"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_csl_basis_hnf_det(case):
    csl = _csl_from_case(case)
    assert abs(integer_det3(csl.basis_hnf)) == case["expected_hnf_det"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS_WITH_EXPECTED_HNF)
def test_csl_basis_hnf_known(case):
    csl = _csl_from_case(case)
    np.testing.assert_array_equal(csl.basis_hnf, case["expected_basis_hnf"])


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS_WITH_EXPECTED_KERNEL_MODULI)
def test_csl_kernel_moduli(case):
    csl = _csl_from_case(case)
    assert csl.diagnostics.kernel_moduli == case["expected_kernel_moduli"]


def test_csl_from_scaled_rotation_rejects_unknown_post_reduce():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))

    with pytest.raises(
        CrystallographyValueError,
        match="unknown post_reduce mode invalid",
    ):
        csl_from_scaled_rotation(
            rot,
            post_reduce="invalid",  # type: ignore[ty:invalid-argument-type]
        )


def test_csl_from_scaled_rotation_translates_snf_failure_to_backend_error():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))

    with patch(
        "GBOpt.crystallography.csl.smith_normal_form_3x3",
        side_effect=ExactNormalFormError("forced SNF failure"),
    ):
        with pytest.raises(CrystallographyBackendError, match="forced SNF failure"):
            csl_from_scaled_rotation(rot)


def test_csl_from_scaled_rotation_lll_preserves_sigma_and_hnf():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))

    raw = csl_from_scaled_rotation(rot, post_reduce="none")
    reduced = csl_from_scaled_rotation(rot, post_reduce="lll")

    assert reduced.sigma == raw.sigma
    np.testing.assert_array_equal(reduced.basis_hnf, raw.basis_hnf)
    check = verify_coincidence_basis(rot, reduced.basis_hnf, sigma=reduced.sigma)
    assert check.ok is True


# --------------------------------------------------------------------------------------
# verify_coincidence_basis
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_verify_coincidence_basis_accepts_valid_csl_basis(case):
    rot = _rotation_from_case(case)
    csl = _csl_from_case(case)

    check = verify_coincidence_basis(rot, csl.basis_hnf, sigma=case["expected_sigma"])

    assert check.ok is True
    assert check.det_basis == case["expected_sigma"]
    assert check.sigma == case["expected_sigma"]
    np.testing.assert_array_equal(check.residual_mod_N, np.zeros((3, 3), dtype=object))


def test_verify_coincidence_basis_rejects_non_csl_basis_by_residual():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    non_csl = np.eye(3, dtype=object)

    check = verify_coincidence_basis(rot, non_csl, sigma=5)

    assert check.ok is False
    assert np.any(check.residual_mod_N != 0)
    assert check.det_basis == 1
    assert check.sigma == 5


def test_verify_coincidence_basis_rejects_sigma_mismatch_by_determinant():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)

    check = verify_coincidence_basis(rot, csl.basis_hnf, sigma=3)

    assert check.ok is False
    assert check.det_basis == 5
    assert check.sigma == 3
    np.testing.assert_array_equal(
        check.residual_mod_N,
        np.zeros((3, 3), dtype=object),
    )


@pytest.mark.parametrize(
    "sigma",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("5", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_verify_coincidence_basis_rejects_invalid_sigma(sigma):
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)

    with pytest.raises(
        CrystallographyValueError,
        match="sigma must be a positive integer",
    ):
        verify_coincidence_basis(
            rot,
            csl.basis_hnf,
            sigma=sigma,
        )


def test_verify_coincidence_basis_rejects_non_integer_basis():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))

    with pytest.raises(CrystallographyValueError):
        verify_coincidence_basis(rot, np.eye(3) * 1.5)


# --------------------------------------------------------------------------------------
# inplane_basis_from_csl integration
# --------------------------------------------------------------------------------------


def _assert_basis_vectors_are_in_plane(inplane, plane):
    h = np.asarray(plane, dtype=object)
    assert int(h @ inplane.basis[:, 0]) == 0
    assert int(h @ inplane.basis[:, 1]) == 0


def _assert_basis_vectors_are_linearly_independent(inplane):
    v1 = inplane.basis[:, 0].astype(int)
    v2 = inplane.basis[:, 1].astype(int)
    cross = np.cross(v1, v2)
    assert any(v != 0 for v in cross), "basis vectors are linearly dependent"


def _assert_basis_vectors_are_csl_members(rotation, inplane):
    M = np.asarray(rotation.matrix, dtype=object)
    N = rotation.denominator

    for column in range(2):
        v = np.asarray(inplane.basis[:, column], dtype=object)
        residual = M @ v % N
        np.testing.assert_array_equal(residual, np.zeros(3, dtype=object))


def _assert_inplane_cross_abs(inplane, expected_cross_abs):
    v1 = inplane.basis[:, 0].astype(int)
    v2 = inplane.basis[:, 1].astype(int)
    np.testing.assert_array_equal(
        np.abs(np.cross(v1, v2)),
        np.asarray(expected_cross_abs, dtype=int),
    )


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_vectors_are_in_requested_plane(case):
    inplane = _inplane_from_case(case)

    _assert_basis_vectors_are_in_plane(inplane, case["plane"])


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_vectors_are_linearly_independent(case):
    inplane = _inplane_from_case(case)

    _assert_basis_vectors_are_linearly_independent(inplane)


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_csl_inplane_basis_vectors_are_csl_members(case):
    rot = _rotation_from_case(case)
    inplane = _inplane_from_case(case)

    _assert_basis_vectors_are_csl_members(rot, inplane)


@pytest.mark.parametrize(
    "case",
    INPLANE_EXACT_CSL_SCENARIOS_WITH_EXPECTED_CROSS_ABS,
)
def test_csl_inplane_basis_cross_abs_matches_known_values(case):
    inplane = _inplane_from_case(case)

    _assert_inplane_cross_abs(inplane, case["expected_inplane_cross_abs"])


def test_inplane_csl_sigma5_36deg_plane523_regression_uses_primitive_area():
    # Regression: general planes must use the full primitive plane lattice.
    # A non-primitive null basis produced an unnecessarily large in-plane CSL cell.
    plane = (5, 2, 3)
    rot = quaternion_to_scaled_rotation(SIGMA5_36DEG_QUAT)
    csl = csl_from_scaled_rotation(rot)
    inplane = inplane_basis_from_csl(csl.basis_hnf, plane)

    _assert_basis_vectors_are_in_plane(inplane, plane)
    _assert_basis_vectors_are_linearly_independent(inplane)
    _assert_basis_vectors_are_csl_members(rot, inplane)

    P = np.array(
        [
            plane,
            inplane.basis[:, 0],
            inplane.basis[:, 1],
        ],
        dtype=object,
    )
    assert inplane_area_index(P) == 5


def test_inplane_csl_sigma5_36deg_plane210_regression_has_unit_shortest_vector():
    # Regression: the primitive [2, 1, 0] plane lattice contains [0, 0, 1],
    # which is also a CSL vector for the [001] rotation.
    plane = (2, 1, 0)
    rot = quaternion_to_scaled_rotation(SIGMA5_36DEG_QUAT)
    csl = csl_from_scaled_rotation(rot)
    inplane = inplane_basis_from_csl(csl.basis_hnf, plane)

    _assert_basis_vectors_are_in_plane(inplane, plane)
    _assert_basis_vectors_are_linearly_independent(inplane)
    _assert_basis_vectors_are_csl_members(rot, inplane)
    _assert_inplane_cross_abs(inplane, (2, 1, 0))

    r1, _r2 = gauss_reduce_2d(inplane.basis[:, 0], inplane.basis[:, 1])
    assert np.linalg.norm(np.asarray(r1, dtype=float)) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# dsc_basis
# --------------------------------------------------------------------------------------


def test_dsc_basis_sigma5_001():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    result = dsc_basis(csl.basis_hnf, csl.sigma)
    expected_adj = np.array([[5, 0, 0], [-2, 1, 0], [0, 0, 5]], dtype=object)
    assert result.denominator == 5
    np.testing.assert_array_equal(result.numerator, expected_adj)
    assert integer_det3(result.numerator) == 25


def test_dsc_basis_rejects_sigma_mismatch():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)

    with pytest.raises(
        CrystallographyValueError,
        match="does not equal sigma=3",
    ):
        dsc_basis(csl.basis_hnf, sigma=3)


@pytest.mark.parametrize(
    "sigma",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("5", id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_dsc_basis_rejects_invalid_sigma(sigma):
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)

    with pytest.raises(
        CrystallographyValueError,
        match="sigma must be a positive integer",
    ):
        dsc_basis(csl.basis_hnf, sigma=sigma)


def test_dsc_basis_rejects_lattice_basis():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)

    with pytest.raises(
        CrystallographyNotImplementedError,
        match="non-cubic lattice bases are not implemented",
    ):
        dsc_basis(csl.basis_hnf, csl.sigma, lattice_basis=np.eye(3))


@pytest.mark.parametrize(
    "bad_basis",
    [
        pytest.param(np.ones((2, 2), dtype=object), id="shape"),
        pytest.param(
            np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), id="float"
        ),
    ],
)
def test_dsc_basis_rejects_invalid_csl_basis(bad_basis):
    with pytest.raises(CrystallographyValueError):
        dsc_basis(bad_basis, sigma=1)


def test_dsc_basis_handles_negative_determinant_basis():
    csl_basis = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=object)

    result = dsc_basis(csl_basis, sigma=1)

    np.testing.assert_array_equal(
        csl_basis @ result.numerator,
        np.eye(3, dtype=object),
    )
