# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for crystallographic exactification operations."""

import numpy as np
import pytest

from GBOpt.BoundarySpec import BoundarySpecError, PQSpec
from GBOpt.crystallography.boundary import pq_spec_to_embedding
from GBOpt.crystallography.exactification import (
    _rationalize_direction,
    exactify_five_dof,
)
from GBOpt.crystallography.orientation import (
    orientation_matrices_from_five_dof,
    validate_orientation_matrix,
)
from GBOpt.crystallography.pq import recover_exact_row_rotation_from_paired_pq
from GBOpt.crystallography.types import (
    CrystallographyNotImplementedError,
    CrystallographyValueError,
)

SIGMA5_PARAMS = np.array([np.arctan2(3, 4), 0.0, 0.0, 0.0, 0.0])
SIGMA3_INCLINED_PARAMS = np.array(
    [
        3 * np.pi / 4,
        np.arccos(-1 / 3),
        np.pi / 4,
        np.pi / 4,
        -np.arcsin(1 / np.sqrt(3.0)),
    ]
)

EXACTIFICATION_CASES = [
    pytest.param(
        SIGMA5_PARAMS,
        np.array(
            [
                [5, 0, 0],
                [0, 5, 0],
                [0, 0, 1],
            ],
            dtype=object,
        ),
        np.array(
            [
                [4, -3, 0],
                [3, 4, 0],
                [0, 0, 1],
            ],
            dtype=object,
        ),
        id="sigma5-twist",
    ),
    pytest.param(
        SIGMA3_INCLINED_PARAMS,
        np.array(
            [
                [1, 1, 1],
                [1, 1, -2],
                [-1, 1, 0],
            ],
            dtype=object,
        ),
        np.array(
            [
                [1, 1, 1],
                [-1, -1, 2],
                [1, -1, 0],
            ],
            dtype=object,
        ),
        id="inclined-sigma3",
    ),
]


def _assert_matches_five_dof(P, Q, params):
    expected_left, expected_right = orientation_matrices_from_five_dof(params)
    actual_left = validate_orientation_matrix(P, "P")
    actual_right = validate_orientation_matrix(Q, "Q")

    np.testing.assert_allclose(actual_left[0], expected_left[0], atol=1.0e-12)
    np.testing.assert_allclose(
        actual_left.T @ actual_right,
        expected_left.T @ expected_right,
        atol=1.0e-12,
    )


# --------------------------------------------------------------------------------------
# _rationalize_direction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        pytest.param(
            [2.0, 4.0, 6.0],
            [1, 2, 3],
            id="gcd-reduced",
        ),
        pytest.param(
            [-2.0, -4.0, -6.0],
            [-1, -2, -3],
            id="negative-direction-preserved",
        ),
        pytest.param(
            [0.0, 0.0, -5.0],
            [0, 0, -1],
            id="negative-axis",
        ),
    ],
)
def test_rationalize_direction_returns_primitive_integer_direction(
    direction,
    expected,
):
    result = _rationalize_direction(
        direction,
        max_denominator=100,
        tol=1.0e-12,
        name="direction",
    )

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == object


@pytest.mark.parametrize(
    ("direction", "match"),
    [
        pytest.param(
            [1.0, 2.0],
            r"shape \(3,\)",
            id="wrong-shape",
        ),
        pytest.param(
            [1.0, np.nan, 0.0],
            "finite values",
            id="nan",
        ),
        pytest.param(
            [1.0, np.inf, 0.0],
            "finite values",
            id="infinite",
        ),
        pytest.param(
            [0.0, 0.0, 0.0],
            "nonzero",
            id="zero",
        ),
    ],
)
def test_rationalize_direction_rejects_invalid_direction(direction, match):
    with pytest.raises(CrystallographyValueError, match=match):
        _rationalize_direction(
            direction,
            max_denominator=100,
            tol=1.0e-12,
            name="direction",
        )


def test_rationalize_direction_rejects_approximation_outside_tolerance():
    with pytest.raises(
        CrystallographyValueError,
        match="could not be exactified",
    ):
        _rationalize_direction(
            [1.0, np.sqrt(2.0), 0.0],
            max_denominator=8,
            tol=1.0e-12,
            name="direction",
        )


# --------------------------------------------------------------------------------------
# exactify_five_dof
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("params", "expected_P", "expected_Q"), EXACTIFICATION_CASES)
def test_exactify_five_dof_returns_expected_paired_pq(
    params,
    expected_P,
    expected_Q,
):
    P, Q = exactify_five_dof(params)

    np.testing.assert_array_equal(P, expected_P)
    np.testing.assert_array_equal(Q, expected_Q)
    _assert_matches_five_dof(P, Q, params)

    recovered = recover_exact_row_rotation_from_paired_pq(P, Q)
    recovered_matrix = (
        np.asarray(recovered.matrix, dtype=np.float64) / recovered.denominator
    )
    expected_left, expected_right = orientation_matrices_from_five_dof(params)
    np.testing.assert_allclose(
        recovered_matrix,
        expected_left.T @ expected_right,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(SIGMA5_PARAMS, id="sigma5-twist"),
        pytest.param(SIGMA3_INCLINED_PARAMS, id="inclined-sigma3"),
    ],
)
def test_exactified_pq_round_trips_through_primitive_pq_spec(params):
    P, Q = exactify_five_dof(params)

    embedding = pq_spec_to_embedding(PQSpec(P=P, Q=Q, basis_mode="primitive"))

    expected_left, expected_right = orientation_matrices_from_five_dof(params)
    np.testing.assert_allclose(embedding.R_left[0], expected_left[0], atol=1.0e-12)
    np.testing.assert_allclose(
        embedding.R_left.T @ embedding.R_right,
        expected_left.T @ expected_right,
        atol=1.0e-12,
    )


def test_exactify_five_dof_rejects_non_csl_misorientation():
    with pytest.raises(CrystallographyValueError, match="could not be exactified"):
        exactify_five_dof(np.array([0.0, 0.0, 0.1, 0.0, 0.0]))


def test_exactify_five_dof_rejects_sigma_above_limit():
    with pytest.raises(CrystallographyValueError, match="exceeds max_sigma=4"):
        exactify_five_dof(SIGMA5_PARAMS, max_sigma=4)


def test_exactify_five_dof_rejects_small_max_denominator():
    with pytest.raises(
        CrystallographyValueError,
        match="could not be exactified within max_denominator=2",
    ):
        exactify_five_dof(SIGMA5_PARAMS, max_denominator=2)


def test_exactify_five_dof_rejects_misorientation_outside_angle_tolerance():
    params = SIGMA5_PARAMS.copy()
    params[0] += 1.0e-10

    with pytest.raises(CrystallographyValueError, match="angle_tol=1e-12"):
        exactify_five_dof(params, angle_tol=1.0e-12)


def test_exactify_five_dof_rejects_inexact_boundary_plane():
    params = np.array([0.0, 0.0, 0.0, 0.3, 0.0])

    with pytest.raises(
        CrystallographyValueError,
        match="boundary plane normal could not be exactified",
    ):
        exactify_five_dof(
            params,
            max_denominator=8,
            plane_tol=1.0e-12,
        )


def test_exactify_five_dof_enforces_exact_cell_size_limit():
    with pytest.raises(BoundarySpecError, match="exceeds max_exact_atoms"):
        exactify_five_dof(SIGMA5_PARAMS, max_exact_atoms=4)


def test_exactify_five_dof_rejects_non_cubic_lattice_metric():
    with pytest.raises(
        CrystallographyNotImplementedError,
        match="non-cubic lattice metrics are not implemented",
    ):
        exactify_five_dof(np.zeros(5), lattice_metric=np.eye(3))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        pytest.param("max_exact_atoms", 0, id="nonpositive-max-exact-atoms"),
        pytest.param("max_sigma", True, id="boolean-max-sigma"),
        pytest.param("max_denominator", 1.5, id="noninteger-max-denominator"),
        pytest.param("angle_tol", 0.0, id="nonpositive-angle-tolerance"),
        pytest.param("plane_tol", np.inf, id="nonfinite-plane-tolerance"),
    ],
)
def test_exactify_five_dof_rejects_invalid_bounds(keyword, value):
    with pytest.raises(CrystallographyValueError, match=keyword):
        exactify_five_dof(np.zeros(5), **{keyword: value})


def test_exactify_five_dof_rejects_malformed_params():
    with pytest.raises(CrystallographyValueError, match=r"shape \(5,\)"):
        exactify_five_dof(np.zeros(4))
