# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import math

import numpy as np
import pytest
from crystallography_fixtures import CSL_SCENARIO_DICTS

from GBOpt.crystallography.quaternion import (
    _integer_quaternion_candidate_from_unit,
    integer_quaternion_from_unit,
    normalize_integer_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_scaled_rotation,
)
from GBOpt.crystallography.rotation import assert_scaled_rotation
from GBOpt.crystallography.types import (
    CrystallographyNotImplementedError,
    CrystallographyValueError,
)

# --------------------------------------------------------------------------------------
# Shared inputs
# --------------------------------------------------------------------------------------

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], N=5
SIGMA5_QUAT = (2, 0, 0, 1)
SIGMA5_N = 5
SIGMA5_M = np.array([[3, -4, 0], [4, 3, 0], [0, 0, 5]], dtype=object)

# Sigma5 [001] 36.87 deg -- quat [3, 0, 0, 1], N=10
SIGMA5_36_R = np.array(
    [
        [4 / 5, -3 / 5, 0],
        [3 / 5, 4 / 5, 0],
        [0, 0, 1],
    ]
)

# Sigma5 [001] 53.13 deg unit quaternion
SIGMA5_QUAT_NORM = np.array([2, 0, 0, 1], dtype=float) / np.sqrt(5)
SIGMA5_R_EXPECTED = np.array(
    [
        [3 / 5, -4 / 5, 0],
        [4 / 5, 3 / 5, 0],
        [0, 0, 1],
    ]
)

EXACT_CSL_SCENARIOS = [
    pytest.param(case, id=str(case["id"])) for case in CSL_SCENARIO_DICTS
]

EXACT_CSL_SCENARIOS_WITH_EXPECTED_M = [
    pytest.param(case, id=str(case["id"]))
    for case in CSL_SCENARIO_DICTS
    if "expected_M" in case
]


# --------------------------------------------------------------------------------------
# normalize_integer_quaternion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quat", "expected"),
    [
        pytest.param((4, 0, 0, 2), (2, 0, 0, 1), id="common-factor"),
        pytest.param((0, 0, 0, -3), (0, 0, 0, 1), id="canonical-sign-late"),
        pytest.param((-2, 0, 0, -1), (2, 0, 0, 1), id="canonical-sign-first"),
    ],
)
def test_normalize_integer_quaternion_returns_primitive_canonical_form(quat, expected):
    assert normalize_integer_quaternion(quat) == expected


@pytest.mark.parametrize(
    ("quat", "match"),
    [
        pytest.param([1, 0, 0], r"shape \(4,\)", id="wrong-length"),
        pytest.param([1.5, 0, 0, 1], "integer-valued", id="non-integer-float"),
        pytest.param([True, 0, 0, 1], "not an integer", id="bool"),
        pytest.param([np.nan, 0, 0, 1], "finite", id="nan"),
        pytest.param((0, 0, 0, 0), "zero quaternion", id="zero"),
    ],
)
def test_normalize_integer_quaternion_rejects_invalid_quaternion(quat, match):
    with pytest.raises(CrystallographyValueError, match=match):
        normalize_integer_quaternion(quat)


# --------------------------------------------------------------------------------------
# quaternion_to_scaled_rotation
# --------------------------------------------------------------------------------------


def test_quaternion_to_scaled_rotation_returns_expected_sigma5_rotation():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)

    assert rot.denominator == SIGMA5_N
    np.testing.assert_array_equal(rot.matrix, SIGMA5_M)
    assert rot.source == "quaternion"
    assert rot.quaternion == SIGMA5_QUAT
    assert_scaled_rotation(rot)


def test_quaternion_to_scaled_rotation_canonicalize_controls_common_factor_reduction():
    raw_quat = (4, 0, 0, 2)

    rot_canonicalized = quaternion_to_scaled_rotation(raw_quat, canonicalize=True)
    rot_raw = quaternion_to_scaled_rotation(raw_quat, canonicalize=False)

    assert rot_canonicalized.denominator == SIGMA5_N
    np.testing.assert_array_equal(rot_canonicalized.matrix, SIGMA5_M)

    assert rot_raw.denominator == 20
    np.testing.assert_array_equal(rot_raw.matrix, 4 * SIGMA5_M)


def test_quaternion_to_scaled_rotation_zero_quaternion_raises():
    with pytest.raises(CrystallographyValueError, match="zero quaternion"):
        quaternion_to_scaled_rotation((0, 0, 0, 0))


def test_quaternion_to_scaled_rotation_lattice_metric_raises_not_implemented():
    with pytest.raises(CrystallographyNotImplementedError, match="non-cubic"):
        quaternion_to_scaled_rotation(SIGMA5_QUAT, lattice_metric=np.eye(3))


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_quaternion_to_scaled_rotation_matches_scenario_denominators(case):
    rot = quaternion_to_scaled_rotation(case["q"])

    assert rot.denominator == case["expected_N"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS_WITH_EXPECTED_M)
def test_quaternion_to_scaled_rotation_matches_scenario_numerator_matrices(case):
    rot = quaternion_to_scaled_rotation(case["q"])

    np.testing.assert_array_equal(rot.matrix, case["expected_M"])


# --------------------------------------------------------------------------------------
# integer_quaternion_from_unit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quat", "expected"),
    [
        pytest.param(SIGMA5_QUAT_NORM, SIGMA5_QUAT, id="positive"),
        pytest.param(-SIGMA5_QUAT_NORM, SIGMA5_QUAT, id="negative-canonicalized"),
    ],
)
def test_integer_quaternion_from_unit_returns_primitive_canonical_integer_quaternion(
    quat,
    expected,
):
    result = integer_quaternion_from_unit(quat)

    assert result == expected
    assert math.gcd(*map(abs, result)) == 1


@pytest.mark.parametrize(
    ("quat", "match"),
    [
        pytest.param([1.0, 0.0, 0.0], r"shape \(4,\)", id="wrong-shape"),
        pytest.param([np.nan, 0.0, 0.0, 1.0], "finite", id="nan"),
        pytest.param([np.inf, 0.0, 0.0, 1.0], "finite", id="inf"),
        pytest.param([0.0, 0.0, 0.0, 0.0], "zero quaternion", id="zero"),
    ],
)
def test_integer_quaternion_from_unit_rejects_invalid_unit_quaternions(quat, match):
    with pytest.raises(CrystallographyValueError, match=match):
        integer_quaternion_from_unit(np.array(quat, dtype=float))


@pytest.mark.parametrize(
    "max_denominator",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_integer_quaternion_from_unit_rejects_invalid_max_denominator(
    max_denominator,
):
    with pytest.raises(CrystallographyValueError, match="positive integer"):
        integer_quaternion_from_unit(
            SIGMA5_QUAT_NORM,
            max_denominator=max_denominator,
        )


def test_integer_quaternion_from_unit_unrecoverable_quaternion_raises():
    irrational = np.array([np.sqrt(2), 1.0, 1.0, 1.0], dtype=float)
    irrational /= np.linalg.norm(irrational)

    with pytest.raises(CrystallographyValueError, match="does not match"):
        integer_quaternion_from_unit(irrational, max_denominator=2)


def test_integer_quaternion_from_unit_accepts_numpy_max_denominator():
    result = integer_quaternion_from_unit(
        SIGMA5_QUAT_NORM,
        max_denominator=np.int64(5),  # type: ignore[ty:invalid-argument-type]
    )

    assert result == (2, 0, 0, 1)


def test_integer_quaternion_candidate_does_not_apply_fixed_match_tolerance():
    perturbed = SIGMA5_QUAT_NORM.copy()
    perturbed[0] += 1.0e-8
    perturbed /= np.linalg.norm(perturbed)

    result = _integer_quaternion_candidate_from_unit(
        perturbed,
        max_denominator=3,
    )

    assert result == SIGMA5_QUAT


@pytest.mark.parametrize(
    "quat",
    [
        pytest.param(["bad", 0, 0, 1], id="string-component"),
        pytest.param(object(), id="non-array-object"),
    ],
)
def test_integer_quaternion_from_unit_translates_conversion_errors(quat):
    with pytest.raises(
        CrystallographyValueError,
        match="finite four-component quaternion",
    ):
        integer_quaternion_from_unit(quat)


# --------------------------------------------------------------------------------------
# quaternion_to_rotation_matrix
# --------------------------------------------------------------------------------------


def test_quaternion_to_rotation_matrix_sigma5_matches_expected():
    rotation = quaternion_to_rotation_matrix(SIGMA5_QUAT_NORM)

    np.testing.assert_allclose(rotation, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_integer_quaternion_normalized_internally():
    rotation = quaternion_to_rotation_matrix(np.array([2, 0, 0, 1]))

    np.testing.assert_allclose(rotation, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_integer_valued_float_normalized_internally():
    rotation = quaternion_to_rotation_matrix(np.array([2.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(rotation, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_output_is_proper_rotation():
    rotation = quaternion_to_rotation_matrix(SIGMA5_QUAT_NORM)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-12


def test_quaternion_to_rotation_matrix_identity_quaternion_gives_identity():
    rotation = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_quaternion_to_rotation_matrix_sigma5_36deg_matches_expected():
    quat = np.array([3, 0, 0, 1], dtype=float) / np.sqrt(10)

    rotation = quaternion_to_rotation_matrix(quat)

    np.testing.assert_allclose(rotation, SIGMA5_36_R, atol=1e-12)


@pytest.mark.parametrize(
    ("quat", "match"),
    [
        pytest.param([1, 0, 0], r"shape \(4,\)", id="too-short"),
        pytest.param([[1, 0, 0, 0]], r"shape \(4,\)", id="two-dimensional-row"),
        pytest.param(np.ones((2, 2)), r"shape \(4,\)", id="two-by-two"),
        pytest.param([0.0, 0.0, 0.0, 0.0], "non-zero", id="zero"),
        pytest.param([np.nan, 0.0, 0.0, 1.0], "finite values", id="nan"),
        pytest.param([np.inf, 0.0, 0.0, 1.0], "finite values", id="inf"),
        pytest.param(
            [1.5, 0.0, 0.0, 1.0],
            "exact integer quaternion",
            id="non-unit-non-integer",
        ),
    ],
)
def test_quaternion_to_rotation_matrix_rejects_invalid_quaternions(
    quat,
    match,
):
    with pytest.raises(CrystallographyValueError, match=match):
        quaternion_to_rotation_matrix(
            np.array(quat, dtype=float),
        )


@pytest.mark.parametrize(
    "quat",
    [
        pytest.param(["bad", 0, 0, 1], id="string-component"),
        pytest.param(object(), id="non-array-object"),
    ],
)
def test_quaternion_to_rotation_matrix_translates_conversion_errors(quat):
    with pytest.raises(
        CrystallographyValueError,
        match="finite four-component quaternion",
    ):
        quaternion_to_rotation_matrix(quat)
