# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import math
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from GBOpt.crystallography.orientation import (
    build_mixed_orientations,
    build_symmetric_tilt_orientations,
    build_tilt_orientations,
    build_twist_orientations,
    five_dof_from_axis_angle,
    five_dof_from_orientation_matrices,
    inclination_from_normal,
    normalize_direction,
    orientation_matrices_from_five_dof,
    validate_orientation_matrix,
)
from GBOpt.crystallography.types import CrystallographyValueError


def _assert_proper_orientation(matrix: np.ndarray, *, atol: float = 1.0e-10) -> None:
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=atol, rtol=0.0)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=atol)


def _axis_rotation(axis: object, angle_deg: float) -> np.ndarray:
    axis_hat = np.asarray(axis, dtype=float)
    axis_hat /= np.linalg.norm(axis_hat)
    return Rotation.from_rotvec(axis_hat * math.radians(angle_deg)).as_matrix()


def _inclination_matrix(params: np.ndarray) -> np.ndarray:
    theta, phi = params[3:]
    return (
        Rotation.from_euler("z", phi) * Rotation.from_euler("y", theta)
    ).as_matrix()


# --------------------------------------------------------------------------------------
# Floating-point direction and orientation validation
# --------------------------------------------------------------------------------------


def test_normalize_direction_returns_unit_copy_without_mutating_input():
    direction = np.array([3.0, 4.0, 0.0])
    original = direction.copy()

    normalized = normalize_direction(direction)

    np.testing.assert_array_equal(direction, original)
    np.testing.assert_allclose(normalized, [0.6, 0.8, 0.0])
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("direction", "match"),
    [
        pytest.param([0.0, 0.0, 0.0], "nonzero", id="zero"),
        pytest.param([1.0, 2.0], "shape", id="wrong-shape"),
        pytest.param([1.0, np.nan, 0.0], "finite", id="nan"),
        pytest.param([1.0, np.inf, 0.0], "finite", id="infinite"),
    ],
)
def test_normalize_direction_rejects_invalid_vectors(direction, match):
    with pytest.raises(CrystallographyValueError, match=match):
        normalize_direction(direction)


@pytest.mark.parametrize("tol", [0.0, -1.0, np.inf, np.nan, True])
def test_normalize_direction_rejects_invalid_tolerance(tol):
    with pytest.raises(CrystallographyValueError):
        normalize_direction([1.0, 0.0, 0.0], tol=tol)


def test_validate_orientation_matrix_normalizes_scaled_rows():
    matrix = np.diag([2.0, 3.0, 4.0])

    normalized = validate_orientation_matrix(matrix)

    np.testing.assert_allclose(normalized, np.eye(3))
    _assert_proper_orientation(normalized)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        pytest.param(np.eye(2), "shape", id="wrong-shape"),
        pytest.param(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "nonzero",
            id="zero-row",
        ),
        pytest.param(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "orthogonal",
            id="nonorthogonal",
        ),
        pytest.param(np.diag([-1.0, 1.0, 1.0]), "right-handed", id="left-handed"),
        pytest.param(
            [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]],
            "finite",
            id="nonfinite",
        ),
    ],
)
def test_validate_orientation_matrix_rejects_invalid_matrices(matrix, match):
    with pytest.raises(CrystallographyValueError, match=match):
        validate_orientation_matrix(matrix)


# --------------------------------------------------------------------------------------
# Floating-point P/Q builders
# --------------------------------------------------------------------------------------


def test_build_tilt_orientations_satisfies_row_rotation_contract():
    normal = np.array([3.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle_deg = 36.869898

    P, Q = build_tilt_orientations(normal, axis, angle_deg)

    _assert_proper_orientation(P)
    _assert_proper_orientation(Q)
    np.testing.assert_allclose(P[0], normal / np.linalg.norm(normal))
    np.testing.assert_allclose(P[1], axis)
    np.testing.assert_allclose(Q, P @ _axis_rotation(axis, angle_deg), atol=1.0e-12)


def test_build_tilt_orientations_rejects_axis_outside_boundary_plane():
    with pytest.raises(
        CrystallographyValueError,
        match="must lie in the boundary plane",
    ):
        build_tilt_orientations([1, 0, 0], [1, 0, 1], 30.0)


def test_build_symmetric_tilt_orientations_mirrors_boundary_normals():
    median = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle_deg = 2.0 * math.degrees(math.atan(1.0 / 5.0))

    P, Q = build_symmetric_tilt_orientations(median, axis, angle_deg)

    _assert_proper_orientation(P)
    _assert_proper_orientation(Q)
    np.testing.assert_allclose(P[0], np.array([5.0, 1.0, 0.0]) / math.sqrt(26.0))
    np.testing.assert_allclose(Q[0], np.array([5.0, -1.0, 0.0]) / math.sqrt(26.0))
    np.testing.assert_allclose(Q, P @ _axis_rotation(axis, angle_deg), atol=1.0e-12)


def test_build_symmetric_tilt_orientations_rejects_nonperpendicular_axis():
    with pytest.raises(CrystallographyValueError, match="perpendicular"):
        build_symmetric_tilt_orientations([1, 0, 0], [1, 0, 1], 30.0)


def test_build_twist_orientations_preserves_boundary_normal():
    normal = np.array([1.0, 1.0, 1.0])
    angle_deg = 60.0

    P, Q = build_twist_orientations(normal, angle_deg)

    _assert_proper_orientation(P)
    _assert_proper_orientation(Q)
    np.testing.assert_allclose(P[0], normal / np.linalg.norm(normal))
    np.testing.assert_allclose(Q[0], P[0], atol=1.0e-12)
    np.testing.assert_allclose(Q, P @ _axis_rotation(normal, angle_deg), atol=1.0e-12)


def test_build_twist_orientations_projects_custom_reference():
    P, _Q = build_twist_orientations(
        [0.0, 0.0, 1.0],
        45.0,
        in_plane_reference=[1.0, 1.0, 5.0],
    )

    np.testing.assert_allclose(P[1], [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0])


def test_build_twist_orientations_rejects_parallel_reference():
    with pytest.raises(CrystallographyValueError, match="must not be parallel"):
        build_twist_orientations(
            [0.0, 0.0, 1.0],
            45.0,
            in_plane_reference=[0.0, 0.0, 2.0],
        )


def test_build_mixed_orientations_uses_axis_projection_by_default():
    normal = np.array([3.0, 1.0, 1.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle_deg = 36.869898

    P, Q = build_mixed_orientations(normal, axis, angle_deg)

    normal_hat = normal / np.linalg.norm(normal)
    projected_axis = axis - np.dot(axis, normal_hat) * normal_hat
    projected_axis /= np.linalg.norm(projected_axis)

    _assert_proper_orientation(P)
    _assert_proper_orientation(Q)
    np.testing.assert_allclose(P[0], normal_hat)
    np.testing.assert_allclose(P[1], projected_axis)
    np.testing.assert_allclose(Q, P @ _axis_rotation(axis, angle_deg), atol=1.0e-12)


@pytest.mark.parametrize(
    ("normal", "axis", "match"),
    [
        pytest.param([1, 0, 0], [0, 1, 0], "twist component", id="pure-tilt"),
        pytest.param([1, 0, 0], [2, 0, 0], "tilt component", id="pure-twist"),
    ],
)
def test_build_mixed_orientations_rejects_pure_boundaries(normal, axis, match):
    with pytest.raises(CrystallographyValueError, match=match):
        build_mixed_orientations(normal, axis, 30.0)


# --------------------------------------------------------------------------------------
# Five-DOF conversion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "normal",
    [
        pytest.param([1.0, 0.0, 0.0], id="lab-x"),
        pytest.param([3.0, 1.0, 2.0], id="general"),
        pytest.param([0.0, 1.0, 0.0], id="positive-y-singularity"),
        pytest.param([0.0, -1.0, 0.0], id="negative-y-singularity"),
    ],
)
def test_inclination_from_normal_reconstructs_unit_normal(normal):
    theta, phi = inclination_from_normal(normal)
    params = np.array([0.0, 0.0, 0.0, theta, phi])

    reconstructed = _inclination_matrix(params)[0]

    np.testing.assert_allclose(
        reconstructed,
        np.asarray(normal, dtype=float) / np.linalg.norm(normal),
        atol=1.0e-12,
    )


def test_five_dof_from_axis_angle_reconstructs_rotation_and_inclination():
    axis = np.array([1.0, 1.0, 1.0])
    normal = np.array([3.0, 1.0, 2.0])
    angle_deg = 60.0

    params = five_dof_from_axis_angle(axis, angle_deg, normal)

    reconstructed_rotation = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    np.testing.assert_allclose(
        reconstructed_rotation,
        _axis_rotation(axis, angle_deg),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _inclination_matrix(params)[0],
        normal / np.linalg.norm(normal),
        atol=1.0e-12,
    )


def test_five_dof_from_axis_angle_suppresses_scipy_gimbal_lock_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = five_dof_from_axis_angle([0, 0, 1], 30.0, [1, 0, 0])

    reconstructed = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    np.testing.assert_allclose(reconstructed, _axis_rotation([0, 0, 1], 30.0))


def test_five_dof_from_orientation_matrices_accepts_scaled_rows():
    angle_deg = 36.869898
    P_unit, Q_unit = build_tilt_orientations([3, 1, 0], [0, 0, 1], angle_deg)
    P = P_unit * np.array([2.0, 3.0, 4.0])[:, np.newaxis]
    Q = Q_unit * np.array([5.0, 6.0, 7.0])[:, np.newaxis]

    params = five_dof_from_orientation_matrices(P, Q)

    reconstructed_rotation = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    np.testing.assert_allclose(
        reconstructed_rotation,
        P_unit.T @ Q_unit,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(_inclination_matrix(params)[0], P_unit[0], atol=1.0e-12)


def test_five_dof_from_orientation_matrices_warns_and_uses_p_normal():
    P = np.eye(3)
    Q = _axis_rotation([0, 0, 1], 20.0)

    with pytest.warns(UserWarning, match=r"differs from P\[0\]"):
        params = five_dof_from_orientation_matrices(
            P,
            Q,
            boundary_normal=[0, 1, 0],
        )

    np.testing.assert_allclose(_inclination_matrix(params)[0], P[0], atol=1.0e-12)


@pytest.mark.parametrize("normal_warning_deg", [-1.0, np.nan, np.inf, True])
def test_five_dof_from_orientation_matrices_rejects_invalid_warning_threshold(
    normal_warning_deg,
):
    with pytest.raises(CrystallographyValueError):
        five_dof_from_orientation_matrices(
            np.eye(3),
            np.eye(3),
            normal_warning_deg=normal_warning_deg,
        )


# --------------------------------------------------------------------------------------
# orientation_matrices_from_five_dof
# --------------------------------------------------------------------------------------


def test_orientation_matrices_from_five_dof_reconstructs_frames():
    params = np.array(
        [
            np.radians(20.0),
            np.radians(35.0),
            np.radians(-10.0),
            np.radians(15.0),
            np.radians(-25.0),
        ]
    )

    R_left, R_right = orientation_matrices_from_five_dof(params)

    expected_misorientation = Rotation.from_euler(
        "ZXZ",
        params[:3],
    ).as_matrix()
    expected_left = (
        Rotation.from_euler("z", params[4])
        * Rotation.from_euler("y", params[3])
    ).as_matrix()

    np.testing.assert_allclose(R_left, expected_left, atol=1e-12)
    np.testing.assert_allclose(
        R_left.T @ R_right,
        expected_misorientation,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        pytest.param(
            [],
            r"params must have shape \(5,\)",
            id="empty",
        ),
        pytest.param(
            [0.0, 0.0, 0.0, 0.0],
            r"params must have shape \(5,\)",
            id="too-short",
        ),
        pytest.param(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            r"params must have shape \(5,\)",
            id="too-long",
        ),
        pytest.param(
            [[0.0, 0.0, 0.0, 0.0, 0.0]],
            r"params must have shape \(5,\)",
            id="two-dimensional",
        ),
        pytest.param(
            [np.nan, 0.0, 0.0, 0.0, 0.0],
            "params must contain only finite values",
            id="nan",
        ),
        pytest.param(
            [0.0, np.inf, 0.0, 0.0, 0.0],
            "params must contain only finite values",
            id="positive-infinity",
        ),
        pytest.param(
            [0.0, 0.0, -np.inf, 0.0, 0.0],
            "params must contain only finite values",
            id="negative-infinity",
        ),
    ],
)
def test_orientation_matrices_from_five_dof_rejects_invalid_params(
    params,
    message,
):
    with pytest.raises(CrystallographyValueError, match=message):
        orientation_matrices_from_five_dof(params)
