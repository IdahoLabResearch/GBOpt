# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Floating-point crystallographic orientation-frame construction and conversion.

Provides validation and construction of row-wise grain orientation matrices, geometric
tilt, twist, and mixed-boundary frame builders, and conversion of axis-angle or paired
orientation descriptions to GBMaker's five-DOF representation.

The matrices handled here are floating-point orientation frames. They are not exact
integer P/Q matrices and are not suitable as exact ``PQSpec`` input without a separate
exactification step.
"""

from __future__ import annotations

import math
import warnings
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from .types import CrystallographyValueError

_DEFAULT_VECTOR_TOL = 1.0e-12
_DEFAULT_ORIENTATION_TOL = 1.0e-10
_DEFAULT_NORMAL_WARNING_DEG = 1.0

# ---------------------------------------------------------------------------
# Floating-point row-orientation construction and conversion
# ---------------------------------------------------------------------------


def _finite_float(value: object, name: str) -> float:
    """Return *value* as a finite float.

    :param value: Candidate scalar value.
    :param name: Parameter name used in the error message.
    :return: Finite floating-point value.
    :raises CrystallographyValueError: If the value is boolean, non-numeric, or
        non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise CrystallographyValueError(
            f"{name} must be a finite real number; got {value!r}."
        )

    result = float(value)
    if not math.isfinite(result):
        raise CrystallographyValueError(
            f"{name} must be a finite real number; got {value!r}."
        )
    return result


def _validated_tolerance(value: object, name: str) -> float:
    """Validate and return a strictly positive numerical tolerance.

    Boolean values are rejected even though ``bool`` is a subclass of ``int``. All other
    real scalar values are converted to a Python ``float`` and must be finite and
    greater than zero.

    :param value: Candidate tolerance value. It must be a finite, non-boolean real
        scalar greater than zero.
    :param name: Human-readable name for the tolerance. This value is included in
        validation error messages to identify the invalid argument.
    :return: The validated tolerance as a finite, strictly positive Python ``float``.
    :raises CrystallographyValueError: If ``value`` is boolean, is not a real scalar,
        cannot be represented as a finite ``float``, or is less than or equal to zero.
    """
    result = _finite_float(value, name)
    if result <= 0.0:
        raise CrystallographyValueError(
            f"{name} must be strictly positive; got {value!r}."
        )
    return result


def _vector3(value: object, name: str) -> np.ndarray:
    """Convert a value to a finite three-component floating-point vector.

    The input is converted to a NumPy array with ``numpy.float64`` dtype. This function
    validates only the vector's shape and finiteness; it does not normalize the vector
    or require it to be nonzero.

    :param value: Array-like value expected to contain exactly three numeric components.
    :param name: Human-readable name for the vector. This value is included in
        validation error messages to identify the invalid argument.
    :return: A one-dimensional ``numpy.float64`` array with shape ``(3,)``. The returned
        vector retains the input magnitude and is not normalized.
    :raises CrystallographyValueError: If ``value`` cannot be converted to a
        floating-point NumPy array, does not have shape ``(3,)``, or contains one or
        more NaN or infinite components.
    """
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CrystallographyValueError(
            f"{name} must be a finite three-component vector."
        ) from exc

    if vector.shape != (3,):
        raise CrystallographyValueError(
            f"{name} must have shape (3,); got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise CrystallographyValueError(f"{name} must contain only finite values.")
    return vector


def normalize_direction(
    direction: object,
    name: str = "direction",
    *,
    tol: float = _DEFAULT_VECTOR_TOL,
) -> np.ndarray:
    """Return a normalized copy of a nonzero three-component direction.

    :param direction: Candidate three-component direction.
    :param name: Parameter name used in error messages. Optional, defaults to
        ``"direction"``.
    :param tol: Minimum accepted vector norm. Keyword parameter, optional, defaults to
        ``1e-12``.
    :return: Unit-length ``float64`` direction.
    :raises CrystallographyValueError: If the direction has the wrong shape, contains
        non-finite values, or has norm no greater than ``tol``.
    """
    tol = _validated_tolerance(tol, "tol")
    vector = _vector3(direction, name)
    norm = float(np.linalg.norm(vector))
    if norm <= tol:
        raise CrystallographyValueError(
            f"{name} must be nonzero; got norm {norm:.3e}."
        )
    return vector / norm


def validate_orientation_matrix(
    matrix: object,
    name: str = "orientation matrix",
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> np.ndarray:
    """Return a normalized, validated row-wise orientation matrix.

    Rows identify the crystal directions aligned with the lab-frame x, y, and z axes.
    The rows must be finite, nonzero, mutually orthogonal, and right-handed. Row lengths
    need not be one on input; the returned matrix is row-normalized.

    :param matrix: Candidate row-wise orientation matrix.
    :param name: Matrix name used in error messages. Optional.
    :param tol: Absolute orthogonality and determinant tolerance. Keyword parameter,
        optional, defaults to ``1e-10``.
    :return: Normalized ``float64`` matrix with shape ``(3, 3)``.
    :raises CrystallographyValueError: If the matrix is malformed or is not a proper
        row-wise orientation matrix.
    """
    tol = _validated_tolerance(tol, "tol")
    try:
        candidate = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CrystallographyValueError(
            f"{name} must be a finite 3 by 3 matrix."
        ) from exc

    if candidate.shape != (3, 3):
        raise CrystallographyValueError(
            f"{name} must have shape (3, 3); got {candidate.shape}."
        )
    if not np.all(np.isfinite(candidate)):
        raise CrystallographyValueError(f"{name} must contain only finite values.")

    norms = np.linalg.norm(candidate, axis=1)
    if np.any(norms <= tol):
        row = int(np.flatnonzero(norms <= tol)[0])
        raise CrystallographyValueError(
            f"{name} row {row} must be nonzero; got norm {norms[row]:.3e}."
        )

    normalized = candidate / norms[:, np.newaxis]
    gram_error = float(
        np.max(np.abs(normalized @ normalized.T - np.eye(3, dtype=np.float64)))
    )
    if gram_error > tol:
        raise CrystallographyValueError(
            f"{name} rows must be mutually orthogonal after normalization; "
            f"maximum Gram-matrix error is {gram_error:.3e}."
        )

    determinant = float(np.linalg.det(normalized))
    if abs(determinant - 1.0) > tol:
        raise CrystallographyValueError(
            f"{name} must be right-handed after normalization; "
            f"determinant is {determinant:.12g}."
        )

    return normalized


def _project_into_plane(
    direction: object,
    normal: np.ndarray,
    *,
    name: str,
    tol: float,
) -> np.ndarray:
    """Project a direction into the plane perpendicular to a unit normal.

    The supplied direction is first validated and normalized. Its component parallel to
    ``normal`` is then removed, and the remaining in-plane component is normalized.

    This private helper assumes that ``normal`` has already been validated as a finite
    unit vector with shape ``(3,)``.

    :param direction: Array-like three-component direction to project. It need not be
        normalized, but it must be finite and nonzero.
    :param normal: Validated unit vector with shape ``(3,)`` defining the normal of the
        target plane.
    :param name: Human-readable name for ``direction``. This value is included in
        validation error messages to identify the invalid argument.
    :param tol: Strictly positive numerical tolerance used when validating the direction
        magnitude and determining whether the projected component is effectively zero.
    :return: A unit-length ``numpy.float64`` vector with shape ``(3,)`` that lies in the
        plane perpendicular to ``normal``.
    :raises CrystallographyValueError: If ``tol`` is invalid; if ``direction`` cannot be
        interpreted as a finite, nonzero three-component vector; or if ``direction`` is
        parallel to ``normal`` within ``tol`` and therefore has no nonzero in-plane
        projection.
    """
    direction_hat = normalize_direction(direction, name, tol=tol)
    projected = direction_hat - np.dot(direction_hat, normal) * normal
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= tol:
        raise CrystallographyValueError(
            f"{name} must not be parallel to the boundary normal."
        )
    return projected / projected_norm


def _default_in_plane_reference(
    normal: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    """Choose a stable default direction in the plane perpendicular to a normal.

    The Cartesian basis vector least aligned with ``normal`` is selected as a seed. That
    seed is projected into the plane perpendicular to ``normal`` and normalized.
    Selecting the least-aligned basis vector avoids an unstable projection from a seed
    that is nearly parallel to the normal.

    This private helper assumes that ``normal`` has already been validated as a finite
    unit vector with shape ``(3,)``.

    :param normal: Validated unit vector with shape ``(3,)`` defining the boundary-plane
        normal.
    :param tol: Strictly positive numerical tolerance used when projecting and
        normalizing the selected Cartesian seed.
    :return: A unit-length ``numpy.float64`` vector with shape ``(3,)`` lying in the
        plane perpendicular to ``normal``.
    :raises CrystallographyValueError: If ``tol`` is invalid or if the selected
        Cartesian seed cannot be projected to a nonzero in-plane direction within
        ``tol``.
    """
    basis = np.eye(3, dtype=np.float64)
    seed = basis[int(np.argmin(np.abs(basis @ normal)))]
    return _project_into_plane(seed, normal, name="in-plane reference", tol=tol)


def _orientation_from_normal_and_in_plane(
    boundary_normal: object,
    in_plane_direction: object,
    *,
    project_in_plane: bool,
    tol: float,
    name: str,
) -> np.ndarray:
    """Construct a right-handed row-wise orientation matrix.

    Row 0 is the normalized boundary normal. Row 1 is derived from the supplied in-plane
    direction. Row 2 is the normalized cross product of rows 0 and 1, completing a
    right-handed basis.

    When ``project_in_plane`` is true, the component of ``in_plane_direction`` parallel
    to the boundary normal is removed before row 1 is normalized. When it is false, the
    supplied direction must already be perpendicular to the boundary normal within
    ``tol``.

    :param boundary_normal: Array-like three-component boundary normal used to construct
        row 0. It need not be normalized, but it must be finite and nonzero.
    :param in_plane_direction: Array-like three-component direction used to construct
        row 1. It need not be normalized, but it must be finite and nonzero.
    :param project_in_plane: Whether to project ``in_plane_direction`` into the plane
        perpendicular to ``boundary_normal``. If false, the function instead requires
        the two normalized directions to be perpendicular within ``tol``.
    :param tol: Strictly positive numerical tolerance used for vector validation,
        normalization, perpendicularity testing, cross-product degeneracy testing, and
        final orientation-matrix validation.
    :param name: Human-readable name assigned to the constructed orientation matrix and
        included in final validation error messages.
    :return: A normalized, right-handed ``numpy.float64`` orientation matrix with shape
        ``(3, 3)`` whose rows represent the boundary normal, first in-plane direction,
        and completing in-plane direction.
    :raises CrystallographyValueError: If ``tol`` is invalid; if either input direction
        is malformed, non-finite, or zero within ``tol``; if ``project_in_plane`` is
        false and the directions are not perpendicular within ``tol``; if the directions
        do not define a nondegenerate basis; or if the constructed matrix fails
        orthogonality or handedness validation.
    """
    normal = normalize_direction(boundary_normal, "boundary normal", tol=tol)
    if project_in_plane:
        in_plane = _project_into_plane(
            in_plane_direction,
            normal,
            name="in-plane direction",
            tol=tol,
        )
    else:
        in_plane = normalize_direction(
            in_plane_direction,
            "in-plane direction",
            tol=tol,
        )
        dot = float(np.dot(normal, in_plane))
        if abs(dot) > tol:
            raise CrystallographyValueError(
                "in-plane direction must be perpendicular to the boundary normal; "
                f"normalized dot product is {dot:.3e}."
            )

    third = np.cross(normal, in_plane)
    third_norm = float(np.linalg.norm(third))
    if third_norm <= tol:
        raise CrystallographyValueError(
            "boundary normal and in-plane direction do not define an orientation."
        )

    return validate_orientation_matrix(
        np.vstack((normal, in_plane, third / third_norm)),
        name,
        tol=tol,
    )


def _axis_angle_rotation(
    rotation_axis: object,
    angle_deg: object,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate an axis-angle description and construct its rotation matrix.

    The rotation axis is converted to a finite three-component vector and normalized.
    The angle is validated as a finite, non-boolean real scalar in degrees. SciPy's
    rotation-vector convention is then used to construct the corresponding proper
    rotation matrix.

    :param rotation_axis: Array-like three-component rotation axis in crystal
        coordinates. It need not be normalized, but it must be finite and nonzero.
    :param angle_deg: Rotation angle in degrees. It must be a finite, non-boolean real
        scalar.
    :param tol: Strictly positive numerical tolerance used when validating and
        normalizing ``rotation_axis``.
    :return: A tuple ``(axis_hat, rotation, angle)`` where ``axis_hat`` is the
        normalized ``numpy.float64`` rotation axis with shape ``(3,)``, ``rotation`` is
        the corresponding proper ``numpy.float64`` rotation matrix with shape ``(3,
        3)``, and ``angle`` is the validated angle as a Python ``float`` in degrees.
    :raises CrystallographyValueError: If ``tol`` is invalid; if ``rotation_axis``
        cannot be interpreted as a finite, nonzero three-component vector; or if
        ``angle_deg`` is boolean, non-real, or non-finite.
    """
    axis = normalize_direction(rotation_axis, "rotation axis", tol=tol)
    angle = _finite_float(angle_deg, "angle_deg")
    rotation = Rotation.from_rotvec(axis * math.radians(angle)).as_matrix()
    return axis, rotation, angle


def build_tilt_orientations(
    left_boundary_normal: object,
    tilt_axis: object,
    angle_deg: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-orientation matrices for a general pure-tilt boundary.

    ``left_boundary_normal`` is the boundary normal expressed in the left crystal. The
    tilt axis must lie in the boundary plane. The right orientation is generated by the
    requested relative rotation. This constructor does not label the result symmetric;
    symmetry requires an independently specified median plane and is handled by
    :func:`build_symmetric_tilt_orientations`.

    The returned floating-point matrices are suitable for five-DOF conversion. They are
    not guaranteed to be exact integer ``PQSpec`` input.

    :param left_boundary_normal: Boundary normal in left-grain crystal coordinates.
    :param tilt_axis: Common tilt axis in crystal coordinates.
    :param angle_deg: Relative misorientation angle in degrees.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :return: ``(P, Q)`` normalized row-orientation matrices.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = normalize_direction(
        left_boundary_normal,
        "left boundary normal",
        tol=tol,
    )
    axis, rotation, _ = _axis_angle_rotation(tilt_axis, angle_deg, tol=tol)
    dot = float(np.dot(normal, axis))
    if abs(dot) > tol:
        raise CrystallographyValueError(
            "tilt axis must lie in the boundary plane; normalized normal-axis dot "
            f"product is {dot:.3e}."
        )

    P = _orientation_from_normal_and_in_plane(
        normal,
        axis,
        project_in_plane=False,
        tol=tol,
        name="P",
    )
    Q = validate_orientation_matrix(P @ rotation, "Q", tol=tol)
    return P, Q


def build_symmetric_tilt_orientations(
    median_boundary_normal: object,
    tilt_axis: object,
    angle_deg: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a geometrically symmetric pure-tilt orientation pair.

    ``median_boundary_normal`` is the bisector normal about which the left and right
    crystal boundary normals are placed at equal and opposite half angles. For example,
    a median normal ``[1, 0, 0]``, tilt axis ``[0, 0, 1]``, and angle ``2*atan(1/5)``
    produce normals parallel to ``[5, 1, 0]`` and ``[5, -1, 0]``.

    The returned floating-point matrices are not exact integer ``PQSpec`` input unless a
    separate exactification step succeeds.

    :param median_boundary_normal: Symmetry-bisector boundary normal.
    :param tilt_axis: Common tilt axis, perpendicular to the median normal.
    :param angle_deg: Total left-to-right misorientation angle in degrees.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :return: ``(P, Q)`` normalized row-orientation matrices.
    """
    tol = _validated_tolerance(tol, "tol")
    median = normalize_direction(
        median_boundary_normal,
        "median boundary normal",
        tol=tol,
    )
    axis, full_rotation, angle = _axis_angle_rotation(
        tilt_axis,
        angle_deg,
        tol=tol,
    )
    dot = float(np.dot(median, axis))
    if abs(dot) > tol:
        raise CrystallographyValueError(
            "tilt axis must be perpendicular to the median boundary normal; "
            f"normalized dot product is {dot:.3e}."
        )

    left_half_rotation = Rotation.from_rotvec(
        axis * math.radians(-0.5 * angle)
    ).as_matrix()
    right_half_rotation = Rotation.from_rotvec(
        axis * math.radians(0.5 * angle)
    ).as_matrix()
    left_normal = median @ left_half_rotation
    right_normal = median @ right_half_rotation

    P = _orientation_from_normal_and_in_plane(
        left_normal,
        axis,
        project_in_plane=False,
        tol=tol,
        name="P",
    )
    Q = _orientation_from_normal_and_in_plane(
        right_normal,
        axis,
        project_in_plane=False,
        tol=tol,
        name="Q",
    )

    if not np.allclose(P @ full_rotation, Q, atol=10.0 * tol, rtol=0.0):
        raise CrystallographyValueError(
            "symmetric-tilt construction failed its relative-rotation invariant."
        )

    left_deviation = P[0] - np.dot(P[0], median) * median
    right_deviation = Q[0] - np.dot(Q[0], median) * median
    if not np.allclose(
        left_deviation,
        -right_deviation,
        atol=10.0 * tol,
        rtol=0.0,
    ):
        raise CrystallographyValueError(
            "symmetric-tilt construction failed its mirror-normal invariant."
        )

    return P, Q


def build_twist_orientations(
    boundary_normal: object,
    angle_deg: object,
    *,
    in_plane_reference: object | None = None,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-orientation matrices for a pure-twist boundary.

    The rotation axis is the boundary normal. ``in_plane_reference`` controls the
    otherwise arbitrary lab-y direction of the left grain. When omitted, a stable
    Cartesian seed is selected automatically.

    :param boundary_normal: Common boundary normal and twist axis.
    :param angle_deg: Relative twist angle in degrees.
    :param in_plane_reference: Optional direction used to define the left in-plane
        basis. It is projected into the boundary plane.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :return: ``(P, Q)`` normalized row-orientation matrices.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = normalize_direction(boundary_normal, "boundary normal", tol=tol)
    _, rotation, _ = _axis_angle_rotation(normal, angle_deg, tol=tol)

    if in_plane_reference is None:
        in_plane = _default_in_plane_reference(normal, tol=tol)
    else:
        in_plane = _project_into_plane(
            in_plane_reference,
            normal,
            name="in-plane reference",
            tol=tol,
        )

    P = _orientation_from_normal_and_in_plane(
        normal,
        in_plane,
        project_in_plane=False,
        tol=tol,
        name="P",
    )
    Q = validate_orientation_matrix(P @ rotation, "Q", tol=tol)

    if not np.allclose(P[0], Q[0], atol=10.0 * tol, rtol=0.0):
        raise CrystallographyValueError(
            "twist construction failed to preserve the boundary normal."
        )
    return P, Q


def build_mixed_orientations(
    left_boundary_normal: object,
    rotation_axis: object,
    angle_deg: object,
    *,
    in_plane_reference: object | None = None,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-orientation matrices for a mixed tilt/twist boundary.

    A mixed boundary requires the rotation axis to have both a component normal to the
    boundary and a component in its plane. By default, the in-plane projection of the
    rotation axis defines the left lab-y direction. A different in-plane reference may
    be supplied explicitly.

    :param left_boundary_normal: Boundary normal in left-grain crystal coordinates.
    :param rotation_axis: Mixed-character misorientation axis.
    :param angle_deg: Relative misorientation angle in degrees.
    :param in_plane_reference: Optional left-grain in-plane basis seed.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :return: ``(P, Q)`` normalized row-orientation matrices.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = normalize_direction(
        left_boundary_normal,
        "left boundary normal",
        tol=tol,
    )
    axis, rotation, _ = _axis_angle_rotation(rotation_axis, angle_deg, tol=tol)
    normal_component = abs(float(np.dot(normal, axis)))
    if normal_component <= tol:
        raise CrystallographyValueError(
            "mixed boundary requires a nonzero twist component; the rotation axis "
            "lies in the boundary plane."
        )
    if 1.0 - normal_component <= tol:
        raise CrystallographyValueError(
            "mixed boundary requires a nonzero tilt component; the rotation axis is "
            "parallel to the boundary normal."
        )

    reference = axis if in_plane_reference is None else in_plane_reference
    in_plane = _project_into_plane(
        reference,
        normal,
        name="in-plane reference",
        tol=tol,
    )
    P = _orientation_from_normal_and_in_plane(
        normal,
        in_plane,
        project_in_plane=False,
        tol=tol,
        name="P",
    )
    Q = validate_orientation_matrix(P @ rotation, "Q", tol=tol)
    return P, Q


def inclination_from_normal(
    boundary_normal: object,
    *,
    tol: float = _DEFAULT_VECTOR_TOL,
) -> tuple[float, float]:
    """Return GBMaker inclination angles ``(theta, phi)`` for a boundary normal.

    GBMaker uses ``Rincl = Rz(phi) @ Ry(theta)`` and row-vector coordinates. Under that
    convention the first row of ``Rincl`` is the crystal direction aligned with lab x.

    :param boundary_normal: Boundary normal in crystal coordinates.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :return: ``(theta, phi)`` in radians.
    """
    normal = normalize_direction(boundary_normal, "boundary normal", tol=tol)
    nx, ny, nz = normal
    phi = float(math.asin(float(np.clip(-ny, -1.0, 1.0))))
    theta = 0.0 if abs(abs(ny) - 1.0) <= tol else float(math.atan2(nz, nx))
    return theta, phi


def _zxz_euler_angles(rotation: np.ndarray) -> np.ndarray:
    """Extract intrinsic ZXZ Euler angles from a proper rotation matrix.

    SciPy emits a gimbal-lock warning when the middle ZXZ angle is singular. In that
    case, SciPy still returns a valid canonical Euler-angle representative, so this
    helper suppresses only that warning and returns the resulting angles unchanged.

    This private helper expects ``rotation`` to have already been constructed or
    validated as a finite proper rotation matrix.

    :param rotation: Finite proper rotation matrix with shape ``(3, 3)`` to convert
        using SciPy's intrinsic ``ZXZ`` Euler convention.
    :return: A ``numpy.ndarray`` with shape ``(3,)`` containing ``[alpha, beta, gamma]``
        in radians. At a gimbal-lock singularity, the returned values are SciPy's
        canonical representative and are not a unique Euler decomposition.
    :raises ValueError: If ``rotation`` does not have an acceptable matrix shape or
        cannot be interpreted by SciPy as a valid rotation.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Gimbal lock detected.*",
            category=UserWarning,
        )
        return Rotation.from_matrix(rotation).as_euler("ZXZ")


def five_dof_from_axis_angle(
    rotation_axis: object,
    angle_deg: object,
    boundary_normal: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> np.ndarray:
    """Convert an axis-angle boundary description to five-DOF parameters.

    The axis-angle pair defines the crystal-frame misorientation. Its proper rotation
    matrix is decomposed using the intrinsic ZXZ Euler convention to obtain ``alpha``,
    ``beta``, and ``gamma``. The left-grain boundary normal independently determines the
    GBMaker inclination angles ``theta`` and ``phi``.

    The returned array therefore has the form ``[alpha, beta, gamma, theta, phi]``, with
    all angles expressed in radians.

    :param rotation_axis: Array-like three-component misorientation axis in crystal
        coordinates. It need not be normalized, but it must be finite and nonzero.
    :param angle_deg: Misorientation angle in degrees. It must be a finite, non-boolean
        real scalar.
    :param boundary_normal: Array-like three-component left-grain boundary normal in
        crystal coordinates. It need not be normalized, but it must be finite and
        nonzero.
    :param tol: Strictly positive numerical tolerance used when validating and
        normalizing the rotation axis and boundary normal. Keyword-only; defaults to
        ``_DEFAULT_ORIENTATION_TOL``.
    :return: A one-dimensional ``numpy.float64`` array with shape ``(5,)`` containing
        ``[alpha, beta, gamma, theta, phi]`` in radians.
    :raises CrystallographyValueError: If ``tol`` is invalid; if ``rotation_axis`` or
        ``boundary_normal`` cannot be interpreted as a finite, nonzero three-component
        vector; or if ``angle_deg`` is boolean, non-real, or non-finite.
    :raises ValueError: If SciPy cannot convert the constructed rotation matrix to an
        intrinsic ZXZ Euler-angle representation.
    """
    tol = _validated_tolerance(tol, "tol")
    _, rotation, _ = _axis_angle_rotation(rotation_axis, angle_deg, tol=tol)
    alpha, beta, gamma = _zxz_euler_angles(rotation)
    theta, phi = inclination_from_normal(boundary_normal, tol=tol)
    return np.array([alpha, beta, gamma, theta, phi], dtype=np.float64)


def five_dof_from_orientation_matrices(
    P: object,
    Q: object,
    boundary_normal: object | None = None,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
    normal_warning_deg: float = _DEFAULT_NORMAL_WARNING_DEG,
) -> np.ndarray:
    """Convert row-wise orientation matrices to GBMaker five-DOF parameters.

    ``P`` and ``Q`` may contain scaled crystallographic directions; each row is
    normalized during validation. ``boundary_normal`` is an optional consistency check
    only. The inclination is always derived from normalized ``P[0]``.

    :param P: Left-grain row-wise orientation matrix.
    :param Q: Right-grain row-wise orientation matrix.
    :param boundary_normal: Optional independently supplied boundary normal.
    :param tol: Numerical tolerance. Keyword parameter, optional.
    :param normal_warning_deg: Angular discrepancy above which a ``UserWarning`` is
        emitted for ``boundary_normal``. Keyword parameter, optional, defaults to one
        degree.
    :return: Five angles ``[alpha, beta, gamma, theta, phi]`` in radians.
    """
    tol = _validated_tolerance(tol, "tol")
    warning_deg = _finite_float(normal_warning_deg, "normal_warning_deg")
    if warning_deg < 0.0:
        raise CrystallographyValueError(
            "normal_warning_deg must be non-negative."
        )

    P_norm = validate_orientation_matrix(P, "P", tol=tol)
    Q_norm = validate_orientation_matrix(Q, "Q", tol=tol)
    rotation = P_norm.T @ Q_norm
    alpha, beta, gamma = _zxz_euler_angles(rotation)

    normal_from_P = P_norm[0]
    if boundary_normal is not None:
        supplied = normalize_direction(
            boundary_normal,
            "boundary normal",
            tol=tol,
        )
        cosine = float(np.clip(np.dot(normal_from_P, supplied), -1.0, 1.0))
        discrepancy_deg = math.degrees(math.acos(cosine))
        if discrepancy_deg > warning_deg:
            warnings.warn(
                "Supplied boundary normal differs from P[0] by "
                f"{discrepancy_deg:.2f} degrees; using P[0] for inclination.",
                UserWarning,
                stacklevel=2,
            )

    theta, phi = inclination_from_normal(normal_from_P, tol=tol)
    return np.array([alpha, beta, gamma, theta, phi], dtype=np.float64)


def orientation_matrices_from_five_dof(
    params: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GBMaker five-DOF parameters to row-orientation matrices.

    The five parameters use GBMaker's legacy convention
    ``[alpha, beta, gamma, theta, phi]``. The first three values define the
    crystal-frame misorientation using intrinsic ZXZ Euler angles. The final two
    values define the left-grain inclination matrix as
    ``Rz(phi) @ Ry(theta)``.

    The right-grain orientation is constructed as
    ``R_right = R_left @ R_misorientation``.

    :param params: Array-like sequence containing
        ``[alpha, beta, gamma, theta, phi]`` in radians.
    :param tol: Strictly positive numerical tolerance used when validating the
        resulting row-orientation matrices. Keyword-only, optional, defaults to
        ``_DEFAULT_ORIENTATION_TOL``.
    :return: A tuple ``(R_left, R_right)`` containing normalized, proper,
        row-wise orientation matrices with shape ``(3, 3)``.
    :raises CrystallographyValueError: If ``params`` cannot be converted to a
        finite floating-point sequence of length five, if ``tol`` is invalid,
        or if either resulting matrix fails row-orientation validation.
    """
    tol = _validated_tolerance(tol, "tol")

    try:
        values = np.asarray(params, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CrystallographyValueError(
            "params must be a finite five-element sequence."
        ) from exc

    if values.shape != (5,):
        raise CrystallographyValueError(
            f"params must have shape (5,); got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise CrystallographyValueError(
            "params must contain only finite values."
        )

    alpha, beta, gamma, theta, phi = values

    misorientation = Rotation.from_euler(
        "ZXZ",
        [alpha, beta, gamma],
    ).as_matrix()
    left = (
        Rotation.from_euler("z", phi)
        * Rotation.from_euler("y", theta)
    ).as_matrix()
    right = left @ misorientation

    return (
        validate_orientation_matrix(left, "R_left", tol=tol),
        validate_orientation_matrix(right, "R_right", tol=tol),
    )


__all__ = [
    "build_mixed_orientations",
    "build_symmetric_tilt_orientations",
    "build_tilt_orientations",
    "build_twist_orientations",
    "five_dof_from_axis_angle",
    "five_dof_from_orientation_matrices",
    "inclination_from_normal",
    "normalize_direction",
    "validate_orientation_matrix",
]
