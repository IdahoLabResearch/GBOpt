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
# Validation and normalization
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
    return _normalize_direction(direction, name, tol=tol)


def _normalize_direction(
    direction: object,
    name: str,
    *,
    tol: float,
) -> np.ndarray:
    """Normalize a direction using a prevalidated tolerance.

    The direction is converted to a finite ``float64`` vector, checked against the
    minimum permitted norm, and normalized. Unlike ``normalize_direction``, this helper
    does not validate ``tol``.

    :param direction: Array-like object containing exactly three finite numeric
        components.
    :param name: Human-readable argument name used in validation error messages.
    :param tol: Prevalidated finite, strictly positive lower bound for the vector norm.
        Keyword argument.
    :return: Unit-length ``float64`` vector with shape ``(3,)``.
    :raises CrystallographyValueError: If ``direction`` is malformed, contains a
        non-finite value, or has norm less than or equal to ``tol``.
    """
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
    They must be finite, nonzero, mutually orthogonal, and right-handed. Row lengths
    need not be one on input; the returned matrix is row-normalized.

    :param matrix: Candidate row-wise orientation matrix.
    :param name: Matrix name used in error messages. Optional, defaults to
        ``"orientation matrix"``.
    :param tol: Absolute tolerance used for row norms, orthogonality, and determinant
        validation. Keyword argument, optional, defaults to
        ``_DEFAULT_ORIENTATION_TOL``.
    :return: Normalized ``float64`` matrix with shape ``(3, 3)``.
    :raises CrystallographyValueError: If ``tol`` is invalid or ``matrix`` is malformed,
        non-finite, degenerate, nonorthogonal, or not right-handed.
    """
    tol = _validated_tolerance(tol, "tol")
    return _validate_orientation_matrix(matrix, name, tol=tol)


def _validate_orientation_matrix(
    matrix: object,
    name: str,
    *,
    tol: float,
) -> np.ndarray:
    """Normalize and validate a row-wise orientation matrix.

    The candidate is converted to ``float64``, checked for shape and finite values, and
    normalized row by row. The normalized rows must form a mutually orthogonal,
    right-handed frame. This helper assumes that ``tol`` has already been validated.

    :param matrix: Candidate orientation matrix with shape ``(3, 3)``. Its rows may have
        arbitrary nonzero magnitudes.
    :param name: Human-readable matrix name used in validation error messages.
    :param tol: Prevalidated absolute tolerance used for row norms, Gram-matrix error,
        and determinant error. Keyword argument.
    :return: Row-normalized ``float64`` matrix with shape ``(3, 3)``.
    :raises CrystallographyValueError: If ``matrix`` cannot be converted, has the wrong
        shape, contains non-finite or zero rows, is not orthogonal within ``tol``, or is
        not right-handed within ``tol``.
    """
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
    zero_rows = norms <= tol
    if np.any(zero_rows):
        row = int(np.flatnonzero(zero_rows)[0])
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

    determinant = float(np.dot(normalized[0], np.cross(normalized[1], normalized[2])))
    if abs(determinant - 1.0) > tol:
        raise CrystallographyValueError(
            f"{name} must be right-handed after normalization; "
            f"determinant is {determinant:.12g}."
        )

    return normalized


# ---------------------------------------------------------------------------
# Orientation-frame construction
# ---------------------------------------------------------------------------


def _project_into_plane(
    direction: object,
    normal: np.ndarray,
    *,
    name: str,
    tol: float,
) -> np.ndarray:
    """Project a direction into the plane perpendicular to a unit normal.

    The direction is validated, projected directly, and normalized. This helper assumes
    that ``normal`` is a finite unit vector with shape ``(3,)`` and that ``tol`` has
    already been validated.

    :param direction: Three-component direction to project. It need not be normalized,
        but it must be finite and nonzero.
    :param normal: Unit vector with shape ``(3,)`` defining the target plane.
    :param name: Human-readable name for ``direction`` used in validation errors.
        Keyword argument.
    :param tol: Positive tolerance used to detect zero input and a vanishing projected
        component. Keyword argument.
    :return: Unit-length ``float64`` vector with shape ``(3,)`` lying in the plane
        perpendicular to ``normal``.
    :raises CrystallographyValueError: If ``direction`` is malformed, non-finite,
        effectively zero, or parallel to ``normal`` within ``tol``.
    """
    vector = _vector3(direction, name)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= tol:
        raise CrystallographyValueError(
            f"{name} must be nonzero; got norm {vector_norm:.3e}."
        )

    projected = vector - np.dot(vector, normal) * normal
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= tol * vector_norm:
        raise CrystallographyValueError(
            f"{name} must not be parallel to the boundary normal."
        )
    return projected / projected_norm


def _default_in_plane_reference(
    normal: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    """Choose a stable default direction perpendicular to a unit normal.

    The Cartesian basis vector least aligned with ``normal`` is selected as a seed,
    projected into the perpendicular plane, and normalized. This helper assumes that
    ``normal`` is a finite unit vector and that ``tol`` has already been validated.

    :param normal: Unit boundary-plane normal with shape ``(3,)``.
    :param tol: Positive tolerance used while projecting and normalizing the selected
        seed. Keyword argument.
    :return: Unit-length ``float64`` vector with shape ``(3,)`` perpendicular to
        ``normal``.
    :raises CrystallographyValueError: If the selected seed cannot be projected to a
        nonzero direction within ``tol``.
    """
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(normal)))] = 1.0
    return _project_into_plane(seed, normal, name="in-plane reference", tol=tol)


def _orientation_from_unit_vectors(
    normal: np.ndarray,
    in_plane: np.ndarray,
) -> np.ndarray:
    """Construct a row-wise orientation matrix from two trusted unit vectors.

    The boundary normal becomes the first row and the supplied in-plane direction
    becomes the second row. Their normalized cross product becomes the third row,
    producing a right-handed orientation frame.

    This helper performs no input validation. Callers must provide finite, unit-length
    vectors with shape ``(3,)`` that are mutually perpendicular. Supplying zero,
    parallel, non-unit, or malformed vectors may produce a non-orthonormal matrix or
    non-finite values.

    :param normal: Finite unit vector with shape ``(3,)`` defining the boundary normal
        and first orientation row.
    :param in_plane: Finite unit vector with shape ``(3,)`` perpendicular to ``normal``
        and defining the second orientation row.
    :return: ``numpy.float64`` matrix with shape ``(3, 3)`` whose rows are ``normal``,
        ``in_plane``, and the normalized cross product ``normal x in_plane``.
    """
    third = np.cross(normal, in_plane)
    third /= np.linalg.norm(third)
    return np.stack((normal, in_plane, third))


def _axis_angle_rotation(
    rotation_axis: object,
    angle_deg: object,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate an axis-angle description and construct its rotation matrix.

    The rotation axis is normalized, the angle is validated as a finite non-boolean real
    value in degrees, and SciPy's rotation-vector convention is used to construct the
    corresponding proper rotation matrix. This helper assumes that ``tol`` has already
    been validated as finite and strictly positive.

    :param rotation_axis: Three-component rotation axis in crystal coordinates. It need
        not be normalized, but it must be finite and nonzero.
    :param angle_deg: Rotation angle in degrees.
    :param tol: Positive tolerance used when normalizing ``rotation_axis``. Keyword
        argument.
    :return: Tuple ``(axis_hat, rotation, angle)`` containing the normalized ``float64``
        axis, the corresponding 3 by 3 rotation matrix, and the validated angle as a
        Python ``float``.
    :raises CrystallographyValueError: If ``rotation_axis`` is malformed, non-finite, or
        zero, or if ``angle_deg`` is Boolean, non-real, or non-finite.
    """
    axis = _normalize_direction(rotation_axis, "rotation axis", tol=tol)
    angle = _finite_float(angle_deg, "angle_deg")
    rotation = Rotation.from_rotvec(axis * math.radians(angle)).as_matrix()
    return axis, rotation, angle


# ---------------------------------------------------------------------------
# Boundary orientation builders
# ---------------------------------------------------------------------------


def build_tilt_orientations(
    left_boundary_normal: object,
    tilt_axis: object,
    angle_deg: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-orientation matrices for a general pure-tilt boundary.

    ``left_boundary_normal`` is expressed in the left crystal, and ``tilt_axis`` must
    lie in its boundary plane. The right orientation is generated by the requested
    relative rotation. Symmetric placement around a median plane is handled separately
    by ``build_symmetric_tilt_orientations``.

    The returned floating-point matrices are suitable for five-DOF conversion but are
    not guaranteed to be exact integer ``PQSpec`` input.

    :param left_boundary_normal: Boundary normal in left-grain crystal coordinates.
    :param tilt_axis: Common tilt axis in crystal coordinates.
    :param angle_deg: Relative misorientation angle in degrees.
    :param tol: Numerical tolerance used for vector and geometric validation. Keyword
        argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :return: Tuple ``(P, Q)`` of normalized row-orientation matrices.
    :raises CrystallographyValueError: If an input is invalid or the normalized tilt
        axis does not lie in the boundary plane within ``tol``.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = _normalize_direction(
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

    P = _orientation_from_unit_vectors(normal, axis)
    Q = P @ rotation
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
    boundary normals are placed at equal and opposite half angles. For example, median
    normal ``[1, 0, 0]``, tilt axis ``[0, 0, 1]``, and angle ``2*atan(1/5)`` produce
    normals parallel to ``[5, 1, 0]`` and ``[5, -1, 0]``.

    The returned floating-point matrices are not exact integer ``PQSpec`` input unless a
    separate exactification step succeeds.

    :param median_boundary_normal: Symmetry-bisector boundary normal.
    :param tilt_axis: Common tilt axis perpendicular to the median normal.
    :param angle_deg: Total left-to-right misorientation angle in degrees.
    :param tol: Numerical tolerance used for vector and invariant validation. Keyword
        argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :return: Tuple ``(P, Q)`` of normalized row-orientation matrices.
    :raises CrystallographyValueError: If an input is invalid, the tilt axis is not
        perpendicular to the median normal, or a symmetric-construction invariant fails.
    """
    tol = _validated_tolerance(tol, "tol")
    median = _normalize_direction(
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

    right_half_rotation = Rotation.from_rotvec(
        axis * math.radians(0.5 * angle)
    ).as_matrix()
    left_normal = median @ right_half_rotation.T
    right_normal = median @ right_half_rotation

    P = _orientation_from_unit_vectors(left_normal, axis)
    Q = P @ full_rotation

    if not np.allclose(Q[0], right_normal, atol=10.0 * tol, rtol=0.0):
        raise CrystallographyValueError(
            "symmetric-tilt construction failed its right-normal invariant."
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
    :param in_plane_reference: Direction used to define the left-grain in-plane basis.
        The direction is projected into the boundary plane. Keyword argument, optional,
        defaults to ``None``.
    :param tol: Numerical tolerance used for vector and invariant validation. Keyword
        argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :return: Tuple ``(P, Q)`` of normalized row-orientation matrices.
    :raises CrystallographyValueError: If an input is invalid, the supplied in-plane
        reference has no nonzero projection, or the constructed rotation fails to
        preserve the boundary normal.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = _normalize_direction(boundary_normal, "boundary normal", tol=tol)
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

    P = _orientation_from_unit_vectors(normal, in_plane)
    Q = P @ rotation

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
    rotation axis defines the left-grain lab-y direction. A different reference may be
    supplied explicitly.

    :param left_boundary_normal: Boundary normal in left-grain crystal coordinates.
    :param rotation_axis: Mixed-character misorientation axis.
    :param angle_deg: Relative misorientation angle in degrees.
    :param in_plane_reference: Optional left-grain in-plane basis seed. Keyword
        argument, optional, defaults to ``None``.
    :param tol: Numerical tolerance used for vector and character validation. Keyword
        argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :return: Tuple ``(P, Q)`` of normalized row-orientation matrices.
    :raises CrystallographyValueError: If an input is invalid, the rotation axis lacks
        either a tilt or twist component, or the selected reference has no nonzero
        in-plane projection.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = _normalize_direction(
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
    P = _orientation_from_unit_vectors(normal, in_plane)
    Q = P @ rotation
    return P, Q


# ---------------------------------------------------------------------------
# Five-degree-of-freedom conversions
# ---------------------------------------------------------------------------


def inclination_from_normal(
    boundary_normal: object,
    *,
    tol: float = _DEFAULT_VECTOR_TOL,
) -> tuple[float, float]:
    """Return GBMaker inclination angles for a boundary normal.

    GBMaker uses ``Rincl = Rz(phi) @ Ry(theta)`` with row-vector coordinates. Under that
    convention, the first row of ``Rincl`` is the crystal direction aligned with the
    lab-frame x-axis.

    :param boundary_normal: Boundary normal in crystal coordinates.
    :param tol: Numerical tolerance used to validate and normalize the normal. Keyword
        argument, optional, defaults to ``_DEFAULT_VECTOR_TOL``.
    :return: Tuple ``(theta, phi)`` in radians.
    :raises CrystallographyValueError: If ``tol`` is invalid or ``boundary_normal`` is
        malformed, non-finite, or effectively zero.
    """
    tol = _validated_tolerance(tol, "tol")
    normal = _normalize_direction(boundary_normal, "boundary normal", tol=tol)
    return _inclination_from_unit_normal(normal, tol=tol)


def _inclination_from_unit_normal(
    normal: np.ndarray,
    *,
    tol: float,
) -> tuple[float, float]:
    """Return inclination angles for an already normalized boundary normal.

    At the positive- or negative-y singularity, ``theta`` is set to zero as a canonical
    representative.

    :param normal: Unit boundary normal with shape ``(3,)``.
    :param tol: Tolerance used to identify the y-axis singularity. Keyword argument.
    :return: Tuple ``(theta, phi)`` in radians.
    """
    nx, ny, nz = normal
    phi = math.asin(np.clip(-ny, -1.0, 1.0))
    theta = 0.0 if abs(abs(ny) - 1.0) <= tol else math.atan2(nz, nx)
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
    matrix is decomposed using intrinsic ZXZ Euler angles to obtain ``alpha``, ``beta``,
    and ``gamma``. The left-grain boundary normal independently determines the
    inclination angles ``theta`` and ``phi``.

    :param rotation_axis: Three-component misorientation axis in crystal coordinates. It
        need not be normalized, but it must be finite and nonzero.
    :param angle_deg: Misorientation angle in degrees.
    :param boundary_normal: Three-component left-grain boundary normal in crystal
        coordinates. It need not be normalized, but it must be finite and nonzero.
    :param tol: Positive numerical tolerance used to validate and normalize the rotation
        axis and boundary normal. Keyword argument, optional, defaults to
        ``_DEFAULT_ORIENTATION_TOL``.
    :return: ``float64`` array with shape ``(5,)`` containing ``[alpha, beta, gamma,
        theta, phi]`` in radians.
    :raises CrystallographyValueError: If ``tol`` is invalid, either direction is
        malformed, non-finite, or zero, or ``angle_deg`` is Boolean, non-real, or
        non-finite.
    :raises ValueError: If SciPy cannot convert the constructed rotation matrix to an
        intrinsic ZXZ Euler-angle representation.
    """
    tol = _validated_tolerance(tol, "tol")
    _, rotation, _ = _axis_angle_rotation(rotation_axis, angle_deg, tol=tol)
    alpha, beta, gamma = _zxz_euler_angles(rotation)
    normal = _normalize_direction(boundary_normal, "boundary normal", tol=tol)
    theta, phi = _inclination_from_unit_normal(normal, tol=tol)
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
    only. Inclination is always derived from normalized ``P[0]``.

    :param P: Left-grain row-wise orientation matrix.
    :param Q: Right-grain row-wise orientation matrix.
    :param boundary_normal: Independently supplied boundary normal used for a
        consistency warning, or ``None``. Optional, defaults to ``None``.
    :param tol: Numerical tolerance used for matrix and vector validation. Keyword
        argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :param normal_warning_deg: Angular discrepancy above which a ``UserWarning`` is
        emitted for ``boundary_normal``. Keyword argument, optional, defaults to
        ``_DEFAULT_NORMAL_WARNING_DEG``.
    :return: ``float64`` array containing ``[alpha, beta, gamma, theta, phi]`` in
        radians.
    :raises CrystallographyValueError: If a tolerance is invalid, either matrix is not a
        proper row-wise orientation matrix, the supplied boundary normal is invalid, or
        ``normal_warning_deg`` is negative.
    """
    tol = _validated_tolerance(tol, "tol")
    warning_deg = _finite_float(normal_warning_deg, "normal_warning_deg")
    if warning_deg < 0.0:
        raise CrystallographyValueError(
            "normal_warning_deg must be non-negative."
        )

    P_norm = _validate_orientation_matrix(P, "P", tol=tol)
    Q_norm = _validate_orientation_matrix(Q, "Q", tol=tol)
    rotation = P_norm.T @ Q_norm
    alpha, beta, gamma = _zxz_euler_angles(rotation)

    normal_from_P = P_norm[0]
    if boundary_normal is not None:
        supplied = _normalize_direction(
            boundary_normal,
            "boundary normal",
            tol=tol,
        )
        cosine = float(np.clip(np.dot(normal_from_P, supplied), -1.0, 1.0))
        warning_cosine = (
            math.cos(math.radians(warning_deg)) if warning_deg < 180.0 else -1.0
        )
        if cosine < warning_cosine:
            discrepancy_deg = math.degrees(math.acos(cosine))
            warnings.warn(
                "Supplied boundary normal differs from P[0] by "
                f"{discrepancy_deg:.2f} degrees; using P[0] for inclination.",
                UserWarning,
                stacklevel=2,
            )

    theta, phi = _inclination_from_unit_normal(normal_from_P, tol=tol)
    return np.array([alpha, beta, gamma, theta, phi], dtype=np.float64)


def orientation_matrices_from_five_dof(
    params: object,
    *,
    tol: float = _DEFAULT_ORIENTATION_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GBMaker five-DOF parameters to row-orientation matrices.

    The five values follow the legacy convention ``[alpha, beta, gamma, theta, phi]``.
    The first three define the crystal-frame misorientation using intrinsic ZXZ Euler
    angles. The final two define the left-grain inclination matrix as ``Rz(phi) @
    Ry(theta)``. The right-grain orientation is constructed as ``R_right = R_left @
    R_misorientation``.

    :param params: Sequence containing ``[alpha, beta, gamma, theta, phi]`` in radians.
    :param tol: Positive numerical tolerance used to validate the resulting orientation
        matrices. Keyword argument, optional, defaults to ``_DEFAULT_ORIENTATION_TOL``.
    :return: Tuple ``(R_left, R_right)`` containing normalized proper row-wise
        orientation matrices with shape ``(3, 3)``.
    :raises CrystallographyValueError: If ``params`` cannot be converted to a finite
        sequence of length five, ``tol`` is invalid, or either resulting matrix fails
        row-orientation validation.
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
    left = Rotation.from_euler("ZY", [phi, theta]).as_matrix()
    right = left @ misorientation

    return (
        _validate_orientation_matrix(left, "R_left", tol=tol),
        _validate_orientation_matrix(right, "R_right", tol=tol),
    )


__all__ = [
    "normalize_direction",
    "validate_orientation_matrix",
    "build_tilt_orientations",
    "build_symmetric_tilt_orientations",
    "build_twist_orientations",
    "build_mixed_orientations",
    "inclination_from_normal",
    "five_dof_from_axis_angle",
    "five_dof_from_orientation_matrices",
    "orientation_matrices_from_five_dof",
]
