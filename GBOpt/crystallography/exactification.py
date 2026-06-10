# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Rationalization of approximate boundaries into exact crystallographic forms."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from ._guards import _require_cubic
from .csl import csl_from_scaled_rotation
from .embedding import (
    _paired_pq_from_direction_rows,
    orthogonal_embedding_from_row_rotation_and_plane,
)
from .integer import as_positive_int, row_gcd_reduce
from .orientation import orientation_matrices_from_five_dof
from .quaternion import integer_quaternion_from_unit, quaternion_to_scaled_rotation
from .rotation import transpose_rotation_convention
from .types import CrystallographyValueError

_DEFAULT_MAX_SIGMA = 10_000
_DEFAULT_MAX_DENOMINATOR = 10_001
_DEFAULT_ANGLE_TOL = 1.0e-9
_DEFAULT_PLANE_TOL = 1.0e-9


def _positive_tolerance(value: object, name: str) -> float:
    """Validate and return a finite, strictly positive tolerance.

    Python and NumPy real scalars are accepted and converted to a Python ``float``.
    Boolean scalars, non-real values, NaN, infinity, and values less than or equal to
    zero are rejected. No upper bound is imposed: these tolerances are absolute error
    thresholds rather than fractions or probabilities, so values greater than or equal
    to one remain well-defined even though they may make an exactification check very
    permissive.

    :param value: Candidate scalar tolerance to validate.
    :param name: Parameter name used to identify the invalid value in error messages.
    :return: ``value`` converted to a finite, strictly positive Python ``float``.
    :raises CrystallographyValueError: If ``value`` is Boolean, is not a real scalar,
        cannot be represented as a finite ``float``, or is less than or equal to zero.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise CrystallographyValueError(
            f"{name} must be a finite positive real number; got {value!r}."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CrystallographyValueError(
            f"{name} must be a finite positive real number; got {value!r}."
        )
    return result


def _rationalize_direction(
    direction: object,
    *,
    max_denominator: int,
    tol: float,
    name: str,
) -> np.ndarray:
    """Recover a primitive integer direction from a floating-point vector.

    The input is interpreted as a direction, so its magnitude is discarded. Component
    ratios are formed relative to the largest-magnitude normalized component, which
    avoids dividing by a small coordinate, and each ratio is approximated by a
    :class:`fractions.Fraction` whose denominator does not exceed ``max_denominator``.
    The ratios are lifted to a common integer vector, reduced by the GCD of its
    components, and oriented to point into the same hemisphere as the input vector.

    The recovered primitive direction is accepted only when its normalized components
    differ from the normalized input by no more than ``tol`` in the infinity norm.
    ``max_denominator`` and ``tol`` are expected to have been validated by the public
    caller before this private helper is invoked.

    :param direction: Array-like floating-point direction with shape ``(3,)``. The
        vector must contain only finite values and must be nonzero.
    :param max_denominator: Largest denominator permitted for each rationalized
        component ratio. Keyword argument, required.
    :param tol: Maximum permitted absolute component error between the normalized input
        direction and normalized recovered integer direction. Keyword argument,
        required.
    :param name: Human-readable parameter name used in validation and exactification
        error messages. Keyword argument, required.
    :return: Primitive object-dtype integer vector with shape ``(3,)`` and a sign chosen
        so that its dot product with the normalized input direction is nonnegative.
    :raises CrystallographyValueError: If ``direction`` cannot be converted to a finite
        three-component vector, is the zero vector, or cannot be rationalized within
        ``tol`` using denominators bounded by ``max_denominator``.
    """
    try:
        vector = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CrystallographyValueError(
            f"{name} must be a finite three-component direction."
        ) from exc

    if vector.shape != (3,):
        raise CrystallographyValueError(
            f"{name} must have shape (3,); got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise CrystallographyValueError(f"{name} must contain only finite values.")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise CrystallographyValueError(f"{name} must be nonzero.")
    target = vector / norm

    reference_index = int(np.argmax(np.abs(target)))
    reference = float(target[reference_index])
    fractions = [
        Fraction(float(component) / reference).limit_denominator(max_denominator)
        for component in target
    ]
    denominator_lcm = math.lcm(*(fraction.denominator for fraction in fractions))
    integer_direction = np.array(
        [int(fraction * denominator_lcm) for fraction in fractions],
        dtype=object,
    )
    integer_direction = row_gcd_reduce(integer_direction)

    recovered = np.asarray(integer_direction, dtype=np.float64)
    recovered /= np.linalg.norm(recovered)
    if float(np.dot(recovered, target)) < 0.0:
        integer_direction = -integer_direction
        recovered = -recovered

    error = float(np.max(np.abs(recovered - target)))
    if error > tol:
        raise CrystallographyValueError(
            f"{name} could not be exactified within {tol=}; "
            f"maximum component error is {error:.3e}."
        )

    return integer_direction


def exactify_five_dof(
    params: object,
    *,
    max_exact_atoms: int = 10_000,
    max_sigma: int = _DEFAULT_MAX_SIGMA,
    max_denominator: int = _DEFAULT_MAX_DENOMINATOR,
    angle_tol: float = _DEFAULT_ANGLE_TOL,
    plane_tol: float = _DEFAULT_PLANE_TOL,
    lattice_metric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactify five-DOF boundary parameters into paired integer P/Q matrices.

    ``params`` is first converted to left- and right-grain row-orientation matrices using
    the package five-DOF convention. Their relative misorientation is rationalized
    through an integer Hamilton quaternion, converted to an exact row-convention scaled
    rotation, and rejected unless its angular error and CSL sigma satisfy the requested
    bounds. The left-grain boundary normal is independently rationalized to a primitive
    integer direction. An orthogonal exact embedding supplies suitable boundary-plane
    and in-plane directions, which are finally rescaled into exactly paired ``P`` and
    ``Q`` rows.

    The returned matrices are proper row-wise integer orientation frames and satisfy
    ``Q == P @ R`` exactly for the recovered rational row rotation ``R``. Equivalently,
    :func:`recover_exact_row_rotation_from_paired_pq` recovers the same scaled rotation
    used during construction. The matrices are suitable for
    ``PQSpec(P=P, Q=Q, basis_mode="primitive")``.

    :param params: Five-DOF parameters ``[alpha, beta, gamma, theta, phi]`` in radians,
        interpreted by :func:`orientation_matrices_from_five_dof`.
    :param max_exact_atoms: Positive upper bound used by the orthogonal embedding path
        for the exact in-plane cell and the absolute determinants of the candidate P/Q
        supercells. Keyword argument, optional, defaults to ``10000``.
    :param max_sigma: Positive upper bound on the coincidence-site-lattice sigma of the
        exactified misorientation. Keyword argument, optional, defaults to ``10000``.
    :param max_denominator: Positive upper bound used both when recovering the integer
        quaternion and when rationalizing boundary-normal component ratios. Keyword
        argument, optional, defaults to ``10001``.
    :param angle_tol: Finite, strictly positive maximum geodesic rotation error, in
        radians, between the approximate five-DOF misorientation and the recovered exact
        misorientation. Keyword argument, optional, defaults to ``1e-9``.
    :param plane_tol: Finite, strictly positive maximum absolute component error between
        the normalized floating-point boundary normal and normalized recovered primitive
        integer normal. Keyword argument, optional, defaults to ``1e-9``.
    :param lattice_metric: Reserved lattice metric argument. Only ``None``, representing
        the currently supported implicit cubic metric, is accepted. Keyword argument,
        optional, defaults to ``None``.
    :return: ``(P, Q)`` as object-dtype 3 by 3 integer matrices suitable for primitive
        ``PQSpec`` construction and exactly paired under the recovered row rotation.
    :raises CrystallographyValueError: If a bound or tolerance is invalid; if ``params``
        is malformed; if the misorientation cannot be rationalized within
        ``max_denominator`` and ``angle_tol``; if its CSL sigma exceeds ``max_sigma``;
        if the boundary normal cannot be rationalized within ``plane_tol``; or if final
        exact P/Q pairing validation fails.
    :raises CrystallographyNotImplementedError: If ``lattice_metric`` is not ``None``.
    :raises BoundarySpecError: If the exact in-plane cell or candidate P/Q supercell
        exceeds ``max_exact_atoms`` in the orthogonal embedding path.
    """
    _require_cubic(lattice_metric)
    max_exact_atoms = as_positive_int(max_exact_atoms, "max_exact_atoms")
    max_sigma = as_positive_int(max_sigma, "max_sigma")
    max_denominator = as_positive_int(max_denominator, "max_denominator")
    angle_tol = _positive_tolerance(angle_tol, "angle_tol")
    plane_tol = _positive_tolerance(plane_tol, "plane_tol")

    left, right = orientation_matrices_from_five_dof(params)
    approximate_rotation = left.T @ right

    scalar_last = Rotation.from_matrix(approximate_rotation).as_quat()
    unit_quaternion = scalar_last[[3, 0, 1, 2]]
    try:
        integer_quaternion = integer_quaternion_from_unit(
            unit_quaternion,
            max_denominator=max_denominator,
        )
    except CrystallographyValueError as exc:
        raise CrystallographyValueError(
            "Five-DOF misorientation could not be exactified within "
            f"{max_denominator=}."
        ) from exc

    row_rotation = quaternion_to_scaled_rotation(integer_quaternion)
    exact_rotation = (
        np.asarray(row_rotation.matrix, dtype=np.float64)
        / row_rotation.denominator
    )
    angular_error = float(
        Rotation.from_matrix(exact_rotation.T @ approximate_rotation).magnitude()
    )
    if angular_error > angle_tol:
        raise CrystallographyValueError(
            "Five-DOF misorientation could not be exactified within "
            f"{angle_tol=}; angular error is {angular_error:.3e} radians."
        )

    column_rotation = transpose_rotation_convention(row_rotation)
    csl = csl_from_scaled_rotation(column_rotation)
    if csl.sigma > max_sigma:
        raise CrystallographyValueError(
            f"Exactified CSL sigma {csl.sigma} exceeds {max_sigma=}."
        )

    plane = _rationalize_direction(
        left[0],
        max_denominator=max_denominator,
        tol=plane_tol,
        name="boundary plane normal",
    )

    embedding = orthogonal_embedding_from_row_rotation_and_plane(
        row_rotation,
        plane,
        source="five_dof",
        max_exact_atoms=max_exact_atoms,
    )
    if embedding.P is None or embedding.metadata is None:
        raise CrystallographyValueError(
            "Orthogonal exactification did not produce P rows and primitive metadata."
        )

    return _paired_pq_from_direction_rows(
        embedding.P,
        row_rotation,
        primitive_area_index=embedding.metadata.primitive_area_index,
    )


__all__ = ["exactify_five_dof"]
