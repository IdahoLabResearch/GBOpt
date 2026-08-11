# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Rationalization of approximate boundaries into exact crystallographic forms."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ._guards import _require_cubic
from ._limits import (
    DEFAULT_MAX_PQ_DETERMINANT,
    DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
)
from .csl import csl_from_scaled_rotation
from .embedding import (
    _exact_embedding_from_precomputed_csl,
    _paired_pq_from_direction_rows,
)
from .integer import as_positive_int, row_gcd_reduce
from .orientation import orientation_matrices_from_five_dof
from .quaternion import (
    _integer_quaternion_candidate_from_unit,
    quaternion_to_scaled_rotation,
)
from .rotation import transpose_rotation_convention
from .types import CrystallographyValueError

_DEFAULT_MAX_SIGMA = 10_000
_DEFAULT_MAX_DENOMINATOR = 10_001
_DEFAULT_ANGLE_TOL = 1.0e-9
_DEFAULT_PLANE_TOL = 1.0e-9


def _positive_tolerance(value: object, name: str) -> float:
    """Validate and return a finite, strictly positive tolerance.

    :param value: Candidate scalar tolerance to validate.
    :param name: Parameter name used in validation errors.
    :return: ``value`` converted to a finite, strictly positive Python ``float``.
    :raises CrystallographyValueError: If ``value`` is Boolean, non-real, non-finite,
        or less than or equal to zero.
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


def _unit_float_direction(direction: object, name: str) -> np.ndarray:
    """Return a scale-safe unit vector for a finite nonzero 3-vector.

    The vector is divided by its largest absolute component before its Euclidean norm is
    evaluated. This prevents overflow for very large finite vectors and underflow for
    nonzero subnormal vectors. Magnitude is intentionally ignored: exactification treats
    the input only as a direction and therefore applies no minimum-norm threshold.

    :param direction: Candidate three-component direction.
    :param name: Human-readable argument name used in validation errors.
    :return: Unit-length ``float64`` vector with shape ``(3,)``.
    :raises CrystallographyValueError: If ``direction`` cannot be converted to a finite
        three-component vector or is exactly zero.
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

    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise CrystallographyValueError(f"{name} must be nonzero.")

    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _unit_integer_vector(values: Sequence[int]) -> np.ndarray:
    """Return a scale-safe unit vector for a nonzero integer sequence.

    The values are scaled by their largest absolute component before normalization,
    avoiding unnecessary overflow during conversion to floating point.

    :param values: Nonzero sequence of integer components.
    :return: Unit-length ``float64`` vector with the same number of components as
        ``values``.
    :raises CrystallographyValueError: If every component is zero.
    """
    integers = tuple(int(value) for value in values)
    scale = max(map(abs, integers))
    if scale == 0:
        raise CrystallographyValueError("integer direction must be nonzero.")

    scaled = np.fromiter(
        (value / scale for value in integers),
        dtype=np.float64,
        count=len(integers),
    )
    return scaled / np.linalg.norm(scaled)


def _rationalize_direction(
    direction: object,
    *,
    max_denominator: int,
    tol: float,
    name: str,
) -> np.ndarray:
    """Recover a primitive integer direction from a floating-point vector.

    The input magnitude is discarded. Component ratios are formed relative to the
    largest-magnitude unit-vector component, rationalized with bounded denominators,
    lifted to a common integer vector, and GCD-reduced. The candidate is accepted only
    when its unit direction differs from the requested direction by no more than ``tol``
    in the infinity norm.

    ``max_denominator`` and ``tol`` are assumed to have been validated by the caller.

    :param direction: Finite nonzero array-like direction with shape ``(3,)``.
    :param max_denominator: Largest denominator permitted for each component ratio.
        Keyword argument.
    :param tol: Maximum permitted unit-component error in the infinity norm. Keyword
        argument.
    :param name: Human-readable argument name used in validation errors. Keyword
        argument.
    :return: Primitive object-dtype integer direction with the input hemisphere
        retained.
    :raises CrystallographyValueError: If the direction is invalid or cannot be
        rationalized within the requested bounds.
    """
    target = _unit_float_direction(direction, name)

    reference_index = int(np.argmax(np.abs(target)))
    reference = float(target[reference_index])
    fractions = [
        Fraction(float(component) / reference).limit_denominator(max_denominator)
        for component in target
    ]
    denominator_lcm = math.lcm(*(fraction.denominator for fraction in fractions))
    integer_direction = row_gcd_reduce(
        np.array(
            [int(fraction * denominator_lcm) for fraction in fractions],
            dtype=object,
        )
    )

    recovered = _unit_integer_vector(integer_direction)
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


def _quaternion_angular_error(
    target: np.ndarray,
    integer_quaternion: Sequence[int],
) -> float:
    """Return the geodesic rotation error for an integer quaternion candidate.

    Quaternion sign equivalence is accounted for before the error is calculated.

    :param target: Target unit quaternion in Hamilton scalar-first order.
    :param integer_quaternion: Nonzero integer quaternion candidate in Hamilton
        scalar-first order.
    :return: Geodesic rotation error in radians.
    :raises CrystallographyValueError: If ``integer_quaternion`` is the zero quaternion.
    """
    recovered = _unit_integer_vector(integer_quaternion)
    if float(np.dot(recovered, target)) < 0.0:
        recovered = -recovered

    chord = float(np.linalg.norm(recovered - target))
    return 4.0 * math.asin(min(1.0, 0.5 * chord))


def exactify_five_dof(
    params: object,
    *,
    max_primitive_area_index: int = DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
    max_pq_determinant: int = DEFAULT_MAX_PQ_DETERMINANT,
    max_sigma: int = _DEFAULT_MAX_SIGMA,
    max_denominator: int = _DEFAULT_MAX_DENOMINATOR,
    angle_tol: float = _DEFAULT_ANGLE_TOL,
    plane_tol: float = _DEFAULT_PLANE_TOL,
    lattice_metric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactify five-DOF boundary parameters into paired integer P/Q matrices.

    The five-DOF parameters are first converted to floating-point left- and right-grain
    row-orientation matrices. Their relative misorientation is represented as a unit
    quaternion and rationalized to a bounded-denominator primitive integer quaternion.
    The candidate is accepted only when its geodesic rotation error does not exceed
    ``angle_tol``.

    The accepted quaternion defines an exact row-convention scaled rotation. Its
    column-convention transpose is used to construct a CSL whose Sigma value must not
    exceed ``max_sigma``. The left-grain boundary normal is independently rationalized
    to a primitive integer direction whose unit-component error must not exceed
    ``plane_tol``.

    An exact embedding is selected from the constructed CSL. Its directions are then
    converted into minimally scaled paired P/Q rows that recover the same exact row
    rotation. ``max_primitive_area_index`` bounds the minimal in-plane CSL topology,
    while ``max_pq_determinant`` bounds both the selected embedding matrices and the
    final paired matrices.

    :param params: Five-DOF parameters ``[alpha, beta, gamma, theta, phi]`` in radians.
    :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area index.
        Keyword argument, optional, defaults to ``DEFAULT_MAX_PRIMITIVE_AREA_INDEX``.
    :param max_pq_determinant: Maximum permitted absolute determinant of each exact P/Q
        matrix produced during embedding selection and final row pairing. Keyword
        argument, optional, defaults to ``DEFAULT_MAX_PQ_DETERMINANT``.
    :param max_sigma: Maximum permitted CSL coincidence index. Keyword argument,
        optional, defaults to ``_DEFAULT_MAX_SIGMA``.
    :param max_denominator: Maximum denominator used when rationalizing quaternion and
        boundary-normal component ratios. Keyword argument, optional, defaults to
        ``_DEFAULT_MAX_DENOMINATOR``.
    :param angle_tol: Maximum permitted geodesic misorientation error in radians.
        Keyword argument, optional, defaults to ``_DEFAULT_ANGLE_TOL``.
    :param plane_tol: Maximum permitted infinity-norm component error between the
        requested unit boundary normal and its recovered primitive integer direction.
        Keyword argument, optional, defaults to ``_DEFAULT_PLANE_TOL``.
    :param lattice_metric: Reserved non-cubic lattice metric. Only ``None`` is currently
        supported. Keyword argument, optional, defaults to ``None``.
    :return: Exactly paired object-dtype integer matrices ``(P, Q)`` suitable for
        ``PQSpec(P=P, Q=Q, basis_mode="primitive")``.
    :raises CrystallographyValueError: If a bound or tolerance is invalid, the
        misorientation or plane cannot be exactified within the requested limits, the
        derived Sigma value is too large, or exact embedding construction does not
        produce the required integer rows and metadata.
    :raises CrystallographyNotImplementedError: If ``lattice_metric`` is not ``None``.
    :raises CrystallographyBackendError: If exact CSL normal-form construction fails.
    :raises BoundarySpecError: If exact embedding construction fails or an exact-cell
        limit is exceeded.
    """
    _require_cubic(lattice_metric)
    max_primitive_area_index = as_positive_int(
        max_primitive_area_index,
        "max_primitive_area_index",
    )
    max_pq_determinant = as_positive_int(
        max_pq_determinant,
        "max_pq_determinant",
    )
    max_sigma = as_positive_int(max_sigma, "max_sigma")
    max_denominator = as_positive_int(max_denominator, "max_denominator")
    angle_tol = _positive_tolerance(angle_tol, "angle_tol")
    plane_tol = _positive_tolerance(plane_tol, "plane_tol")

    left, right = orientation_matrices_from_five_dof(params)
    approximate_rotation = left.T @ right

    scalar_last = Rotation.from_matrix(approximate_rotation).as_quat()
    unit_quaternion = scalar_last[[3, 0, 1, 2]]
    try:
        integer_quaternion = _integer_quaternion_candidate_from_unit(
            unit_quaternion,
            max_denominator=max_denominator,
        )
    except CrystallographyValueError as exc:
        raise CrystallographyValueError(
            "Five-DOF misorientation could not produce an integer quaternion "
            f"candidate with {max_denominator=}: {exc}"
        ) from exc

    angular_error = _quaternion_angular_error(unit_quaternion, integer_quaternion)
    if angular_error > angle_tol:
        raise CrystallographyValueError(
            "Five-DOF misorientation could not be exactified with "
            f"{max_denominator=} within {angle_tol=}; angular error is "
            f"{angular_error:.3e} radians."
        )

    row_rotation = quaternion_to_scaled_rotation(integer_quaternion)
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

    embedding = _exact_embedding_from_precomputed_csl(
        row_rotation,
        plane,
        csl,
        source="five_dof",
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )
    if embedding.P is None or embedding.metadata is None:
        raise CrystallographyValueError(
            "Exact embedding did not produce integer direction rows and primitive "
            "cell metadata."
        )

    return _paired_pq_from_direction_rows(
        embedding.P,
        row_rotation,
        primitive_area_index=embedding.metadata.primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )


__all__ = ["exactify_five_dof"]
