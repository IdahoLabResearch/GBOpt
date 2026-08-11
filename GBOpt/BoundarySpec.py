# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Boundary specification dataclasses and canonical embedding types.

Defines the user-facing boundary specification dataclasses, boundary-spec validation
errors, primitive-cell metadata, and the ``BoundaryEmbedding`` container returned by
crystallography adapters. Validation normalizes mutable array-like inputs into immutable
Python tuples or read-only NumPy arrays and raises ``BoundarySpecError`` subclasses for
boundary-spec field failures.
"""

import math
import operator
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from GBOpt.Utils import integer_linalg as ilinalg

_BOUNDARY_EMBEDDING_SOURCES = ("pq", "csl", "five_dof")

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class BoundarySpecError(Exception):
    """Base class for all boundary-spec validation failures."""


class BoundarySpecTypeError(BoundarySpecError, TypeError):
    """Raised when a boundary-spec field has the wrong type."""


class BoundarySpecValueError(BoundarySpecError, ValueError):
    """Raised when a boundary-spec field has an invalid value."""


class BoundarySpecOrthogonalityError(BoundarySpecError):
    """Raised when a derived rotation matrix fails an orthogonality check."""


def _translate_exact_integer_error(exc: ilinalg.ExactIntegerError) -> BoundarySpecError:
    """Translate exact-integer utility errors into boundary-spec errors.

    :param exc: Exception raised by ``GBOpt.Utils.integer_linalg``.
    :return: ``BoundarySpecError`` subclass preserving the original message.
    """
    if isinstance(exc, ilinalg.ExactIntegerTypeError):
        return BoundarySpecTypeError(str(exc))
    return BoundarySpecValueError(str(exc))


# ---------------------------------------------------------------------------
# Boundary-format dataclasses
# ---------------------------------------------------------------------------

def _is_bool_like(value) -> bool:
    """Return whether ``value`` is a Python or NumPy boolean scalar.

    :param value: Value to test.
    :return: ``True`` when ``value`` is a Python ``bool`` or NumPy ``bool_``.
    """
    return isinstance(value, (bool, np.bool_))


def _as_object_array(values, name: str) -> np.ndarray:
    """Return ``values`` as an object array, translating array-conversion failures.

    :param values: Array-like input to convert.
    :param name: Field name used in the conversion error message.
    :return: Object-dtype NumPy array produced from ``values``.
    :raises BoundarySpecTypeError: If ``values`` cannot be converted to a NumPy object
        array.
    """
    try:
        return np.asarray(values, dtype=object)
    except (ValueError, TypeError) as exc:
        raise BoundarySpecTypeError(
            f"{name} cannot be converted to an array: {exc}") from exc


def _as_finite_float_array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a shape-checked finite float array, rejecting booleans before coercion.

    :param values: Array-like input to validate and convert.
    :param shape: Required NumPy shape for the returned array.
    :param name: Field name used in error messages.
    :return: Float ndarray with shape ``shape`` and only finite entries.
    :raises BoundarySpecValueError: If the converted object array has the wrong shape or
        contains ``NaN`` or infinity.
    :raises BoundarySpecTypeError: If any entry is boolean or numeric conversion fails.
    """
    arr_obj = _as_object_array(values, name)

    if arr_obj.shape != shape:
        raise BoundarySpecValueError(
            f"{name} must have shape {shape}; got {arr_obj.shape}.")

    for index, value in np.ndenumerate(arr_obj):
        if _is_bool_like(value):
            raise BoundarySpecTypeError(f"{name}{index} must not be boolean.")

    try:
        arr = arr_obj.astype(float)
    except (ValueError, TypeError) as exc:
        raise BoundarySpecTypeError(
            f"{name} cannot be converted to a numeric array: {exc}"
        ) from exc

    if not np.all(np.isfinite(arr)):
        raise BoundarySpecValueError(
            f"{name} contains non-finite entries (NaN or inf).")

    return arr


def _as_finite_float_scalar(value, name: str) -> float:
    """Return a finite Python float, rejecting booleans before coercion.

    :param value: Scalar value to validate and convert.
    :param name: Field name used in error messages.
    :return: ``value`` converted to a finite Python ``float``.
    :raises BoundarySpecTypeError: If ``value`` is boolean or cannot be converted to
        ``float``.
    :raises BoundarySpecValueError: If ``value`` converts to ``NaN`` or infinity.
    """
    if _is_bool_like(value):
        raise BoundarySpecTypeError(f"{name} must not be boolean.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundarySpecTypeError(
            f"{name} must be a finite float; got {value!r}.") from exc

    if not np.isfinite(number):
        raise BoundarySpecValueError(f"{name} must be finite; got {number}.")

    return number


def _validate_nonzero_int_vector(seq, name: str, length: int = 3) -> tuple[int, ...]:
    """Validate ``seq`` as an exact, finite, integer-valued, nonzero vector.

    :param seq: Array-like sequence to validate.
    :param name: Name used in error messages.
    :param length: Expected number of elements. Keyword argument, optional, defaults to
        ``3``.
    :return: Tuple of Python ``int`` values.
    :raises BoundarySpecTypeError: If exact-integer validation fails because an entry
        has an invalid type.
    :raises BoundarySpecValueError: If exact-integer validation fails because of shape
        or value issues, or if the vector is all-zero.
    """
    try:
        values = ilinalg.as_int_vector(seq, length, name)
    except ilinalg.ExactIntegerError as exc:
        raise _translate_exact_integer_error(exc) from exc

    if not any(values):
        raise BoundarySpecValueError(f"{name} must not be all-zero.")

    return values


def _validate_pq_matrix(matrix, name: str) -> tuple[tuple[int, int, int], ...]:
    """Validate ``matrix`` as an exact, nonsingular 3 by 3 integer matrix.

    Called from ``PQSpec.__post_init__`` for each of ``P`` and ``Q``.

    :param matrix: Array-like input to validate.
    :param name: Name used in error messages, for example ``"P"`` or ``"Q"``.
    :return: Tuple-of-tuples copy of ``matrix`` as Python ``int`` values.
    :raises BoundarySpecTypeError: If exact-integer validation fails because an entry
        has an invalid type.
    :raises BoundarySpecValueError: If exact-integer validation fails because of shape
        or value issues, or if the matrix is singular.
    """
    field_name = f"PQSpec.{name}"
    try:
        arr = ilinalg.as_int_array(matrix, (3, 3), field_name)
    except ilinalg.ExactIntegerError as exc:
        raise _translate_exact_integer_error(exc) from exc

    if ilinalg.det3_int_checked(arr) == 0:
        raise BoundarySpecValueError(f"{field_name} is singular.")

    return (
        (int(arr[0, 0]), int(arr[0, 1]), int(arr[0, 2])),
        (int(arr[1, 0]), int(arr[1, 1]), int(arr[1, 2])),
        (int(arr[2, 0]), int(arr[2, 1]), int(arr[2, 2])),
    )


def _primitive_integer_tuple(values: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``values`` divided by their common component GCD.

    :param values: Tuple of integer components to reduce.
    :return: Tuple with the common component GCD divided out.
    :raises BoundarySpecValueError: If ``values`` is the all-zero tuple.
    """
    gcd_value = math.gcd(*(abs(value) for value in values))
    if gcd_value == 0:
        raise BoundarySpecValueError("integer tuple must not be all-zero.")
    return tuple(value // gcd_value for value in values)


def _sigma_from_primitive_integer_quaternion(quat: tuple[int, ...]) -> int:
    """Return the Sigma value implied by a primitive integer quaternion.

    The value is the odd part of ``w**2 + x**2 + y**2 + z**2`` after repeatedly dividing
    out factors of two.

    :param quat: Primitive integer quaternion in Hamilton scalar-first order.
    :return: Sigma value implied by ``quat``.
    :raises BoundarySpecValueError: If ``quat`` does not contain exactly four
        components.
    """
    if len(quat) != 4:
        raise BoundarySpecValueError("quat must have length 4.")
    norm_sq = sum(value * value for value in quat)
    while norm_sq > 1 and norm_sq % 2 == 0:
        norm_sq //= 2
    return int(norm_sq)


def _validate_positive_integer(value, name: str) -> int:
    """Return ``value`` as a positive Python ``int``.

    Boolean values are rejected before integer coercion so ``True`` and ``False`` are
    not accepted as numeric inputs.

    :param value: Scalar to validate.
    :param name: Name used in error messages.
    :return: Positive Python ``int``.
    :raises BoundarySpecTypeError: If ``value`` is boolean.
    :raises BoundarySpecValueError: If ``value`` is not an integer or is not positive.
    """
    if _is_bool_like(value):
        raise BoundarySpecTypeError(f"{name} must not be boolean.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise BoundarySpecValueError(
            f"{name} must be a positive integer"
        ) from exc
    if integer <= 0:
        raise BoundarySpecValueError(f"{name} must be a positive integer")
    return integer


def _validate_bool(value, name: str) -> bool:
    """Return ``value`` as a Python ``bool`` after rejecting non-boolean scalars.

    :param value: Scalar value to validate.
    :param name: Name used in error messages.
    :return: ``value`` converted to Python ``bool``.
    :raises BoundarySpecTypeError: If ``value`` is not a Python or NumPy boolean scalar.
    """
    if not _is_bool_like(value):
        raise BoundarySpecTypeError(f"{name} must be a bool; got {value!r}.")
    return bool(value)


def _frozen_float_matrix(matrix, name: str) -> np.ndarray:
    """Return a copied, finite, read-only 3 by 3 float matrix.

    :param matrix: Array-like input to copy and validate.
    :param name: Field name used in error messages.
    :return: Read-only 3 by 3 float ndarray.
    :raises BoundarySpecTypeError: If ``matrix`` cannot be converted to a float array.
    :raises BoundarySpecValueError: If ``matrix`` is not shape ``(3, 3)`` or contains
        non-finite entries.
    """
    try:
        arr = np.array(matrix, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise BoundarySpecTypeError(
            f"{name} cannot be converted to a float matrix: {exc}"
        ) from exc

    if arr.shape != (3, 3):
        raise BoundarySpecValueError(f"{name} must have shape (3, 3); got {arr.shape}.")

    if not np.all(np.isfinite(arr)):
        raise BoundarySpecValueError(
            f"{name} contains non-finite entries (NaN or inf).")

    arr.setflags(write=False)
    return arr


def _frozen_optional_int_matrix(matrix, name: str) -> np.ndarray | None:
    """Return ``None`` or a copied, read-only exact 3 by 3 integer matrix.

    :param matrix: Matrix to validate, or ``None``.
    :param name: Field name used in error messages.
    :return: ``None`` when ``matrix`` is ``None``; otherwise a read-only object-dtype
        integer ndarray.
    :raises BoundarySpecTypeError: If exact-integer validation fails because an entry
        has an invalid type.
    :raises BoundarySpecValueError: If exact-integer validation fails because of shape
        or value issues.
    """
    if matrix is None:
        return None

    try:
        arr = ilinalg.as_int_array(matrix, (3, 3), name)
    except ilinalg.ExactIntegerError as exc:
        raise _translate_exact_integer_error(exc) from exc

    arr = np.array(arr, dtype=object, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class FiveDOFSpec:
    """Five-degree-of-freedom boundary specified by Euler-angle parameters.

    :param params: Five-element sequence ``[alpha, beta, gamma, theta, phi]`` of finite
        floats, in radians. ``alpha``, ``beta``, and ``gamma`` are ZXZ Euler angles
        defining the inter-grain misorientation. ``theta`` is the inclination rotation
        about the y-axis, and ``phi`` is the inclination rotation about the z-axis.
    """
    params: Sequence[float]

    def __post_init__(self) -> None:
        """Normalize and store immutable five-degree-of-freedom parameters."""
        arr = _as_finite_float_array(self.params, (5,), "FiveDOFSpec.params")
        object.__setattr__(self, "params", tuple(float(x) for x in arr))


@dataclass(frozen=True, slots=True)
class PQSpec:
    """Boundary specified directly by reference-grain ``P`` and paired-grain ``Q`` rows.

    :param P: 3 by 3 row-wise orientation matrix for the reference grain. Rows are
        Miller-index directions and must form a nonsingular exact integer matrix.
    :param Q: 3 by 3 row-wise orientation matrix for the paired grain, with the same
        validation requirements as ``P``.
    :param basis_mode: Whether ``P``/``Q`` rows represent a ``"primitive"`` supercell
        whose rotation should be recovered and reduced to the minimal in-plane cell, or
        a ``"supplied"`` cell that should be canonicalized as-is. Keyword argument,
        optional, defaults to ``"primitive"``.
    """
    P: Sequence[Sequence[int | float]]
    Q: Sequence[Sequence[int | float]]
    basis_mode: Literal["primitive", "supplied"] = "primitive"

    def __post_init__(self) -> None:
        """Validate ``P``/``Q`` matrices and store the normalized basis mode.

        :raises BoundarySpecValueError: If ``basis_mode`` is not ``"primitive"`` or
            ``"supplied"``.
        """
        P = _validate_pq_matrix(self.P, "P")
        Q = _validate_pq_matrix(self.Q, "Q")

        if self.basis_mode not in ("primitive", "supplied"):
            raise BoundarySpecValueError(
                "PQSpec.basis_mode must be either 'primitive' or 'supplied'; "
                f"got {self.basis_mode!r}."
            )

        object.__setattr__(self, "P", P)
        object.__setattr__(self, "Q", Q)


@dataclass(frozen=True, kw_only=True, slots=True)
class _CSLSpecBase:
    """Shared axis, plane, and optional sigma fields for CSL boundary specs.

    :param axis: Rotation axis as integer Miller indices. Keyword argument, required.
    :param plane: Boundary-plane normal as integer Miller indices. Keyword argument,
        required.
    :param sigma: Optional expected sigma index. Keyword argument, optional, defaults to
        ``None``.
    """
    axis: Sequence[int]
    plane: Sequence[int]
    sigma: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize shared CSL axis, plane, and sigma fields."""
        axis = _validate_nonzero_int_vector(self.axis, "axis")
        plane = _validate_nonzero_int_vector(self.plane, "plane")

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane", plane)

        if self.sigma is not None:
            sigma = _validate_positive_integer(self.sigma, "sigma")
            object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True, kw_only=True, slots=True)
class CSLExactSpec(_CSLSpecBase):
    """Exact CSL boundary specified by axis, plane, and integer quaternion.

    :param axis: Rotation axis as integer Miller indices ``[u, v, w]``; for example,
        ``[0, 0, 1]`` for a ``[001]``-axis tilt boundary. Keyword argument, required.
    :param plane: Boundary-plane normal as integer Miller indices ``[h, k, l]`` in grain
        1's crystal frame; for example, ``[1, 0, 0]`` for a ``(100)`` boundary plane.
        Keyword argument, required.
    :param quat: Integer quaternion in Hamilton scalar-first order ``[w, x, y, z]``. All
        four components must be integers; the unit quaternion is obtained by dividing by
        the norm. The identity quaternion is valid and represents a zero-misorientation
        Sigma 1 case. Keyword argument, required; the dataclass default of ``None``
        exists only to satisfy inheritance and raises ``BoundarySpecValueError`` if not
        supplied.
    :param sigma: Expected sigma index for the CSL boundary. When supplied, it is
        validated against the quaternion-derived value. Keyword argument, optional,
        defaults to ``None``.
    """

    # default=None required by dataclass inheritance; __post_init__ enforces it.
    quat: Sequence[int] = field(default=None)  # type: ignore[ty:invalid-assignment]

    def __post_init__(self) -> None:
        """Validate quaternion requirements and consistency with axis and sigma.

        :raises BoundarySpecValueError: If ``quat`` is missing, exact-integer quaternion
            validation fails, the quaternion vector part is not parallel to ``axis``, or
            ``sigma`` disagrees with the quaternion-derived value.
        :raises BoundarySpecTypeError: If exact-integer quaternion validation fails
            because an entry has an invalid type.
        """
        _CSLSpecBase.__post_init__(self)

        if self.quat is None:
            raise BoundarySpecValueError("CSLExactSpec.quat is required.")

        quat = _validate_nonzero_int_vector(self.quat, "quat", length=4)
        quat = _primitive_integer_tuple(quat)
        object.__setattr__(self, "quat", quat)

        quat_vec = quat[1:]
        if any(quat_vec):
            try:
                cross = ilinalg.cross_int3(quat_vec, self.axis)
            except ilinalg.ExactIntegerError as exc:
                raise _translate_exact_integer_error(exc) from exc

            if any(int(value) for value in cross):
                raise BoundarySpecValueError(
                    f"Quaternion vector part {list(quat_vec)} is not parallel to "
                    f"axis {list(self.axis)}. The axis must match the rotation axis "
                    "encoded in the quaternion."
                )

        if self.sigma is not None:
            derived_sigma = _sigma_from_primitive_integer_quaternion(quat)
            if derived_sigma != self.sigma:
                raise BoundarySpecValueError(
                    f"Sigma mismatch: quaternion {list(quat)} gives "
                    f"sigma={derived_sigma}, but sigma={self.sigma} was provided."
                )


@dataclass(frozen=True, kw_only=True, slots=True)
class CSLApproxSpec(_CSLSpecBase):
    """Approximate CSL boundary specified by rotation axis, boundary plane, and angle.

    :param axis: Rotation axis as integer Miller indices ``[u, v, w]``. Keyword
        argument, required.
    :param plane: Boundary-plane normal as integer Miller indices ``[h, k, l]`` in grain
        1's crystal frame. Keyword argument, required.
    :param angle_deg: Misorientation angle in degrees. Keyword argument, required; the
        dataclass default of ``None`` exists only to satisfy inheritance and raises
        ``BoundarySpecValueError`` if not supplied.
    :param sigma: Nominal sigma value for the boundary. Informational only for
        approximate construction and not validated against ``angle_deg``. Keyword
        argument, optional, defaults to ``None``.
    """
    angle_deg: float = field(default=None)  # type: ignore[ty:invalid-assignment]

    def __post_init__(self) -> None:
        """Validate and store the approximate misorientation angle.

        :raises BoundarySpecValueError: If ``angle_deg`` is not supplied.
        """
        _CSLSpecBase.__post_init__(self)

        if self.angle_deg is None:
            raise BoundarySpecValueError("CSLApproxSpec.angle_deg is required.")

        angle = _as_finite_float_scalar(self.angle_deg, "CSLApproxSpec.angle_deg")
        object.__setattr__(self, "angle_deg", angle)


@dataclass(frozen=True, slots=True)
class PrimitiveCellMetadata:
    """Primitive boundary-defining cell metadata for exact embeddings.

    Stores boundary-topology quantities produced by the CSL or PQ adapter, not crystal
    structure data. ``UnitCell`` holds the atomic basis and lattice parameter; this
    class holds boundary-level geometry: optional caller-input area provenance, minimal
    primitive CSL area, optional returned orientation-row area, boundary-plane normal,
    and exact rotation denominator.

    ``orientation_area_index`` describes the area of the returned
    ``BoundaryEmbedding.P`` orientation rows when reported. It is descriptive only and is
    not required to be an integer multiple of ``primitive_area_index`` because
    orthogonal embeddings may use orientation rows rather than a CSL supercell basis.

    :param basis_mode: ``"primitive"``or ``"supplied"`` construction path that produced
        this metadata.
    :param input_area_index: In-plane area index of caller-provided ``P`` rows, when
        available.
    :param primitive_area_index: Minimal in-plane area index of the primitive CSL cell.
    :param input_reduction_index: ``input_area_index // primitive_area_index`` when
        ``input_area_index`` is available; otherwise ``None``.
    :param orientation_area_index: In-plane area index of the returned
        ``BoundaryEmbedding.P`` rows, when useful to report.
    :param plane: Primitive boundary-plane normal ``(h, k, l)``.
    :param rotation_denominator: Denominator ``N`` of the exact scaled rotation that
        produced this boundary.
    :param conventional_cell_multiplier: Atom-count multiplier ``2 *
        primitive_area_index`` for converting a conventional-cell atom count to a
        primitive bicrystal atom count.
    """

    basis_mode: Literal["primitive", "supplied"]
    input_area_index: int | None
    primitive_area_index: int
    input_reduction_index: int | None
    orientation_area_index: int | None
    plane: tuple[int, int, int]
    rotation_denominator: int
    conventional_cell_multiplier: int

    def __post_init__(self) -> None:
        """Validate primitive-cell metadata consistency and normalize stored values.

        :raises BoundarySpecValueError: If basis_mode is invalid, if an area index is not
            positive, if an area index is not a multiple of primitive_area_index, if a
            reduction index is inconsistent with its area indices, or if
            conventional_cell_multiplier is not twice primitive_area_index.
        """
        if self.basis_mode not in ("primitive", "supplied"):
            raise BoundarySpecValueError(
                "PrimitiveCellMetadata.basis_mode must be either "
                f"'primitive' or 'supplied'; got {self.basis_mode!r}."
            )

        primitive_area_index = _validate_positive_integer(
            self.primitive_area_index,
            "PrimitiveCellMetadata.primitive_area_index",
        )
        rotation_denominator = _validate_positive_integer(
            self.rotation_denominator,
            "PrimitiveCellMetadata.rotation_denominator",
        )
        conventional_cell_multiplier = _validate_positive_integer(
            self.conventional_cell_multiplier,
            "PrimitiveCellMetadata.conventional_cell_multiplier",
        )
        plane = _validate_nonzero_int_vector(
            self.plane,
            "PrimitiveCellMetadata.plane",
            length=3,
        )

        input_area_index = None
        input_reduction_index = None
        if self.input_area_index is not None:
            input_area_index = _validate_positive_integer(
                self.input_area_index,
                "PrimitiveCellMetadata.input_area_index",
            )
            if input_area_index % primitive_area_index != 0:
                raise BoundarySpecValueError(
                    "PrimitiveCellMetadata.input_area_index must be an integer "
                    "multiple of primitive_area_index."
                )

            expected_input_reduction_index = input_area_index // primitive_area_index
            if self.input_reduction_index is None:
                raise BoundarySpecValueError(
                    "PrimitiveCellMetadata.input_reduction_index is required when "
                    "input_area_index is provided."
                )

            input_reduction_index = _validate_positive_integer(
                self.input_reduction_index,
                "PrimitiveCellMetadata.input_reduction_index",
            )
            if input_reduction_index != expected_input_reduction_index:
                raise BoundarySpecValueError(
                    "PrimitiveCellMetadata.input_reduction_index must equal "
                    "input_area_index // primitive_area_index; got "
                    f"{input_reduction_index}, expected {expected_input_reduction_index}."
                )
        elif self.input_reduction_index is not None:
            raise BoundarySpecValueError(
                "PrimitiveCellMetadata.input_reduction_index must be None when "
                "input_area_index is None."
            )

        expected_multiplier = 2 * primitive_area_index
        if conventional_cell_multiplier != expected_multiplier:
            raise BoundarySpecValueError(
                "PrimitiveCellMetadata.conventional_cell_multiplier must equal "
                f"2 * primitive_area_index; got {conventional_cell_multiplier}, "
                f"expected {expected_multiplier}."
            )

        orientation_area_index = None
        if self.orientation_area_index is not None:
            orientation_area_index = _validate_positive_integer(
                self.orientation_area_index,
                "PrimitiveCellMetadata.orientation_area_index",
            )

        object.__setattr__(self, "input_area_index", input_area_index)
        object.__setattr__(self, "primitive_area_index", primitive_area_index)
        object.__setattr__(self, "input_reduction_index", input_reduction_index)
        object.__setattr__(self, "orientation_area_index", orientation_area_index)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "rotation_denominator", rotation_denominator)
        object.__setattr__(
            self,
            "conventional_cell_multiplier",
            conventional_cell_multiplier,
        )


@dataclass(frozen=True, slots=True)
class BoundaryEmbedding:
    """Canonical internal representation produced by every input adapter.

    ``P`` and ``Q`` are exact row-wise orientation matrices, or ``None`` for
    approximate-only paths. ``R_left`` and ``R_right`` are floating-point rotation
    matrices matching ``GBMaker``'s internal convention. ``exact`` and ``coherent``
    describe the construction path and interface type. ``source`` names the originating
    format.

    :param P: Exact row-wise orientation matrix for the reference grain, or ``None`` on
        approximate-only paths.
    :param Q: Exact row-wise orientation matrix for the paired grain, or ``None`` on
        approximate-only paths.
    :param R_left: Floating-point rotation matrix for the reference grain in
        ``GBMaker``'s internal convention.
    :param R_right: Floating-point rotation matrix for the paired grain in ``GBMaker``'s
        internal convention.
    :param exact: ``True`` when ``P``/``Q`` are exact integer matrices rather than
        derived from a floating-point angle.
    :param coherent: ``True`` when the embedding represents a coherent, non-relaxed
        boundary construction.
    :param source: Originating spec format: ``"pq"``, ``"csl"``, or ``"five_dof"``.
    :param metadata: Primitive-cell metadata when available. Keyword argument, optional,
        defaults to ``None``.
    """
    P: np.ndarray | None
    Q: np.ndarray | None
    R_left: np.ndarray
    R_right: np.ndarray
    exact: bool
    coherent: bool
    source: str
    metadata: PrimitiveCellMetadata | None = None

    def __post_init__(self) -> None:
        """Validate, freeze, and store canonical ``BoundaryEmbedding`` fields.

        :raises BoundarySpecValueError: If ``source`` is unsupported or ``exact=True``
            is used without both ``P`` and ``Q``.
        :raises BoundarySpecTypeError: If ``metadata`` is neither
            ``PrimitiveCellMetadata`` nor ``None``.
        """
        R_left = _frozen_float_matrix(self.R_left, "R_left")
        R_right = _frozen_float_matrix(self.R_right, "R_right")

        P = _frozen_optional_int_matrix(self.P, "P")
        Q = _frozen_optional_int_matrix(self.Q, "Q")

        exact = _validate_bool(self.exact, "exact")
        coherent = _validate_bool(self.coherent, "coherent")

        object.__setattr__(self, "source", self.source)
        object.__setattr__(self, "metadata", self.metadata)

        if self.source not in _BOUNDARY_EMBEDDING_SOURCES:
            expected = ", ".join(repr(value) for value in _BOUNDARY_EMBEDDING_SOURCES)
            raise BoundarySpecValueError(
                f"source must be one of {expected}; got {self.source!r}."
            )

        if self.metadata is not None and not isinstance(
            self.metadata,
            PrimitiveCellMetadata,
        ):
            raise BoundarySpecTypeError(
                "metadata must be a PrimitiveCellMetadata instance or None; "
                f"got {type(self.metadata).__name__}."
            )

        if exact and (P is None or Q is None):
            raise BoundarySpecValueError(
                "Exact BoundaryEmbedding requires P and Q matrices.")

        object.__setattr__(self, "R_left", R_left)
        object.__setattr__(self, "R_right", R_right)
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "coherent", coherent)


__all__ = [
    "BoundarySpecError",
    "BoundarySpecTypeError",
    "BoundarySpecValueError",
    "BoundarySpecOrthogonalityError",
    "FiveDOFSpec",
    "PQSpec",
    "CSLExactSpec",
    "CSLApproxSpec",
    "PrimitiveCellMetadata",
    "BoundaryEmbedding",
]
