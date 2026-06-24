# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared data types, type aliases, and exceptions for crystallography operations.

Contains dataclasses, named tuples, type aliases, and the exception hierarchy used
across the crystallography package. No arithmetic or validation logic belongs here; this
module is a pure data-definition layer imported by all other crystallography modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


class CrystallographyError(Exception):
    """Base for all crystallography errors."""


class CrystallographyValueError(CrystallographyError, ValueError):
    """Invalid input to a crystallography function."""


class CrystallographyBackendError(CrystallographyError):
    """Computation failed in an exact normal-form routine."""


class CrystallographyDivisibilityError(CrystallographyValueError):
    """A rational exact result was not integer-valued when required."""


class CrystallographyNotImplementedError(CrystallographyError, NotImplementedError):
    """Operation is defined but not yet implemented."""


_IntArray: TypeAlias = NDArray[np.object_]
# Miller index triplet or other 3-component integer vector
Int3: TypeAlias = tuple[int, int, int]
# Hamilton-order integer quaternion (w, x, y, z)
Int4: TypeAlias = tuple[int, int, int, int]
# Post-reduction mode for CSL basis construction
ReductionMode: TypeAlias = Literal["none", "lll"]


def _is_integer_scalar(value: Any) -> bool:
    """Return whether value is an integer scalar, excluding bool.

    :param value: Value to test.
    :return: ``True`` when value is a Python or NumPy integer scalar but not bool.
    """
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _require_int(value: Any, name: str) -> int:
    """Return value as ``int`` after validating it is an integer scalar.

    :param value: Candidate integer scalar.
    :param name: Field name used in error messages.
    :return: Value converted to a Python ``int``.
    :raises CrystallographyValueError: If value is not an integer scalar.
    """
    if not _is_integer_scalar(value):
        raise CrystallographyValueError(f"{name} must be an integer; got {value!r}.")
    return int(value)


def _require_positive_int(value: Any, name: str) -> int:
    """Return value as ``int`` after validating it is positive.

    :param value: Candidate positive integer scalar.
    :param name: Field name used in error messages.
    :return: Value converted to a Python ``int``.
    :raises CrystallographyValueError: If value is not a positive integer.
    """
    integer = _require_int(value, name)
    if integer <= 0:
        raise CrystallographyValueError(
            f"{name} must be a positive integer; got {value!r}."
        )
    return integer


def _require_nonnegative_int(value: Any, name: str) -> int:
    """Return value as ``int`` after validating it is nonnegative.

    :param value: Candidate nonnegative integer scalar.
    :param name: Field name used in error messages.
    :return: Value converted to a Python ``int``.
    :raises CrystallographyValueError: If value is not a nonnegative integer.
    """
    integer = _require_int(value, name)
    if integer < 0:
        raise CrystallographyValueError(
            f"{name} must be a nonnegative integer; got {value!r}."
        )
    return integer


def _require_bool(value: Any, name: str) -> bool:
    """Return value as ``bool`` after validating it is a boolean scalar.

    :param value: Candidate boolean scalar.
    :param name: Field name used in error messages.
    :return: Value converted to a Python ``bool``.
    :raises CrystallographyValueError: If value is not a boolean scalar.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise CrystallographyValueError(f"{name} must be a bool; got {value!r}.")
    return bool(value)


def _require_int_tuple(values: Any, length: int, name: str) -> tuple[int, ...]:
    """Return a fixed-length tuple of Python integers.

    :param values: Candidate sequence of integer scalars.
    :param length: Required tuple length.
    :param name: Field name used in error messages.
    :return: Tuple of Python integers.
    :raises CrystallographyValueError: If shape or integer validation fails.
    """
    arr = np.asarray(values, dtype=object)
    if arr.shape != (length,):
        raise CrystallographyValueError(
            f"{name} must have shape ({length},); got {arr.shape}."
        )
    return tuple(_require_int(value, f"{name}[{idx}]") for idx, value in enumerate(arr))


def _require_int_array(array: ArrayLike, shape: tuple[int, ...], name: str) -> _IntArray:
    """Return a read-only object-dtype integer array with a required shape.

    :param array: Candidate array-like field value.
    :param shape: Required NumPy shape.
    :param name: Field name used in error messages.
    :return: Read-only object-dtype array of Python integers.
    :raises CrystallographyValueError: If shape or integer validation fails.
    """
    arr = np.asarray(array, dtype=object)
    if arr.shape != shape:
        raise CrystallographyValueError(
            f"{name} must have shape {shape}; got {arr.shape}."
        )

    normalized = np.empty(shape, dtype=object)
    for index, value in np.ndenumerate(arr):
        normalized[index] = _require_int(value, f"{name}{index}")

    normalized.setflags(write=False)
    return normalized


def _array_repr(array: ArrayLike, *, name: str) -> str:
    """Return a compact single-line representation for an ndarray field.

    The representation includes shape, dtype, writeability, and a compact value preview
    suitable for logging. Whitespace is normalized so one dataclass instance generally
    occupies one log line.

    :param array: Array-like field value to represent.
    :param name: Field name to include in the returned representation.
    :return: Single-line representation of the array field.
    """
    arr = np.asarray(array)
    values = np.array2string(
        arr,
        separator=", ",
        threshold=18,
        edgeitems=2,
        max_line_width=10_000,
    )
    values = " ".join(values.split())
    return (
        f"{name}=array(shape={arr.shape}, dtype={arr.dtype}, "
        f"writeable={arr.flags.writeable}, values={values})"
    )


@dataclass(frozen=True, slots=True, eq=False)
class ScaledRotation:
    """Exact scaled rotation ``R = M / N``.

    Project convention: ``M / N`` is the row-vector multiplier used by P/Q embeddings::

        ``q_row = p_row @ M / N``

    Column-vector CSL routines must receive the transposed numerator ``M.T``.

    :param denominator: Positive integer denominator.
    :param matrix: 3 by 3 integer numerator matrix in row-vector convention.
    :param source: Input source used to construct the rotation.
    :param quaternion: Primitive Hamilton-order integer quaternion when the rotation
        came from quaternion input; otherwise ``None``.
    """

    denominator: int
    matrix: _IntArray = field(repr=False)
    source: Literal["quaternion", "matrix", "five_dof"]
    quaternion: Int4 | None = None

    def __post_init__(self) -> None:
        """Validate and freeze scaled-rotation fields.

        :raises CrystallographyValueError: If source is not one of ``"quaternion"``,
            ``"matrix"``, or ``"five_dof"``.
        """
        object.__setattr__(
            self,
            "denominator",
            _require_positive_int(self.denominator, "denominator"),
        )
        object.__setattr__(
            self,
            "matrix",
            _require_int_array(self.matrix, (3, 3), "matrix"),
        )

        if self.source not in ("quaternion", "matrix", "five_dof"):
            raise CrystallographyValueError(
                "source must be one of 'quaternion', 'matrix', or 'five_dof'; "
                f"got {self.source!r}."
            )

        if self.quaternion is not None:
            object.__setattr__(
                self,
                "quaternion",
                _require_int_tuple(self.quaternion, 4, "quaternion"),
            )

    def __repr__(self) -> str:
        """Return a compact single-line representation of this scaled rotation.

        :return: String including denominator, source, quaternion, and matrix metadata.
        """
        return (
            "ScaledRotation("
            f"denominator={self.denominator!r}, "
            f"source={self.source!r}, "
            f"quaternion={self.quaternion!r}, "
            f"{_array_repr(self.matrix, name='matrix')}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class SmithDiagnostics:
    """Smith normal-form diagnostics for a scaled rotation.

    The full SNF factor matrices are not stored here because no downstream computation
    uses them; only the domain-derived quantities are carried forward.

    :param diagonal: The three diagonal entries of the Smith normal form of the rotation
        numerator matrix ``M``, as an ``Int3``.
    :param kernel_moduli: Per-axis moduli ``N / gcd(di, N)`` for each SNF diagonal entry
        ``di``, where ``N`` is the scaled rotation denominator. Their product is the CSL
        sigma value.
    """

    diagonal: Int3
    kernel_moduli: Int3

    def __post_init__(self) -> None:
        """Validate and normalize Smith normal-form diagnostic fields.

        :raises CrystallographyValueError: If any diagonal entry is negative or any
            kernel modulus is non-positive.
        """
        diagonal = _require_int_tuple(self.diagonal, 3, "diagonal")
        kernel_moduli = _require_int_tuple(self.kernel_moduli, 3, "kernel_moduli")

        if any(value < 0 for value in diagonal):
            raise CrystallographyValueError(
                f"diagonal entries must be nonnegative; got {diagonal}."
            )
        if any(value <= 0 for value in kernel_moduli):
            raise CrystallographyValueError(
                f"kernel_moduli entries must be positive; got {kernel_moduli}."
            )

        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "kernel_moduli", kernel_moduli)


@dataclass(frozen=True, slots=True, eq=False)
class CSLResult:
    """Complete CSL construction result.

    :param rotation: Validated scaled rotation used to build the CSL.
    :param sigma: CSL index; the number of coincident lattice sites per primitive unit
        cell. Equals ``abs(det(basis_hnf))``.
    :param basis: Exposed 3 by 3 CSL basis, optionally post-reduced.
    :param basis_hnf: Canonical column-HNF CSL basis used for verification.
    :param diagnostics: Smith normal-form derived diagnostic quantities.
    """

    rotation: ScaledRotation
    sigma: int
    basis: _IntArray = field(repr=False)
    basis_hnf: _IntArray = field(repr=False)
    diagnostics: SmithDiagnostics

    def __post_init__(self) -> None:
        """Validate and freeze CSL result arrays and scalar fields."""
        object.__setattr__(self, "sigma", _require_positive_int(self.sigma, "sigma"))
        object.__setattr__(
            self,
            "basis",
            _require_int_array(self.basis, (3, 3), "basis"),
        )
        object.__setattr__(
            self,
            "basis_hnf",
            _require_int_array(self.basis_hnf, (3, 3), "basis_hnf"),
        )

    def __repr__(self) -> str:
        """Return a compact single-line debug representation.

        The nested rotation is summarized by denominator and source rather than expanded
        in full so CSL result logs remain readable.

        :return: String representation including sigma, rotation summary, diagnostics,
            and compact CSL basis summaries.
        """
        return (
            "CSLResult("
            f"sigma={self.sigma!r}, "
            f"rotation_denominator={self.rotation.denominator!r}, "
            f"rotation_source={self.rotation.source!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"{_array_repr(self.basis, name='basis')}, "
            f"{_array_repr(self.basis_hnf, name='basis_hnf')}"
            ")"
        )


@dataclass(frozen=True, slots=True, eq=False)
class InPlaneBasis:
    """Primitive in-plane CSL basis and its CSL-column coefficients.

    :param basis: 3 by 2 integer matrix whose columns are in-plane CSL vectors.
    :param coefficients: 3 by 2 integer coefficient matrix satisfying ``csl_basis @
        coefficients == basis``. It expresses each in-plane CSL vector as a linear
        combination of the CSL basis columns.
    :param plane_covector: Primitive integer plane normal.
    """

    basis: _IntArray = field(repr=False)
    coefficients: _IntArray = field(repr=False)
    plane_covector: Int3

    def __post_init__(self) -> None:
        """Validate and freeze in-plane basis fields."""
        object.__setattr__(
            self,
            "basis",
            _require_int_array(self.basis, (3, 2), "basis"),
        )
        object.__setattr__(
            self,
            "coefficients",
            _require_int_array(self.coefficients, (3, 2), "coefficients"),
        )
        object.__setattr__(
            self,
            "plane_covector",
            _require_int_tuple(self.plane_covector, 3, "plane_covector"),
        )

    def __repr__(self) -> str:
        """Return a compact single-line representation of this in-plane basis.

        :return: String including plane covector and array metadata.
        """
        return (
            "InPlaneBasis("
            f"plane_covector={self.plane_covector!r}, "
            f"{_array_repr(self.basis, name='basis')}, "
            f"{_array_repr(self.coefficients, name='coefficients')}"
            ")"
        )


@dataclass(frozen=True, slots=True, eq=False)
class DSCBasis:
    """Rational DSC basis represented by an integer numerator and denominator.

    :param numerator: 3 by 3 integer numerator matrix for the DSC basis.
    :param denominator: Positive denominator; equals ``sigma`` for cubic CSLs.
    :param sigma: CSL index associated with this DSC basis.
    """

    numerator: _IntArray = field(repr=False)
    denominator: int
    sigma: int

    def __post_init__(self) -> None:
        """Validate and freeze DSC basis fields."""
        object.__setattr__(
            self,
            "numerator",
            _require_int_array(self.numerator, (3, 3), "numerator"),
        )
        object.__setattr__(
            self,
            "denominator",
            _require_positive_int(self.denominator, "denominator"),
        )
        object.__setattr__(
            self,
            "sigma",
            _require_positive_int(self.sigma, "sigma"),
        )

    def __repr__(self) -> str:
        """Return a compact single-line representation of this DSC basis.

        :return: String including denominator, sigma, and numerator metadata.
        """
        return (
            "DSCBasis("
            f"denominator={self.denominator!r}, "
            f"sigma={self.sigma!r}, "
            f"{_array_repr(self.numerator, name='numerator')}"
            ")"
        )


@dataclass(frozen=True, slots=True, eq=False)
class CoincidenceCheck:
    """Exact coincidence-lattice membership check result.

    :param ok: ``True`` when every tested basis vector is coincident and, if supplied,
        the determinant matches ``sigma``.
    :param residual_mod_N: 3 by 3 integer array of element-wise remainders of ``M @ C
        mod N``, where ``M`` is the rotation numerator matrix, ``C`` is the checked CSL
        basis, and ``N`` is the rotation denominator. All entries are zero when
        membership holds.
    :param det_basis: Absolute determinant of the checked basis.
    :param sigma: Expected sigma used for determinant validation, or ``None``.
    """

    ok: bool
    residual_mod_N: _IntArray = field(repr=False)
    det_basis: int
    sigma: int | None

    def __post_init__(self) -> None:
        """Validate and freeze coincidence-check fields."""
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(
            self,
            "residual_mod_N",
            _require_int_array(self.residual_mod_N, (3, 3), "residual_mod_N"),
        )
        object.__setattr__(
            self,
            "det_basis",
            _require_nonnegative_int(self.det_basis, "det_basis"),
        )
        if self.sigma is not None:
            object.__setattr__(
                self,
                "sigma",
                _require_positive_int(self.sigma, "sigma"),
            )

    def __repr__(self) -> str:
        """Return a compact single-line representation of this coincidence check.

        :return: String including ok, det_basis, sigma, and residual metadata.
        """
        return (
            "CoincidenceCheck("
            f"ok={self.ok!r}, "
            f"det_basis={self.det_basis!r}, "
            f"sigma={self.sigma!r}, "
            f"{_array_repr(self.residual_mod_N, name='residual_mod_N')}"
            ")"
        )


__all__ = (
    "CrystallographyError",
    "CrystallographyValueError",
    "CrystallographyBackendError",
    "CrystallographyDivisibilityError",
    "CrystallographyNotImplementedError",
    "Int3",
    "Int4",
    "ReductionMode",
    "ScaledRotation",
    "SmithDiagnostics",
    "CSLResult",
    "InPlaneBasis",
    "DSCBasis",
    "CoincidenceCheck",
)
