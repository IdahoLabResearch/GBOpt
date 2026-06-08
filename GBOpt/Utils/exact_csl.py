# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer CSL arithmetic for cubic Bravais coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    column_hnf_3x3,
    hnf_2d_supercells,
    smith_normal_form_3x3,
)


Int3 = tuple[int, int, int]
Int4 = tuple[int, int, int, int]
ReductionMode = Literal["none", "lll"]
NormalFormBackend = Literal["auto", "pure_python", "sympy"]


class ExactCSLError(Exception):
    """Base for all exact-CSL arithmetic errors."""


class ExactCSLValueError(ExactCSLError, ValueError):
    """Invalid input to an exact-CSL function."""


class ExactCSLBackendError(ExactCSLError):
    """Requested exact normal-form backend is unavailable."""


class ExactCSLNotImplementedError(ExactCSLError, NotImplementedError):
    """Operation is defined but not yet implemented."""


@dataclass(frozen=True)
class ScaledRotation:
    """Exact scaled rotation ``R = M / N``."""

    N: int
    M: np.ndarray
    source: Literal["quaternion", "matrix", "five_dof"]
    quaternion: Int4 | None = None


@dataclass(frozen=True)
class SmithDiagnostics:
    """Smith normal-form diagnostics for a scaled rotation."""

    U: np.ndarray
    D: np.ndarray
    V: np.ndarray
    diagonal: Int3
    kernel_moduli: Int3


@dataclass(frozen=True)
class CSLResult:
    """Complete CSL construction result."""

    rotation: ScaledRotation
    sigma: int
    basis: np.ndarray
    basis_hnf: np.ndarray
    diagnostics: SmithDiagnostics


@dataclass(frozen=True)
class InPlaneBasis:
    """Primitive in-plane CSL basis and its CSL-column coefficients."""

    basis: np.ndarray
    coefficients: np.ndarray
    plane_covector: Int3
    case_id: Literal[1, 2, 3]


@dataclass(frozen=True)
class DSCBasis:
    """Rational DSC basis represented by an integer numerator and denominator."""

    numerator: np.ndarray
    denominator: int
    sigma: int


@dataclass(frozen=True)
class CoincidenceCheck:
    """Exact coincidence-lattice membership check result."""

    ok: bool
    residual_mod_N: np.ndarray
    det_basis: int
    sigma: int | None


@dataclass(frozen=True)
class ExactifiedFiveDOF:
    """Reserved result type for future five-DOF exactification."""

    P: np.ndarray
    Q: np.ndarray
    rotation: ScaledRotation
    csl: CSLResult
    max_error: float


def _check_lattice_metric(metric: np.ndarray | None) -> None:
    """Reject non-cubic metric inputs reserved for a later extension."""
    if metric is not None:
        raise ExactCSLNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )


def _as_int_vector(values: ArrayLike, length: int, name: str) -> tuple[int, ...]:
    """Return an exact integer vector from an array-like input."""
    arr = np.asarray(values)
    if arr.shape != (length,):
        raise ExactCSLValueError(f"{name} must have shape ({length},); got {arr.shape}.")
    result: list[int] = []
    for i, value in enumerate(arr):
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ExactCSLValueError(f"{name}[{i}]={value!r} is not an integer.") from exc
        if value != integer:
            raise ExactCSLValueError(
                f"{name}[{i}]={value!r} is not exactly integer-valued."
            )
        result.append(integer)
    return tuple(result)


def _as_int_matrix(A: ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return an exact integer matrix with object dtype."""
    arr = np.asarray(A)
    if arr.shape != shape:
        raise ExactCSLValueError(f"{name} must have shape {shape}; got {arr.shape}.")
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        value = arr[index]
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ExactCSLValueError(
                f"{name}{index}={value!r} is not an integer."
            ) from exc
        if value != integer:
            raise ExactCSLValueError(
                f"{name}{index}={value!r} is not exactly integer-valued."
            )
        out[index] = integer
    return out


def _det3(A: ArrayLike) -> int:
    """Return the exact determinant of a 3 by 3 integer matrix."""
    matrix = np.asarray(A, dtype=object)
    return int(
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def _adjugate_3x3(A: ArrayLike) -> np.ndarray:
    """Return the exact adjugate of a 3 by 3 integer matrix."""
    matrix = np.asarray(A, dtype=object)
    out = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            rows = [r for r in range(3) if r != j]
            cols = [c for c in range(3) if c != i]
            minor = (
                matrix[rows[0], cols[0]] * matrix[rows[1], cols[1]]
                - matrix[rows[0], cols[1]] * matrix[rows[1], cols[0]]
            )
            out[i, j] = minor if (i + j) % 2 == 0 else -minor
    return out


def _gcd_many(values: ArrayLike) -> int:
    """Return the gcd of the absolute values in *values*."""
    gcd_value = 0
    for value in np.asarray(values, dtype=object).flat:
        gcd_value = math.gcd(gcd_value, abs(int(value)))
    return gcd_value


def _row_gcd_reduce(row: ArrayLike) -> np.ndarray:
    """Divide an integer row by the gcd of its entries."""
    arr = np.asarray(row, dtype=object)
    gcd_value = _gcd_many(arr)
    if gcd_value <= 1:
        return arr.astype(object)
    return np.array([int(value) // gcd_value for value in arr], dtype=object)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``x*a + y*b == g``."""
    a = int(a)
    b = int(b)
    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return (
        old_r,
        old_s if a >= 0 else -old_s,
        old_t if b >= 0 else -old_t,
    )


def _primitive_plane(h: ArrayLike) -> Int3:
    """Return a primitive integer plane covector."""
    vec = list(_as_int_vector(h, 3, "h"))
    gcd_value = 0
    for value in vec:
        gcd_value = math.gcd(gcd_value, abs(value))
    if gcd_value == 0:
        raise ExactCSLValueError("plane covector h must not be the zero vector.")
    return tuple(value // gcd_value for value in vec)  # type: ignore[return-value]


def normalize_integer_quaternion(q: tuple) -> Int4:
    """Return the canonical primitive representative of an integer quaternion."""
    values = list(_as_int_vector(q, 4, "q"))
    gcd_value = 0
    for value in values:
        gcd_value = math.gcd(gcd_value, abs(value))
    if gcd_value == 0:
        raise ExactCSLValueError("q is the zero quaternion.")
    values = [value // gcd_value for value in values]
    if tuple(values) < (0, 0, 0, 0):
        values = [-value for value in values]
    return tuple(values)  # type: ignore[return-value]


def quaternion_to_scaled_rotation(
    q: tuple,
    *,
    canonicalize: bool = True,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Construct the exact Euler-Rodrigues scaled rotation for an integer quaternion."""
    _check_lattice_metric(lattice_metric)
    quat = normalize_integer_quaternion(q) if canonicalize else _as_int_vector(q, 4, "q")
    w, x, y, z = quat
    N = w * w + x * x + y * y + z * z
    if N == 0:
        raise ExactCSLValueError("q is the zero quaternion.")

    M = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=object,
    )
    rotation = ScaledRotation(N=N, M=M, source="quaternion", quaternion=quat)
    _assert_scaled_rotation(rotation)
    return rotation


def _assert_scaled_rotation(rotation: ScaledRotation) -> None:
    """Raise if ``rotation`` fails exact scaled-rotation identities."""
    N = int(rotation.N)
    M = np.asarray(rotation.M, dtype=object)
    gram = M @ M.T
    expected = (N * N) * np.eye(3, dtype=object)
    if not np.array_equal(gram, expected):
        raise ExactCSLValueError("scaled rotation is not exactly orthogonal.")
    det = _det3(M)
    if det != N ** 3:
        raise ExactCSLValueError(
            f"scaled rotation determinant {det} does not equal N^3={N ** 3}."
        )


def validate_scaled_rotation_matrix(
    M_in: np.ndarray,
    N: int | None = None,
    *,
    source: Literal["matrix", "five_dof"] = "matrix",
    reduce_common_factor: bool = False,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Validate a user-supplied integer scaled rotation matrix."""
    _check_lattice_metric(lattice_metric)
    M = _as_int_matrix(M_in, (3, 3), "M_in")
    expected_N = None if N is None else int(N)
    if N is not None and expected_N != N:
        raise ExactCSLValueError(f"N must be an integer; got {N!r}.")

    if reduce_common_factor:
        gcd_value = _gcd_many(M)
        if gcd_value > 1:
            if expected_N is not None and expected_N % gcd_value != 0:
                raise ExactCSLValueError(
                    "common matrix factor does not divide the supplied N."
                )
            M = np.array([int(value) // gcd_value for value in M.flat], dtype=object).reshape(3, 3)
            if expected_N is not None:
                expected_N //= gcd_value

    gram = M @ M.T
    diagonal = [int(gram[i, i]) for i in range(3)]
    if diagonal[0] <= 0 or diagonal[1:] != diagonal[:1] * 2:
        raise ExactCSLValueError("M @ M.T does not have equal positive diagonal entries.")
    for i in range(3):
        for j in range(3):
            if i != j and gram[i, j] != 0:
                raise ExactCSLValueError("M @ M.T has nonzero off-diagonal entries.")
    derived_N = math.isqrt(diagonal[0])
    if derived_N * derived_N != diagonal[0]:
        raise ExactCSLValueError("M @ M.T diagonal is not a perfect square.")
    det = _det3(M)
    if det != derived_N ** 3:
        raise ExactCSLValueError(
            f"det(M)={det} does not equal N^3={derived_N ** 3}."
        )
    if expected_N is not None and expected_N != derived_N:
        raise ExactCSLValueError(
            f"supplied N={expected_N} does not match derived N={derived_N}."
        )
    return ScaledRotation(N=derived_N, M=M, source=source)


def sigma_from_snf_diagonal(N: int, diagonal: tuple) -> tuple[int, Int3]:
    """Derive true Sigma and kernel moduli from an SNF diagonal and scale ``N``."""
    scale = int(N)
    if scale <= 0:
        raise ExactCSLValueError(f"N must be positive; got {N!r}.")
    diag = _as_int_vector(diagonal, 3, "diagonal")
    moduli: list[int] = []
    sigma = 1
    for value in diag:
        modulus = scale // math.gcd(abs(int(value)), scale)
        if modulus <= 0:
            raise ExactCSLValueError("SNF-derived kernel modulus is not positive.")
        moduli.append(modulus)
        sigma *= modulus
    if sigma <= 0:
        raise ExactCSLValueError("SNF-derived sigma is not positive.")
    return sigma, tuple(moduli)  # type: ignore[return-value]


def csl_from_scaled_rotation(
    rotation: ScaledRotation,
    *,
    backend: NormalFormBackend = "auto",
    post_reduce: ReductionMode = "none",
) -> CSLResult:
    """Construct a canonical CSL basis from an exact scaled rotation."""
    if backend not in ("auto", "pure_python", "sympy"):
        raise ExactCSLBackendError(f"unknown normal-form backend {backend!r}.")
    if backend == "sympy":
        try:
            import sympy  # noqa: F401
        except ImportError as exc:
            raise ExactCSLBackendError("requested SymPy backend is unavailable.") from exc
    if post_reduce not in ("none", "lll"):
        raise ExactCSLValueError(f"unknown post_reduce mode {post_reduce!r}.")

    M = _as_int_matrix(rotation.M, (3, 3), "rotation.M")
    checked_rotation = ScaledRotation(
        N=int(rotation.N),
        M=M,
        source=rotation.source,
        quaternion=rotation.quaternion,
    )
    _assert_scaled_rotation(checked_rotation)
    try:
        snf = smith_normal_form_3x3(M)
        diagonal = tuple(int(snf.D[i, i]) for i in range(3))
        sigma, kernel_moduli = sigma_from_snf_diagonal(checked_rotation.N, diagonal)
        scales = np.diag(np.array(kernel_moduli, dtype=object))
        raw_basis = snf.V @ scales
        basis_hnf = column_hnf_3x3(raw_basis)
    except ExactNormalFormError as exc:
        raise ExactCSLBackendError(str(exc)) from exc

    exposed_basis = lll_reduce(basis_hnf) if post_reduce == "lll" else raw_basis
    diagnostics = SmithDiagnostics(
        U=snf.U,
        D=snf.D,
        V=snf.V,
        diagonal=diagonal,  # type: ignore[arg-type]
        kernel_moduli=kernel_moduli,
    )
    result = CSLResult(
        rotation=checked_rotation,
        sigma=sigma,
        basis=exposed_basis,
        basis_hnf=basis_hnf,
        diagnostics=diagnostics,
    )
    check = verify_coincidence_basis(checked_rotation, basis_hnf, sigma=sigma)
    if not check.ok:
        raise ExactCSLValueError("constructed CSL basis failed exact verification.")
    return result


def _independent_rank2(B: np.ndarray) -> bool:
    """Return True if a 3 by 2 integer matrix has rank two."""
    v1 = B[:, 0]
    v2 = B[:, 1]
    cross = np.array(
        [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ],
        dtype=object,
    )
    return any(value != 0 for value in cross)


def _cross3(v1: ArrayLike, v2: ArrayLike) -> np.ndarray:
    """Return the exact cross product of two integer 3-vectors."""
    a = np.asarray(v1, dtype=object)
    b = np.asarray(v2, dtype=object)
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=object,
    )


def _primitive_null_coefficients(covector: ArrayLike) -> np.ndarray:
    """Return primitive integer columns spanning ``covector @ x == 0``."""
    vec = np.asarray(covector, dtype=object).copy()
    V = np.eye(3, dtype=object)

    for i in range(2):
        nonzero = next((j for j in range(i, 3) if vec[j] != 0), None)
        if nonzero is None:
            break
        if nonzero != i:
            vec[[i, nonzero]] = vec[[nonzero, i]]
            V[:, [i, nonzero]] = V[:, [nonzero, i]]
        for j in range(i + 1, 3):
            if vec[j] == 0:
                continue
            g, a, b = _extended_gcd(int(vec[i]), int(vec[j]))
            c = int(vec[i]) // g
            d = int(vec[j]) // g
            old_i = V[:, i].copy()
            old_j = V[:, j].copy()
            V[:, i] = a * old_i + b * old_j
            V[:, j] = -d * old_i + c * old_j
            vec[i] = g
            vec[j] = 0

    coeffs = V[:, 1:3].astype(object)
    target = np.asarray(covector, dtype=object)
    cross = _cross3(coeffs[:, 0], coeffs[:, 1])
    dot = sum(int(cross[i]) * int(target[i]) for i in range(3))
    if dot < 0:
        coeffs[:, 1] = -coeffs[:, 1]
    return coeffs


def _saturate_coefficients(coeffs: np.ndarray, covector: np.ndarray) -> np.ndarray:
    """Replace a nonprimitive rank-two coefficient basis with a primitive one."""
    cross = _cross3(coeffs[:, 0], coeffs[:, 1])
    primitive = np.asarray(covector, dtype=object)
    factor: int | None = None
    for i in range(3):
        if primitive[i] == 0:
            if cross[i] != 0:
                factor = None
                break
            continue
        value = int(cross[i])
        base = int(primitive[i])
        if value % base != 0:
            factor = None
            break
        current = value // base
        if factor is None:
            factor = current
        elif factor != current:
            factor = None
            break
    if factor is None or abs(factor) <= 1:
        return coeffs
    return _primitive_null_coefficients(primitive)


def _verify_projected_basis(B: np.ndarray) -> np.ndarray:
    """Validate and return a rank-two projected basis."""
    if not _independent_rank2(B):
        raise ExactCSLValueError("in-plane CSL vectors are linearly dependent.")
    return B.astype(object)


def inplane_basis_from_csl(
    csl_basis: np.ndarray,
    h: tuple,
    *,
    saturate: bool = True,
    lattice_metric: np.ndarray | None = None,
) -> InPlaneBasis:
    """Find two CSL vectors lying in the integer plane ``h^T v = 0``."""
    _check_lattice_metric(lattice_metric)
    C = _as_int_matrix(csl_basis, (3, 3), "csl_basis")
    plane = _primitive_plane(h)
    h_vec = np.array(plane, dtype=object)
    d = C.T @ h_vec
    zero_indices = [i for i in range(3) if d[i] == 0]
    coeffs = np.zeros((3, 2), dtype=object)

    if len(zero_indices) >= 2:
        coeffs[zero_indices[0], 0] = 1
        coeffs[zero_indices[1], 1] = 1
        case_id: Literal[1, 2, 3] = 1
    elif len(zero_indices) == 1:
        zero = zero_indices[0]
        other = [i for i in range(3) if i != zero]
        a, b = other
        g = math.gcd(abs(int(d[a])), abs(int(d[b])))
        coeffs[zero, 0] = 1
        coeffs[a, 1] = int(d[b]) // g
        coeffs[b, 1] = -int(d[a]) // g
        case_id = 2
    else:
        d0, d1, d2 = (int(d[0]), int(d[1]), int(d[2]))
        g01 = math.gcd(abs(d0), abs(d1))
        g12 = math.gcd(abs(d1), abs(d2))
        coeffs[:, 0] = np.array([-d1 // g01, d0 // g01, 0], dtype=object)
        coeffs[:, 1] = np.array([0, -d2 // g12, d1 // g12], dtype=object)
        if not _independent_rank2(coeffs):
            g02 = math.gcd(abs(d0), abs(d2))
            coeffs[:, 0] = np.array([-d1 // g01, d0 // g01, 0], dtype=object)
            coeffs[:, 1] = np.array([-d2 // g02, 0, d0 // g02], dtype=object)
        case_id = 3

    if saturate:
        coeffs = _saturate_coefficients(coeffs, d)
    basis = C @ coeffs
    basis = _verify_projected_basis(basis)
    residual = h_vec @ basis
    if residual[0] != 0 or residual[1] != 0:
        raise ExactCSLValueError("constructed in-plane basis is not in the plane.")
    return InPlaneBasis(
        basis=basis,
        coefficients=coeffs,
        plane_covector=plane,
        case_id=case_id,
    )


def enumerate_inplane_hnf_supercells(
    inplane_basis: np.ndarray,
    index: int,
) -> list[np.ndarray]:
    """Return all index-``n`` supercells of an in-plane CSL basis."""
    basis = _as_int_matrix(inplane_basis, (3, 2), "inplane_basis")
    return [basis @ H for H in hnf_2d_supercells(index)]


def dsc_basis(
    csl_basis: np.ndarray,
    sigma: int,
    *,
    lattice_basis: np.ndarray | None = None,
) -> DSCBasis:
    """Return the cubic DSC basis numerator ``adj(C)`` and denominator Sigma."""
    if lattice_basis is not None:
        raise ExactCSLNotImplementedError(
            "non-cubic lattice bases are not implemented"
        )
    C = _as_int_matrix(csl_basis, (3, 3), "csl_basis")
    sigma_int = int(sigma)
    if sigma_int != sigma or sigma_int <= 0:
        raise ExactCSLValueError(f"sigma must be a positive integer; got {sigma!r}.")
    det = _det3(C)
    if abs(det) != sigma_int:
        raise ExactCSLValueError(
            f"|det(csl_basis)|={abs(det)} does not equal sigma={sigma_int}."
        )
    numerator = _adjugate_3x3(C)
    if det < 0:
        numerator = -numerator
    if not np.array_equal(C @ numerator, sigma_int * np.eye(3, dtype=object)):
        raise ExactCSLValueError("DSC adjugate check failed.")
    return DSCBasis(numerator=numerator, denominator=sigma_int, sigma=sigma_int)


def verify_coincidence_basis(
    rotation: ScaledRotation,
    csl_basis: np.ndarray,
    *,
    sigma: int | None = None,
) -> CoincidenceCheck:
    """Check exact CSL membership and optional determinant index."""
    M = _as_int_matrix(rotation.M, (3, 3), "rotation.M")
    C = _as_int_matrix(csl_basis, (3, 3), "csl_basis")
    N = int(rotation.N)
    if N <= 0:
        raise ExactCSLValueError(f"rotation.N must be positive; got {rotation.N!r}.")
    residual = (M @ C) % N
    det_basis = abs(_det3(C))
    ok = not any(value != 0 for value in residual.flat)
    if sigma is not None:
        expected = int(sigma)
        if expected != sigma or expected <= 0:
            raise ExactCSLValueError(f"sigma must be a positive integer; got {sigma!r}.")
        ok = ok and det_basis == expected
    else:
        expected = None
    return CoincidenceCheck(
        ok=ok,
        residual_mod_N=residual.astype(object),
        det_basis=det_basis,
        sigma=expected,
    )


def pq_from_csl_plane(
    rotation: ScaledRotation,
    inplane: InPlaneBasis,
    *,
    row_reduce: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct raw integer P/Q matrices from an exact in-plane CSL basis."""
    M = _as_int_matrix(rotation.M, (3, 3), "rotation.M")
    N = int(rotation.N)
    normal = np.array(inplane.plane_covector, dtype=object)
    p1 = np.asarray(inplane.basis[:, 0], dtype=object)
    p2 = np.asarray(inplane.basis[:, 1], dtype=object)
    images: list[np.ndarray] = []
    for vector in (p1, p2):
        numerator = M @ vector
        if any(int(value) % N != 0 for value in numerator):
            raise ExactCSLValueError("in-plane vector image is not divisible by N.")
        images.append(np.array([int(value) // N for value in numerator], dtype=object))

    P = np.array([normal, p1, p2], dtype=object)
    Q = np.array([normal, images[0], images[1]], dtype=object)
    if row_reduce:
        P = np.array([_row_gcd_reduce(row) for row in P], dtype=object)
        Q = np.array([_row_gcd_reduce(row) for row in Q], dtype=object)
    return P, Q


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """Return a lattice-equivalent basis; optional shortening is deferred."""
    warnings.warn(
        "lll_reduce is not yet implemented; returning input unchanged.",
        UserWarning,
        stacklevel=2,
    )
    if not (0.25 < float(delta) <= 1.0):
        raise ExactCSLValueError("delta must be in the interval (0.25, 1.0].")
    return _as_int_matrix(B, (3, 3), "B").copy()


def exactify_five_dof(
    params: np.ndarray,
    *,
    lattice: str = "sc",
    max_sigma: int = 200,
    max_denominator: int = 4096,
    angle_tol: float = 1e-8,
    plane_tol: float = 1e-8,
    lattice_basis: np.ndarray | None = None,
) -> ExactifiedFiveDOF:
    """Stage-E hook for future floating five-DOF exactification."""
    raise ExactCSLNotImplementedError(
        "exactify_five_dof is not yet implemented (Stage E). "
        "Use CSLExactSpec with an explicit integer quaternion, or "
        "GBMaker.from_boundary_spec(..., mode='approximate') for float input."
    )


def integer_quaternion_from_unit(q: ArrayLike, max_denominator: int = 10001) -> Int4:
    """Recover a primitive integer quaternion proportional to a unit quaternion."""
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ExactCSLValueError(f"q must have shape (4,); got {arr.shape}.")
    nonzero = [i for i, value in enumerate(arr) if abs(float(value)) > 1e-14]
    if not nonzero:
        raise ExactCSLValueError("q is the zero quaternion.")
    ref = nonzero[0]
    fractions: list[Fraction] = []
    for value in arr:
        if abs(float(value)) <= 1e-14:
            fractions.append(Fraction(0, 1))
        else:
            fractions.append(Fraction(float(value) / float(arr[ref])).limit_denominator(max_denominator))
    denominator_lcm = 1
    for value in fractions:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    ints = [int(value * denominator_lcm) for value in fractions]
    return normalize_integer_quaternion(tuple(ints))
