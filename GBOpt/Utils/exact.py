# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact boundary and CSL utilities for canonical bicrystal construction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import warnings
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    PQSpec,
    PrimitiveCellMetadata,
)
from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    _as_int_matrix as _inf_as_int_matrix,
    _det3 as _inf_det3,
    _dot_int,
    _int_adj3 as _inf_int_adj3,
    _row_gcd_reduce as _row_gcd_reduce_int,
    column_hnf_3x3,
    hnf_2d_supercells,
    primitive_integer_null_basis_3d,
    smith_normal_form_3x3,
)


Int3 = tuple[int, int, int]
Int4 = tuple[int, int, int, int]
ReductionMode = Literal["none", "lll"]


class ExactCSLError(Exception):
    """Base for all exact-CSL arithmetic errors."""


class ExactCSLValueError(ExactCSLError, ValueError):
    """Invalid input to an exact-CSL function."""


class ExactCSLBackendError(ExactCSLError):
    """Computation failed in an exact normal-form routine."""


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
    """Smith normal-form diagnostics for a scaled rotation.

    Only the domain-specific derived quantities are stored here.  The full
    ``U``, ``D``, ``V`` matrices from the SNF decomposition are available as
    the local ``snf`` variable inside ``csl_from_scaled_rotation`` but are not
    carried forward since no downstream computation uses them.

    :param diagonal: The SNF diagonal entries ``(d0, d1, d2)``.
    :param kernel_moduli: Per-axis moduli ``N / gcd(di, N)`` used to derive sigma.
    """

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


def _as_int_array(A: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return *A* as a shape-checked object array of Python integers.

    Thin wrapper over ``integer_normal_forms._as_int_matrix`` that translates
    ``ExactNormalFormError`` to ``ExactCSLValueError`` for consistency with this
    module's exception hierarchy.

    :param A: Input array-like of any shape.
    :param shape: Expected shape (1D or 2D).
    :param name: Name used in error messages.
    :return: Object-dtype ndarray of Python integers.
    :raises ExactCSLValueError: If shape mismatches or any entry is non-integer.
    """
    try:
        return _inf_as_int_matrix(A, shape, name)
    except ExactNormalFormError as exc:
        raise ExactCSLValueError(str(exc)) from exc


def _as_int_vector(values: ArrayLike, length: int, name: str) -> tuple[int, ...]:
    """Return an exact integer tuple from a 1D array-like input.

    Delegates to ``_as_int_array`` with ``shape=(length,)`` and converts to a
    plain Python tuple of ints for use in contexts that require hashable sequences.

    :param values: 1D array-like of integer-valued entries.
    :param length: Expected number of elements.
    :param name: Name used in error messages.
    :return: Tuple of Python ints.
    :raises ExactCSLValueError: If length mismatches or any entry is non-integer.
    """
    arr = _as_int_array(values, (length,), name)
    return tuple(int(v) for v in arr)


def _det3(A: ArrayLike) -> int:
    """Return the exact determinant of a 3 by 3 integer matrix.

    Delegates to ``integer_normal_forms._det3``.
    """
    try:
        return _inf_det3(_as_int_array(A, (3, 3), "A"))
    except ExactNormalFormError as exc:
        raise ExactCSLValueError(str(exc)) from exc


def _row_gcd_reduce_object(row: ArrayLike) -> np.ndarray:
    """Divide an integer row by the gcd of its entries; return object-dtype array.

    Delegates to ``integer_normal_forms._row_gcd_reduce``.
    """
    return _row_gcd_reduce_int(row)


def _check_lattice_metric(metric: np.ndarray | None) -> None:
    """Reject non-cubic metric inputs reserved for a later extension."""
    if metric is not None:
        raise ExactCSLNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )


def _primitive_plane(h: ArrayLike) -> Int3:
    """Return a primitive integer plane covector."""
    vec = list(_as_int_vector(h, 3, "h"))
    gcd_value = math.gcd(*[abs(v) for v in vec])
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
    quat = (
        normalize_integer_quaternion(q)
        if canonicalize
        else _as_int_vector(q, 4, "q")
    )
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
    M = _as_int_array(M_in, (3, 3), "M_in")
    expected_N = None if N is None else int(N)
    if N is not None and expected_N != N:
        raise ExactCSLValueError(f"N must be an integer; got {N!r}.")

    if reduce_common_factor:
        gcd_value = math.gcd(*[abs(int(v)) for v in M.flat])
        if gcd_value > 1:
            if expected_N is not None and expected_N % gcd_value != 0:
                raise ExactCSLValueError(
                    "common matrix factor does not divide the supplied N."
                )
            M = np.array(
                [int(value) // gcd_value for value in M.flat],
                dtype=object,
            ).reshape(3, 3)
            if expected_N is not None:
                expected_N //= gcd_value

    gram = M @ M.T
    diagonal = [int(gram[i, i]) for i in range(3)]
    if diagonal[0] <= 0 or diagonal[1:] != diagonal[:1] * 2:
        raise ExactCSLValueError(
            "M @ M.T does not have equal positive diagonal entries."
        )
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
    post_reduce: ReductionMode = "none",
) -> CSLResult:
    """Construct a canonical CSL basis from an exact scaled rotation.

    :param rotation: Validated scaled rotation (from ``quaternion_to_scaled_rotation``).
    :param post_reduce: Optional post-reduction mode. ``"none"`` returns the raw
        SNF-derived basis; ``"lll"`` applies Lenstra-Lenstra-Lovasz reduction to
        yield shorter basis vectors.
    :return: Complete CSL construction result.
    :raises ExactCSLValueError: On invalid input or if exact verification fails.
    :raises ExactCSLBackendError: If the internal SNF or HNF computation fails.
    """
    if post_reduce not in ("none", "lll"):
        raise ExactCSLValueError(f"unknown post_reduce mode {post_reduce!r}.")

    M = _as_int_array(rotation.M, (3, 3), "rotation.M")
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


def _primitive_null_coefficients(covector: ArrayLike) -> np.ndarray:
    """Return primitive integer columns spanning ``covector @ x == 0``."""
    try:
        return primitive_integer_null_basis_3d(covector)
    except ExactNormalFormError as exc:
        raise ExactCSLValueError(str(exc)) from exc


def _saturate_coefficients(coeffs: np.ndarray, covector: np.ndarray) -> np.ndarray:
    """Replace a nonprimitive rank-two coefficient basis with a primitive one."""
    cross = np.cross(coeffs[:, 0].astype(int), coeffs[:, 1].astype(int)).astype(object)
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
    cross = np.cross(B[:, 0].astype(int), B[:, 1].astype(int))
    if not any(v != 0 for v in cross):
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
    C = _as_int_array(csl_basis, (3, 3), "csl_basis")
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
        cross = np.cross(coeffs[:, 0].astype(int), coeffs[:, 1].astype(int))
        if not any(v != 0 for v in cross):
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
    basis = _as_int_array(inplane_basis, (3, 2), "inplane_basis")
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
    C = _as_int_array(csl_basis, (3, 3), "csl_basis")
    sigma_int = int(sigma)
    if sigma_int != sigma or sigma_int <= 0:
        raise ExactCSLValueError(f"sigma must be a positive integer; got {sigma!r}.")
    det = _det3(C)
    if abs(det) != sigma_int:
        raise ExactCSLValueError(
            f"|det(csl_basis)|={abs(det)} does not equal sigma={sigma_int}."
        )
    numerator = np.array(_inf_int_adj3(C), dtype=object)
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
    M = _as_int_array(rotation.M, (3, 3), "rotation.M")
    C = _as_int_array(csl_basis, (3, 3), "csl_basis")
    N = int(rotation.N)
    if N <= 0:
        raise ExactCSLValueError(f"rotation.N must be positive; got {rotation.N!r}.")
    residual = (M @ C) % N
    det_basis = abs(_det3(C))
    ok = not any(value != 0 for value in residual.flat)
    if sigma is not None:
        expected = int(sigma)
        if expected != sigma or expected <= 0:
            raise ExactCSLValueError(
                f"sigma must be a positive integer; got {sigma!r}."
            )
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
    M = _as_int_array(rotation.M, (3, 3), "rotation.M")
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
        P = np.array([_row_gcd_reduce_object(row) for row in P], dtype=object)
        Q = np.array([_row_gcd_reduce_object(row) for row in Q], dtype=object)
    return P, Q


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """Return an LLL-reduced basis spanning the same lattice as the columns of B.

    Applies the Lenstra-Lenstra-Lovasz algorithm to find a basis of short,
    nearly-orthogonal vectors.  For a 3-column input the algorithm terminates
    in O(log ||B||) iterations.

    The output columns are the same integer lattice as the input but with
    shorter, more orthogonal basis vectors.  The transformation is exact:
    integer column operations are tracked throughout.

    :param B: Full-rank 3 by 3 integer matrix whose columns are basis vectors.
    :param delta: Lovasz condition parameter in ``(0.25, 1.0]``.  Smaller
        values allow more aggressive swaps; ``delta=0.75`` is the classical
        Lenstra-Lenstra-Lovasz choice.
    :return: LLL-reduced 3 by 3 integer matrix (object dtype).
    :raises ExactCSLValueError: If ``delta`` is out of range or B is singular.
    """
    if not (0.25 < float(delta) <= 1.0):
        raise ExactCSLValueError("delta must be in the interval (0.25, 1.0].")
    M = _as_int_array(B, (3, 3), "B")
    if _det3(M) == 0:
        raise ExactCSLValueError(
            "lll_reduce requires a full-rank (non-singular) basis."
        )

    # Work with float column vectors; all reduction steps use integer rounding
    # so the result is guaranteed to be integer-valued.
    vecs = [M[:, i].astype(float) for i in range(3)]

    def _gram_schmidt():
        """Return orthogonal basis (bstar) and Gram-Schmidt coefficients (mu)."""
        bstar = [vecs[0].copy()]
        mu = [[0.0] * 3 for _ in range(3)]
        for i in range(1, 3):
            proj = vecs[i].copy()
            for j in range(i):
                denom = np.dot(bstar[j], bstar[j])
                mu[i][j] = (np.dot(vecs[i], bstar[j]) / denom) if denom > 1e-30 else 0.0
                proj -= mu[i][j] * bstar[j]
            bstar.append(proj)
        return bstar, mu

    k = 1
    while k < 3:
        bstar, mu = _gram_schmidt()

        # Size reduction: replace vecs[k] with vecs[k] - round(mu[k][j]) * vecs[j].
        for j in range(k - 1, -1, -1):
            r = round(mu[k][j])
            if r != 0:
                vecs[k] = vecs[k] - float(r) * vecs[j]
                bstar, mu = _gram_schmidt()

        # Lovasz condition: check if swap of vecs[k] and vecs[k-1] is needed.
        bk_sq = np.dot(bstar[k], bstar[k])
        bkm1_sq = np.dot(bstar[k - 1], bstar[k - 1])
        if bkm1_sq > 1e-30 and bk_sq >= (delta - mu[k][k - 1] ** 2) * bkm1_sq:
            k += 1
        else:
            vecs[k], vecs[k - 1] = vecs[k - 1].copy(), vecs[k].copy()
            k = max(k - 1, 1)

    return np.column_stack([np.round(v).astype(int) for v in vecs]).astype(object)


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
            fractions.append(
                Fraction(float(value) / float(arr[ref])).limit_denominator(
                    max_denominator
                )
            )
    denominator_lcm = 1
    for value in fractions:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    ints = [int(value * denominator_lcm) for value in fractions]
    return normalize_integer_quaternion(tuple(ints))


# ---------------------------------------------------------------------------
# Internal helpers for canonicalization
# ---------------------------------------------------------------------------


def _row_gcd_reduce(row: np.ndarray) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components.

    Thin float-casting wrapper over ``integer_normal_forms._row_gcd_reduce``,
    which returns object dtype.  GBMaker's rotation-matrix pipeline expects
    float arrays, so the conversion is done here rather than at every call site.
    """
    return _row_gcd_reduce_int(np.round(row).astype(int)).astype(float)


def _assert_integer_rows(M: np.ndarray, name: str) -> None:
    """Raise BoundarySpecError if any row of M is not close to integer-valued."""
    for i, row in enumerate(M):
        if not np.allclose(row, np.round(row), atol=1e-9, rtol=0.0):
            raise BoundarySpecError(
                f"{name} row {i} {row} is not integer-valued. "
                "P/Q rows must be integer Miller indices."
            )


def _first_nonzero_sign(row: np.ndarray) -> int:
    """Return the sign of the first nonzero component, or 0 for an all-zero row."""
    for v in row:
        if v != 0:
            return 1 if v > 0 else -1
    return 0


def _canonical_inplane_key(row: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Return the deterministic sort key for an in-plane orientation row."""
    row_int = np.asarray(row, dtype=int)
    if _first_nonzero_sign(row_int) < 0:
        row_int = -row_int
    return _dot_int(row_int, row_int), tuple(int(v) for v in row_int)


def _gauss_reduce_2d(
    v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lagrange 2D lattice reduction using integer arithmetic.

    Returns (shorter, longer).

    Inputs are expected to be integer-valued (as produced by _row_gcd_reduce).
    Integer dot products and integer rounding are used throughout to avoid
    floating-point precision loss for large Miller indices.

    .. note::
        Shortness is measured by the Euclidean inner product, which assumes an
        orthonormal basis. For non-cubic systems, pass vectors in Cartesian
        coordinates; passing raw Miller-index vectors in a non-cubic cell will
        give an incorrect reduced basis.
    """
    a = np.round(v1).astype(int)
    b = np.round(v2).astype(int)
    for _ in range(200):
        aa = _dot_int(a, a)
        bb = _dot_int(b, b)
        if bb < aa:
            a, b = b, a
            aa = bb
        if aa == 0:
            break
        ab = _dot_int(a, b)
        t = (ab + aa // 2) // aa
        if t == 0:
            break
        b = b - t * a
    else:
        warnings.warn(
            "Gauss reduction did not converge in 200 iterations; "
            "result may be unreduced",
            stacklevel=3,
        )
    return a, b


def _canonicalize_matrix(M: np.ndarray) -> np.ndarray:
    """Return the canonical form of a single 3x3 orientation matrix.

    Sign convention:
    - Row 0 (boundary normal): first nonzero component positive.
      Compensating negation of row 2 preserves the determinant.
    - Row 1 (first in-plane direction): first nonzero component positive.
      Compensating negation of row 2 preserves the determinant.
    - Row 2 sign: absorbs all compensating negations above, then a final
      determinant check ensures right-handedness.  Row 2 is never given
      an independent sign convention; its sign is fully derived.
    """
    # GCD-reduce each row
    rows = [_row_gcd_reduce(M[i]) for i in range(3)]

    # Gauss-reduce the in-plane rows (1 and 2), then GCD-reduce each.
    # The reduction may swap rows, which can flip the determinant.
    r1, r2 = _gauss_reduce_2d(rows[1], rows[2])
    rows[1] = _row_gcd_reduce(r1)
    rows[2] = _row_gcd_reduce(r2)

    # Deterministic in-plane row ordering: put the row with the larger
    # (norm_sq, canonical_sign_tuple) key into rows[1].  "Canonical sign"
    # negates a row whose first nonzero component is negative so that the lex
    # comparison is independent of sign convention.  Sorting here -- before any
    # sign fixing -- ensures equivalent in-plane bases that differ only by row
    # order produce the same canonical matrix.
    if _canonical_inplane_key(rows[1]) < _canonical_inplane_key(rows[2]):
        rows[1], rows[2] = rows[2], rows[1]

    # Fix row 0 sign: negate rows 0 AND 2 together so det is unchanged.
    if _first_nonzero_sign(rows[0]) < 0:
        rows[0] = -rows[0]
        rows[2] = -rows[2]

    # Fix row 1 sign: negate rows 1 AND 2 together so det is unchanged.
    if _first_nonzero_sign(rows[1]) < 0:
        rows[1] = -rows[1]
        rows[2] = -rows[2]

    # Final right-handedness check using integer triple product to avoid
    # float comparison.  Gauss reduction may have swapped rows (flipping
    # det), and the two compensating negations above may have cancelled.
    det_sign = int(np.dot(
        rows[0].astype(int),
        np.cross(rows[1].astype(int), rows[2].astype(int)),
    ))
    if det_sign < 0:
        rows[2] = -rows[2]

    return np.array(rows, dtype=float)


def _gauss_reduce_2d_paired(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-reduce P in-plane rows while applying the same row ops to Q.

    All four inputs must be 1D integer-valued arrays of the same length.
    The integer contract is enforced by the caller (``_canonicalize_pq_paired``
    calls ``_assert_integer_rows`` on P and Q before slicing into rows).
    Dot products use ``_dot_int`` to avoid int64 overflow for large entries.

    :param p1: First in-plane row of P.
    :param p2: Second in-plane row of P.
    :param q1: First in-plane row of Q (paired with p1).
    :param q2: Second in-plane row of Q (paired with p2).
    :return: ``(p1_reduced, p2_reduced, q1_reduced, q2_reduced)``.
    :raises BoundarySpecError: If any input is not a 1D array or if the
        input arrays have incompatible shapes.
    """
    for name, v in [("p1", p1), ("p2", p2), ("q1", q1), ("q2", q2)]:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise BoundarySpecError(
                f"_gauss_reduce_2d_paired: {name} must be a 1D array; "
                f"got shape {arr.shape}."
            )
    lengths = {np.asarray(v).shape[0] for v in (p1, p2, q1, q2)}
    if len(lengths) != 1:
        raise BoundarySpecError(
            "_gauss_reduce_2d_paired: p1, p2, q1, q2 must all have the same "
            f"length; got lengths {[np.asarray(v).shape[0] for v in (p1, p2, q1, q2)]}."
        )
    a = np.round(p1).astype(int)
    b = np.round(p2).astype(int)
    qa = np.round(q1).astype(int)
    qb = np.round(q2).astype(int)
    for _ in range(200):
        aa = _dot_int(a, a)
        bb = _dot_int(b, b)
        if bb < aa:
            a, b = b, a
            qa, qb = qb, qa
            aa = bb
        if aa == 0:
            break
        ab = _dot_int(a, b)
        t = (ab + aa // 2) // aa
        if t == 0:
            break
        b = b - t * a
        qb = qb - t * qa
    else:
        warnings.warn(
            "Paired Gauss reduction did not converge in 200 iterations; "
            "result may be unreduced",
            stacklevel=3,
        )
    return a, b, qa, qb


def _canonicalize_pq_paired(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize the orientation convention for an ordered P/Q bicrystal.

    P is treated as the reference grain and Q is transformed only by paired
    row swaps/sign flips so that P/Q row correspondence is preserved. This is
    not a canonical representative of the physical grain-boundary equivalence
    class (grain exchange, crystal symmetry, and translation equivalences are
    not handled).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    _assert_integer_rows(P, "P")
    _assert_integer_rows(Q, "Q")

    p_rows = [
        _row_gcd_reduce(P[0]),
        np.round(P[1]).astype(int).astype(float),
        np.round(P[2]).astype(int).astype(float),
    ]
    q_rows = [
        _row_gcd_reduce(Q[0]),
        np.round(Q[1]).astype(int).astype(float),
        np.round(Q[2]).astype(int).astype(float),
    ]

    p1, p2, q1, q2 = _gauss_reduce_2d_paired(
        p_rows[1], p_rows[2], q_rows[1], q_rows[2]
    )
    # GCD-reduce each in-plane row independently after Gauss reduction,
    # matching _canonicalize_matrix, so scaled-but-equivalent inputs produce
    # identical canonical output (e.g. [2,0,0] -> [1,0,0]).  Independent
    # reduction is valid because direction indices carry no meaningful scaling.
    p_rows[1] = _row_gcd_reduce(p1)
    p_rows[2] = _row_gcd_reduce(p2)
    q_rows[1] = _row_gcd_reduce(q1)
    q_rows[2] = _row_gcd_reduce(q2)

    if _canonical_inplane_key(p_rows[1]) < _canonical_inplane_key(p_rows[2]):
        p_rows[1], p_rows[2] = p_rows[2], p_rows[1]
        q_rows[1], q_rows[2] = q_rows[2], q_rows[1]

    if _first_nonzero_sign(p_rows[0]) < 0:
        p_rows[0] = -p_rows[0]
        p_rows[2] = -p_rows[2]
        q_rows[0] = -q_rows[0]
        q_rows[2] = -q_rows[2]

    if _first_nonzero_sign(p_rows[1]) < 0:
        p_rows[1] = -p_rows[1]
        p_rows[2] = -p_rows[2]
        q_rows[1] = -q_rows[1]
        q_rows[2] = -q_rows[2]

    det_sign = int(np.dot(
        p_rows[0].astype(int),
        np.cross(p_rows[1].astype(int), p_rows[2].astype(int)),
    ))
    if det_sign < 0:
        p_rows[2] = -p_rows[2]
        q_rows[2] = -q_rows[2]

    P_canon = np.array(p_rows, dtype=float)
    Q_canon = np.array(q_rows, dtype=float)
    for name, M in [("P", P_canon), ("Q", Q_canon)]:
        if any(not np.any(row) for row in M):
            raise BoundarySpecError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )
    return P_canon, Q_canon


def _inplane_area_index(P: np.ndarray) -> int:
    """Return the integer area index of P's in-plane rows in P[0]'s plane.

    Validates that P[1] and P[2] are actually in the plane defined by P[0]
    (i.e., ``dot(P[1], plane) == 0`` and ``dot(P[2], plane) == 0``).  Also
    validates that all entries are close to integers before rounding.

    :param P: 3x3 integer-valued orientation matrix (row 0 = plane normal,
        rows 1-2 = in-plane vectors).
    :return: Positive integer area index.
    :raises BoundarySpecError: If P[0] is zero, P[1]/P[2] are not in-plane,
        rows are not integer-valued, or the area index is zero.
    """
    P_arr = np.asarray(P, dtype=float)
    if not np.allclose(P_arr, np.round(P_arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecError(
            "P rows must be integer-valued for area-index computation; "
            f"got non-integer entries in P={P_arr.tolist()}."
        )
    P_int = np.round(P_arr).astype(int)
    plane = _row_gcd_reduce(P_int[0]).astype(int)
    denom = _dot_int(plane, plane)
    if denom == 0:
        raise BoundarySpecError("Cannot compute area index for a zero boundary plane.")
    for row_idx in (1, 2):
        proj = _dot_int(P_int[row_idx], plane)
        if proj != 0:
            raise BoundarySpecError(
                f"P row {row_idx} {P_int[row_idx].tolist()} is not in the boundary "
                f"plane {plane.tolist()} (dot product = {proj}, expected 0). "
                "P[1] and P[2] must be integer lattice vectors lying in the plane "
                "defined by the primitive normal P[0]."
            )
    cross = np.cross(P_int[1], P_int[2])
    numer = abs(_dot_int(cross, plane))
    if numer % denom != 0:
        raise BoundarySpecError(
            "In-plane rows do not define an integer area index for the boundary plane."
        )
    index = numer // denom
    if index == 0:
        raise BoundarySpecError(
            "In-plane area index is zero; P[1] and P[2] may be parallel or zero."
        )
    return int(index)


def _primitive_metadata(
    *,
    basis_mode: str,
    supplied_area_index: int,
    primitive_area_index: int,
    plane: np.ndarray,
    rotation_denominator: int,
) -> PrimitiveCellMetadata:
    if supplied_area_index % primitive_area_index == 0:
        reduction_index = supplied_area_index // primitive_area_index
    else:
        raise BoundarySpecError(
            "supplied_area_index must be an integer multiple of "
            "primitive_area_index when reporting primitive-cell metadata; "
            f"got supplied_area_index={supplied_area_index}, "
            f"primitive_area_index={primitive_area_index}."
        )
    return PrimitiveCellMetadata(
        basis_mode=basis_mode,  # type: ignore[arg-type]
        supplied_area_index=int(supplied_area_index),
        primitive_area_index=int(primitive_area_index),
        reduction_index=int(reduction_index),
        plane=tuple(int(x) for x in plane),
        rotation_denominator=int(rotation_denominator),
        conventional_cell_multiplier=int(2 * primitive_area_index),
    )


def _recover_row_rotation_from_pq(P: np.ndarray, Q: np.ndarray) -> ScaledRotation:
    """Recover exact row-convention scaled rotation from paired P/Q rows."""
    _assert_integer_rows(P, "P")
    _assert_integer_rows(Q, "Q")
    P_int = np.round(P).astype(object)
    Q_int = np.round(Q).astype(object)
    det_P = _int_det3(P_int)
    if det_P == 0:
        raise BoundarySpecError("Cannot recover rotation from singular P matrix.")
    adj_P = np.asarray(_int_adj3(P_int), dtype=object)
    numerator = adj_P @ Q_int
    denominator = int(det_P)
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator

    fractions = [
        Fraction(int(value), denominator)
        for value in np.asarray(numerator, dtype=object).flat
    ]
    scale = 1
    for value in fractions:
        scale = math.lcm(scale, value.denominator)
    M = np.array(
        [int(value * scale) for value in fractions],
        dtype=object,
    ).reshape(3, 3)
    try:
        return validate_scaled_rotation_matrix(M, N=scale)
    except ExactCSLError as exc:
        raise BoundarySpecError(
            "P/Q paired rows do not recover an exact proper rotation."
        ) from exc


def _row_rotation_image(
    row: np.ndarray,
    rotation: ScaledRotation,
    *,
    require_divisible: bool,
) -> np.ndarray:
    """Apply a row-convention scaled rotation to an integer row."""
    row_obj = np.asarray(row, dtype=object)
    M = np.asarray(rotation.M, dtype=object)
    numerator = row_obj @ M
    N = int(rotation.N)
    if all(int(value) % N == 0 for value in numerator):
        return np.array([int(value) // N for value in numerator], dtype=int)
    if require_divisible:
        raise BoundarySpecError(
            "Primitive in-plane CSL vector image is not integer-valued under "
            "the recovered rotation."
        )
    return _row_gcd_reduce(numerator).astype(int)


def _primitive_embedding_from_row_rotation(
    row_rotation: ScaledRotation,
    plane: np.ndarray,
    *,
    source: str,
    supplied_area_index: int | None = None,
    max_exact_atoms: int | None = None,
) -> BoundaryEmbedding:
    """Build a primitive paired P/Q embedding from a row-convention rotation."""
    plane_int = _row_gcd_reduce(np.round(plane).astype(int)).astype(int)
    try:
        column_rotation = validate_scaled_rotation_matrix(
            np.asarray(row_rotation.M, dtype=object).T,
            N=row_rotation.N,
        )
        csl = csl_from_scaled_rotation(column_rotation)
        inplane = inplane_basis_from_csl(
            csl.basis_hnf,
            tuple(int(x) for x in plane_int),
        )
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    p1 = np.asarray(inplane.basis[:, 0], dtype=int)
    p2 = np.asarray(inplane.basis[:, 1], dtype=int)
    q0 = _row_rotation_image(plane_int, row_rotation, require_divisible=False)
    q1 = _row_rotation_image(p1, row_rotation, require_divisible=True)
    q2 = _row_rotation_image(p2, row_rotation, require_divisible=True)

    P_raw = np.array([plane_int, p1, p2], dtype=float)
    Q_raw = np.array([q0, q1, q2], dtype=float)
    P_canon, Q_canon = _canonicalize_pq_paired(P_raw, Q_raw)
    primitive_area_index = _inplane_area_index(P_canon)
    if max_exact_atoms is not None and primitive_area_index > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL area index ({primitive_area_index}) exceeds "
            f"max_exact_atoms={max_exact_atoms}. Use mode='approximate' or "
            "increase the limit."
        )

    supplied_index = (
        primitive_area_index
        if supplied_area_index is None
        else int(supplied_area_index)
    )
    metadata = _primitive_metadata(
        basis_mode="primitive",
        supplied_area_index=supplied_index,
        primitive_area_index=primitive_area_index,
        plane=plane_int,
        rotation_denominator=int(row_rotation.N),
    )

    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, R in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(R) - 1.0) < 1e-10):
            raise BoundarySpecOrthogonalityError(
                f"{r_name} derived from primitive {source} input is not a "
                "proper rotation matrix (R @ R.T != I or det != 1)."
            )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source=source,
        metadata=metadata,
    )


def _warn_pq_primitive_fallback(spec: PQSpec, reason: BoundarySpecError) -> None:
    """Warn that primitive PQ reconstruction failed before legacy fallback."""
    P_list = np.asarray(spec.P, dtype=float).tolist()
    Q_list = np.asarray(spec.Q, dtype=float).tolist()
    warnings.warn(
        "PQSpec with basis_mode='primitive' could not reconstruct a primitive "
        "in-plane CSL basis; falling back to supplied-basis canonicalization. "
        f"P={P_list}, Q={Q_list}. Reason: {reason}",
        UserWarning,
        stacklevel=3,
    )

# ---------------------------------------------------------------------------
# Integer membership kernel for exact supercell construction
# ---------------------------------------------------------------------------


def _int_det3(M) -> int:
    """Compute the determinant of a 3 by 3 integer matrix.

    Delegates to ``integer_normal_forms._det3`` after validating integer entries.
    Translates ``ExactNormalFormError`` to ``ValueError`` for backward compatibility.

    :param M: 3 by 3 array-like with integer-valued entries.
    :return: Integer determinant.
    :raises ValueError: If any entry is not integer-valued.
    """
    try:
        return _inf_det3(_inf_as_int_matrix(M, (3, 3), "M"))
    except ExactNormalFormError as exc:
        raise ValueError(str(exc)) from exc


def _int_adj3(M) -> list:
    """Compute the adjugate of a 3 by 3 integer matrix as a list-of-lists.

    Delegates to ``integer_normal_forms._int_adj3``.
    Translates ``ExactNormalFormError`` to ``ValueError`` for backward compatibility.

    :param M: 3 by 3 array-like with integer-valued entries.
    :return: 3 by 3 list-of-lists representing adj(M).
    :raises ValueError: If any entry is not integer-valued.
    """
    try:
        return _inf_int_adj3(M)
    except ExactNormalFormError as exc:
        raise ValueError(str(exc)) from exc


def _integer_membership(
    n,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Test whether integer conventional-cell origin *n* lies inside the repeated
    supercell.

    Fractional supercell coordinates are ``u = n @ S^-1 = (n @ adj(S)) / det(S)``.
    Origin *n* is accepted when ``0 <= u[i] < repeat[i]`` for each axis, which in
    integer arithmetic becomes ``0 <= u_num[i] < repeat[i] * |det(S)|``, where
    ``u_num = n @ adj(S)`` (sign-flipped when ``det(S) < 0`` so the inequality
    direction is preserved).

    The exact construction path always produces ``det(S) > 0``; the sign-flip
    branch supports the general helper for testing purposes.

    :param n: Integer 3-vector (conventional-cell coordinates of the origin).
    :param adj_S: Adjugate of S as a 3x3 list-of-lists (from ``_int_adj3``).
    :param det_S: Integer determinant of S (from ``_int_det3``).
    :param repeat_x: Number of repeats along the x (boundary-normal) direction.
    :param repeat_y: Number of repeats along the y (in-plane) direction.
    :param repeat_z: Number of repeats along the z (in-plane) direction.
    :return: True if *n* is accepted.
    """
    abs_det = abs(det_S)
    # u_num[j] = sum_k n[k] * adj_S[k][j]   (row-vector @ matrix)
    u_num = [
        sum(int(n[k]) * adj_S[k][j] for k in range(3))
        for j in range(3)
    ]
    if det_S < 0:
        u_num = [-u for u in u_num]
    return (
        0 <= u_num[0] < repeat_x * abs_det
        and 0 <= u_num[1] < repeat_y * abs_det
        and 0 <= u_num[2] < repeat_z * abs_det
    )


def build_supercell_matrix(P: np.ndarray) -> np.ndarray:
    """Build the integer supercell matrix *S* = [s0; s1; s2] from canonical P.

    For a canonical orientation matrix P whose rows have already been
    GCD-reduced and made right-handed by ``canonicalize_pq``:
    - ``s1 = P[1]`` -- in-plane period along lab y (integer Miller indices)
    - ``s2 = P[2]`` -- in-plane period along lab z
    - ``s0 = gcd_reduce(cross(s1, s2))`` -- boundary-normal stacking period

    For canonical right-handed P, ``cross(s1, s2)`` is parallel to ``P[0]``
    and after GCD reduction equals it exactly, so *S* = P.

    A clear error is raised if S is non-integer or singular (det = 0).

    :param P: 3x3 canonical orientation matrix (integer-valued rows).
    :return: 3x3 integer ndarray S with rows [s0, s1, s2].
    :raises BoundarySpecError: If P rows are not integer-valued.
    :raises ValueError: If the resulting S is singular (det = 0).
    """
    _assert_integer_rows(P, "P (supercell matrix)")
    S = np.round(P).astype(int)
    det = _int_det3(S)
    if det == 0:
        raise ValueError(
            f"Supercell matrix S derived from P is singular (det=0). "
            f"P = {P.tolist()}. The in-plane rows P[1], P[2] must be "
            "linearly independent."
        )
    return S


def enumerate_supercell_origins(
    S: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> np.ndarray:
    """Enumerate all integer conventional-cell origins inside the repeated supercell.

    The repeated supercell is spanned by ``repeat_x*s0``, ``repeat_y*s1``,
    ``repeat_z*s2``.  Candidates are drawn from the integer bounding box of the
    8 parallelepiped corners, padded by one lattice step.  Membership is tested
    with ``_integer_membership`` -- no floating-point selection is used.

    :param S: 3x3 integer supercell matrix (rows = s0, s1, s2).
    :param repeat_x: Number of repeats along s0.
    :param repeat_y: Number of repeats along s1.
    :param repeat_z: Number of repeats along s2.
    :return: Array of shape (N, 3) of accepted integer origins, where
        ``N == repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If the accepted count does not match the expected value.
    """
    S_int = np.round(S).astype(int)
    s0 = S_int[0]
    s1 = S_int[1]
    s2 = S_int[2]
    det_S = _int_det3(S_int)
    adj_S = _int_adj3(S_int)

    # Bounding box from the 8 parallelepiped corners
    corners = np.array([
        i * repeat_x * s0 + j * repeat_y * s1 + k * repeat_z * s2
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ], dtype=int)
    lo = corners.min(axis=0) - 1
    hi = corners.max(axis=0) + 1

    ranges = [np.arange(lo[d], hi[d] + 1) for d in range(3)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)

    accepted = [
        tuple(row)
        for row in grid
        if _integer_membership(row, adj_S, det_S, repeat_x, repeat_y, repeat_z)
    ]

    expected = repeat_x * repeat_y * repeat_z * abs(det_S)
    if len(accepted) != expected:
        raise ValueError(
            f"enumerate_supercell_origins: expected {expected} origins "
            f"(repeat={repeat_x},{repeat_y},{repeat_z}, |det|={abs(det_S)}) "
            f"but got {len(accepted)}.  S = {S_int.tolist()}"
        )
    return np.array(accepted, dtype=int)


def validate_and_normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate that quat is an integer quaternion and return its normalized form.

    A rotation quaternion encodes a rotation by angle theta about a unit axis
    n_hat = (nx, ny, nz) in Hamilton scalar-first order ``[w, x, y, z]``,
    where ``w = cos(theta/2)`` and ``(x, y, z) = sin(theta/2) * n_hat``. For a CSL
    grain boundary the rotation angle is rational, so the quaternion
    components can be exact integers; the unit quaternion is obtained by
    dividing by the norm.

    :param quat: Candidate integer quaternion in Hamilton order [w, x, y, z],
        shape (4,).
    :returns: Normalized (unit-length) quaternion as ``np.ndarray`` of shape
        (4,) with dtype float.
    :raises BoundarySpecError: If any component is non-integer or the
        quaternion is zero.
    """
    arr = np.asarray(quat, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != 4:
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {arr.shape}."
        )
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecError(
            f"Quaternion components must be integer-valued; got {arr}. "
            "CSLExactSpec requires an integer quaternion [a, b, c, d]."
        )
    int_q = np.round(arr).astype(int)
    norm_sq = int(np.dot(int_q, int_q))
    if norm_sq == 0:
        raise BoundarySpecError(
            "Quaternion is the zero vector; a non-zero integer quaternion is required."
        )
    return arr / np.sqrt(float(norm_sq))


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Delegates to ``scipy.spatial.transform.Rotation`` using scalar-last order
    internally; the reordering is handled here so callers always use Hamilton
    scalar-first order. The quaternion must already be normalized -- call
    ``validate_and_normalize_quaternion`` first.

    :param quat: Normalized unit quaternion in Hamilton scalar-first order
        [w, x, y, z], shape (4,).
    :returns: Rotation matrix ``R`` of shape (3, 3) such that
        ``v_rotated = v @ R.T`` (row-vector convention).
    """
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {q.shape}."
        )
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-12, rtol=0.0):
        raise BoundarySpecError(
            "Quaternion must be normalized before conversion; call "
            "validate_and_normalize_quaternion first."
        )
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def validate_sigma(quat: np.ndarray, sigma: int) -> None:
    """Validate that sigma derived from quat matches the user-supplied value.

    Sigma (Sigma) is the reciprocal density of coincidence sites for a CSL grain
    boundary. For an integer quaternion ``q = [w, x, y, z]``, sigma is the
    odd part of ``N = w^2 + x^2 + y^2 + z^2`` -- that is, ``N`` divided by its
    largest power-of-2 factor. For example: ``q = [2, 0, 0, 1]`` gives
    ``N = 5``, so ``sigma = 5``; ``q = [3, 0, 0, 1]`` gives ``N = 10 = 2*5``,
    so ``sigma = 5``. Validation is an exact integer equality check.

    :param quat: Integer quaternion (unnormalized) in Hamilton order
        [w, x, y, z], shape (4,).
    :param sigma: Expected sigma value to validate against.
    :raises BoundarySpecError: If the derived sigma does not match.
    """
    arr = np.asarray(quat, dtype=float)
    if arr.shape != (4,):
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {arr.shape}."
        )
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecError(
            f"Quaternion components must be integer-valued; got {arr}. "
            "CSLExactSpec requires an integer quaternion [a, b, c, d]."
        )
    int_q = np.round(arr).astype(int)
    if int(np.dot(int_q, int_q)) == 0:
        raise BoundarySpecError(
            "Quaternion is zero; sigma cannot be derived from a zero quaternion."
        )
    try:
        rot = quaternion_to_scaled_rotation(tuple(int(x) for x in int_q))
        csl = csl_from_scaled_rotation(rot)
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    derived = csl.sigma
    if derived != int(sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {int_q.tolist()} gives "
            f"sigma={derived}, but sigma={sigma} was provided."
        )


def _recover_sigma_from_rotation(R: np.ndarray, max_sigma: int = 10001) -> int:
    """Return N such that N*R is an integer matrix (N = sigma or a multiple).

    For a rotation matrix produced from an integer quaternion [w,x,y,z] with
    norm-squared N = w^2+x^2+y^2+z^2, every entry of R is a rational number whose
    denominator (in lowest terms) divides N.  We recover N as the LCM of the
    denominators of all matrix entries, using ``Fraction.limit_denominator`` to
    convert each floating-point entry to its exact rational form.

    This is O(9) -- one Fraction call per matrix entry -- rather than a linear
    search up to max_sigma.

    :param R: 3x3 rotation matrix whose entries are rationals with
        denominator <= max_sigma.
    :param max_sigma: Upper bound passed to ``limit_denominator``; the search
        raises if the true denominator exceeds this value.
    :return: LCM of all entry denominators = the common scaling factor N.
    :raises BoundarySpecError: If any entry's denominator exceeds max_sigma, which
        indicates R was not produced from an integer quaternion with sigma < max_sigma.
    """
    denom = 1
    for entry in R.flat:
        if abs(entry) > 1e-10:
            frac = Fraction(entry).limit_denominator(max_sigma)
            denom = math.lcm(denom, frac.denominator)
    if denom == 0:
        raise BoundarySpecError(
            "Could not recover sigma from rotation matrix (all entries near zero)."
        )
    return denom


def _plane_null_basis(
    plane_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a primitive integer basis for the null space of ``plane_int``.

    Finds ``e1``, ``e2`` in Z^3 such that:

    * ``dot(plane_int, e1) == 0`` and ``dot(plane_int, e2) == 0``
      (both vectors lie in the boundary plane), and
    * ``cross(e1, e2) == plane_int`` (the basis spans the full integer
      plane lattice without gaps).

    The construction applies unimodular column operations to ``[h, k, l]``
    until it becomes ``[1, 0, 0]``, tracking the transformations in V in
    GL_3(Z).  Because V is unimodular, columns 1 and 2 of V are exactly the
    primitive null vectors.

    :param plane_int: Primitive boundary-plane normal as an integer 3-vector.
        Must be non-zero and GCD-reduced (``gcd(|h|, |k|, |l|) == 1``).
        Callers are responsible for reducing via ``_row_gcd_reduce`` first.
    :return: ``(e1, e2)`` as float arrays.
    :raises ValueError: If ``plane_int`` is the zero vector or not primitive.
    """
    vec = np.array(
        [int(plane_int[0]), int(plane_int[1]), int(plane_int[2])], dtype=int
    )
    if not any(vec):
        raise ValueError(
            "_plane_null_basis: plane_int must not be the zero vector."
        )
    if math.gcd(math.gcd(abs(int(vec[0])), abs(int(vec[1]))), abs(int(vec[2]))) != 1:
        raise ValueError(
            f"_plane_null_basis: plane_int {vec.tolist()} is not primitive "
            "(gcd of components != 1). Call _row_gcd_reduce first."
        )

    try:
        basis = primitive_integer_null_basis_3d(vec)
    except ExactNormalFormError as exc:
        raise ValueError(str(exc)) from exc
    return basis[:, 0].astype(float), basis[:, 1].astype(float)


def solve_inplane_csl(
    axis: np.ndarray,
    plane: np.ndarray,
    R: np.ndarray,
    max_exact_atoms: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the exact in-plane CSL basis from rotation axis, boundary plane, and R.

    A CSL (Coincidence Site Lattice) vector is an integer lattice vector v of
    grain 1 that is also a lattice vector of grain 2, i.e. ``v @ R`` is an
    integer vector (row-vector convention).  This function finds the two
    shortest such vectors that lie in the boundary plane ``plane * v = 0``.

    The search works in the 2-D integer lattice spanned by the two null-space
    basis vectors of ``plane`` (see ``_plane_null_basis``).  Every candidate
    ``v = s*e1 + t*e2`` is tested against the CSL condition
    ``(v @ M_int) % N == 0`` where ``M_int = round(N*R)`` and N is the sigma
    value recovered from R.

    :param axis: Integer Miller rotation axis [u v w] (used for documentation
        only; the CSL condition is derived solely from R).
    :param plane: Integer Miller boundary-plane normal [h k l] in grain 1's
        crystal frame.
    :param R: Exact rotation matrix from ``quaternion_to_rotation_matrix``.
    :param max_exact_atoms: Guard on cell size.  Raises if the area of the
        in-plane CSL unit cell (``|v1 x v2|``) exceeds this value, which would
        produce an impractically large simulation cell.
    :return: ``(v1, v2)`` -- two linearly independent in-plane CSL vectors,
        before Gauss reduction.  Pass to ``reduce_2d_basis`` to get the
        shortest basis.
    :raises BoundarySpecError: If fewer than two independent in-plane CSL
        vectors are found within the search range, or if the cell exceeds
        ``max_exact_atoms``.
    """
    plane_int = _row_gcd_reduce(np.round(plane).astype(int))
    try:
        N = _recover_sigma_from_rotation(R)
        M_int = np.round(N * np.asarray(R, dtype=float)).astype(int)
        row_rotation = validate_scaled_rotation_matrix(M_int.T, N=N)
        csl = csl_from_scaled_rotation(row_rotation)
        inplane = inplane_basis_from_csl(
            csl.basis_hnf, tuple(int(x) for x in plane_int))
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    v1 = np.asarray(inplane.basis[:, 0], dtype=float)
    v2 = np.asarray(inplane.basis[:, 1], dtype=float)
    area = np.linalg.norm(np.cross(v1, v2))
    if area > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL cell area ({area:.1f}) exceeds max_exact_atoms="
            f"{max_exact_atoms}.  Use mode='approximate' or increase the limit."
        )
    return v1, v2


def reduce_2d_basis(
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gauss-Lagrange 2-D lattice reduction to a pair of in-plane vectors.

    Returns the shortest basis for the lattice spanned by v1 and v2, with the
    shorter vector first.  Delegates to the internal ``_gauss_reduce_2d`` helper.

    :param v1: First in-plane basis vector (integer-valued).
    :param v2: Second in-plane basis vector (integer-valued).
    :return: ``(r1, r2)`` -- reduced basis, ``||r1|| <= ||r2||``.
    """
    r1, r2 = _gauss_reduce_2d(v1, v2)
    return r1.astype(float), r2.astype(float)


def canonicalize_pq(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical forms of the P and Q orientation matrices.

    Canonicalization rules:
    - rows must be integer-valued (within 1e-9 absolute tolerance, rtol=0)
    - each row is divided by the GCD of its absolute components
    - matrices are right-handed (positive determinant after normalization)
    - row 0 is the boundary normal
    - rows 1-2 form a deterministic Gauss-reduced in-plane basis; the row
      with the larger (norm_sq, canonical-sign lex) key is placed in row 1
    - sign convention: first nonzero component of rows 0 and 1 is positive
    - equivalent inputs (same directions, different scalings, sign flips, or
      in-plane row ordering) canonicalize identically

    :param P: Row-wise orientation matrix for the left grain, shape (3, 3).
    :param Q: Row-wise orientation matrix for the right grain, shape (3, 3).
    :returns: ``(P_canon, Q_canon)`` -- canonicalized orientation matrices.
    :raises BoundarySpecError: If any row of P or Q is not integer-valued, or
        if canonicalization produces a zero row.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    _assert_integer_rows(P, "P")
    _assert_integer_rows(Q, "Q")
    P_canon = _canonicalize_matrix(P)
    Q_canon = _canonicalize_matrix(Q)
    for name, M in [("P", P_canon), ("Q", Q_canon)]:
        if any(not np.any(row) for row in M):
            raise BoundarySpecError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )
    return P_canon, Q_canon


def pq_spec_to_embedding(spec: PQSpec) -> BoundaryEmbedding:
    """Convert a validated PQSpec to a BoundaryEmbedding.

    :param spec: A validated PQSpec.
    :returns: Canonical embedding with ``exact=True``, ``coherent=True``,
        ``source="pq"``. P and Q are in canonical form. R_left and R_right
        are derived by normalizing each row of canonical P and Q to unit length.
        Equivalent PQSpecs (differing only by row scaling, sign convention, or
        in-plane basis choice) always produce identical BoundaryEmbeddings.
    :raises BoundarySpecError: If P or Q rows are not integer-valued, produce
        a zero row after canonicalization, or do not form a proper rotation
        matrix after row-normalization.
    """
    P_raw = np.asarray(spec.P, dtype=float)
    Q_raw = np.asarray(spec.Q, dtype=float)
    row_rotation = None
    if spec.basis_mode == "primitive":
        try:
            row_rotation = _recover_row_rotation_from_pq(P_raw, Q_raw)
        except BoundarySpecError as exc:
            # Existing public PQSpec examples often provide orientation rows
            # rather than paired supercell rows (for example P=I and scaled Q
            # direction rows).  Those cannot define A=inv(P)@Q as a proper
            # rotation, so preserve the legacy supplied-basis behavior.
            _warn_pq_primitive_fallback(spec, exc)
            row_rotation = None
        if row_rotation is not None:
            try:
                supplied_area_index = _inplane_area_index(P_raw)
                return _primitive_embedding_from_row_rotation(
                    row_rotation,
                    _row_gcd_reduce(np.round(P_raw[0]).astype(int)),
                    source="pq",
                    supplied_area_index=supplied_area_index,
                )
            except BoundarySpecError as exc:
                _warn_pq_primitive_fallback(spec, exc)

    if spec.basis_mode == "supplied" or row_rotation is not None:
        P_canon, Q_canon = _canonicalize_pq_paired(P_raw, Q_raw)
    else:
        P_canon, Q_canon = canonicalize_pq(P_raw, Q_raw)
    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, R in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(R) - 1.0) < 1e-10):
            raise BoundarySpecError(
                f"{r_name} derived from P/Q is not a proper rotation matrix "
                "(R @ R.T != I or det != 1). Ensure P/Q rows are mutually "
                "orthogonal integer Miller directions."
            )
    metadata = None
    if row_rotation is None:
        try:
            row_rotation = _recover_row_rotation_from_pq(P_raw, Q_raw)
        except BoundarySpecError:
            row_rotation = None
    if row_rotation is not None:
        supplied_area_index = _inplane_area_index(P_canon)
        metadata = _primitive_metadata(
            basis_mode="supplied",
            supplied_area_index=supplied_area_index,
            primitive_area_index=supplied_area_index,
            plane=_row_gcd_reduce(np.round(P_canon[0]).astype(int)),
            rotation_denominator=int(row_rotation.N),
        )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source="pq",
        metadata=metadata,
    )


def csl_spec_to_embedding(spec, max_exact_atoms: int = 10_000) -> BoundaryEmbedding:
    """Convert a validated CSLExactSpec to a BoundaryEmbedding.

    **How P and Q are constructed.**  In GBMaker's convention each row of a
    grain's orientation matrix records which crystal Miller direction aligns
    with the corresponding lab axis: row 0 = lab x (boundary normal), row 1 =
    lab y, row 2 = lab z.

    For grain 1 we fix the boundary normal (``plane``) as row 0 and fill rows
    1-2 with the two cross-product null-basis vectors of that plane (see
    ``_plane_null_basis``).  For grain 2 each row is obtained by applying the
    misorientation matrix M_int to the corresponding integer row of P and
    GCD-reducing the result::

        Q[row i] = gcd_reduce(P[row i] @ M_int)

    where ``M_int = round(N * R)`` and N is recovered from R.  This formula
    is equivalent to rotating each lab axis from grain 1's crystal frame into
    grain 2's crystal frame -- exactly what R_right encodes.  After
    ``canonicalize_pq`` the resulting matrices are identical to what a
    ``PQSpec`` with the same boundary would produce, enabling the cross-format
    round-trip test.

    :param spec: A ``CSLExactSpec`` instance (quat is required).
    :param max_exact_atoms: Passed to ``solve_inplane_csl`` as the cell-size
        guard.  Raises ``BoundarySpecError`` if the in-plane CSL cell would be
        larger than this.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``,
        ``source="csl"``.
    :raises BoundarySpecError: On invalid quaternion, sigma mismatch, missing
        CSL for the given plane, or cell too large.
    """
    if spec.quat is None:
        raise BoundarySpecError("CSLExactSpec.quat is required.")

    quat_arr = np.asarray(spec.quat, dtype=float)
    quat_norm = validate_and_normalize_quaternion(quat_arr)

    try:
        rot = quaternion_to_scaled_rotation(
            tuple(int(x) for x in np.round(quat_arr).astype(int)))
        csl = csl_from_scaled_rotation(rot)
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    if spec.sigma is not None and csl.sigma != int(spec.sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {np.round(quat_arr).astype(int).tolist()} "
            f"gives sigma={csl.sigma}, but sigma={spec.sigma} was provided."
        )

    plane_int = _row_gcd_reduce(
        np.round(np.asarray(spec.plane, dtype=float)).astype(int))
    try:
        row_rotation = validate_scaled_rotation_matrix(
            rot.M,
            N=rot.N,
            reduce_common_factor=True,
        )
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    plane_row = np.asarray(plane_int, dtype=object)
    image = plane_row @ np.asarray(row_rotation.M, dtype=object)
    preserves_plane = (
        all(int(value) % row_rotation.N == 0 for value in image)
        and np.array_equal(
            _row_gcd_reduce(
                np.array([int(value) // row_rotation.N for value in image], dtype=int)
            ),
            plane_int,
        )
    )
    if preserves_plane:
        primitive_embedding = None
        try:
            primitive_embedding = _primitive_embedding_from_row_rotation(
                row_rotation,
                plane_int,
                source="csl",
                max_exact_atoms=max_exact_atoms,
            )
        except BoundarySpecOrthogonalityError:
            # Some plane-preserving rotations (e.g. Sigma 3 [111]) have a
            # primitive in-plane CSL basis that is not an orthogonal lab y/z
            # pair.  GBMaker's exact path requires proper row-orthogonal
            # rotation matrices, so those cases fall through to the orthogonal
            # construction below.
            primitive_embedding = None
        if primitive_embedding is not None:
            return primitive_embedding

    R = quaternion_to_rotation_matrix(quat_norm)
    N = _recover_sigma_from_rotation(R)
    M_int = np.round(N * R).astype(int)

    # Find the minimal in-plane CSL basis (raises if none exists or cell is too large).
    # Use the shortest CSL in-plane vector as e1 so the max_exact_atoms guard
    # applies to the same basis that P will actually use.
    v1, v2 = solve_inplane_csl(
        np.asarray(spec.axis, dtype=float),
        np.asarray(spec.plane, dtype=float),
        R,
        max_exact_atoms=max_exact_atoms,
    )
    r1, _r2 = reduce_2d_basis(v1, v2)
    e1 = _row_gcd_reduce(r1)
    # e2 = plane_int x e1 is orthogonal to both, keeping P rows mutually
    # orthogonal so the normalized rows form a proper rotation matrix.
    e2 = _row_gcd_reduce(np.cross(plane_int, e1).astype(int))
    P = np.array([
        plane_int.astype(float),
        e1.astype(float),
        e2.astype(float),
    ])

    # Build Q: rotate each lab axis from grain 1 into grain 2's crystal frame.
    Q = np.array([
        _row_gcd_reduce(P[i].astype(int) @ M_int).astype(float)
        for i in range(3)
    ])

    P_canon, Q_canon = canonicalize_pq(P, Q)

    # Re-check the cell size using the actual constructed matrices.
    # solve_inplane_csl guards |v1 x v2| (the CSL in-plane area), but e2 is
    # defined as plane x e1 rather than the second CSL vector, so det(P)
    # equals |plane|^2*|e1|^2, which can exceed the CSL area by a factor of
    # |plane|^2 for non-(100) boundaries.
    det_P = abs(_int_det3(np.round(P_canon).astype(int)))
    det_Q = abs(_int_det3(np.round(Q_canon).astype(int)))
    if max(det_P, det_Q) > max_exact_atoms:
        raise BoundarySpecError(
            f"CSL supercell exceeds max_exact_atoms={max_exact_atoms}: "
            f"|det(P)|={det_P}, |det(Q)|={det_Q}. "
            "Use mode='approximate' or increase the limit."
        )

    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, Rm in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(Rm) - 1.0) < 1e-10):
            raise BoundarySpecError(
                f"{r_name} derived from CSLExactSpec is not a proper rotation matrix "
                "(R @ R.T != I or det != 1). Check that axis, plane, and quat are "
                "mutually consistent."
            )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source="csl",
    )


def csl_approx_spec_to_embedding(spec) -> BoundaryEmbedding:
    """Convert a CSLApproxSpec to a BoundaryEmbedding using the approximate path.

    Constructs floating-point R_left and R_right from the given plane and
    axis/angle misorientation.  P and Q are set to None (no exact integer
    matrices are available).

    R_left is built so that its first row is the unit boundary-plane normal.
    The remaining two rows are completed via Gram-Schmidt using the two
    non-dominant axis-aligned unit vectors, giving a proper rotation.
    R_right = R_left @ R_mis, where R_mis is the rotation about the given
    axis by angle_deg.

    :param spec: A ``CSLApproxSpec`` instance.
    :return: ``BoundaryEmbedding`` with ``exact=False``, ``coherent=True``,
        ``source="csl"``.
    """
    plane = np.asarray(spec.plane, dtype=float)
    plane_unit = plane / np.linalg.norm(plane)

    axis = np.asarray(spec.axis, dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    angle_rad = float(spec.angle_deg) * np.pi / 180.0
    R_mis = Rotation.from_rotvec(axis_unit * angle_rad).as_matrix()

    # Build R_left: row 0 = plane unit normal; rows 1-2 = orthogonal in-plane
    # directions.  e2 = plane x e1 is orthogonal to both by construction.
    plane_int = _row_gcd_reduce(np.round(plane).astype(int))
    e1, _ = _plane_null_basis(plane_int)
    e1 = _row_gcd_reduce(e1)
    e2 = _row_gcd_reduce(np.cross(plane_int, e1).astype(int))
    e1_unit = e1.astype(float) / np.linalg.norm(e1)
    e2_unit = e2.astype(float) / np.linalg.norm(e2)

    R_left = np.array([plane_unit, e1_unit, e2_unit])
    R_right = R_left @ R_mis

    return BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=R_left,
        R_right=R_right,
        exact=False,
        coherent=True,
        source="csl",
    )


def primitive_bicrystal_atom_count(
    embedding: BoundaryEmbedding,
    atoms_per_conventional_cell: int,
) -> int:
    """Return the primitive boundary-defining bicrystal atom count.

    The count is separate from the expanded GBMaker simulation cell size:
    ``2 * primitive_area_index * atoms_per_conventional_cell``.
    """
    if embedding.metadata is None:
        raise BoundarySpecError(
            "BoundaryEmbedding has no primitive-cell metadata to report."
        )
    atoms = int(atoms_per_conventional_cell)
    if atoms <= 0 or atoms != atoms_per_conventional_cell:
        raise BoundarySpecError(
            "atoms_per_conventional_cell must be a positive integer."
        )
    return int(embedding.metadata.conventional_cell_multiplier * atoms)
