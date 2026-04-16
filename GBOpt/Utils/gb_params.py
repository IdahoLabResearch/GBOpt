# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Utility to convert grain boundary crystallographic descriptions into the
5-element misorientation array expected by GBOpt's GBMaker class.

Usage
-----
    python GBOpt/Utils/gb_params.py axis_angle --axis 1 -1 0 --angle 70.53 --normal 1 1 1
    python GBOpt/Utils/gb_params.py orientation --P 2 2 2 1 -1 0 1 1 -2 \
                                                --Q 2 2 2 -1 1 0 -1 -1 2
    python GBOpt/Utils/gb_params.py self_test
"""

import argparse
import sys
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def normalize_rows(M: np.ndarray) -> np.ndarray:
    """
    Row-normalize a 3x3 matrix.

    :param M: 3x3 matrix whose rows will be normalized.
    :return: Row-normalized copy of M.
    :raises ValueError: If any row has zero magnitude.
    """
    M = np.asarray(M, dtype=float)
    norms = np.linalg.norm(M, axis=1)
    if np.any(norms < 1e-14):
        raise ValueError("Orientation matrix has a zero-length row.")
    return M / norms[:, np.newaxis]


def validate_orientation_matrix(
    M: np.ndarray,
    name: str,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Validate that *M* is a proper row-wise orientation matrix.

    The rows must be non-zero, mutually orthonormal, and right-handed. A
    normalized copy is returned.

    :param M: Candidate 3x3 orientation matrix.
    :param name: Label used in error messages.
    :param tol: Numerical tolerance for orthonormality and handedness.
    :return: Row-normalized, validated matrix.
    :raises ValueError: If the matrix is not a proper orientation matrix.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix.")

    M_norm = normalize_rows(M)
    orth_err = float(np.max(np.abs(M_norm @ M_norm.T - np.eye(3))))
    det = float(np.linalg.det(M_norm))

    if orth_err > tol:
        raise ValueError(
            f"{name} rows must be mutually orthogonal after normalization "
            f"(max err = {orth_err:.3e})."
        )
    if abs(det - 1.0) > tol:
        raise ValueError(
            f"{name} must be right-handed with det = +1 after normalization "
            f"(det = {det:.6f})."
        )

    return M_norm


def inclination_from_normal(n: np.ndarray) -> tuple[float, float]:
    """
    Compute GBMaker inclination angles (theta, phi) from a boundary normal.

    GBMaker applies rotations to row-vector positions via ``x_lab = x_crystal @
    Rincl.T``. Under that convention, the first row of
    ``Rincl = Rz(phi) @ Ry(theta)`` is the crystal direction aligned with the
    lab x-axis, i.e. the grain-1 boundary normal.

    The first row of ``Rz(phi) @ Ry(theta)`` is
    ``[cos(phi)*cos(theta), -sin(phi), cos(phi)*sin(theta)]``, so::

        phi   = arcsin(-n_hat[1])
        theta = arctan2(n_hat[2], n_hat[0])

    :param n: Boundary normal direction as a 3-element array [h, k, l].
    :return: ``(theta, phi)`` in radians.
    :raises ValueError: If n is a zero vector.
    """
    n = np.asarray(n, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        raise ValueError("Boundary normal must be a non-zero vector.")
    n_hat = n / norm
    nx, ny, nz = n_hat

    phi = float(np.arcsin(np.clip(-ny, -1.0, 1.0)))

    if abs(ny) >= 1.0 - 1e-10:
        theta = 0.0
    else:
        theta = float(np.arctan2(nz, nx))

    return theta, phi


# ---------------------------------------------------------------------------
# Public conversion functions
# ---------------------------------------------------------------------------

def from_axis_angle(
    axis: np.ndarray,
    angle_deg: float,
    boundary_normal: np.ndarray,
) -> np.ndarray:
    """
    Compute the GBMaker misorientation array from a rotation axis, angle,
    and boundary normal.

    :param axis: Rotation axis [u, v, w] in crystal coordinates (need not be
        a unit vector).
    :param angle_deg: Misorientation angle in degrees.
    :param boundary_normal: Crystal direction [h, k, l] aligned with the GB
        normal (lab x-axis).
    :return: 5-element array ``[alpha, beta, gamma, theta, phi]`` in radians.
    :raises ValueError: If axis is a zero vector.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        raise ValueError("Rotation axis must be a non-zero vector.")
    axis_hat = axis / norm
    angle_rad = np.radians(angle_deg)

    Rmis = Rotation.from_rotvec(axis_hat * angle_rad).as_matrix()
    alpha, beta, gamma = Rotation.from_matrix(Rmis).as_euler("ZXZ")
    theta, phi = inclination_from_normal(
        np.asarray(boundary_normal, dtype=float)
    )

    return np.array([alpha, beta, gamma, theta, phi])


def from_orientation_matrices(
    P: np.ndarray,
    Q: np.ndarray,
    boundary_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the GBMaker misorientation array from P and Q orientation matrices.

    Each matrix has **rows** equal to the crystal directions for the lab x, y,
    and z axes of that grain. The boundary normal is taken from ``P[0]``; if
    ``boundary_normal`` is supplied it is used only as a consistency check.

    With GBMaker's row-vector convention, a grain orientation matrix satisfies
    ``x_lab = x_crystal @ P_norm.T``. If grain 2 is misoriented from grain 1 by
    ``Rmis`` in the crystal frame, then ``Q_norm = P_norm @ Rmis`` and thus::

        Rmis = P_norm.T @ Q_norm

    :param P: 3x3 orientation matrix for grain 1 (left grain).
    :param Q: 3x3 orientation matrix for grain 2 (right grain).
    :param boundary_normal: Optional override for the boundary normal. When
        provided it must match ``P[0]`` within 1°; a warning is printed
        otherwise.
    :return: 5-element array ``[alpha, beta, gamma, theta, phi]`` in radians.
    """
    P_norm = validate_orientation_matrix(P, "P")
    Q_norm = validate_orientation_matrix(Q, "Q")

    Rmis = P_norm.T @ Q_norm
    alpha, beta, gamma = Rotation.from_matrix(Rmis).as_euler("ZXZ")

    normal_from_P = P_norm[0]
    if boundary_normal is not None:
        n = np.asarray(boundary_normal, dtype=float)
        n_hat = n / np.linalg.norm(n)
        cos_angle = np.clip(np.dot(normal_from_P, n_hat), -1.0, 1.0)
        angle_err = np.degrees(np.arccos(cos_angle))
        if angle_err > 1.0:
            print(
                f"WARNING: supplied --normal deviates from P[0] by "
                f"{angle_err:.2f}° — using P[0] for inclination.",
                file=sys.stderr,
            )

    theta, phi = inclination_from_normal(normal_from_P)
    return np.array([alpha, beta, gamma, theta, phi])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    params: np.ndarray,
    boundary_normal: np.ndarray,
    P_norm: Optional[np.ndarray] = None,
    Q_norm: Optional[np.ndarray] = None,
    reference_Rmis: Optional[np.ndarray] = None,
) -> list[str]:
    """
    Run sanity checks on the computed misorientation array.

    :param params: 5-element array ``[alpha, beta, gamma, theta, phi]``.
    :param boundary_normal: Boundary normal direction (need not be a unit
        vector).
    :param P_norm: Row-normalized P matrix (orientation mode only).
    :param Q_norm: Row-normalized Q matrix (orientation mode only).
    :param reference_Rmis: Optional reference misorientation matrix.
    :return: List of result strings, each prefixed with '✓' or '✗'.
    """
    alpha, beta, gamma, theta, phi = params
    checks = []

    Rincl = (
        Rotation.from_euler("z", phi) * Rotation.from_euler("y", theta)
    ).as_matrix()
    n_hat = boundary_normal / np.linalg.norm(boundary_normal)
    normal_err = float(np.linalg.norm(Rincl[0, :] - n_hat))
    mark = "✓" if normal_err < 1e-10 else "✗"
    checks.append(
        f"{mark} Rincl[0,:] matches boundary normal  "
        f"(err = {normal_err:.3e})"
    )

    Rmis = Rotation.from_euler("ZXZ", [alpha, beta, gamma]).as_matrix()
    det = float(np.linalg.det(Rmis))
    ortho_err = float(np.max(np.abs(Rmis.T @ Rmis - np.eye(3))))
    mark = "✓" if abs(det - 1.0) < 1e-10 and ortho_err < 1e-10 else "✗"
    checks.append(
        f"{mark} Rmis is proper rotation  "
        f"(det = {det:.4f}, max col-ortho err = {ortho_err:.3e})"
    )

    if reference_Rmis is not None:
        matrix_err = float(np.max(np.abs(Rmis - reference_Rmis)))
        delta_deg = float(
            np.degrees(
                np.linalg.norm(
                    Rotation.from_matrix(reference_Rmis.T @ Rmis).as_rotvec()
                )
            )
        )
        mark = "✓" if matrix_err < 1e-10 and delta_deg < 1e-8 else "✗"
        checks.append(
            f"{mark} ZXZ reconstruction matches source rotation  "
            f"(max matrix err = {matrix_err:.3e}, angle err = {delta_deg:.4f}°)"
        )

    beta_deg = float(np.degrees(beta))
    if abs(beta_deg) < 1.0 or abs(abs(beta_deg) - 180.0) < 1.0:
        checks.append(
            f"  WARNING: β = {beta_deg:.2f}° is near 0° or 180° "
            f"(ZXZ gimbal lock — α and γ are not uniquely determined)"
        )

    if P_norm is not None and Q_norm is not None:
        cos_a = float(np.clip(np.dot(P_norm[0], Q_norm[0]), -1.0, 1.0))
        normal_angle = float(np.degrees(np.arccos(cos_a)))
        gb_type = "symmetric" if normal_angle < 1e-6 else "asymmetric"
        mark = "✓" if normal_angle < 1e-6 else "~"
        checks.append(
            f"{mark} GB boundary plane type: {gb_type}  "
            f"(P[0] vs Q[0] angular diff = {normal_angle:.4f}°)"
        )

        for name, mat in [("P", P_norm), ("Q", Q_norm)]:
            orth = float(np.max(np.abs(mat @ mat.T - np.eye(3))))
            mark = "✓" if orth < 1e-10 else "✗"
            checks.append(
                f"{mark} {name}_norm rows are orthonormal  "
                f"(max err = {orth:.3e})"
            )

    return checks


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _symbolic(rad: float, tol: float = 1e-6) -> str:
    """Return a symbolic name for *rad* if it can be expressed as a rational
    multiple of π (denominator ≤ 24) or as arctan/arccos/arcsin of a simple
    rational or square-root argument."""

    frac = Fraction(rad / np.pi).limit_denominator(24)
    if abs(float(frac) * np.pi - rad) < tol:
        n, d = frac.numerator, frac.denominator
        if n == 0:
            return "0"
        sign = "-" if n < 0 else ""
        a = abs(n)
        coeff = "" if a == 1 else str(a)
        return f"{sign}{coeff}π/{d}" if d != 1 else f"{sign}{coeff}π"

    pos_args: list[tuple[float, str]] = []
    for d in range(1, 9):
        for n in range(1, 3 * d + 1):
            pos_args.append((n / d, f"{n}/{d}" if d != 1 else str(n)))
    for k in range(2, 8):
        sq = float(np.sqrt(k))
        for n in range(1, 9):
            pos_args.append((n / sq, f"{n}/√{k}" if n != 1 else f"1/√{k}"))
        for d in range(1, 9):
            pos_args.append((sq / d, f"√{k}/{d}" if d != 1 else f"√{k}"))

    for arg, label in pos_args:
        v = float(np.arctan(arg))
        if abs(rad - v) < tol:
            return f"arctan({label})"
        if abs(rad + v) < tol:
            return f"-arctan({label})"

    for sign, s_lbl in [(1, ""), (-1, "-")]:
        for arg, label in pos_args:
            sarg = sign * arg
            if abs(sarg) > 1.0:
                continue
            for fn, fn_name in [(np.arccos, "arccos"), (np.arcsin, "arcsin")]:
                if abs(rad - float(fn(sarg))) < tol:
                    return f"{fn_name}({s_lbl}{label})"

    return ""


def _fmt_angle(rad: float) -> str:
    sym = _symbolic(rad)
    sym_str = f"  [{sym}]" if sym else ""
    return f"{rad:+.6f} rad  ({np.degrees(rad):+8.2f}°){sym_str}"


def format_output(
    params: np.ndarray,
    input_summary: str,
    checks: list[str],
) -> str:
    """
    Build the human-readable output string.

    :param params: 5-element misorientation array.
    :param input_summary: One-line description of the inputs used.
    :param checks: Validation result strings from :func:`validate`.
    :return: Formatted output string ready for printing.
    """
    alpha, beta, gamma, theta, phi = params
    array_str = ", ".join(f"{v:.6f}" for v in params)

    lines = [
        "",
        "=== GBOpt Misorientation Parameters ===",
        "",
        f"Input:  {input_summary}",
        "",
        "Misorientation (ZXZ, crystal frame):",
        f"  α = {_fmt_angle(alpha)}",
        f"  β = {_fmt_angle(beta)}",
        f"  γ = {_fmt_angle(gamma)}",
        "",
        "Inclination:",
        f"  θ = {_fmt_angle(theta)}",
        f"  φ = {_fmt_angle(phi)}",
        "",
        "Validation:",
    ]
    lines.extend(f"  {c}" for c in checks)
    lines += [
        "",
        f"misorientation = np.array([{array_str}])",
        "",
    ]
    return "\n".join(lines)


def _assert_rotation_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Raise AssertionError if two rotation matrices differ beyond tolerance."""
    delta = Rotation.from_matrix(expected.T @ actual)
    angle_err = np.linalg.norm(delta.as_rotvec())
    if angle_err >= 1e-10:
        raise AssertionError(
            f"Rotation mismatch: angular error = {np.degrees(angle_err):.6e} deg"
        )


def _orientation_matrix(normal: np.ndarray, in_plane: np.ndarray) -> np.ndarray:
    """Build a row-wise orientation matrix from a boundary normal and in-plane seed."""
    x_dir = np.asarray(normal, dtype=float)
    y_seed = np.asarray(in_plane, dtype=float)
    x_dir /= np.linalg.norm(x_dir)
    y_seed -= np.dot(y_seed, x_dir) * x_dir
    y_seed /= np.linalg.norm(y_seed)
    z_dir = np.cross(x_dir, y_seed)
    return np.vstack((x_dir, y_seed, z_dir))


def run_self_test() -> None:
    """
    Run standalone regression checks for representative boundary types.
    """
    cases: list[str] = []

    angle_deg = 36.869898
    normal = np.array([3.0, 1.0, 0.0])
    params = from_axis_angle([0.0, 0.0, 1.0], angle_deg, normal)
    expected = np.array(
        [
            np.radians(angle_deg),
            0.0,
            0.0,
            0.0,
            -np.arctan(1.0 / 3.0),
        ]
    )
    np.testing.assert_allclose(params, expected, atol=1e-8)
    Rincl = (
        Rotation.from_euler("z", params[4]) * Rotation.from_euler("y", params[3])
    ).as_matrix()
    np.testing.assert_allclose(Rincl[0, :], normal / np.linalg.norm(normal), atol=1e-10)
    cases.append("Sigma5 symmetric tilt")

    P = _orientation_matrix([2.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    Rmis_expected = Rotation.from_rotvec(
        np.array([0.0, 0.0, np.radians(angle_deg)])
    ).as_matrix()
    Q = P @ Rmis_expected
    params = from_orientation_matrices(P, Q)
    Rmis = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    Rincl = (
        Rotation.from_euler("z", params[4]) * Rotation.from_euler("y", params[3])
    ).as_matrix()
    _assert_rotation_close(Rmis, Rmis_expected)
    np.testing.assert_allclose(Rincl[0, :], P[0], atol=1e-10)
    cases.append("Asymmetric tilt")

    angle_deg = 45.0
    axis = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 0.0, 1.0])
    params = from_axis_angle(axis, angle_deg, normal)
    Rmis = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    Rmis_expected = Rotation.from_rotvec(axis * np.radians(angle_deg)).as_matrix()
    Rincl = (
        Rotation.from_euler("z", params[4]) * Rotation.from_euler("y", params[3])
    ).as_matrix()
    _assert_rotation_close(Rmis, Rmis_expected)
    np.testing.assert_allclose(Rincl[0, :], normal, atol=1e-10)
    cases.append("Twist")

    angle_deg = 60.0
    axis = np.array([1.0, 1.0, 1.0])
    axis /= np.linalg.norm(axis)
    normal = np.array([1.0, 1.0, 1.0])
    params = from_axis_angle(axis, angle_deg, normal)
    Rmis = Rotation.from_euler("ZXZ", params[:3]).as_matrix()
    Rmis_expected = Rotation.from_rotvec(axis * np.radians(angle_deg)).as_matrix()
    Rincl = (
        Rotation.from_euler("z", params[4]) * Rotation.from_euler("y", params[3])
    ).as_matrix()
    _assert_rotation_close(Rmis, Rmis_expected)
    np.testing.assert_allclose(Rincl[0, :], normal / np.linalg.norm(normal), atol=1e-10)
    cases.append("Sigma3 coherent twin")

    print("Standalone regression checks passed:")
    for case in cases:
        print(f"  - {case}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert grain boundary crystallographic descriptions into the "
            "5-element misorientation array for GBOpt's GBMaker."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    aa = sub.add_parser(
        "axis_angle",
        help="Derive parameters from rotation axis, angle, and boundary normal.",
    )
    aa.add_argument(
        "--axis",
        nargs=3,
        type=float,
        metavar=("U", "V", "W"),
        required=True,
        help="Rotation axis [u v w] in crystal coordinates.",
    )
    aa.add_argument(
        "--angle",
        type=float,
        required=True,
        metavar="DEG",
        help="Misorientation angle in degrees.",
    )
    aa.add_argument(
        "--normal",
        nargs=3,
        type=float,
        metavar=("H", "K", "L"),
        required=True,
        help="Boundary normal direction [h k l] in crystal coordinates.",
    )

    ori = sub.add_parser(
        "orientation",
        help=(
            "Derive parameters from P and Q orientation matrices "
            "(rows = crystal directions for lab x/y/z axes)."
        ),
    )
    ori.add_argument(
        "--P",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help=(
            "Grain 1 orientation matrix, 9 values row-major "
            "(e.g. --P 2 2 2  1 -1 0  1 1 -2)."
        ),
    )
    ori.add_argument(
        "--Q",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help="Grain 2 orientation matrix, 9 values row-major.",
    )
    ori.add_argument(
        "--normal",
        nargs=3,
        type=float,
        metavar=("H", "K", "L"),
        default=None,
        help=(
            "Optional boundary normal override for consistency check. "
            "If omitted, P[0] is used."
        ),
    )

    sub.add_parser(
        "self_test",
        help="Run standalone regression checks for representative GB types.",
    )

    return parser


def main() -> None:
    """Entry point for the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "self_test":
        run_self_test()
        return

    if args.mode == "axis_angle":
        axis = np.array(args.axis)
        normal = np.array(args.normal)
        params = from_axis_angle(axis, args.angle, normal)
        reference_Rmis = Rotation.from_rotvec(
            axis / np.linalg.norm(axis) * np.radians(args.angle)
        ).as_matrix()
        input_summary = (
            f"axis={axis.tolist()}  angle={args.angle}°  normal={normal.tolist()}"
        )
        P_norm = Q_norm = None

    else:
        P = np.array(args.P).reshape(3, 3)
        Q = np.array(args.Q).reshape(3, 3)
        normal_arg = (
            np.array(args.normal) if args.normal is not None else None
        )
        params = from_orientation_matrices(P, Q, normal_arg)
        P_norm = normalize_rows(P)
        Q_norm = normalize_rows(Q)
        reference_Rmis = P_norm.T @ Q_norm
        normal = P_norm[0]
        input_summary = f"P={P.tolist()}  Q={Q.tolist()}"

    checks = validate(params, normal, P_norm, Q_norm, reference_Rmis)
    print(format_output(params, input_summary, checks))


if __name__ == "__main__":
    main()
