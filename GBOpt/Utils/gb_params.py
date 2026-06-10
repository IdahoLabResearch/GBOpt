# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Utility to convert grain-boundary descriptions into GBOpt core formats.

Usage
-----
    python GBOpt/Utils/gb_params.py axis_angle --axis 1 -1 0 --angle 70.53 --normal 1 1 1
    python GBOpt/Utils/gb_params.py orientation --P 2 2 2 1 -1 0 1 1 -2 \
                                                --Q 2 2 2 -1 1 0 -1 -1 2
    python GBOpt/Utils/gb_params.py csl --axis 0 0 1 --plane 1 0 0 --quat 2 0 0 1
    python GBOpt/Utils/gb_params.py canonicalize --P ... --Q ...
    python GBOpt/Utils/gb_params.py self_test
"""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from GBOpt.BoundarySpec import (  # noqa: E402
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.Utils.gb_exact import (  # noqa: E402
    canonicalize_pq,
    csl_approx_spec_to_embedding,
    csl_spec_to_embedding,
    exactify_five_dof,
)

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
        provided it must match ``P[0]`` within 1 deg; a warning is printed
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
                f"{angle_err:.2f} deg — using P[0] for inclination.",
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
            f"(max matrix err = {matrix_err:.3e}, angle err = {delta_deg:.4f} deg)"
        )

    beta_deg = float(np.degrees(beta))
    if abs(beta_deg) < 1.0 or abs(abs(beta_deg) - 180.0) < 1.0:
        checks.append(
            f"  WARNING: beta = {beta_deg:.2f} deg is near 0 deg or 180 deg "
            f"(ZXZ gimbal lock — alpha and gamma are not uniquely determined)"
        )

    if P_norm is not None and Q_norm is not None:
        cos_a = float(np.clip(np.dot(P_norm[0], Q_norm[0]), -1.0, 1.0))
        normal_angle = float(np.degrees(np.arccos(cos_a)))
        gb_type = "symmetric" if normal_angle < 1e-6 else "asymmetric"
        mark = "✓" if normal_angle < 1e-6 else "~"
        checks.append(
            f"{mark} GB boundary plane type: {gb_type}  "
            f"(P[0] vs Q[0] angular diff = {normal_angle:.4f} deg)"
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
    multiple of pi (denominator <= 24) or as arctan/arccos/arcsin of a simple
    rational or square-root argument."""

    frac = Fraction(rad / np.pi).limit_denominator(24)
    if abs(float(frac) * np.pi - rad) < tol:
        n, d = frac.numerator, frac.denominator
        if n == 0:
            return "0"
        sign = "-" if n < 0 else ""
        a = abs(n)
        coeff = "" if a == 1 else str(a)
        return f"{sign}{coeff}pi/{d}" if d != 1 else f"{sign}{coeff}pi"

    pos_args: list[tuple[float, str]] = []
    for d in range(1, 9):
        for n in range(1, 3 * d + 1):
            pos_args.append((n / d, f"{n}/{d}" if d != 1 else str(n)))
    for k in range(2, 8):
        sq = float(np.sqrt(k))
        for n in range(1, 9):
            pos_args.append((n / sq, f"{n}/sqrt({k})" if n != 1 else f"1/sqrt({k})"))
        for d in range(1, 9):
            pos_args.append((sq / d, f"sqrt({k})/{d}" if d != 1 else f"sqrt({k})"))

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
    return f"{rad:+.6f} rad  ({np.degrees(rad):+8.2f} deg){sym_str}"


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
        f"  alpha = {_fmt_angle(alpha)}",
        f"  beta = {_fmt_angle(beta)}",
        f"  gamma = {_fmt_angle(gamma)}",
        "",
        "Inclination:",
        f"  theta = {_fmt_angle(theta)}",
        f"  phi = {_fmt_angle(phi)}",
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


def _clean_float(value: float, tol: float = 5e-13) -> float:
    """Return a JSON-friendly float with tiny signed zeros removed."""
    value = float(value)
    if abs(value) < tol:
        return 0.0
    return value


def _array_to_float_list(values: np.ndarray) -> list[float]:
    return [_clean_float(v) for v in np.asarray(values, dtype=float).ravel()]


def _json_number(value: float) -> int | float:
    """Use ints in JSON when a value is effectively integer-valued."""
    value = _clean_float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return int(rounded)
    return value


def _matrix_to_json(M: np.ndarray) -> list[list[int | float]]:
    arr = np.asarray(M, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix; got shape {arr.shape}.")
    return [[_json_number(v) for v in row] for row in arr]


def _int_vector(values, name: str, length: int) -> list[int]:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have length {length}; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be integer-valued.")
    return [int(v) for v in np.round(arr).astype(int)]


def _five_dof_payload(params: np.ndarray | list[float]) -> dict:
    spec = FiveDOFSpec(params)
    return {
        "format": "five_dof",
        "params": _array_to_float_list(np.asarray(spec.params, dtype=float)),
        "units": "radians",
    }


def _pq_payload(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    basis_mode: str = "primitive",
) -> dict:
    payload = {
        "format": "pq",
        "P": _matrix_to_json(P),
        "Q": _matrix_to_json(Q),
        "basis_mode": basis_mode,
    }
    PQSpec(payload["P"], payload["Q"], basis_mode=basis_mode)
    return payload


def _csl_payload(
    axis,
    plane,
    *,
    quat=None,
    angle_deg=None,
    sigma=None,
    max_exact_atoms: int = 10_000,
) -> dict:
    axis_int = _int_vector(axis, "axis", 3)
    plane_int = _int_vector(plane, "plane", 3)
    sigma_int = None if sigma is None else int(sigma)
    if sigma is not None and sigma_int <= 0:
        raise ValueError("sigma must be a positive integer.")

    if quat is not None:
        quat_int = _int_vector(quat, "quat", 4)
        spec = CSLExactSpec(
            axis=axis_int,
            plane=plane_int,
            sigma=sigma_int,
            quat=quat_int,
        )
        csl_spec_to_embedding(spec, max_exact_atoms=max_exact_atoms)
        payload = {
            "format": "csl",
            "exact": True,
            "axis": axis_int,
            "plane": plane_int,
            "quat": quat_int,
        }
    else:
        if angle_deg is None:
            raise ValueError("csl requires either quat or angle_deg.")
        angle = float(angle_deg)
        spec = CSLApproxSpec(
            axis=axis_int,
            plane=plane_int,
            sigma=sigma_int,
            angle_deg=angle,
        )
        csl_approx_spec_to_embedding(spec)
        payload = {
            "format": "csl",
            "exact": False,
            "axis": axis_int,
            "plane": plane_int,
            "angle_deg": _clean_float(angle),
        }

    if sigma_int is not None:
        payload["sigma"] = sigma_int
    return payload


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_core_payload(args) -> dict:
    if args.input_file is not None:
        with open(args.input_file, encoding="utf-8") as stream:
            payload = json.load(stream)
    elif args.input_json is not None:
        payload = json.loads(args.input_json)
    else:
        payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Core-format input must be a JSON object.")
    return payload


def _normalize_core_payload(
    payload: dict,
    *,
    max_exact_atoms: int = 10_000,
) -> dict:
    fmt = payload.get("format")
    if fmt == "five_dof":
        return _five_dof_payload(payload["params"])
    if fmt == "pq":
        return _pq_payload(
            np.asarray(payload["P"], dtype=float),
            np.asarray(payload["Q"], dtype=float),
            basis_mode=payload.get("basis_mode", "primitive"),
        )
    if fmt == "csl":
        return _csl_payload(
            payload["axis"],
            payload["plane"],
            quat=payload.get("quat"),
            angle_deg=payload.get("angle_deg"),
            sigma=payload.get("sigma"),
            max_exact_atoms=max_exact_atoms,
        )
    raise ValueError("Core-format input must have format 'five_dof', 'pq', or 'csl'.")


def _csl_spec_from_payload(
    payload: dict,
    *,
    max_exact_atoms: int = 10_000,
):
    payload = _normalize_core_payload(payload, max_exact_atoms=max_exact_atoms)
    if payload["format"] != "csl":
        raise ValueError("Expected a csl payload.")
    if payload["exact"]:
        return CSLExactSpec(
            axis=payload["axis"],
            plane=payload["plane"],
            sigma=payload.get("sigma"),
            quat=payload["quat"],
        )
    return CSLApproxSpec(
        axis=payload["axis"],
        plane=payload["plane"],
        sigma=payload.get("sigma"),
        angle_deg=payload["angle_deg"],
    )


def _five_dof_from_embedding(embedding) -> dict:
    Rmis = embedding.R_left.T @ embedding.R_right
    alpha, beta, gamma = Rotation.from_matrix(Rmis).as_euler("ZXZ")
    theta, phi = inclination_from_normal(embedding.R_left[0])
    return _five_dof_payload(np.array([alpha, beta, gamma, theta, phi]))


def _convert_payload(
    payload: dict,
    target: str,
    *,
    max_exact_atoms: int = 10_000,
) -> dict:
    payload = _normalize_core_payload(payload, max_exact_atoms=max_exact_atoms)
    source = payload["format"]
    if source == target:
        return payload

    if target == "five_dof":
        if source == "pq":
            params = from_orientation_matrices(
                np.asarray(payload["P"], dtype=float),
                np.asarray(payload["Q"], dtype=float),
            )
            return _five_dof_payload(params)
        if source == "csl":
            spec = _csl_spec_from_payload(
                payload,
                max_exact_atoms=max_exact_atoms,
            )
            if payload["exact"]:
                embedding = csl_spec_to_embedding(
                    spec,
                    max_exact_atoms=max_exact_atoms,
                )
            else:
                embedding = csl_approx_spec_to_embedding(spec)
            return _five_dof_from_embedding(embedding)

    if target == "pq":
        if source == "csl" and payload["exact"]:
            embedding = csl_spec_to_embedding(
                _csl_spec_from_payload(
                    payload,
                    max_exact_atoms=max_exact_atoms,
                ),
                max_exact_atoms=max_exact_atoms,
            )
            return _pq_payload(embedding.P, embedding.Q, basis_mode="primitive")
        if source == "five_dof":
            P, Q = exactify_five_dof(
                np.asarray(payload["params"], dtype=float),
                max_exact_atoms=max_exact_atoms,
            )
            return _pq_payload(P, Q, basis_mode="primitive")

    raise ValueError(f"Conversion from {source!r} to {target!r} is not available.")


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
            "Convert grain boundary crystallographic descriptions into GBOpt "
            "core formats (five_dof, pq, csl)."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    aa = sub.add_parser(
        "axis_angle",
        help="Convert rotation axis, angle, and boundary normal to five_dof.",
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
    aa.add_argument(
        "--format",
        choices=("json", "human"),
        default="json",
        help="Output JSON core format by default; use human for the legacy report.",
    )

    ori = sub.add_parser(
        "orientation",
        help=(
            "Convert P and Q orientation matrices "
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
    ori.add_argument(
        "--target",
        choices=("five_dof", "pq"),
        default="five_dof",
        help="Core output format. five_dof preserves the legacy conversion behavior.",
    )
    ori.add_argument(
        "--format",
        choices=("json", "human"),
        default="json",
        help="Output JSON core format by default; use human for the legacy report.",
    )

    csl = sub.add_parser(
        "csl",
        help="Validate and emit an exact or approximate CSL core-format spec.",
    )
    csl.add_argument("--axis", nargs=3, type=int,
                     metavar=("U", "V", "W"), required=True)
    csl.add_argument("--plane", nargs=3, type=int,
                     metavar=("H", "K", "L"), required=True)
    csl_kind = csl.add_mutually_exclusive_group(required=True)
    csl_kind.add_argument(
        "--quat",
        nargs=4,
        type=int,
        metavar=("W", "X", "Y", "Z"),
        help="Integer quaternion [w x y z] for an exact CSL spec.",
    )
    csl_kind.add_argument(
        "--angle",
        type=float,
        metavar="DEG",
        help="Approximate misorientation angle in degrees.",
    )
    csl.add_argument("--sigma", type=int, default=None, help="Optional positive sigma.")
    csl.add_argument(
        "--max-exact-atoms",
        type=int,
        default=10_000,
        help="Exact CSL cell-size guard used while validating --quat input.",
    )

    conv = sub.add_parser(
        "convert",
        help="Convert a JSON core-format spec to another supported core format.",
    )
    conv_in = conv.add_mutually_exclusive_group()
    conv_in.add_argument("--input-json", help="Input core-format JSON object.")
    conv_in.add_argument("--input-file", help="Path to an input core-format JSON file.")
    conv.add_argument(
        "--to",
        choices=("five_dof", "pq", "csl"),
        required=True,
        help="Target core format.",
    )
    conv.add_argument(
        "--max-exact-atoms",
        type=int,
        default=10_000,
        help="Exact CSL/exactification cell-size guard.",
    )

    exact = sub.add_parser(
        "exactify",
        help="Exactify five_dof parameters through the Stage E hook.",
    )
    exact.add_argument(
        "--params",
        nargs=5,
        type=float,
        metavar=("ALPHA", "BETA", "GAMMA", "THETA", "PHI"),
        required=True,
        help="five_dof parameters in radians.",
    )
    exact.add_argument(
        "--max-exact-atoms",
        type=int,
        default=10_000,
        help="Exactification cell-size guard.",
    )

    canon = sub.add_parser(
        "canonicalize",
        help="Canonicalize exact P/Q matrices using GBOpt's canonicalize_pq routine.",
    )
    canon.add_argument(
        "--P",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help="Grain 1 orientation matrix, 9 row-major values.",
    )
    canon.add_argument(
        "--Q",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help="Grain 2 orientation matrix, 9 row-major values.",
    )

    sub.add_parser(
        "self_test",
        help="Run standalone regression checks for representative GB types.",
    )

    return parser


def main() -> int:
    """Entry point for the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "self_test":
        run_self_test()
        return 0

    try:
        if args.mode == "axis_angle":
            axis = np.array(args.axis)
            normal = np.array(args.normal)
            params = from_axis_angle(axis, args.angle, normal)
            if args.format == "json":
                _print_payload(_five_dof_payload(params))
                return 0

            reference_Rmis = Rotation.from_rotvec(
                axis / np.linalg.norm(axis) * np.radians(args.angle)
            ).as_matrix()
            input_summary = (
                f"axis={axis.tolist()}  angle={args.angle} deg  normal={normal.tolist()}"
            )
            checks = validate(params, normal, None, None, reference_Rmis)
            print(format_output(params, input_summary, checks))
            return 0

        if args.mode == "orientation":
            P = np.array(args.P).reshape(3, 3)
            Q = np.array(args.Q).reshape(3, 3)
            normal_arg = (
                np.array(args.normal) if args.normal is not None else None
            )
            if args.target == "pq":
                if args.format == "human":
                    parser.error(
                        "--format human is only supported with --target five_dof.")
                _print_payload(_pq_payload(P, Q))
                return 0

            params = from_orientation_matrices(P, Q, normal_arg)
            if args.format == "json":
                _print_payload(_five_dof_payload(params))
                return 0

            P_norm = normalize_rows(P)
            Q_norm = normalize_rows(Q)
            reference_Rmis = P_norm.T @ Q_norm
            normal = P_norm[0]
            input_summary = f"P={P.tolist()}  Q={Q.tolist()}"
            checks = validate(params, normal, P_norm, Q_norm, reference_Rmis)
            print(format_output(params, input_summary, checks))
            return 0

        if args.mode == "csl":
            _print_payload(
                _csl_payload(
                    args.axis,
                    args.plane,
                    quat=args.quat,
                    angle_deg=args.angle,
                    sigma=args.sigma,
                    max_exact_atoms=args.max_exact_atoms,
                )
            )
            return 0

        if args.mode == "convert":
            payload = _load_core_payload(args)
            _print_payload(
                _convert_payload(
                    payload,
                    args.to,
                    max_exact_atoms=args.max_exact_atoms,
                )
            )
            return 0

        if args.mode == "exactify":
            try:
                P, Q = exactify_five_dof(
                    np.asarray(args.params, dtype=float),
                    max_exact_atoms=args.max_exact_atoms,
                )
            except NotImplementedError:
                _print_payload(
                    {
                        "status": "not_implemented",
                        "message": (
                            "five_dof exactification is not yet implemented; "
                            "this is the Stage E hook."
                        ),
                    }
                )
                return 0
            _print_payload(_pq_payload(P, Q, basis_mode="primitive"))
            return 0

        if args.mode == "canonicalize":
            P = np.array(args.P).reshape(3, 3)
            Q = np.array(args.Q).reshape(3, 3)
            P_canon, Q_canon = canonicalize_pq(P, Q)
            _print_payload(_pq_payload(P_canon, Q_canon, basis_mode="primitive"))
            return 0

    except (BoundarySpecError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    parser.error(f"Unhandled mode: {args.mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
