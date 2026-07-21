# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Thin CLI for converting grain-boundary descriptions into GBOpt core formats.

Examples
--------
Convert an axis-angle description to five-DOF JSON::

    python GBOpt/Utils/gb_params.py axis_angle \
        --axis 1 -1 0 --angle 70.53 --normal 1 1 1

Convert row-wise orientation matrices to five-DOF JSON::

    python GBOpt/Utils/gb_params.py orientation \
        --P 2 2 2 1 -1 0 1 1 -2 \
        --Q 2 2 2 -1 1 0 -1 -1 2

Validate and emit an exact CSL specification::

    python GBOpt/Utils/gb_params.py csl \
        --axis 0 0 1 --plane 1 0 0 --quat 2 0 0 1

Canonicalize exact integer P/Q matrices::

    python GBOpt/Utils/gb_params.py canonicalize --P ... --Q ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.crystallography.boundary import (
    csl_approx_spec_to_embedding,
    csl_exact_spec_to_embedding,
    five_dof_spec_to_embedding,
    pq_spec_to_embedding,
)
from GBOpt.crystallography.exactification import exactify_five_dof
from GBOpt.crystallography.orientation import (
    five_dof_from_axis_angle,
    five_dof_from_orientation_matrices,
    normalize_direction,
    validate_orientation_matrix,
)
from GBOpt.crystallography.pq import canonicalize_pq_paired
from GBOpt.crystallography.types import CrystallographyValueError

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_JSON_ZERO_TOL = 5.0e-13
_REPORT_TOL = 1.0e-10

# ---------------------------------------------------------------------------
# Human-readable reporting
# ---------------------------------------------------------------------------


def _validation_report(
    params: object,
    boundary_normal: object,
    *,
    P: object | None = None,
    Q: object | None = None,
) -> list[str]:
    """Build human-readable validation messages for five-DOF parameters.

    Domain conversion is delegated to :func:`five_dof_spec_to_embedding`, which applies
    the package's authoritative five-DOF convention. The report then measures whether
    the resulting left-grain frame reproduces the requested boundary normal and whether
    the relative left-to-right frame is a proper rotation. Optional ``P`` and ``Q``
    matrices add an informational comparison of their boundary-normal rows.

    :param params: Five values ``[alpha, beta, gamma, theta, phi]`` in radians accepted
        by :class:`FiveDOFSpec`.
    :param boundary_normal: Array-like three-component boundary normal that the
        inclination portion of ``params`` is expected to reproduce.
    :param P: Optional left-grain row-orientation matrix. It must be supplied together
        with ``Q`` and is normalized and validated before use. Keyword-only, optional,
        defaults to ``None``.
    :param Q: Optional right-grain row-orientation matrix. It must be supplied together
        with ``P`` and is normalized and validated before use. Keyword-only, optional,
        defaults to ``None``.
    :return: Ordered list of formatted ``PASS``, ``FAIL``, ``NOTE``, and ``INFO``
        messages suitable for inclusion in the human-readable report.
    :raises ValueError: If exactly one of ``P`` and ``Q`` is supplied.
    """
    spec = FiveDOFSpec(params)
    embedding = five_dof_spec_to_embedding(spec)
    values = np.asarray(spec.params, dtype=np.float64)
    beta = float(values[1])
    checks: list[str] = []

    expected_normal = normalize_direction(boundary_normal, "boundary normal")
    normal_error = float(np.linalg.norm(embedding.R_left[0] - expected_normal))
    mark = "PASS" if normal_error < _REPORT_TOL else "FAIL"
    checks.append(
        f"{mark}: inclination reproduces the boundary normal (error={normal_error:.3e})"
    )

    misorientation = embedding.R_left.T @ embedding.R_right
    determinant = float(np.linalg.det(misorientation))
    orthogonality_error = float(
        np.max(np.abs(misorientation.T @ misorientation - np.eye(3)))
    )
    mark = (
        "PASS"
        if abs(determinant - 1.0) < _REPORT_TOL and orthogonality_error < _REPORT_TOL
        else "FAIL"
    )
    checks.append(
        f"{mark}: misorientation is a proper rotation "
        f"(det={determinant:.12g}, orthogonality error={orthogonality_error:.3e})"
    )

    beta_deg = math.degrees(beta)
    if abs(beta_deg) < 1.0 or abs(abs(beta_deg) - 180.0) < 1.0:
        checks.append(
            "NOTE: beta is near a ZXZ singularity; alpha and gamma are not "
            "individually unique."
        )

    if P is not None or Q is not None:
        if P is None or Q is None:
            raise ValueError("P and Q must be supplied together.")
        P_norm = validate_orientation_matrix(P, "P")
        Q_norm = validate_orientation_matrix(Q, "Q")
        cosine = float(np.clip(np.dot(P_norm[0], Q_norm[0]), -1.0, 1.0))
        normal_angle_deg = math.degrees(math.acos(cosine))
        checks.append(
            "INFO: left/right crystal boundary normals differ by "
            f"{normal_angle_deg:.6f} deg."
        )

    return checks


def _simple_symbolic_arguments() -> list[tuple[float, str]]:
    """Generate candidate arguments for inverse-trigonometric angle formatting.

    The candidates include small positive rational values and simple square-root
    expressions. Labels are ASCII strings so report output is portable across terminals.

    :return: A list of ``(value, label)`` pairs consumed by :func:`_symbolic`, where
        ``value`` is the floating-point argument and ``label`` is its display
        representation.
    """
    candidates: dict[str, float] = {}

    for denominator in range(1, 9):
        for numerator in range(1, 3 * denominator + 1):
            label = str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
            candidates[label] = numerator / denominator

    for root in range(2, 8):
        square_root = math.sqrt(root)
        for numerator in range(1, 9):
            label = f"1/sqrt({root})" if numerator == 1 else f"{numerator}/sqrt({root})"
            candidates[label] = numerator / square_root
        for denominator in range(1, 9):
            label = (
                f"sqrt({root})" if denominator == 1 else f"sqrt({root})/{denominator}"
            )
            candidates[label] = square_root / denominator

    return [(value, label) for label, value in candidates.items()]


def _multiple_expression(
    function_name: str,
    argument_label: str,
    multiplier: int,
) -> str:
    """Format a signed integer multiple of an inverse-trigonometric expression.

    :param function_name: Function label, such as ``"arctan"``, ``"arccos"``, or
        ``"arcsin"``.
    :param argument_label: Preformatted symbolic argument inserted inside the function
        call.
    :param multiplier: Signed nonzero integer coefficient applied to the function value.
    :return: A compact expression such as ``"arctan(1/5)"``, ``"-arcsin(1/2)"``, or
        ``"2*arccos(1/3)"``.
    """
    base = f"{function_name}({argument_label})"
    if multiplier == 1:
        return base
    if multiplier == -1:
        return f"-{base}"
    return f"{multiplier}*{base}"


def _symbolic(rad: float, tol: float = 1.0e-6) -> str:
    """Recognize a compact symbolic representation of an angle.

    The search first considers rational multiples of pi with denominator at most 24. It
    then considers signed integer multiples from -4 through 4 of ``arctan``, ``arccos``,
    and ``arcsin`` values evaluated at the candidate arguments returned by
    :func:`_simple_symbolic_arguments`.

    :param rad: Angle in radians to match against the supported symbolic forms.
    :param tol: Absolute matching tolerance in radians. Optional; defaults to ``1e-6``.
    :return: The first matching symbolic expression, or the empty string when no
        candidate is within ``tol``.
    """
    rad = float(rad)
    fraction = Fraction(rad / math.pi).limit_denominator(24)
    if abs(float(fraction) * math.pi - rad) < tol:
        numerator = fraction.numerator
        denominator = fraction.denominator
        if numerator == 0:
            return "0"
        sign = "-" if numerator < 0 else ""
        coefficient = "" if abs(numerator) == 1 else str(abs(numerator))
        pi_term = f"{sign}{coefficient}pi"
        return pi_term if denominator == 1 else f"{pi_term}/{denominator}"

    arguments = _simple_symbolic_arguments()
    for argument, label in arguments:
        values = (("arctan", math.atan(argument)),)
        for function_name, value in values:
            for multiplier in range(-4, 5):
                if multiplier == 0:
                    continue
                if abs(rad - multiplier * value) < tol:
                    return _multiple_expression(function_name, label, multiplier)

    for argument, label in arguments:
        for signed_argument, signed_label in (
            (argument, label),
            (-argument, f"-{label}"),
        ):
            if abs(signed_argument) > 1.0:
                continue
            for function_name, function in (
                ("arccos", math.acos),
                ("arcsin", math.asin),
            ):
                value = function(signed_argument)
                for multiplier in range(-4, 5):
                    if multiplier == 0:
                        continue
                    if abs(rad - multiplier * value) < tol:
                        return _multiple_expression(
                            function_name,
                            signed_label,
                            multiplier,
                        )

    return ""


def _format_angle(rad: float) -> str:
    """Format one angle for the human-readable report.

    :param rad: Angle in radians.
    :return: A string containing signed radians, signed degrees, and, when recognized by
        :func:`_symbolic`, a bracketed symbolic suffix.
    """
    symbolic = _symbolic(rad)
    suffix = f" [{symbolic}]" if symbolic else ""
    return f"{rad:+.6f} rad ({math.degrees(rad):+8.2f} deg){suffix}"


def format_human_output(
    params: object,
    input_summary: str,
    checks: Sequence[str],
) -> str:
    """Build the legacy human-readable five-DOF report.

    :param params: Five values ``[alpha, beta, gamma, theta, phi]`` in radians accepted
        by :class:`FiveDOFSpec`.
    :param input_summary: One-line description of the source values used to derive the
        parameters.
    :param checks: Ordered sequence of preformatted validation messages included under
        the report's ``Validation`` heading.
    :return: Complete multi-line report ready to write to standard output.
    """
    spec = FiveDOFSpec(params)
    alpha, beta, gamma, theta, phi = np.asarray(
        spec.params,
        dtype=np.float64,
    )
    array_text = ", ".join(f"{value:.6f}" for value in spec.params)

    lines = [
        "",
        "=== GBOpt Misorientation Parameters ===",
        "",
        f"Input: {input_summary}",
        "",
        "Misorientation (ZXZ, crystal frame):",
        f"  alpha = {_format_angle(float(alpha))}",
        f"  beta  = {_format_angle(float(beta))}",
        f"  gamma = {_format_angle(float(gamma))}",
        "",
        "Inclination:",
        f"  theta = {_format_angle(float(theta))}",
        f"  phi   = {_format_angle(float(phi))}",
        "",
        "Validation:",
    ]
    lines.extend(f"  {check}" for check in checks)
    lines.extend(
        [
            "",
            f"misorientation = np.array([{array_text}])",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core-format parsing and serialization
# ---------------------------------------------------------------------------


def _json_float(value: object, tol: float = _JSON_ZERO_TOL) -> float:
    """Convert a validated numeric value to stable JSON floating-point output.

    Core-domain validation occurs before this serializer is called, normally through a
    boundary-spec dataclass. This helper performs only presentation normalization by
    converting the value to a Python ``float`` and replacing tiny signed values with
    positive ``0.0``.

    :param value: Previously validated finite numeric scalar to serialize.
    :param tol: Nonnegative magnitude threshold below which the result is replaced by
        ``0.0``. Optional, defaults to ``_JSON_ZERO_TOL``.
    :return: Python ``float`` suitable for JSON serialization.
    :raises ValueError: If ``value`` converts to NaN or infinity, or if ``tol`` is
        negative.
    """
    if tol < 0.0:
        raise ValueError(f"tol must be nonnegative; got {tol!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite real number; got {value!r}.")
    return 0.0 if abs(result) < tol else result


def _five_dof_payload(spec: FiveDOFSpec) -> dict[str, Any]:
    """Serialize a validated five-DOF specification.

    :param spec: Validated :class:`FiveDOFSpec` whose parameters are expressed in
        radians.
    :return: JSON-safe dictionary with ``format="five_dof"``, normalized floating-point
        ``params``, and ``units="radians"``.
    :raises TypeError: If ``spec`` is not a :class:`FiveDOFSpec` instance.
    """
    if not isinstance(spec, FiveDOFSpec):
        raise TypeError(f"spec must be FiveDOFSpec; got {type(spec).__name__}.")
    return {
        "format": "five_dof",
        "params": [_json_float(value) for value in spec.params],
        "units": "radians",
    }


def _pq_payload(spec: PQSpec) -> dict[str, Any]:
    """Serialize a validated exact P/Q specification.

    Exact-integer validation and nonsingularity checks are owned by :class:`PQSpec`,
    which delegates to :mod:`GBOpt.Utils.integer_linalg`. This helper only converts the
    immutable normalized tuples stored by the spec into JSON lists.

    :param spec: Validated :class:`PQSpec` containing exact integer ``P`` and ``Q``
        matrices and a normalized basis mode.
    :return: JSON-safe dictionary with ``format="pq"``, nested integer ``P`` and ``Q``
        rows, and the validated ``basis_mode``.
    :raises TypeError: If ``spec`` is not a :class:`PQSpec` instance.
    """
    if not isinstance(spec, PQSpec):
        raise TypeError(f"spec must be PQSpec; got {type(spec).__name__}.")
    return {
        "format": "pq",
        "P": [[int(value) for value in row] for row in spec.P],
        "Q": [[int(value) for value in row] for row in spec.Q],
        "basis_mode": spec.basis_mode,
    }


def _csl_payload(spec: CSLExactSpec | CSLApproxSpec) -> dict[str, Any]:
    """Serialize a validated exact or approximate CSL specification.

    :param spec: Validated :class:`CSLExactSpec` or :class:`CSLApproxSpec` instance.
    :return: JSON-safe tagged CSL dictionary. Exact specifications contain ``quat`` and
        ``exact=true``; approximate specifications contain ``angle_deg`` and
        ``exact=false``. ``sigma`` is included only when present on the spec.
    :raises TypeError: If ``spec`` is neither a :class:`CSLExactSpec` nor a
        :class:`CSLApproxSpec`.
    """
    payload: dict[str, Any] = {
        "format": "csl",
        "axis": [int(value) for value in spec.axis],
        "plane": [int(value) for value in spec.plane],
    }
    if isinstance(spec, CSLExactSpec):
        payload.update(
            exact=True,
            quat=[int(value) for value in spec.quat],
        )
    elif isinstance(spec, CSLApproxSpec):
        payload.update(
            exact=False,
            angle_deg=_json_float(spec.angle_deg),
        )
    else:
        raise TypeError(
            f"spec must be CSLExactSpec or CSLApproxSpec; got {type(spec).__name__}."
        )

    if spec.sigma is not None:
        payload["sigma"] = int(spec.sigma)
    return payload


def _build_csl_spec(
    axis: object,
    plane: object,
    *,
    quat: object | None = None,
    angle_deg: object | None = None,
    sigma: object | None = None,
) -> CSLExactSpec | CSLApproxSpec:
    """Construct one validated exact or approximate CSL specification.

    Exactly one of ``quat`` and ``angle_deg`` selects the CSL variant. All exact integer
    shape, type, nonzero, quaternion-axis, and Sigma checks are delegated to the
    boundary-spec dataclasses and ultimately to :mod:`GBOpt.Utils.integer_linalg`.

    :param axis: Three-component exact integer rotation axis accepted by the CSL spec
        dataclasses.
    :param plane: Three-component exact integer boundary-plane normal accepted by the
        CSL spec dataclasses.
    :param quat: Optional four-component exact integer quaternion ``[w, x, y, z]`` for
        an exact CSL. Keyword-only, mutually exclusive with ``angle_deg``, defaults to
        ``None``.
    :param angle_deg: Optional finite rotation angle in degrees for an approximate CSL.
        Keyword-only, mutually exclusive with ``quat``, defaults to ``None``.
    :param sigma: Optional positive exact integer Sigma value. Keyword-only, defaults to
        ``None``.
    :return: Validated :class:`CSLExactSpec` when ``quat`` is supplied, otherwise a
        validated :class:`CSLApproxSpec`.
    :raises ValueError: If both or neither of ``quat`` and ``angle_deg`` are supplied.
    """
    if (quat is None) == (angle_deg is None):
        raise ValueError("csl requires exactly one of quat or angle_deg.")

    if quat is not None:
        return CSLExactSpec(axis=axis, plane=plane, sigma=sigma, quat=quat)

    return CSLApproxSpec(axis=axis, plane=plane, sigma=sigma, angle_deg=angle_deg)


def _normalize_csl_payload(
    payload: Mapping[str, object],
) -> tuple[CSLExactSpec | CSLApproxSpec, dict[str, Any]]:
    """Normalize and validate a CSL core-format mapping.

    The ``exact`` discriminator must agree with the variant field: exact payloads
    require ``quat`` and prohibit ``angle_deg``; approximate payloads require
    ``angle_deg`` and prohibit ``quat``. For backward compatibility, ``exact`` may be
    omitted when exactly one variant field makes the intended type unambiguous.

    :param payload: Mapping containing ``axis``, ``plane``, optional ``sigma``, and a
        consistent exactness discriminator plus exactly one of ``quat`` or
        ``angle_deg``.
    :return: Tuple ``(spec, normalized_payload)`` containing the validated exact or
        approximate CSL specification followed by its JSON-safe dictionary.
    :raises ValueError: If ``exact`` is non-boolean or the variant fields are missing or
        contradictory.
    """
    has_quat = payload.get("quat") is not None
    has_angle = payload.get("angle_deg") is not None
    exact_value = payload.get("exact")

    if exact_value is None:
        if has_quat == has_angle:
            raise ValueError(
                "A csl payload without 'exact' must contain exactly one of 'quat' "
                "or 'angle_deg'."
            )
        exact = has_quat
    else:
        if not isinstance(exact_value, (bool, np.bool_)):
            raise ValueError("csl field 'exact' must be boolean.")
        exact = bool(exact_value)

    if exact:
        if not has_quat or has_angle:
            raise ValueError(
                "An exact csl payload requires 'quat' and must not contain 'angle_deg'."
            )
        spec = _build_csl_spec(
            payload["axis"],
            payload["plane"],
            quat=payload["quat"],
            sigma=payload.get("sigma"),
        )
    else:
        if not has_angle or has_quat:
            raise ValueError(
                "An approximate csl payload requires 'angle_deg' and must not contain "
                "'quat'."
            )
        spec = _build_csl_spec(
            payload["axis"],
            payload["plane"],
            angle_deg=payload["angle_deg"],
            sigma=payload.get("sigma"),
        )

    return spec, _csl_payload(spec)


def _normalize_core_payload(
    payload: object,
) -> tuple[
    dict[str, Any],
    FiveDOFSpec | PQSpec | CSLExactSpec | CSLApproxSpec,
]:
    """Normalize and validate any supported GBOpt core-format payload.

    The CLI owns only tagged-JSON schema dispatch. Field validation is delegated to the
    corresponding boundary-spec dataclass.

    :param payload: Candidate mapping whose ``format`` field identifies ``"five_dof"``,
        ``"pq"``, or ``"csl"`` data.
    :return: Tuple ``(normalized_payload, spec)`` containing a JSON-safe dictionary and
        its validated boundary-spec object.
    :raises ValueError: If ``payload`` is not a mapping or names an unsupported format.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Core-format input must be a JSON object.")

    format_name = payload.get("format")
    if format_name == "five_dof":
        spec = FiveDOFSpec(payload["params"])  # type: ignore[arg-type]
        return _five_dof_payload(spec), spec

    if format_name == "pq":
        basis_mode = payload.get("basis_mode", "primitive")
        spec = PQSpec(
            payload["P"],  # type: ignore[arg-type]
            payload["Q"],  # type: ignore[arg-type]
            basis_mode=basis_mode,  # type: ignore[arg-type]
        )
        return _pq_payload(spec), spec

    if format_name == "csl":
        spec, normalized = _normalize_csl_payload(payload)
        return normalized, spec

    raise ValueError("Core-format input must have format 'five_dof', 'pq', or 'csl'.")


def _embedding_from_spec(
    spec: FiveDOFSpec | PQSpec | CSLExactSpec | CSLApproxSpec,
    *,
    max_exact_atoms: int,
) -> BoundaryEmbedding:
    """Convert a validated boundary specification to its package embedding.

    :param spec: Validated five-DOF, P/Q, exact CSL, or approximate CSL specification.
    :param max_exact_atoms: Exact-cell size bound forwarded only to exact CSL embedding
        construction. Keyword-only.
    :return: Boundary embedding produced by the corresponding crystallography adapter.
    :raises TypeError: If ``spec`` has an unsupported type.
    """
    if isinstance(spec, FiveDOFSpec):
        return five_dof_spec_to_embedding(spec)
    if isinstance(spec, PQSpec):
        return pq_spec_to_embedding(spec)
    if isinstance(spec, CSLExactSpec):
        return csl_exact_spec_to_embedding(
            spec,
            max_exact_atoms=max_exact_atoms,
        )
    if isinstance(spec, CSLApproxSpec):
        return csl_approx_spec_to_embedding(spec)
    raise TypeError(f"Unsupported boundary spec type: {type(spec).__name__}.")


def _five_dof_spec_from_embedding(embedding: BoundaryEmbedding) -> FiveDOFSpec:
    """Recover a validated five-DOF specification from a boundary embedding.

    Relative rotation and inclination recovery are delegated to
    :func:`five_dof_from_orientation_matrices`; this CLI module does not duplicate the
    row-orientation convention or Euler-angle extraction.

    :param embedding: Boundary embedding whose ``R_left`` and ``R_right`` matrices
        describe the left- and right-grain row orientations.
    :return: Validated :class:`FiveDOFSpec` recovered from the embedding frames.
    """
    params = five_dof_from_orientation_matrices(
        embedding.R_left,
        embedding.R_right,
    )
    return FiveDOFSpec(params)


def _convert_payload(
    payload: object,
    target: str,
    *,
    max_exact_atoms: int = 10_000,
) -> dict[str, Any]:
    """Convert a supported core-format payload to another representation.

    Tagged-JSON normalization is performed once. Domain conversion is then delegated to
    boundary-spec embedding adapters, P/Q orientation conversion, or the exactification
    hook. Exact CSL conversion constructs only one embedding per requested conversion.

    :param payload: Input object in ``five_dof``, ``pq``, or ``csl`` core format.
    :param target: Requested output format, currently ``"five_dof"`` or ``"pq"``.
    :param max_exact_atoms: Exact-cell size bound forwarded to exact CSL embedding and
        five-DOF exactification operations. Keyword-only, defaults to ``10_000``.
    :return: Normalized JSON-safe payload in ``target`` format. If source and target are
        equal, the normalized source payload is returned.
    :raises ValueError: If ``target`` is unsupported, the requested conversion is
        unavailable, or five-DOF exactification is not implemented.
    """
    normalized, spec = _normalize_core_payload(payload)
    source = normalized["format"]
    if source == target:
        return normalized

    if target == "five_dof":
        embedding = _embedding_from_spec(
            spec,
            max_exact_atoms=max_exact_atoms,
        )
        return _five_dof_payload(_five_dof_spec_from_embedding(embedding))

    if target == "pq":
        if isinstance(spec, CSLExactSpec):
            embedding = _embedding_from_spec(
                spec,
                max_exact_atoms=max_exact_atoms,
            )
            if embedding.P is None or embedding.Q is None:
                raise ValueError("Exact CSL embedding did not provide P/Q matrices.")
            return _pq_payload(PQSpec(embedding.P, embedding.Q, basis_mode="primitive"))

        if isinstance(spec, FiveDOFSpec):
            try:
                P, Q = exactify_five_dof(
                    np.asarray(spec.params, dtype=np.float64),
                    max_exact_atoms=max_exact_atoms,
                )
            except NotImplementedError as exc:
                raise ValueError(
                    "Conversion from 'five_dof' to 'pq' requires five_dof "
                    "exactification, which is not implemented."
                ) from exc
            return _pq_payload(PQSpec(P, Q, basis_mode="primitive"))

    raise ValueError(f"Conversion from {source!r} to {target!r} is not available.")


def _load_core_payload(args: argparse.Namespace) -> object:
    """Load one JSON core-format payload from the CLI-selected input source.

    Input precedence is ``input_file``, then ``input_json``, then standard input. The
    parser's mutually exclusive options normally ensure that no more than one explicit
    source is set.

    :param args: Parsed :class:`argparse.Namespace` exposing ``input_file`` and
        ``input_json`` attributes. Each attribute may be ``None``.
    :return: The Python object decoded from the selected JSON source.
    """
    if args.input_file is not None:
        with open(args.input_file, encoding="utf-8") as stream:
            return json.load(stream)
    if args.input_json is not None:
        return json.loads(args.input_json)
    return json.load(sys.stdin)


def _print_payload(payload: Mapping[str, object]) -> None:
    """Serialize and print one core-format payload as formatted JSON.

    :param payload: Mapping to encode using two-space indentation and lexicographically
        sorted keys.
    """
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _command_axis_angle(args: argparse.Namespace) -> int:
    """Handle the ``axis_angle`` CLI subcommand.

    :param args: Parsed :class:`argparse.Namespace` exposing ``axis``, ``angle``,
        ``normal``, and ``format``. The axis and normal each contain three numeric
        components, the angle is measured in degrees, and the format is ``"json"`` or
        ``"human"``.
    :return: Process status code ``0`` after writing the selected output format.
    """
    axis = np.asarray(args.axis, dtype=np.float64)
    normal = np.asarray(args.normal, dtype=np.float64)
    spec = FiveDOFSpec(five_dof_from_axis_angle(axis, args.angle, normal))

    if args.format == "json":
        _print_payload(_five_dof_payload(spec))
        return 0

    checks = _validation_report(spec.params, normal)
    summary = f"axis={axis.tolist()} angle={args.angle} deg normal={normal.tolist()}"
    print(format_human_output(spec.params, summary, checks))
    return 0


def _command_orientation(args: argparse.Namespace) -> int:
    """Handle the ``orientation`` CLI subcommand.

    :param args: Parsed :class:`argparse.Namespace` exposing row-major ``P`` and ``Q``
        sequences of nine numeric values, optional three-component ``normal``, and
        ``format`` equal to ``"json"`` or ``"human"``.
    :return: Process status code ``0`` after writing the selected output format.
    """
    P = np.asarray(args.P, dtype=np.float64).reshape(3, 3)
    Q = np.asarray(args.Q, dtype=np.float64).reshape(3, 3)
    normal = None if args.normal is None else np.asarray(args.normal, dtype=np.float64)
    spec = FiveDOFSpec(five_dof_from_orientation_matrices(P, Q, normal))

    if args.format == "json":
        _print_payload(_five_dof_payload(spec))
        return 0

    P_norm = validate_orientation_matrix(P, "P")
    Q_norm = validate_orientation_matrix(Q, "Q")
    checks = _validation_report(
        spec.params,
        P_norm[0],
        P=P_norm,
        Q=Q_norm,
    )
    summary = f"P={P.tolist()} Q={Q.tolist()}"
    print(format_human_output(spec.params, summary, checks))
    return 0


def _command_csl(args: argparse.Namespace) -> int:
    """Handle the ``csl`` CLI subcommand.

    The handler delegates field validation to a CSL boundary-spec dataclass, constructs
    the corresponding embedding as a feasibility check, and emits the normalized spec.

    :param args: Parsed :class:`argparse.Namespace` exposing integer ``axis`` and
        ``plane`` vectors, exactly one of ``quat`` or ``angle``, optional ``sigma``, and
        the ``max_exact_atoms`` bound.
    :return: Process status code ``0`` after successful validation and JSON emission.
    """
    spec = _build_csl_spec(
        args.axis,
        args.plane,
        quat=args.quat,
        angle_deg=args.angle,
        sigma=args.sigma,
    )
    _embedding_from_spec(spec, max_exact_atoms=args.max_exact_atoms)
    _print_payload(_csl_payload(spec))
    return 0


def _command_convert(args: argparse.Namespace) -> int:
    """Handle the ``convert`` CLI subcommand.

    :param args: Parsed :class:`argparse.Namespace` exposing ``input_file``,
        ``input_json``, target format ``to``, and ``max_exact_atoms``.
    :return: Process status code ``0`` after printing the converted JSON payload.
    """
    payload = _load_core_payload(args)
    _print_payload(
        _convert_payload(
            payload,
            args.to,
            max_exact_atoms=args.max_exact_atoms,
        )
    )
    return 0


def _command_exactify(args: argparse.Namespace) -> int:
    """Handle the ``exactify`` CLI subcommand.

    :param args: Parsed :class:`argparse.Namespace` exposing five floating-point
        ``params`` in radians and the ``max_exact_atoms`` bound.
    :return: Process status code ``0`` after printing the exact P/Q payload.
    :raises ValueError: If parameters cannot be converted to floating point or
        exactification is not implemented.
    """
    five_dof = FiveDOFSpec(args.params)
    try:
        P, Q = exactify_five_dof(
            np.asarray(five_dof.params, dtype=np.float64),
            max_exact_atoms=args.max_exact_atoms,
        )
    except NotImplementedError as exc:
        raise ValueError("five_dof exactification is not implemented.") from exc

    _print_payload(_pq_payload(PQSpec(P, Q, basis_mode="primitive")))
    return 0


def _command_canonicalize(args: argparse.Namespace) -> int:
    """Handle the ``canonicalize`` CLI subcommand.

    :param args: Parsed :class:`argparse.Namespace` exposing row-major exact integer
        ``P`` and ``Q`` sequences, each containing nine components.
    :return: Process status code ``0`` after printing the paired canonical P/Q payload.
    """
    P = np.asarray(args.P, dtype=object).reshape(3, 3)
    Q = np.asarray(args.Q, dtype=object).reshape(3, 3)
    P_canon, Q_canon = canonicalize_pq_paired(P, Q)
    _print_payload(_pq_payload(PQSpec(P_canon, Q_canon, basis_mode="primitive")))
    return 0


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    """Add the shared output-format option to a subcommand parser.

    :param parser: :class:`argparse.ArgumentParser` for a subcommand supporting JSON and
        legacy human-readable output. The parser is modified in place by adding a
        ``--format`` option whose default is ``"json"``.
    """
    parser.add_argument(
        "--format",
        choices=("json", "human"),
        default="json",
        help="Output JSON by default; use human for the legacy report.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the complete ``gb_params`` command-line parser.

    :return: Configured top-level :class:`argparse.ArgumentParser` containing the
        ``axis_angle``, ``orientation``, ``csl``, ``convert``, ``exactify``, and
        ``canonicalize`` subcommands, their options, and their registered handler
        functions.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert grain-boundary crystallographic descriptions into GBOpt core "
            "formats."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    axis_angle = subparsers.add_parser(
        "axis_angle",
        help="Convert a rotation axis, angle, and boundary normal to five_dof.",
    )
    axis_angle.add_argument(
        "--axis",
        nargs=3,
        type=float,
        metavar=("U", "V", "W"),
        required=True,
    )
    axis_angle.add_argument(
        "--angle",
        type=float,
        metavar="DEG",
        required=True,
    )
    axis_angle.add_argument(
        "--normal",
        nargs=3,
        type=float,
        metavar=("H", "K", "L"),
        required=True,
    )
    _add_output_format(axis_angle)
    axis_angle.set_defaults(handler=_command_axis_angle)

    orientation = subparsers.add_parser(
        "orientation",
        help="Convert row-wise P and Q orientation matrices to five_dof.",
    )
    orientation.add_argument(
        "--P",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help="Left-grain orientation matrix, nine row-major values.",
    )
    orientation.add_argument(
        "--Q",
        nargs=9,
        type=float,
        metavar="V",
        required=True,
        help="Right-grain orientation matrix, nine row-major values.",
    )
    orientation.add_argument(
        "--normal",
        nargs=3,
        type=float,
        metavar=("H", "K", "L"),
        default=None,
        help="Optional boundary-normal consistency check; P[0] remains authoritative.",
    )
    _add_output_format(orientation)
    orientation.set_defaults(handler=_command_orientation)

    csl = subparsers.add_parser(
        "csl",
        help="Validate and emit an exact or approximate CSL core-format spec.",
    )
    csl.add_argument(
        "--axis",
        nargs=3,
        type=int,
        metavar=("U", "V", "W"),
        required=True,
    )
    csl.add_argument(
        "--plane",
        nargs=3,
        type=int,
        metavar=("H", "K", "L"),
        required=True,
    )
    csl_kind = csl.add_mutually_exclusive_group(required=True)
    csl_kind.add_argument(
        "--quat",
        nargs=4,
        type=int,
        metavar=("W", "X", "Y", "Z"),
    )
    csl_kind.add_argument("--angle", type=float, metavar="DEG")
    csl.add_argument("--sigma", type=int, default=None)
    csl.add_argument("--max-exact-atoms", type=int, default=10_000)
    csl.set_defaults(handler=_command_csl)

    convert = subparsers.add_parser(
        "convert",
        help="Convert a JSON core-format spec to another supported format.",
    )
    convert_input = convert.add_mutually_exclusive_group()
    convert_input.add_argument("--input-json")
    convert_input.add_argument("--input-file")
    convert.add_argument("--to", choices=("five_dof", "pq"), required=True)
    convert.add_argument("--max-exact-atoms", type=int, default=10_000)
    convert.set_defaults(handler=_command_convert)

    exactify = subparsers.add_parser(
        "exactify",
        help="Exactify five_dof parameters through the exactification hook.",
    )
    exactify.add_argument(
        "--params",
        nargs=5,
        type=float,
        metavar=("ALPHA", "BETA", "GAMMA", "THETA", "PHI"),
        required=True,
    )
    exactify.add_argument("--max-exact-atoms", type=int, default=10_000)
    exactify.set_defaults(handler=_command_exactify)

    canonicalize = subparsers.add_parser(
        "canonicalize",
        help="Canonicalize paired exact integer P/Q matrices.",
    )
    canonicalize.add_argument(
        "--P",
        nargs=9,
        type=int,
        metavar="V",
        required=True,
    )
    canonicalize.add_argument(
        "--Q",
        nargs=9,
        type=int,
        metavar="V",
        required=True,
    )
    canonicalize.set_defaults(handler=_command_canonicalize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``gb_params`` command-line interface.

    The selected handler returns an integer process status. Expected domain, payload,
    I/O, and conversion failures are converted to :mod:`argparse` errors so command-line
    callers receive a diagnostic and exit status ``2``.

    :param argv: Optional argument sequence excluding the executable name. When
        ``None``, :mod:`argparse` reads from ``sys.argv``. Defaults to ``None``.
    :return: Process status ``0`` when the selected command succeeds. The trailing ``2``
        is a defensive return for static analysis because :meth:`ArgumentParser.error`
        raises ``SystemExit``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except (
        BoundarySpecError,
        CrystallographyValueError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    return 2


if __name__ == "__main__":
    sys.exit(main())
