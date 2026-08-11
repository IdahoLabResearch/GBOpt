# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import CSLApproxSpec, CSLExactSpec, FiveDOFSpec, PQSpec
from GBOpt.crystallography.boundary import csl_exact_spec_to_embedding
from GBOpt.crystallography.pq import canonicalize_pq_paired
from GBOpt.Utils import gb_params

REPO_ROOT = Path(__file__).resolve().parents[1]
GB_PARAMS_SCRIPT = REPO_ROOT / "GBOpt" / "Utils" / "gb_params.py"

FIVE_DOF_ZERO_SOURCE = {
    "format": "five_dof",
    "params": [0, 0, 0, 0, 0],
}
IDENTITY_PQ_SOURCE = {
    "format": "pq",
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Q": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "basis_mode": "primitive",
}
EXACT_CSL_SOURCE = {
    "format": "csl",
    "exact": True,
    "axis": [0, 0, 1],
    "plane": [1, 0, 0],
    "quat": [2, 0, 0, 1],
    "sigma": 5,
}
APPROXIMATE_CSL_SOURCE = {
    "format": "csl",
    "exact": False,
    "axis": [0, 0, 1],
    "plane": [1, 0, 0],
    "angle_deg": 17.3,
}


def _run_cli(
    *args: object,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the standalone ``gb_params`` script in a subprocess.

    :param args: Command-line arguments passed to the script.
    :param input_text: Optional text supplied to the subprocess standard input.
    :return: Completed subprocess result with captured text output.
    """
    return subprocess.run(
        [sys.executable, str(GB_PARAMS_SCRIPT), *map(str, args)],
        cwd=REPO_ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _decode_json_output(stdout: str, stderr: str = "") -> dict[str, Any]:
    """Decode a JSON object from command output with an informative test failure.

    :param stdout: Standard output expected to contain one JSON object.
    :param stderr: Standard error included in a failure diagnostic.
    :return: Decoded JSON object.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "gb_params did not produce valid JSON\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
            f"JSON error: {exc}"
        )

    assert isinstance(payload, dict), (
        f"Expected a JSON object, got {type(payload).__name__}"
    )
    return payload


def _run_main_json(capsys: pytest.CaptureFixture[str], *args: object) -> dict[str, Any]:
    """Run ``gb_params.main`` successfully and decode its JSON output.

    :param capsys: Pytest capture fixture used to collect command output.
    :param args: Command-line arguments passed to ``gb_params.main``.
    :return: Decoded JSON object printed by the command.
    """
    assert gb_params.main([str(value) for value in args]) == 0
    captured = capsys.readouterr()
    return _decode_json_output(captured.out, captured.err)


def _run_main_error(
    capsys: pytest.CaptureFixture[str],
    *args: object,
) -> str:
    """Run ``gb_params.main`` expecting argparse error termination.

    :param capsys: Pytest capture fixture used to collect command output.
    :param args: Command-line arguments passed to ``gb_params.main``.
    :return: Captured standard-error text.
    """
    with pytest.raises(SystemExit) as caught:
        gb_params.main([str(value) for value in args])

    assert caught.value.code == 2
    return capsys.readouterr().err


def _matrix_payload(matrix: np.ndarray) -> list[list[int]]:
    """Convert an exact matrix to nested Python-integer lists.

    :param matrix: Numerical matrix whose entries are exactly integer-valued.
    :return: Nested list preserving Python-sized integer values.
    """
    return [[int(value) for value in row] for row in np.asarray(matrix, dtype=object)]


def _assert_five_dof_payload(payload: dict[str, Any]) -> FiveDOFSpec:
    """Validate and return a five-DOF payload as ``FiveDOFSpec``.

    :param payload: Candidate tagged JSON payload.
    :return: Validated five-DOF specification.
    """
    assert payload["format"] == "five_dof"
    assert payload["units"] == "radians"
    return FiveDOFSpec(payload["params"])


# --------------------------------------------------------------------------------------
# Direct-script and basic command behavior
# --------------------------------------------------------------------------------------


def test_axis_angle_direct_script_outputs_five_dof_core_format():
    result = _run_cli(
        "axis_angle",
        "--axis", 1, 1, 1,
        "--angle", 60,
        "--normal", 1, 1, 1,
    )

    assert result.returncode == 0, result.stderr
    payload = _decode_json_output(result.stdout, result.stderr)
    spec = _assert_five_dof_payload(payload)

    reconstructed = Rotation.from_euler("ZXZ", spec.params[:3]).as_matrix()
    axis = np.ones(3) / np.sqrt(3.0)
    expected = Rotation.from_rotvec(axis * np.deg2rad(60.0)).as_matrix()
    np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12)


def test_orientation_outputs_five_dof_core_format(capsys):
    payload = _run_main_json(
        capsys,
        "orientation",
        "--P", 1, 0, 0, 0, 1, 0, 0, 0, 1,
        "--Q", 0, 1, 0, -1, 0, 0, 0, 0, 1,
    )

    spec = _assert_five_dof_payload(payload)
    reconstructed = Rotation.from_euler("ZXZ", spec.params[:3]).as_matrix()
    np.testing.assert_allclose(
        reconstructed,
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float),
        atol=1.0e-12,
    )


def test_axis_angle_human_output_reports_symbolic_multiple(capsys):
    angle_deg = 2.0 * np.degrees(np.arctan(1.0 / 5.0))

    assert gb_params.main(
        [
            "axis_angle",
            "--axis", "0", "0", "1",
            "--angle", str(angle_deg),
            "--normal", "1", "0", "0",
            "--format", "human",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "GBOpt Misorientation Parameters" in output
    assert "2*arctan(1/5)" in output
    assert "PASS: inclination reproduces the boundary normal" in output


# --------------------------------------------------------------------------------------
# CSL commands
# --------------------------------------------------------------------------------------


def test_csl_outputs_exact_core_format(capsys):
    payload = _run_main_json(
        capsys,
        "csl",
        "--axis", 0, 0, 1,
        "--plane", 1, 0, 0,
        "--quat", 2, 0, 0, 1,
        "--sigma", 5,
    )

    assert payload["format"] == "csl"
    assert payload["exact"] is True
    CSLExactSpec(
        axis=payload["axis"],
        plane=payload["plane"],
        quat=payload["quat"],
        sigma=payload["sigma"],
    )


def test_csl_outputs_approximate_core_format(capsys):
    payload = _run_main_json(
        capsys,
        "csl",
        "--axis", 0, 0, 1,
        "--plane", 1, 0, 0,
        "--angle", 17.3,
    )

    assert payload["format"] == "csl"
    assert payload["exact"] is False
    CSLApproxSpec(
        axis=payload["axis"],
        plane=payload["plane"],
        angle_deg=payload["angle_deg"],
    )


def test_csl_rejects_sigma_mismatch(capsys):
    stderr = _run_main_error(
        capsys,
        "csl",
        "--axis", 0, 0, 1,
        "--plane", 1, 0, 0,
        "--quat", 2, 0, 0, 1,
        "--sigma", 13,
    )

    assert "Sigma mismatch" in stderr


# --------------------------------------------------------------------------------------
# Core-format conversion
# --------------------------------------------------------------------------------------


def test_convert_exact_csl_outputs_pq_core_format(capsys):
    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "pq",
        "--input-json", json.dumps(EXACT_CSL_SOURCE),
    )

    assert payload["format"] == "pq"
    PQSpec(payload["P"], payload["Q"], basis_mode=payload["basis_mode"])

    source_spec = CSLExactSpec(
        axis=EXACT_CSL_SOURCE["axis"],
        plane=EXACT_CSL_SOURCE["plane"],
        quat=EXACT_CSL_SOURCE["quat"],
        sigma=EXACT_CSL_SOURCE["sigma"],
    )
    embedding = csl_exact_spec_to_embedding(source_spec)

    assert embedding.P is not None
    assert embedding.Q is not None
    assert payload["P"] == _matrix_payload(embedding.P)
    assert payload["Q"] == _matrix_payload(embedding.Q)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(EXACT_CSL_SOURCE, id="exact-csl"),
        pytest.param(APPROXIMATE_CSL_SOURCE, id="approximate-csl"),
        pytest.param(IDENTITY_PQ_SOURCE, id="pq"),
    ],
)
def test_convert_supported_sources_to_five_dof(capsys, source):
    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-json", json.dumps(source),
    )

    _assert_five_dof_payload(payload)


def test_convert_same_format_normalizes_five_dof_payload(capsys):
    source = {
        "format": "five_dof",
        "params": [-0.0, 1.0e-16, 0.25, -0.5, 1.0],
    }

    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-json", json.dumps(source),
    )

    assert payload == {
        "format": "five_dof",
        "params": [0.0, 0.0, 0.25, -0.5, 1.0],
        "units": "radians",
    }


def test_convert_same_pq_format_preserves_large_integers(capsys):
    large = 2**53 + 1
    source = {
        "format": "pq",
        "P": [[large, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Q": [[large, 0, 0], [0, 1, 0], [0, 0, 1]],
        "basis_mode": "supplied",
    }

    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "pq",
        "--input-json", json.dumps(source),
    )

    assert payload["P"][0][0] == large
    assert payload["Q"][0][0] == large
    assert isinstance(payload["P"][0][0], int)


def test_convert_accepts_unambiguous_legacy_csl_payload_without_exact(capsys):
    source = dict(EXACT_CSL_SOURCE)
    source.pop("exact")

    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-json", json.dumps(source),
    )

    _assert_five_dof_payload(payload)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        pytest.param(
            {
                "format": "csl",
                "exact": True,
                "axis": [0, 0, 1],
                "plane": [1, 0, 0],
                "angle_deg": 30.0,
            },
            "exact csl payload requires 'quat'",
            id="exact-with-angle",
        ),
        pytest.param(
            {
                "format": "csl",
                "exact": False,
                "axis": [0, 0, 1],
                "plane": [1, 0, 0],
                "quat": [2, 0, 0, 1],
            },
            "approximate csl payload requires 'angle_deg'",
            id="approximate-with-quaternion",
        ),
        pytest.param(
            {
                "format": "csl",
                "axis": [0, 0, 1],
                "plane": [1, 0, 0],
                "quat": [2, 0, 0, 1],
                "angle_deg": 30.0,
            },
            "exactly one of 'quat' or 'angle_deg'",
            id="legacy-ambiguous",
        ),
        pytest.param(
            {
                "format": "csl",
                "exact": "yes",
                "axis": [0, 0, 1],
                "plane": [1, 0, 0],
                "quat": [2, 0, 0, 1],
            },
            "field 'exact' must be boolean",
            id="nonboolean-discriminator",
        ),
    ],
)
def test_convert_rejects_contradictory_csl_payloads(capsys, source, message):
    stderr = _run_main_error(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-json", json.dumps(source),
    )

    assert message in stderr


def test_convert_rejects_noninteger_exact_pq_entry(capsys):
    source = {
        "format": "pq",
        "P": [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Q": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }

    stderr = _run_main_error(
        capsys,
        "convert",
        "--to", "pq",
        "--input-json", json.dumps(source),
    )

    assert "integer" in stderr


def test_convert_rejects_boolean_sigma(capsys):
    source = dict(EXACT_CSL_SOURCE, sigma=True)

    stderr = _run_main_error(
        capsys,
        "convert",
        "--to",
        "five_dof",
        "--input-json",
        json.dumps(source),
    )

    assert "sigma must not be boolean" in stderr


def test_convert_pq_to_five_dof_enforces_max_pq_determinant(capsys):
    source = {
        # fmt:off
        "format": "pq",
        "P": [[0, 0, 1], [3, 1, 0], [-1, 3, 0]],
        "Q": [[0, 0, 1], [3, -1, 0], [1, 3, 0]],
        "basis_mode": "supplied",
        # fmt:on
    }

    stderr = _run_main_error(
        capsys,
        "convert",
        "--to",
        "five_dof",
        "--max-pq-determinant",
        9,
        "--input-json",
        json.dumps(source),
    )

    assert "Exact P/Q determinant exceeds max_pq_determinant=9" in stderr
    assert "|det(P)|=10" in stderr
    assert "|det(Q)|=10" in stderr


# --------------------------------------------------------------------------------------
# Input sources and command errors
# --------------------------------------------------------------------------------------


def test_convert_reads_payload_from_file(capsys, tmp_path):
    input_file = tmp_path / "boundary.json"
    input_file.write_text(json.dumps(FIVE_DOF_ZERO_SOURCE), encoding="utf-8")

    payload = _run_main_json(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-file", input_file,
    )

    _assert_five_dof_payload(payload)


def test_convert_reads_payload_from_standard_input(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(FIVE_DOF_ZERO_SOURCE)))

    payload = _run_main_json(capsys, "convert", "--to", "five_dof")

    _assert_five_dof_payload(payload)


def test_convert_rejects_malformed_json(capsys):
    stderr = _run_main_error(
        capsys,
        "convert",
        "--to", "five_dof",
        "--input-json", "{not-json}",
    )

    assert "Expecting property name" in stderr


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ("exactify", "--params", 0, 0, 0, 0, 0),
            id="exactify-command",
        ),
        pytest.param(
            (
                "convert",
                "--to",
                "pq",
                "--input-json",
                json.dumps(FIVE_DOF_ZERO_SOURCE),
            ),
            id="convert-command",
        ),
    ],
)
def test_five_dof_exactification_commands_output_identity_pq(
    capsys,
    args,
):
    payload = _run_main_json(capsys, *args)

    assert payload == IDENTITY_PQ_SOURCE


def test_convert_rejects_unsupported_csl_target_at_parse_time(capsys):
    stderr = _run_main_error(
        capsys,
        "convert",
        "--to", "csl",
        "--input-json", json.dumps(FIVE_DOF_ZERO_SOURCE),
    )

    assert "invalid choice: 'csl'" in stderr


# --------------------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------------------


def test_canonicalize_outputs_paired_canonical_pq_core_format(capsys):
    P = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=object)
    Q = np.array([[0, 0, 1], [1, 2, 0], [-2, 1, 0]], dtype=object)

    payload = _run_main_json(
        capsys,
        "canonicalize",
        "--P", *P.ravel(),
        "--Q", *Q.ravel(),
    )

    expected_P, expected_Q = canonicalize_pq_paired(P, Q)

    assert payload["format"] == "pq"
    assert payload["basis_mode"] == "primitive"
    assert payload["P"] == _matrix_payload(expected_P)
    assert payload["Q"] == _matrix_payload(expected_Q)
