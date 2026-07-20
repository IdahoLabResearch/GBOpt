# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from GBOpt.BoundarySpec import CSLApproxSpec, CSLExactSpec, FiveDOFSpec, PQSpec
from GBOpt.crystallography.boundary import csl_exact_spec_to_embedding
from GBOpt.crystallography.embedding import canonicalize_pq

REPO_ROOT = Path(__file__).resolve().parents[1]
GB_PARAMS_SCRIPT = REPO_ROOT / "GBOpt" / "Utils" / "gb_params.py"

FIVE_DOF_ZERO_SOURCE = {
    "format": "five_dof",
    "params": [0, 0, 0, 0, 0],
}


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    """Run the gb_params CLI and return the completed subprocess."""
    return subprocess.run(
        [sys.executable, str(GB_PARAMS_SCRIPT), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_cli_json(*args: object) -> dict[str, Any]:
    """Run a successful CLI command and decode its JSON output."""
    result = _run_cli(*args)

    assert result.returncode == 0, (
        f"gb_params exited with status {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "gb_params did not produce valid JSON\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"JSON error: {exc}"
        )

    assert isinstance(payload, dict), (
        f"Expected a JSON object, got {type(payload).__name__}"
    )
    return payload


def _matrix_payload(matrix: np.ndarray) -> list[list[int]]:
    """Convert a numerical matrix to the CLI's nested integer-list format."""
    return np.rint(np.asarray(matrix, dtype=float)).astype(int).tolist()


def _assert_five_dof_payload(payload: dict[str, Any]) -> None:
    """Assert that a payload conforms to the FiveDOF core format."""
    assert payload["format"] == "five_dof"
    FiveDOFSpec(payload["params"])


def test_axis_angle_outputs_five_dof_core_format():
    payload = _run_cli_json(
        "axis_angle",
        "--axis", 1, 1, 1,
        "--angle", 60,
        "--normal", 1, 1, 1,
    )

    _assert_five_dof_payload(payload)
    assert payload["units"] == "radians"


def test_orientation_outputs_five_dof_core_format():
    payload = _run_cli_json(
        "orientation",
        "--P", 1, 0, 0, 0, 1, 0, 0, 0, 1,
        "--Q", 0, 1, 0, 0, 0, 1, 1, 0, 0,
    )

    _assert_five_dof_payload(payload)


def test_csl_outputs_exact_core_format():
    payload = _run_cli_json(
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


def test_csl_outputs_approximate_core_format():
    payload = _run_cli_json(
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


def test_convert_exact_csl_outputs_pq_core_format():
    source = {
        "format": "csl",
        "exact": True,
        "axis": [0, 0, 1],
        "plane": [1, 0, 0],
        "quat": [2, 0, 0, 1],
        "sigma": 5,
    }

    payload = _run_cli_json(
        "convert",
        "--to",
        "pq",
        "--input-json",
        json.dumps(source),
    )

    assert payload["format"] == "pq"

    PQSpec(
        payload["P"],
        payload["Q"],
        basis_mode=payload["basis_mode"],
    )

    source_spec = CSLExactSpec(
        axis=source["axis"],
        plane=source["plane"],
        quat=source["quat"],
        sigma=source["sigma"],
    )
    embedding = csl_exact_spec_to_embedding(source_spec)

    assert embedding.P is not None
    assert embedding.Q is not None
    assert payload["P"] == _matrix_payload(embedding.P)
    assert payload["Q"] == _matrix_payload(embedding.Q)


def test_exactify_reports_stage_e_hook():
    payload = _run_cli_json(
        "exactify",
        "--params", 0, 0, 0, 0, 0,
    )

    assert payload["status"] == "not_implemented"
    assert "Stage E" in payload["message"]


def test_convert_five_dof_to_pq_reports_exactification_gap():
    result = _run_cli(
        "convert",
        "--to",
        "pq",
        "--input-json",
        json.dumps(FIVE_DOF_ZERO_SOURCE),
    )

    assert result.returncode != 0
    assert "five_dof exactification" in result.stderr
    assert "not yet implemented" in result.stderr


def test_convert_rejects_unsupported_csl_target_at_parse_time():
    result = _run_cli(
        "convert",
        "--to",
        "csl",
        "--input-json",
        json.dumps(FIVE_DOF_ZERO_SOURCE),
    )

    assert result.returncode != 0
    assert "invalid choice: 'csl'" in result.stderr


def test_canonicalize_outputs_canonical_pq_core_format():
    P = np.diag([2.0, 3.0, 4.0])
    Q = np.diag([2.0, 3.0, 4.0])

    payload = _run_cli_json(
        "canonicalize",
        "--P",
        *P.ravel(),
        "--Q",
        *Q.ravel(),
    )

    expected_P, expected_Q = canonicalize_pq(P, Q)

    assert payload["format"] == "pq"
    assert payload["P"] == _matrix_payload(expected_P)
    assert payload["Q"] == _matrix_payload(expected_Q)
