# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from GBOpt.BoundarySpec import CSLApproxSpec, CSLExactSpec, FiveDOFSpec, PQSpec
from GBOpt.Utils.gb_exact import canonicalize_pq, csl_spec_to_embedding


class TestGBParamsCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.script = cls.repo_root / "GBOpt" / "Utils" / "gb_params.py"

    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, str(self.script), *map(str, args)],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    @staticmethod
    def matrix_payload(matrix):
        arr = np.asarray(matrix, dtype=float)
        return [[int(round(v)) for v in row] for row in arr]

    def test_axis_angle_outputs_five_dof_core_format(self):
        payload = self.run_cli(
            "axis_angle",
            "--axis", 1, 1, 1,
            "--angle", 60,
            "--normal", 1, 1, 1,
        )

        self.assertEqual(payload["format"], "five_dof")
        self.assertEqual(payload["units"], "radians")
        FiveDOFSpec(payload["params"])

    def test_orientation_outputs_five_dof_core_format(self):
        payload = self.run_cli(
            "orientation",
            "--P", 1, 0, 0, 0, 1, 0, 0, 0, 1,
            "--Q", 0, 1, 0, 0, 0, 1, 1, 0, 0,
        )

        self.assertEqual(payload["format"], "five_dof")
        FiveDOFSpec(payload["params"])

    def test_csl_outputs_exact_core_format(self):
        payload = self.run_cli(
            "csl",
            "--axis", 0, 0, 1,
            "--plane", 1, 0, 0,
            "--quat", 2, 0, 0, 1,
            "--sigma", 5,
        )

        self.assertEqual(payload["format"], "csl")
        self.assertTrue(payload["exact"])
        CSLExactSpec(
            axis=payload["axis"],
            plane=payload["plane"],
            quat=payload["quat"],
            sigma=payload["sigma"],
        )

    def test_csl_outputs_approximate_core_format(self):
        payload = self.run_cli(
            "csl",
            "--axis", 0, 0, 1,
            "--plane", 1, 0, 0,
            "--angle", 17.3,
        )

        self.assertEqual(payload["format"], "csl")
        self.assertFalse(payload["exact"])
        CSLApproxSpec(
            axis=payload["axis"],
            plane=payload["plane"],
            angle_deg=payload["angle_deg"],
        )

    def test_convert_exact_csl_outputs_pq_core_format(self):
        source = {
            "format": "csl",
            "exact": True,
            "axis": [0, 0, 1],
            "plane": [1, 0, 0],
            "quat": [2, 0, 0, 1],
            "sigma": 5,
        }

        payload = self.run_cli("convert", "--to", "pq", "--input-json", json.dumps(source))

        self.assertEqual(payload["format"], "pq")
        PQSpec(payload["P"], payload["Q"], basis_mode=payload["basis_mode"])
        embedding = csl_spec_to_embedding(
            CSLExactSpec(
                axis=source["axis"],
                plane=source["plane"],
                quat=source["quat"],
                sigma=source["sigma"],
            )
        )
        self.assertEqual(payload["P"], self.matrix_payload(embedding.P))
        self.assertEqual(payload["Q"], self.matrix_payload(embedding.Q))

    def test_exactify_reports_stage_e_hook(self):
        payload = self.run_cli("exactify", "--params", 0, 0, 0, 0, 0)

        self.assertEqual(payload["status"], "not_implemented")
        self.assertIn("Stage E", payload["message"])

    def test_canonicalize_invokes_canonicalize_pq(self):
        P = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
        Q = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
        payload = self.run_cli(
            "canonicalize",
            "--P", *P.ravel().tolist(),
            "--Q", *Q.ravel().tolist(),
        )

        expected_P, expected_Q = canonicalize_pq(P, Q)
        self.assertEqual(payload["format"], "pq")
        self.assertEqual(payload["P"], self.matrix_payload(expected_P))
        self.assertEqual(payload["Q"], self.matrix_payload(expected_Q))


if __name__ == "__main__":
    unittest.main()
