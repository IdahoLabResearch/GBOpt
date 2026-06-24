# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared test data for crystallography test suite.

This module provides shared constants and scenario definitions used across multiple test
files. Import explicitly where needed, since these are data constants rather than pytest
fixtures.

Usage:
    from crystallography_fixtures import CSL_SCENARIO_DICTS, SIGMA5_36_P, ...
"""

import numpy as np

# ---------------------------------------------------------------------------
# P/Q matrix constants
# ---------------------------------------------------------------------------

SIGMA5_36_P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_36_Q = ((4, -3, 0), (3, 4, 0), (0, 0, 1))

SIGMA5_TWIST_LEGACY_P = ((0, 0, 1), (3, 1, 0), (-1, 3, 0))
SIGMA5_TWIST_LEGACY_Q = ((0, 0, 1), (3, -1, 0), (1, 3, 0))

SIGMA5_TWIST_PRIMITIVE_P = ((0, 0, 1), (1, 2, 0), (-2, 1, 0))
SIGMA5_TWIST_PRIMITIVE_Q = ((0, 0, 1), (2, 1, 0), (-1, 2, 0))

# --------------------------------------------------------------------------------------
# CSL scenario dicts
#
# Used by test_crystallography_csl.py and test_crystallography_embedding.py. Each dict
# contains the quaternion, plane, and expected values for a known CSL boundary. Build
# pytest.param objects from these in each test file as:
#
#   EXACT_CSL_SCENARIOS = [
#           pytest.param(d, id=str(d["id"])) for d in CSL_SCENARIO_DICTS
#           ]
# --------------------------------------------------------------------------------------

CSL_SCENARIO_DICTS = [
    {
        "id": "sigma5_001",
        "q": (2, 0, 0, 1),
        "plane": (0, 0, 1),
        "expected_N": 5,
        "expected_M": np.array(
            [[3, -4, 0], [4, 3, 0], [0, 0, 5]], dtype=object
        ),
        "expected_sigma": 5,
        "expected_hnf_det": 5,
        "expected_basis_hnf": np.array(
            [[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object
        ),
        "expected_kernel_moduli": (5, 1, 1)
    },
    {
        "id": "symmetry_quaternion_sigma_one",
        "q": (1, 1, 1, 1),
        "plane": None,
        "expected_N": 4,
        "expected_M": np.array(
            [[0, 0, 4], [4, 0, 0], [0, 4, 0]], dtype=object
        ),
        "expected_sigma": 1,
        "expected_hnf_det": 1,
        "expected_kernel_moduli": (1, 1, 1),
    },
    {
        "id": "sigma3_111",
        "q": (1, 1, 1, 0),
        "plane": (1, 1, 1),
        "expected_N": 3,
        "expected_M": np.array(
            [[1, 2, 2], [2, 1, -2], [-2, 2, -1]], dtype=object
        ),
        "expected_sigma": 3,
        "expected_hnf_det": 3,
        "expected_inplane_cross_abs": np.array([3, 3, 3]),
        "expected_kernel_moduli": (3, 1, 1)
    },
]
