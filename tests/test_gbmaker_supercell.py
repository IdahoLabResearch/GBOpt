# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.crystallography.integer import integer_adj3, integer_det3
from GBOpt.gbmaker_supercell import (
    _integer_membership,
    build_supercell_matrix,
    enumerate_supercell_origins,
)

SIGMA5_RIGHT_GRAIN = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
SIGMA5_RIGHT_GRAIN_ARRAY = np.array(SIGMA5_RIGHT_GRAIN, dtype=int)


@pytest.fixture
def sigma5_membership_data():
    return {
        "det_S": integer_det3(SIGMA5_RIGHT_GRAIN),
        "adj_S": integer_adj3(SIGMA5_RIGHT_GRAIN),
    }


def test_integer_membership_accepts_origin(sigma5_membership_data):
    assert _integer_membership(
        [0, 0, 0],
        sigma5_membership_data["adj_S"],
        sigma5_membership_data["det_S"],
        1,
        1,
        1,
    )


def test_integer_membership_counts_identity_repeated_cell():
    adj_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    det_I = 1

    accepted = sum(
        _integer_membership([x, y, z], adj_I, det_I, 3, 2, 1)
        for x in range(-1, 5)
        for y in range(-1, 4)
        for z in range(-1, 3)
    )

    assert accepted == 3 * 2 * 1


def test_integer_membership_rejects_exclusive_upper_boundary(sigma5_membership_data):
    assert not _integer_membership(
        [0, 0, 1],
        sigma5_membership_data["adj_S"],
        sigma5_membership_data["det_S"],
        1,
        1,
        1,
    )


def test_integer_membership_normalizes_negative_determinant_sign():
    matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    det_S = integer_det3(matrix)

    assert det_S == -1
    assert _integer_membership(
        [0, 0, 0],
        integer_adj3(matrix),
        det_S,
        1,
        1,
        1,
    )


def test_build_supercell_matrix_accepts_identity_p():
    P = np.eye(3, dtype=float)

    S = build_supercell_matrix(P)

    np.testing.assert_array_equal(S, np.eye(3, dtype=int))


def test_build_supercell_matrix_accepts_sigma5_right_grain():
    Q = SIGMA5_RIGHT_GRAIN_ARRAY.astype(float)

    S = build_supercell_matrix(Q)

    np.testing.assert_array_equal(S, SIGMA5_RIGHT_GRAIN_ARRAY)


@pytest.mark.parametrize(
    "P_bad",
    [
        np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[1, 0], [0, 1]], dtype=float),
    ],
)
def test_build_supercell_matrix_rejects_invalid_input(P_bad):
    with pytest.raises(ValueError):
        build_supercell_matrix(P_bad)


def test_build_supercell_matrix_rejects_singular_matrix():
    P = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

    with pytest.raises(ValueError, match="singular"):
        build_supercell_matrix(P)


def test_enumerate_supercell_origins_returns_single_origin_for_identity_cell():
    origins = enumerate_supercell_origins(np.eye(3, dtype=int), 1, 1, 1)

    assert origins.shape == (1, 3)
    np.testing.assert_array_equal(origins[0], [0, 0, 0])


def test_enumerate_supercell_origins_counts_identity_repeated_cell():
    origins = enumerate_supercell_origins(np.eye(3, dtype=int), 2, 2, 2)

    assert len(origins) == 8


def test_enumerate_supercell_origins_counts_sigma5_unit_repeat():
    origins = enumerate_supercell_origins(SIGMA5_RIGHT_GRAIN_ARRAY, 1, 1, 1)

    assert len(origins) == 25


def test_enumerate_supercell_origins_has_no_duplicates():
    origins = enumerate_supercell_origins(SIGMA5_RIGHT_GRAIN_ARRAY, 2, 1, 1)

    assert len({tuple(row) for row in origins.tolist()}) == len(origins)


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 1, 1), (1, 2, 3)])
def test_enumerate_supercell_origins_count_matches_repeats_times_index(repeats):
    origins = enumerate_supercell_origins(SIGMA5_RIGHT_GRAIN_ARRAY, *repeats)

    assert len(origins) == np.prod(repeats) * abs(integer_det3(SIGMA5_RIGHT_GRAIN))


def test_enumerate_supercell_origins_rejects_singular_matrix():
    S = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=int)

    with pytest.raises(ValueError, match="non-singular"):
        enumerate_supercell_origins(S, 1, 1, 1)


@pytest.mark.parametrize("repeats", [(0, 1, 1), (1, -1, 1), (1.0, 1, 1)])
def test_enumerate_supercell_origins_rejects_invalid_repeat_count(repeats):
    with pytest.raises(ValueError, match="positive integer"):
        enumerate_supercell_origins(np.eye(3, dtype=int), *repeats)
