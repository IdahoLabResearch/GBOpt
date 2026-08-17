# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for exact integer supercell construction and enumeration."""

from itertools import product

import numpy as np
import pytest

from GBOpt.crystallography.integer import integer_adj3, integer_det3
from GBOpt.gbmaker_supercell import (
    _integer_membership,
    build_supercell_matrix,
    enumerate_supercell_origins,
    enumerate_supercell_sites,
)
from GBOpt.UnitCell import RationalBasis, UnitCell

# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

IDENTITY_ROWS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_RIGHT_GRAIN_ROWS = ((4, -3, 0), (3, 4, 0), (0, 0, 1))
OBLIQUE_INDEX2_ROWS = ((1, -1, 0), (1, 1, 0), (0, 0, 1))

INVALID_3X3_INTEGER_MATRICES = [
    pytest.param(
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        "integer-valued",
        id="non-integer-entry",
    ),
    pytest.param(
        [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
        "finite",
        id="non-finite-entry",
    ),
    pytest.param(
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        "not an integer",
        id="boolean-entry",
    ),
    pytest.param(
        [[1, 0], [0, 1]],
        "shape",
        id="wrong-shape",
    ),
]


def _int_matrix(rows) -> np.ndarray:
    """Return shared matrix rows as a fresh object-dtype array."""
    return np.array(rows, dtype=object)


def _origin_set(origins: np.ndarray) -> set[tuple[int, int, int]]:
    """Return integer origins as an order-independent set of tuples."""
    return {tuple(int(value) for value in row) for row in origins}


# ---------------------------------------------------------------------------
# _integer_membership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        pytest.param((0, 0, 0), True, id="lower-corner"),
        pytest.param((2, 1, 0), True, id="last-interior-origin"),
        pytest.param((-1, 0, 0), False, id="below-x"),
        pytest.param((0, -1, 0), False, id="below-y"),
        pytest.param((0, 0, -1), False, id="below-z"),
        pytest.param((3, 0, 0), False, id="exclusive-x-upper-bound"),
        pytest.param((0, 2, 0), False, id="exclusive-y-upper-bound"),
        pytest.param((0, 0, 1), False, id="exclusive-z-upper-bound"),
    ],
)
def test_integer_membership_uses_half_open_repeated_identity_cell(origin, expected):
    identity = _int_matrix(IDENTITY_ROWS)

    assert (
        _integer_membership(
            origin,
            integer_adj3(identity),
            integer_det3(identity),
            3,
            2,
            1,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# build_supercell_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orientation_rows",
    [
        pytest.param(IDENTITY_ROWS, id="identity"),
        pytest.param(SIGMA5_RIGHT_GRAIN_ROWS, id="sigma5-right-grain"),
        pytest.param(OBLIQUE_INDEX2_ROWS, id="oblique-index-2"),
    ],
)
def test_build_supercell_matrix_returns_canonical_orientation_rows(orientation_rows):
    orientation = np.array(orientation_rows, dtype=float)

    result = build_supercell_matrix(orientation)

    np.testing.assert_array_equal(result, _int_matrix(orientation_rows))


@pytest.mark.parametrize(
    ("bad_matrix", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_build_supercell_matrix_rejects_invalid_matrix(bad_matrix, match):
    with pytest.raises(ValueError, match=match):
        build_supercell_matrix(bad_matrix)


def test_build_supercell_matrix_preserves_large_exact_integers():
    large = 10**20
    orientation_rows = (
        (1, -large, 0),
        (large, 1, 0),
        (0, 0, 1),
    )

    result = build_supercell_matrix(_int_matrix(orientation_rows))

    np.testing.assert_array_equal(result, _int_matrix(orientation_rows))
    assert result.dtype == object
    assert all(type(value) is int for value in result.flat)


def test_build_supercell_matrix_rejects_singular_array_like():
    singular = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]

    with pytest.raises(ValueError, match="singular"):
        build_supercell_matrix(singular)  # type: ignore[ty:invalid-argument-type]


@pytest.mark.parametrize(
    "noncanonical_rows",
    [
        pytest.param(
            ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
            id="nonprimitive-normal",
        ),
        pytest.param(
            ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
            id="left-handed",
        ),
        pytest.param(
            ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
            id="normal-does-not-match-inplane-cross-product",
        ),
    ],
)
def test_build_supercell_matrix_rejects_noncanonical_orientation_rows(
    noncanonical_rows,
):
    with pytest.raises(ValueError, match="canonical and right-handed"):
        build_supercell_matrix(noncanonical_rows)


# ---------------------------------------------------------------------------
# enumerate_supercell_origins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supercell_rows", "repeats", "expected"),
    [
        pytest.param(
            IDENTITY_ROWS,
            (2, 2, 2),
            set(product(range(2), repeat=3)),
            id="identity-2x2x2",
        ),
        pytest.param(
            OBLIQUE_INDEX2_ROWS,
            (1, 1, 1),
            {(0, 0, 0), (1, 0, 0)},
            id="oblique-index-2",
        ),
    ],
)
def test_enumerate_supercell_origins_returns_known_origin_set(
    supercell_rows,
    repeats,
    expected,
):
    origins = enumerate_supercell_origins(
        _int_matrix(supercell_rows),
        *repeats,
    )

    assert origins.shape == (len(expected), 3)
    assert _origin_set(origins) == expected


def test_enumerate_supercell_origins_preserves_known_oblique_order():
    origins = enumerate_supercell_origins(
        _int_matrix(OBLIQUE_INDEX2_ROWS),
        1,
        1,
        1,
    )

    np.testing.assert_array_equal(
        origins,
        np.array([[0, 0, 0], [1, 0, 0]], dtype=int),
    )


@pytest.mark.parametrize(
    "repeats",
    [
        pytest.param((1, 1, 1), id="unit-repeat"),
        pytest.param((2, 1, 1), id="repeated-normal"),
        pytest.param((1, 2, 3), id="repeated-inplane"),
    ],
)
def test_enumerate_supercell_origins_satisfies_count_uniqueness_and_membership(
    repeats,
):
    supercell = _int_matrix(SIGMA5_RIGHT_GRAIN_ROWS)
    origins = enumerate_supercell_origins(supercell, *repeats)
    determinant = integer_det3(supercell)
    adjugate = np.asarray(integer_adj3(supercell), dtype=object)
    numerators = np.asarray(origins, dtype=object) @ adjugate
    upper_bounds = np.asarray(repeats, dtype=object) * determinant

    assert origins.shape == (int(np.prod(repeats)) * determinant, 3)
    assert len({tuple(int(value) for value in row) for row in origins}) == len(origins)
    assert np.all(numerators >= 0)
    assert np.all(numerators < upper_bounds)


@pytest.mark.parametrize(
    ("bad_supercell", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_enumerate_supercell_origins_rejects_invalid_supercell(
    bad_supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_origins(bad_supercell, 1, 1, 1)


@pytest.mark.parametrize(
    ("supercell", "match"),
    [
        pytest.param(
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            "non-singular",
            id="singular",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "positive determinant",
            id="negative-determinant",
        ),
    ],
)
def test_enumerate_supercell_origins_rejects_invalid_determinant(
    supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_origins(supercell, 1, 1, 1)


@pytest.mark.parametrize(
    ("repeats", "exception", "match"),
    [
        pytest.param((0, 1, 1), ValueError, "repeat_x", id="zero-x"),
        pytest.param((1, -1, 1), ValueError, "repeat_y", id="negative-y"),
        pytest.param((1, 1, 1.5), TypeError, "repeat_z", id="float-z"),
        pytest.param((True, 1, 1), TypeError, "repeat_x", id="boolean-x"),
        pytest.param(
            (1, np.bool_(True), 1),
            TypeError,
            "repeat_y",
            id="numpy-boolean-y",
        ),
    ],
)
def test_enumerate_supercell_origins_rejects_invalid_repeat_count(
    repeats,
    exception,
    match,
):
    with pytest.raises(exception, match=match):
        enumerate_supercell_origins(_int_matrix(IDENTITY_ROWS), *repeats)


def test_enumerate_supercell_origins_accepts_numpy_integer_repeat_counts():
    origins = enumerate_supercell_origins(
        _int_matrix(IDENTITY_ROWS),
        np.int64(2),  # type: ignore[ty:invalid-argument-type]
        np.int64(1),  # type: ignore[ty:invalid-argument-type]
        np.int64(1),  # type: ignore[ty:invalid-argument-type]
    )

    assert _origin_set(origins) == {(0, 0, 0), (1, 0, 0)}


# ---------------------------------------------------------------------------
# Exact decorated-site enumeration
# ---------------------------------------------------------------------------


def _single_site_basis() -> RationalBasis:
    """Return a fresh exact one-site basis for validation-focused tests."""
    return RationalBasis(
        names=("Cu",),
        numerators=np.array([[0, 0, 0]], dtype=object),
        denominator=1,
    )


def test_enumerate_supercell_sites_fluorite_population_and_species():
    cell = UnitCell()
    cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    sites = enumerate_supercell_sites(
        np.eye(3, dtype=object),
        2,
        3,
        1,
        rational_basis=cell.rational_basis,
    )

    assert sites.site_count == 12 * 2 * 3
    counts = np.bincount(sites.basis_indices, minlength=12)
    np.testing.assert_array_equal(counts, np.full(12, 6))
    assert cell.rational_basis is not None
    names = np.asarray(cell.rational_basis.names)[sites.basis_indices]
    assert np.count_nonzero(names == "U") == 24
    assert np.count_nonzero(names == "O") == 48


def test_enumerate_supercell_sites_preserves_nontrivial_quotient_population():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.615, "Cu")

    sites = enumerate_supercell_sites(
        _int_matrix(SIGMA5_RIGHT_GRAIN_ROWS),
        2,
        1,
        1,
        rational_basis=cell.rational_basis,
    )

    assert sites.supercell_index == 25
    assert sites.site_count == 200
    counts = np.bincount(sites.basis_indices, minlength=4)
    np.testing.assert_array_equal(counts, np.full(4, 50))
    coordinates = {
        tuple(int(value) for value in row)
        for row in sites.coordinate_numerators
    }
    assert len(coordinates) == sites.site_count


def test_enumerate_supercell_sites_wraps_decorated_sites_without_loss():
    basis = RationalBasis(
        names=("Cu", "Cu"),
        numerators=np.array([[0, 0, 0], [3, 0, 0]], dtype=object),
        denominator=4,
    )
    supercell = np.array(
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        dtype=object,
    )

    sites = enumerate_supercell_sites(
        supercell,
        1,
        1,
        1,
        rational_basis=basis,
    )

    assert sites.site_count == 2
    np.testing.assert_array_equal(sites.basis_indices, np.array([0, 1]))
    np.testing.assert_array_equal(
        sites.coordinate_numerators,
        np.array([[0, 0, 0], [3, 1, 0]], dtype=object),
    )


def test_enumerate_supercell_sites_is_deterministic():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.615, "Cu")
    supercell = _int_matrix(SIGMA5_RIGHT_GRAIN_ROWS)

    first = enumerate_supercell_sites(
        supercell, 2, 1, 1, rational_basis=cell.rational_basis
    )
    second = enumerate_supercell_sites(
        supercell, 2, 1, 1, rational_basis=cell.rational_basis
    )

    np.testing.assert_array_equal(
        first.coordinate_numerators,
        second.coordinate_numerators,
    )
    np.testing.assert_array_equal(first.basis_indices, second.basis_indices)
    np.testing.assert_array_equal(first.supercell_matrix, second.supercell_matrix)


def test_enumerate_supercell_sites_returns_read_only_defensive_arrays():
    sites = enumerate_supercell_sites(
        np.eye(3, dtype=object),
        1,
        1,
        1,
        rational_basis=_single_site_basis(),
    )

    first_coordinates = sites.coordinate_numerators
    second_coordinates = sites.coordinate_numerators
    basis_indices = sites.basis_indices
    supercell_matrix = sites.supercell_matrix

    assert first_coordinates is not second_coordinates
    np.testing.assert_array_equal(first_coordinates, second_coordinates)
    assert not first_coordinates.flags.writeable
    assert not basis_indices.flags.writeable
    assert not supercell_matrix.flags.writeable

    with pytest.raises(ValueError):
        first_coordinates[0, 0] = 1
    with pytest.raises(ValueError):
        basis_indices[0] = 1
    with pytest.raises(ValueError):
        supercell_matrix[0, 0] = 2


@pytest.mark.parametrize(
    ("rational_basis", "match"),
    [
        pytest.param(
            None,
            "requires UnitCell.rational_basis",
            id="missing",
        ),
        pytest.param(
            object(),
            "must be a validated UnitCell.RationalBasis",
            id="wrong-type",
        ),
    ],
)
def test_enumerate_supercell_sites_requires_validated_rational_basis(
    rational_basis,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            np.eye(3, dtype=object),
            1,
            1,
            1,
            rational_basis=rational_basis,
        )


@pytest.mark.parametrize(
    ("repeats", "exception", "match"),
    [
        pytest.param((0, 1, 1), ValueError, "repeat_x", id="zero"),
        pytest.param((1, 1.5, 1), TypeError, "repeat_y", id="wrong-type"),
    ],
)
def test_enumerate_supercell_sites_rejects_invalid_repeat_count(
    repeats,
    exception,
    match,
):
    with pytest.raises(exception, match=match):
        enumerate_supercell_sites(
            np.eye(3, dtype=object),
            *repeats,
            rational_basis=_single_site_basis(),
        )


@pytest.mark.parametrize(
    ("supercell", "match"),
    [
        pytest.param(
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            "non-singular",
            id="singular",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "positive determinant",
            id="negative-determinant",
        ),
    ],
)
def test_enumerate_supercell_sites_rejects_invalid_determinant(supercell, match):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            supercell,
            1,
            1,
            1,
            rational_basis=_single_site_basis(),
        )
