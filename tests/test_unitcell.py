# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Unit tests for conventional unit-cell construction and exact basis metadata."""

import math

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.UnitCell import UnitCell, UnitCellTypeError, UnitCellValueError

FCC_RECIPROCAL = np.array(
    [
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
)
IDENTITY = np.eye(3)

BUILTIN_STRUCTURE_CASES = (
    {
        "structure": "sc",
        "atoms": "H",
        "radius": 0.5,
        "positions": [[0.0, 0.0, 0.0]],
        "names": ["H"],
        "reciprocal": IDENTITY,
        "ideal_bonds": {(1, 1): 1.0},
        "ratio": {1: 1},
        "rational_denominator": 1,
    },
    {
        "structure": "bcc",
        "atoms": "Fe",
        "radius": math.sqrt(3) * 0.25,
        "positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        "names": ["Fe", "Fe"],
        "reciprocal": np.array(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        ),
        "ideal_bonds": {(1, 1): math.sqrt(3) / 2},
        "ratio": {1: 1},
        "rational_denominator": 2,
    },
    {
        "structure": "fcc",
        "atoms": "Cu",
        "radius": math.sqrt(2) * 0.25,
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        "names": ["Cu"] * 4,
        "reciprocal": FCC_RECIPROCAL,
        "ideal_bonds": {(1, 1): 1 / math.sqrt(2)},
        "ratio": {1: 1},
        "rational_denominator": 2,
    },
    {
        "structure": "diamond",
        "atoms": "C",
        "radius": math.sqrt(3) * 0.125,
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        "names": ["C"] * 8,
        "reciprocal": FCC_RECIPROCAL,
        "ideal_bonds": {(1, 1): math.sqrt(3) / 4},
        "ratio": {1: 1},
        "rational_denominator": 4,
    },
    {
        "structure": "fluorite",
        "atoms": ("Ca", "F"),
        "radius": math.sqrt(3) * 0.125,
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.75],
        ],
        "names": ["Ca"] * 4 + ["F"] * 8,
        "reciprocal": FCC_RECIPROCAL,
        "ideal_bonds": {
            (1, 1): 1 / math.sqrt(2),
            (1, 2): math.sqrt(3) / 4,
            (2, 2): 0.5,
        },
        "ratio": {1: 1, 2: 2},
        "rational_denominator": 4,
    },
    {
        "structure": "rocksalt",
        "atoms": ("Na", "Cl"),
        "radius": 0.25,
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        "names": ["Na"] * 4 + ["Cl"] * 4,
        "reciprocal": FCC_RECIPROCAL,
        "ideal_bonds": {
            (1, 1): 1 / math.sqrt(2),
            (1, 2): 0.5,
            (2, 2): 1 / math.sqrt(2),
        },
        "ratio": {1: 1, 2: 1},
        "rational_denominator": 2,
    },
    {
        "structure": "zincblende",
        "atoms": ("Zn", "S"),
        "radius": math.sqrt(3) * 0.125,
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        "names": ["Zn"] * 4 + ["S"] * 4,
        "reciprocal": FCC_RECIPROCAL,
        "ideal_bonds": {
            (1, 1): 1 / math.sqrt(2),
            (1, 2): math.sqrt(3) / 4,
            (2, 2): 1 / math.sqrt(2),
        },
        "ratio": {1: 1, 2: 1},
        "rational_denominator": 4,
    },
)

MONATOMIC_NEIGHBOR_CASES = (
    ("sc", "Po", 3.345, (1.0, math.sqrt(2), math.sqrt(3), 2.0)),
    ("bcc", "Fe", 2.86, (math.sqrt(3) / 2, 1.0, math.sqrt(2), math.sqrt(11) / 2)),
    ("fcc", "Cu", 3.54, (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2))),
    (
        "diamond",
        "C",
        3.567,
        (math.sqrt(3) / 4, 1 / math.sqrt(2), math.sqrt(11) / 4, 1.0),
    ),
)

BINARY_NEIGHBOR_CASES = (
    (
        "fluorite",
        ("U", "O"),
        5.454,
        (math.sqrt(3) / 4, 1 / math.sqrt(2), math.sqrt(11) / 4, 1.0),
        (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2)),
        (0.5, 1 / math.sqrt(2), math.sqrt(3) / 2, 1.0),
    ),
    (
        "rocksalt",
        ("Na", "Cl"),
        5.454,
        (0.5, 1 / math.sqrt(2), math.sqrt(3) / 2, 1.0),
        (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2)),
        (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2)),
    ),
    (
        "zincblende",
        ("Zn", "S"),
        5.454,
        (math.sqrt(3) / 4, 1 / math.sqrt(2), math.sqrt(11) / 4, 1.0),
        (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2)),
        (1 / math.sqrt(2), 1.0, math.sqrt(6) / 2, math.sqrt(2)),
    ),
)


def _custom_init_kwargs() -> dict:
    return {
        "unit_cell": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "unit_cell_types": ["H", "C"],
        "a0": 1.0,
        "conventional": np.eye(3),
        "reciprocal": np.eye(3),
        "ideal_bond_lengths": {(1, 1): 1.0, (1, 2): 1.0, (2, 2): 1.0},
        "ratio": {1: 1, 2: 1},
    }


def _assert_neighbor_shells(
    cell: UnitCell,
    expected: tuple[float, ...],
    atom_type: int | None = None,
) -> None:
    for shell, expected_distance in enumerate(expected, start=1):
        actual = cell.nn_distance(shell, atom_type)
        assert actual == pytest.approx(expected_distance), f"neighbor shell {shell}"


# ---------------------------------------------------------------------------
# Default state and built-in construction
# ---------------------------------------------------------------------------


def test_default_unit_cell_state():
    cell = UnitCell()

    assert cell.unit_cell == []
    np.testing.assert_array_equal(cell.primitive, np.zeros((3, 3)))
    assert cell.a0 == 1.0
    assert cell.radius == 0.0
    np.testing.assert_array_equal(cell.reciprocal, np.zeros((3, 3)))
    assert cell.type_map == {}
    assert cell.rational_basis is None


@pytest.mark.parametrize(
    "case",
    BUILTIN_STRUCTURE_CASES,
    ids=lambda case: case["structure"],
)
def test_builtin_structure_initialization(case):
    cell = UnitCell()
    cell.init_by_structure(case["structure"], 1.0, case["atoms"])

    assert cell.a0 == 1.0
    assert cell.radius == pytest.approx(case["radius"])
    assert len(cell.unit_cell) == len(case["positions"])
    np.testing.assert_allclose(
        cell.positions(),
        np.asarray(case["positions"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(cell.names(), np.asarray(case["names"]))
    np.testing.assert_allclose(
        cell.reciprocal,
        case["reciprocal"],
        rtol=0.0,
        atol=1e-12,
    )
    assert cell.ideal_bond_lengths == pytest.approx(case["ideal_bonds"])
    assert cell.ratio == case["ratio"]


def test_builtin_positions_scale_with_initial_lattice_parameter():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.54, "Cu")

    coordinates = np.column_stack(
        (cell.asarray()["x"], cell.asarray()["y"], cell.asarray()["z"])
    )
    np.testing.assert_allclose(
        coordinates,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.77, 1.77],
                [1.77, 0.0, 1.77],
                [1.77, 1.77, 0.0],
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_init_by_structure_rejects_unsupported_structure():
    cell = UnitCell()

    with pytest.raises(NotImplementedError, match="not recognized/implemented"):
        cell.init_by_structure("notimplemented", 1.0, "H")


@pytest.mark.parametrize("structure", ["fluorite", "rocksalt", "zincblende"])
def test_binary_structures_require_two_atom_types(structure):
    cell = UnitCell()

    with pytest.raises(UnitCellValueError, match="requires exactly 2 atom types"):
        cell.init_by_structure(structure, 1.0, "H")


# ---------------------------------------------------------------------------
# Custom construction and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "exception", "match"),
    [
        pytest.param(
            "unit_cell_types",
            "H",
            UnitCellValueError,
            "length mismatch",
            id="type-count-mismatch",
        ),
        pytest.param(
            "ratio",
            {1: 0, 2: 2},
            UnitCellValueError,
            "positive ints",
            id="nonpositive-ratio-value",
        ),
        pytest.param(
            "ratio",
            {0: 1, 1: 1},
            UnitCellValueError,
            "positive ints",
            id="nonpositive-ratio-key",
        ),
        pytest.param(
            "ratio",
            "Error",
            UnitCellTypeError,
            "ratio must be a dict",
            id="invalid-ratio-type",
        ),
        pytest.param(
            "reciprocal",
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            UnitCellValueError,
            "incorrect shape for reciprocal vectors",
            id="invalid-reciprocal-shape",
        ),
    ],
)
def test_init_by_custom_rejects_invalid_input(field, value, exception, match):
    kwargs = _custom_init_kwargs()
    kwargs[field] = value

    with pytest.raises(exception, match=match):
        UnitCell().init_by_custom(**kwargs)


def test_custom_initialization_sets_requested_state():
    cell = UnitCell()
    kwargs = _custom_init_kwargs()
    cell.init_by_custom(**kwargs)

    assert cell.a0 == 1.0
    assert len(cell.unit_cell) == 2
    np.testing.assert_array_equal(cell.names(), np.array(["H", "C"]))
    np.testing.assert_allclose(
        cell.positions(), kwargs["unit_cell"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_array_equal(cell.reciprocal, np.eye(3))
    assert cell.ideal_bond_lengths == pytest.approx(kwargs["ideal_bond_lengths"])
    assert cell.ratio == kwargs["ratio"]
    assert cell.type_map == {"H": 1, "C": 2}
    assert cell.rational_basis is None


def test_formula_ratio_is_normalized_species_metadata_for_multicomponent_cell():
    cell = UnitCell()
    species = ["H"] * 2 + ["He"] * 4 + ["Li"] * 6
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 0.55, len(species)),
            np.zeros(len(species)),
            np.zeros(len(species)),
        )
    )
    cell.init_by_custom(
        coordinates,
        species,
        1.0,
        np.eye(3),
        np.eye(3),
        {},
        ratio={1: 2, 2: 4, 3: 6},
    )

    assert cell.formula_ratio == (("H", 1), ("He", 2), ("Li", 3))


# ---------------------------------------------------------------------------
# Conversion, representation, and type mapping
# ---------------------------------------------------------------------------


def test_asarray_returns_expected_atom_records():
    cell = UnitCell()
    cell.init_by_structure("fcc", 1.0, "Cu")
    expected = np.array(
        [
            ("Cu", 0.0, 0.0, 0.0),
            ("Cu", 0.0, 0.5, 0.5),
            ("Cu", 0.5, 0.0, 0.5),
            ("Cu", 0.5, 0.5, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )

    np.testing.assert_array_equal(cell.asarray(), expected)


def test_repr_default_unit_cell():
    assert repr(UnitCell()) == (
        "UnitCell with 0 atoms\n"
        "Lattice parameter (a0): 1.000 Å\n"
        "Radius: 0.000 Å\n"
        "Atoms: []\n"
        "Reciprocal lattice:\n"
        "[[0. 0. 0.]\n [0. 0. 0.]\n [0. 0. 0.]]"
    )


def test_repr_initialized_unit_cell():
    cell = UnitCell()
    cell.init_by_structure("sc", 1.0, "H")

    assert repr(cell) == (
        "UnitCell with 1 atom\n"
        "Lattice parameter (a0): 1.000 Å\n"
        "Radius: 0.500 Å\n"
        "Atoms: ['H': 0.000, 0.000, 0.000]\n"
        "Reciprocal lattice:\n"
        "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
    )


@pytest.mark.parametrize(
    ("type_map", "expected"),
    [
        pytest.param({"O": 1, "U": 2}, {"U": 2, "O": 1}, id="string-to-int"),
        pytest.param({1: "U", 2: "O"}, {"U": 1, "O": 2}, id="int-to-string"),
    ],
)
def test_type_map_accepts_both_mapping_directions(type_map, expected):
    cell = UnitCell()
    cell.init_by_structure("fluorite", 2.0, ("U", "O"))

    cell.type_map = type_map

    assert cell.type_map == expected


@pytest.mark.parametrize(
    ("value", "exception", "match"),
    [
        pytest.param(
            {2: "U", 3: "O"},
            UnitCellValueError,
            "minimum integer value",
            id="integer-keys-do-not-start-at-one",
        ),
        pytest.param(
            {"O": 2, "U": 3},
            UnitCellValueError,
            "minimum integer value",
            id="integer-values-do-not-start-at-one",
        ),
        pytest.param(
            "Error",
            UnitCellTypeError,
            "type_map must be a dict",
            id="not-a-dictionary",
        ),
    ],
)
def test_type_map_rejects_invalid_mappings(value, exception, match):
    cell = UnitCell()
    cell.init_by_structure("fluorite", 2.0, ("U", "O"))

    with pytest.raises(exception, match=match):
        cell.type_map = value


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        pytest.param(None, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], id="default"),
        pytest.param(
            {"F": 1, "Ca": 2},
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            id="custom",
        ),
    ],
)
def test_types_respects_type_map(mapping, expected):
    cell = UnitCell()
    cell.init_by_structure("fluorite", 5.52, ("Ca", "F"))
    if mapping is not None:
        cell.type_map = mapping

    np.testing.assert_array_equal(cell.types(), np.asarray(expected))


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        pytest.param(None, [1, 1, 1, 1], id="default"),
        pytest.param(
            {"Ca": 2, "F": 1},
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            id="custom",
        ),
    ],
)
def test_names_as_ints_respects_type_map(mapping, expected):
    cell = UnitCell()
    if mapping is None:
        cell.init_by_structure("fcc", 1.0, "Cu")
    else:
        cell.init_by_structure("fluorite", 5.52, ("Ca", "F"), type_map=mapping)

    np.testing.assert_array_equal(
        cell.names(asint=True), np.asarray(expected, dtype=int)
    )


# ---------------------------------------------------------------------------
# Lattice-parameter mutation
# ---------------------------------------------------------------------------


def test_a0_rejects_nonpositive_value():
    cell = UnitCell()
    cell.init_by_structure("fcc", 1.0, "Ni")

    with pytest.raises(UnitCellValueError, match="Must be > 0"):
        cell.a0 = -1.0

    assert cell.a0 == 1.0


def test_a0_rescales_geometry_without_changing_rational_basis():
    cell = UnitCell()
    cell.init_by_structure("fcc", 1.0, "Ni")
    basis_before = cell.rational_basis
    assert basis_before is not None
    numerators_before = basis_before.numerators

    cell.a0 = 2.0

    np.testing.assert_allclose(
        cell.primitive,
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    assert cell.radius == pytest.approx(math.sqrt(2) * 0.5)
    np.testing.assert_allclose(
        cell.reciprocal,
        np.array(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
        ),
        rtol=0.0,
        atol=1e-12,
    )
    assert cell.rational_basis is not None
    assert cell.rational_basis.denominator == basis_before.denominator
    assert cell.rational_basis.names == basis_before.names
    np.testing.assert_array_equal(cell.rational_basis.numerators, numerators_before)


# ---------------------------------------------------------------------------
# Nearest-neighbor distances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("structure", "atoms", "scaled_a0", "expected"),
    MONATOMIC_NEIGHBOR_CASES,
    ids=[case[0] for case in MONATOMIC_NEIGHBOR_CASES],
)
def test_monatomic_neighbor_shells_scale_with_lattice_parameter(
    structure, atoms, scaled_a0, expected
):
    cell = UnitCell()
    cell.init_by_structure(structure, 1.0, atoms)
    _assert_neighbor_shells(cell, expected)

    cell.a0 = scaled_a0
    _assert_neighbor_shells(cell, tuple(scaled_a0 * value for value in expected))


@pytest.mark.parametrize(
    ("structure", "atoms", "scaled_a0", "all_types", "type_one", "type_two"),
    BINARY_NEIGHBOR_CASES,
    ids=[case[0] for case in BINARY_NEIGHBOR_CASES],
)
def test_binary_neighbor_shells_scale_with_lattice_parameter(
    structure,
    atoms,
    scaled_a0,
    all_types,
    type_one,
    type_two,
):
    cell = UnitCell()
    cell.init_by_structure(structure, 1.0, atoms)

    _assert_neighbor_shells(cell, all_types)
    _assert_neighbor_shells(cell, type_one, atom_type=1)
    _assert_neighbor_shells(cell, type_two, atom_type=2)

    cell.a0 = scaled_a0
    _assert_neighbor_shells(
        cell, tuple(scaled_a0 * value for value in all_types)
    )
    _assert_neighbor_shells(
        cell, tuple(scaled_a0 * value for value in type_one), atom_type=1
    )
    _assert_neighbor_shells(
        cell, tuple(scaled_a0 * value for value in type_two), atom_type=2
    )


# ---------------------------------------------------------------------------
# Exact rational basis metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    BUILTIN_STRUCTURE_CASES,
    ids=lambda case: case["structure"],
)
def test_builtin_rational_basis_matches_structured_basis(case):
    cell = UnitCell()
    cell.init_by_structure(case["structure"], 2.5, case["atoms"])

    basis = cell.rational_basis
    assert basis is not None
    assert basis.denominator == case["rational_denominator"]
    assert basis.names == tuple(case["names"])

    numerators = basis.numerators
    assert numerators.shape == (len(case["positions"]), 3)
    assert len({tuple(int(value) for value in row) for row in numerators}) == len(
        case["positions"]
    )
    assert all(
        0 <= int(value) < basis.denominator
        for value in numerators.flat
    )
    coordinates = np.column_stack(
        (cell.asarray()["x"], cell.asarray()["y"], cell.asarray()["z"])
    )
    np.testing.assert_allclose(
        coordinates / cell.a0,
        np.asarray(numerators, dtype=float) / basis.denominator,
        rtol=0.0,
        atol=1e-12,
    )


def test_rational_basis_numerators_are_defensive_and_read_only():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.615, "Cu")

    assert cell.rational_basis is not None
    first = cell.rational_basis.numerators
    second = cell.rational_basis.numerators

    assert first is not second
    assert not first.flags.writeable
    assert not second.flags.writeable
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        first[0, 0] = 1
    np.testing.assert_array_equal(
        cell.rational_basis.numerators,
        np.array(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
            dtype=object,
        ),
    )


def test_custom_reinitialization_clears_rational_basis_metadata():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.615, "Cu")
    assert cell.rational_basis is not None

    cell.init_by_custom(**_custom_init_kwargs())

    assert cell.rational_basis is None


def test_builtin_reinitialization_replaces_rational_basis_metadata():
    cell = UnitCell()
    cell.init_by_structure("fcc", 3.615, "Cu")
    fcc_basis = cell.rational_basis

    cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    assert cell.rational_basis is not fcc_basis
    assert cell.rational_basis is not None
    assert cell.rational_basis.denominator == 4
    assert cell.rational_basis.names.count("U") == 4
    assert cell.rational_basis.names.count("O") == 8
