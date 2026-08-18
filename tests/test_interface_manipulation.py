# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused topology-aware termination and interface-separation regressions."""

import numpy as np
import pytest

from GBOpt.BoundarySpec import PQSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorTypeError,
    GBManipulatorValueError,
)

LATTICE_CONSTANT_ANGSTROM = 3.615
SLAB_VACUUM_ANGSTROM = 2.0
INTERFACE_SEPARATION_ANGSTROM = 0.6
MANIPULATOR_SEED = 7

IDENTITY_PQ = PQSpec(
    P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    basis_mode="supplied",
)


def _build_boundary(*, vacuum: float) -> GBMaker:
    return GBMaker.from_boundary_spec(
        LATTICE_CONSTANT_ANGSTROM,
        "fcc",
        "Cu",
        IDENTITY_PQ,
        mode="exact",
        gb_thickness=0.0,
        repeat_factor=2,
        interaction_distance=LATTICE_CONSTANT_ANGSTROM,
        x_dim_min=8.0,
        vacuum=vacuum,
    )


def _expected_grain_labels(gb: GBMaker) -> np.ndarray:
    return np.hstack(
        (
            np.zeros(len(gb.left_grain), dtype=np.int8),
            np.ones(len(gb.right_grain), dtype=np.int8),
        )
    )


def _periodic_interface_gaps(candidate) -> tuple[float, float]:
    atoms = candidate.atoms
    labels = candidate.grain_labels
    left_x = atoms["x"][labels == 0]
    right_x = atoms["x"][labels == 1]
    xlo, xhi = candidate.box_dims[0]

    central_gap = float(np.min(right_x) - np.max(left_x))
    periodic_gap = float((xhi - np.max(right_x)) + (np.min(left_x) - xlo))
    return central_gap, periodic_gap


# ---------------------------------------------------------------------------
# Termination cycling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_zero_cycle_grain_terminations_preserves_parent_atoms(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    before = gb.whole_system.copy()

    atoms = manipulator.cycle_grain_terminations()

    np.testing.assert_array_equal(atoms, before)
    np.testing.assert_array_equal(gb.whole_system, before)


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_zero_termination_candidate_preserves_parent_atoms_and_labels(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    before = gb.whole_system.copy()

    candidate = manipulator.make_termination_candidate()

    np.testing.assert_array_equal(candidate.atoms, before)
    np.testing.assert_array_equal(candidate.grain_labels, _expected_grain_labels(gb))
    np.testing.assert_array_equal(gb.whole_system, before)


def test_interface_candidate_array_properties_are_defensive_and_read_only():
    gb = _build_boundary(vacuum=0.0)
    candidate = GBManipulator(gb, seed=MANIPULATOR_SEED).make_parent_candidate()

    arrays = (
        candidate.atoms,
        candidate.box_dims,
        candidate.left_grain_x_bounds,
        candidate.right_grain_x_bounds,
        candidate.grain_labels,
    )
    for array in arrays:
        assert not array.flags.writeable

    first_atoms = candidate.atoms
    second_atoms = candidate.atoms
    assert first_atoms is not second_atoms
    np.testing.assert_array_equal(first_atoms, second_atoms)

    with pytest.raises(ValueError):
        first_atoms[0]["x"] = 0.0


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_termination_cycle_is_grain_local_and_preserves_population(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    before = gb.whole_system.copy()
    left_width = gb.gb_plane_x - (gb.box_dims[0, 0] + vacuum)
    right_lower = gb.gb_plane_x
    right_upper = gb.box_dims[0, 1] - vacuum
    right_width = right_upper - right_lower

    candidate = manipulator.make_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
        right_dy=0.75,
        right_dz=1.25,
    )
    split = len(gb.left_grain)
    left = candidate.atoms[:split]
    right = candidate.atoms[split:]
    tolerance = candidate.coordinate_tolerance

    expected_left_x = (
        gb.box_dims[0, 0]
        + vacuum
        + np.mod(
            gb.left_grain["x"] + 0.25 - (gb.box_dims[0, 0] + vacuum),
            left_width,
        )
    )
    expected_right_x = right_lower + np.mod(
        gb.right_grain["x"] + 0.5 - right_lower,
        right_width,
    )

    np.testing.assert_allclose(
        left["x"], expected_left_x, atol=tolerance, rtol=0.0
    )
    np.testing.assert_array_equal(left["y"], gb.left_grain["y"])
    np.testing.assert_array_equal(left["z"], gb.left_grain["z"])
    np.testing.assert_array_equal(left["name"], gb.left_grain["name"])
    np.testing.assert_allclose(
        right["x"], expected_right_x, atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        right["y"],
        np.mod(gb.right_grain["y"] + 0.75, gb.y_dim),
        atol=tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        right["z"],
        np.mod(gb.right_grain["z"] + 1.25, gb.z_dim),
        atol=tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(right["name"], gb.right_grain["name"])
    np.testing.assert_array_equal(candidate.atoms["name"], before["name"])
    assert len(candidate.atoms) == len(before)
    np.testing.assert_array_equal(gb.whole_system, before)


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_termination_cycle_by_full_grain_width_reproduces_parent(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    left_width = gb.gb_plane_x - (gb.box_dims[0, 0] + vacuum)
    right_width = (gb.box_dims[0, 1] - vacuum) - gb.gb_plane_x

    candidate = manipulator.make_termination_candidate(
        left_phase_shift=left_width,
        right_phase_shift=right_width,
    )

    np.testing.assert_allclose(
        candidate.atoms["x"],
        gb.whole_system["x"],
        atol=candidate.coordinate_tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(candidate.atoms["name"], gb.whole_system["name"])


# ---------------------------------------------------------------------------
# Interface separation
# ---------------------------------------------------------------------------


def test_periodic_interface_separation_updates_both_interface_gaps():
    gb = _build_boundary(vacuum=0.0)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    base = manipulator.make_translation_candidate(0.0, 0.0)
    base_atoms = base.atoms
    base_box = base.box_dims
    base_central_gap, base_periodic_gap = _periodic_interface_gaps(base)
    separation = INTERFACE_SEPARATION_ANGSTROM

    separated = manipulator.apply_interface_separation(
        base,
        interface_separation=separation,
    )
    split = len(gb.left_grain)
    central_gap, periodic_gap = _periodic_interface_gaps(separated)

    assert separated.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert separated.periodic_outer_x_interface
    assert separated.box_dims[0, 1] == pytest.approx(
        gb.box_dims[0, 1] + 2.0 * separation
    )
    assert separated.gb_plane_x == pytest.approx(gb.gb_plane_x + separation / 2.0)
    np.testing.assert_array_equal(separated.atoms[:split], gb.left_grain)
    np.testing.assert_allclose(
        separated.atoms[split:]["x"],
        gb.right_grain["x"] + separation,
        atol=separated.coordinate_tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        separated.atoms[split:]["name"], gb.right_grain["name"]
    )
    assert separated.left_grain_x_bounds.tolist() == pytest.approx(
        [gb.box_dims[0, 0], gb.gb_plane_x]
    )
    assert separated.right_grain_x_bounds.tolist() == pytest.approx(
        [gb.gb_plane_x + separation, gb.box_dims[0, 1] + separation]
    )
    assert separated.interface_separation == pytest.approx(separation)
    assert central_gap == pytest.approx(base_central_gap + separation)
    assert periodic_gap == pytest.approx(base_periodic_gap + separation)

    np.testing.assert_array_equal(base.atoms, base_atoms)
    np.testing.assert_array_equal(base.box_dims, base_box)
    assert base.interface_separation == pytest.approx(0.0)


def test_slab_interface_separation_preserves_outer_vacuum_widths():
    vacuum = SLAB_VACUUM_ANGSTROM
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    base = manipulator.make_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
    )
    base_atoms = base.atoms
    base_box = base.box_dims
    separation = INTERFACE_SEPARATION_ANGSTROM

    separated = manipulator.apply_interface_separation(
        base,
        interface_separation=separation,
    )
    split = len(gb.left_grain)

    assert separated.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert not separated.periodic_outer_x_interface
    assert separated.box_dims[0, 1] == pytest.approx(gb.box_dims[0, 1] + separation)
    assert separated.gb_plane_x == pytest.approx(gb.gb_plane_x + separation / 2.0)
    np.testing.assert_array_equal(separated.atoms[:split], base.atoms[:split])
    np.testing.assert_allclose(
        separated.atoms[split:]["x"],
        base.atoms[split:]["x"] + separation,
        atol=separated.coordinate_tolerance,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        separated.atoms[split:]["name"], base.atoms[split:]["name"]
    )
    assert separated.left_grain_x_bounds[0] - separated.box_dims[0, 0] == pytest.approx(
        vacuum
    )
    assert separated.box_dims[0, 1] - separated.right_grain_x_bounds[1] == pytest.approx(
        vacuum
    )
    assert separated.interface_separation == pytest.approx(separation)

    np.testing.assert_array_equal(base.atoms, base_atoms)
    np.testing.assert_array_equal(base.box_dims, base_box)
    assert base.interface_separation == pytest.approx(0.0)


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_zero_interface_separation_is_structurally_noop(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    base = manipulator.make_parent_candidate()

    separated = manipulator.apply_interface_separation(
        base,
        interface_separation=0.0,
    )

    np.testing.assert_array_equal(separated.atoms, base.atoms)
    np.testing.assert_array_equal(separated.box_dims, base.box_dims)
    np.testing.assert_array_equal(
        separated.left_grain_x_bounds, base.left_grain_x_bounds
    )
    np.testing.assert_array_equal(
        separated.right_grain_x_bounds, base.right_grain_x_bounds
    )
    np.testing.assert_array_equal(separated.grain_labels, base.grain_labels)
    assert separated.gb_plane_x == pytest.approx(base.gb_plane_x)
    assert separated.interface_separation == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("interface_separation", "exception", "match"),
    [
        pytest.param(-0.1, GBManipulatorValueError, "nonnegative", id="negative"),
        pytest.param(np.nan, GBManipulatorValueError, "finite real", id="nan"),
        pytest.param(np.inf, GBManipulatorValueError, "finite real", id="infinite"),
        pytest.param(True, GBManipulatorTypeError, "finite real", id="boolean"),
        pytest.param("0.5", GBManipulatorTypeError, "finite real", id="string"),
    ],
)
def test_interface_separation_rejects_invalid_values(
    interface_separation,
    exception,
    match,
):
    gb = _build_boundary(vacuum=0.0)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)

    with pytest.raises(exception, match=match):
        manipulator.apply_interface_separation(
            manipulator.make_parent_candidate(),
            interface_separation=interface_separation,
        )


@pytest.mark.parametrize(
    "vacuum",
    [0.0, SLAB_VACUUM_ANGSTROM],
    ids=["periodic", "slab"],
)
def test_interface_separation_rejects_reapplication(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=MANIPULATOR_SEED)
    separated = manipulator.apply_interface_separation(
        manipulator.make_parent_candidate(),
        interface_separation=0.5,
    )

    with pytest.raises(GBManipulatorValueError, match="cannot be reapplied"):
        manipulator.apply_interface_separation(
            separated,
            interface_separation=0.5,
        )
