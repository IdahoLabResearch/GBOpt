# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused topology-aware termination and interface-separation regressions."""

import numpy as np
import pytest

from GBOpt.BoundarySpec import PQSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator, GBManipulatorValueError


IDENTITY_PQ = PQSpec(
    P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    basis_mode="supplied",
)


def _build_boundary(*, vacuum: float) -> GBMaker:
    return GBMaker.from_boundary_spec(
        3.615,
        "fcc",
        "Cu",
        IDENTITY_PQ,
        mode="exact",
        gb_thickness=0.0,
        repeat_factor=2,
        interaction_distance=3.615,
        x_dim_min=8.0,
        vacuum=vacuum,
    )


@pytest.mark.parametrize(
    ("vacuum", "expected"),
    [
        pytest.param(0.0, BoundaryNormalTopology.PERIODIC_BICRYSTAL, id="periodic"),
        pytest.param(2.0, BoundaryNormalTopology.SINGLE_INTERFACE_SLAB, id="slab"),
    ],
)
def test_gbmaker_exposes_topology_from_vacuum(vacuum, expected):
    gb = _build_boundary(vacuum=vacuum)

    assert gb.normal_topology is expected
    assert gb.normal_topology.periodic_outer_x_interface is (
        expected is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )


@pytest.mark.parametrize("vacuum", [0.0, 2.0], ids=["periodic", "slab"])
def test_zero_termination_cycle_is_exact_and_nonmutating(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=7)
    before = gb.whole_system.copy()

    atoms = manipulator.cycle_grain_terminations()
    candidate = manipulator.make_termination_candidate()

    assert np.array_equal(atoms, before)
    assert np.array_equal(candidate.atoms, before)
    assert np.array_equal(gb.whole_system, before)
    assert np.array_equal(
        candidate.grain_labels,
        np.hstack(
            (
                np.zeros(len(gb.left_grain), dtype=np.int8),
                np.ones(len(gb.right_grain), dtype=np.int8),
            )
        ),
    )
    assert not candidate.atoms.flags.writeable
    assert not candidate.grain_labels.flags.writeable


@pytest.mark.parametrize("vacuum", [0.0, 2.0], ids=["periodic", "slab"])
def test_termination_cycle_is_grain_local_and_preserves_population(vacuum):
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=7)
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
    assert np.allclose(left["x"], expected_left_x)
    assert np.allclose(right["x"], expected_right_x)
    assert np.allclose(right["y"], np.mod(gb.right_grain["y"] + 0.75, gb.y_dim))
    assert np.allclose(right["z"], np.mod(gb.right_grain["z"] + 1.25, gb.z_dim))
    assert np.array_equal(candidate.atoms["name"], gb.whole_system["name"])
    assert len(candidate.atoms) == len(gb.whole_system)
    assert np.array_equal(gb.whole_system, np.hstack((gb.left_grain, gb.right_grain)))


def test_periodic_interface_separation_updates_both_interface_gaps():
    gb = _build_boundary(vacuum=0.0)
    manipulator = GBManipulator(gb, seed=7)
    base = manipulator.make_translation_candidate(0.0, 0.0)
    separated = manipulator.apply_interface_separation(
        base,
        interface_separation=0.6,
    )
    split = len(gb.left_grain)

    assert separated.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert separated.periodic_outer_x_interface
    assert separated.box_dims[0, 1] == pytest.approx(gb.box_dims[0, 1] + 1.2)
    assert separated.gb_plane_x == pytest.approx(gb.gb_plane_x + 0.3)
    assert np.array_equal(separated.atoms[:split], gb.left_grain)
    assert np.allclose(separated.atoms[split:]["x"], gb.right_grain["x"] + 0.6)
    assert np.array_equal(separated.atoms[split:]["name"], gb.right_grain["name"])
    assert separated.left_grain_x_bounds.tolist() == pytest.approx(
        [gb.box_dims[0, 0], gb.gb_plane_x]
    )
    assert separated.right_grain_x_bounds.tolist() == pytest.approx(
        [gb.gb_plane_x + 0.6, gb.box_dims[0, 1] + 0.6]
    )
    assert separated.interface_separation == pytest.approx(0.6)


def test_slab_interface_separation_preserves_outer_vacuum_widths():
    vacuum = 2.0
    gb = _build_boundary(vacuum=vacuum)
    manipulator = GBManipulator(gb, seed=7)
    base = manipulator.make_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
    )
    separated = manipulator.apply_interface_separation(
        base,
        interface_separation=0.6,
    )
    split = len(gb.left_grain)

    assert separated.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert not separated.periodic_outer_x_interface
    assert separated.box_dims[0, 1] == pytest.approx(gb.box_dims[0, 1] + 0.6)
    assert separated.gb_plane_x == pytest.approx(gb.gb_plane_x + 0.3)
    assert np.array_equal(separated.atoms[:split], base.atoms[:split])
    assert np.allclose(separated.atoms[split:]["x"], base.atoms[split:]["x"] + 0.6)
    assert separated.left_grain_x_bounds[0] - separated.box_dims[0, 0] == pytest.approx(
        vacuum
    )
    assert separated.box_dims[0, 1] - separated.right_grain_x_bounds[1] == pytest.approx(
        vacuum
    )


def test_interface_separation_rejects_reapplication():
    gb = _build_boundary(vacuum=0.0)
    manipulator = GBManipulator(gb, seed=7)
    separated = manipulator.apply_interface_separation(
        manipulator.make_parent_candidate(),
        interface_separation=0.5,
    )

    with pytest.raises(GBManipulatorValueError, match="cannot be reapplied"):
        manipulator.apply_interface_separation(
            separated,
            interface_separation=0.5,
        )
